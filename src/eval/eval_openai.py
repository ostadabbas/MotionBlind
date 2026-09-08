#!/usr/bin/env python3
"""GPT-5.x eval on MotionBlind / TimeBlind-style sets via the OpenAI Chat Completions API.

Default model: gpt-5.6-luna (nano tier: $0.20/1M in, $1.20/1M out).
The models take no native video. The sampler (uniform / random / HORNet / F2C) picks N
frames, sent as base64 JPEGs in one user turn. The strategies differ only in the indices
they pick. GPT-5.6 is a reasoning model: a small output cap returns an empty string on
every item. The default is --reasoning-effort none with a reserved output budget.
The driver uses Chat Completions, not the Responses API. As of 2026-08 the Responses API
over-reports usage.output_tokens 2-20x on reasoning calls and bills the inflated number.
The reasoning_effort restriction blocks only function tools; this script sends none.

    python src/eval/eval_openai.py --base-path data --data data/data.jsonl \
        --sampling uniform --num-frames 8 --out results/gpt_5_6_luna_mbhuman_uniform8.json

Reads OPENAI_API_KEY / OPENAI_MODEL from the environment or a repo `.env`.
"""
from __future__ import annotations

# torch must be imported before decord, or the process can segfault. frame_io
# imports decord lazily. A uniform/random cell would otherwise pull decord in
# first, and the next `import torch` crashes the job.
try:
    import torch  # noqa: F401
except ImportError:      # pure-API replay envs need neither torch nor decord
    pass

import argparse
import base64
import datetime
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))

from report_run_metrics import confusion_and_f1, format_report  # noqa: E402
from scoring import (  # noqa: E402
    _load_json_list, add_question_suffix, build_answers, extract_answer, get_scores,
)

FRAME_SAMPLERS = ("uniform", "random", "hornet", "f2cfull")

# 429s that retrying cannot fix. Backing off 6 times on an exhausted balance turns a
# 1-second failure into 95 seconds per item. It also hides the cause.
FATAL_API_CODES = ("insufficient_quota", "credit_balance_exhausted", "invalid_api_key",
                   "account_deactivated", "model_not_found",
                   # A 404 means the route does not exist for this key/model, e.g. an
                   # OpenRouter ":batch" variant, which is reachable only via /api/beta/
                   # batches. Retries cannot fix a missing endpoint.
                   "NotFoundError", "Error code: 404")

# API-error reprs can embed the request (and thus the Authorization header). Those strings
# get printed to SLURM logs and stored in the predictions JSON, so scrub them at the source.
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|Bearer\s+\S+)")


def redact(text) -> str:
    return _SECRET_RE.sub("<REDACTED>", str(text))


def load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v


def parse_args():
    # allow_abbrev=False so `--api-key sk-...` cannot prefix-match
    # `--api-key-file`; it is rejected outright with guidance instead.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--model", default=None, help="OpenAI model id (default: $OPENAI_MODEL or gpt-5.6-luna)")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Environment variable holding the key (e.g. INKLING_API for a hosted "
                        "Inkling endpoint). Keeps secrets out of argv, unlike --api-key.")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint (default: $OPENAI_BASE_URL, else OpenAI). "
                        "Point this at a self-hosted vLLM server to run Inkling or any other "
                        "OpenAI-API model through the same sampling/replay machinery.")
    p.add_argument("--api-key", default=None, help=argparse.SUPPRESS)   # trap, see below
    p.add_argument(
        "--api-key-file",
        default=None,
        help="Read the key from a file (chmod 600). Prefer this or $OPENAI_API_KEY over any "
        "flag that puts the secret on a command line. `ps` exposes argv to every other "
        "user on a shared node.",
    )
    p.add_argument("--base-path", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--sampling", default="uniform", choices=list(FRAME_SAMPLERS),
                   help="f2cfull = paper Frames-to-Clips (arXiv:2510.02262).")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", default="none",
                   choices=["none", "no_video", "shuffled_frames", "reversed_frames"])
    p.add_argument("--prompt-template", default="default",
                   choices=["default", "framed", "bare", "cot"],
                   help="default/framed = prefixed with 'The images below are frames from "
                        "one video, in temporal order.' (what every existing result used); "
                        "bare = the question alone; cot = framed plus describe-then-answer.")
    p.add_argument("--system-prompt", default="none",
                   choices=["none", "debias", "temporal", "both"],
                   help="System message prepended to the chat. 'none' = no system message "
                        "(matches every existing OpenAI result). 'debias' = anti-yes-bias, "
                        "'temporal' = attend to event order/timing, 'both' = combined. "
                        "Text is identical to eval_motion.py/eval_qwen3.py so the lever is "
                        "comparable across open and API models. This is a separate "
                        "experiment axis from --prompt-template; hold frames fixed when "
                        "sweeping it.")
    p.add_argument(
        "--detail",
        default="low",
        choices=["low", "high", "original", "auto"],
        help="Image detail bound. GPT-5.x bills by 32px patches, so resolution costs "
        "tokens at every level: low fits 512x512 (~173 tok for these frames), high fits "
        "2048x2048 capped at 2500 patches (~404 tok), original/auto are uncapped.",
    )
    p.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["unset", "none", "low", "medium", "high", "xhigh", "max"],
        help="GPT-5.x reasoning budget. Default 'none': this is a yes/no classification "
        "task and CoT lowers MotionBlind I_Acc. Reasoning tokens bill as "
        "output at 6x the input rate. Anything above 'none' needs a big --max-tokens. "
        "Use 'unset' to omit the parameter entirely; self-hosted vLLM servers reject "
        "reasoning_effort unless the model's parser supports it.",
    )
    p.add_argument("--token-param", default="max_completion_tokens",
                   choices=["max_completion_tokens", "max_tokens"],
                   help="Output-cap parameter name. GPT-5.x reasoning models require "
                        "max_completion_tokens; Tinker and most vLLM servers want max_tokens.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Omitted by default (GPT-5.x reasoning models reject it). Set 0.0 for "
                        "a vLLM-served model where greedy decoding is wanted.")
    p.add_argument("--max-side", type=int, default=768, help="Longest JPEG side sent to the API.")
    p.add_argument(
        "--selections",
        default=None,
        help="Replay a frozen selections file. Works for every "
        "sampler, needs no GPU/checkpoint/CLIP, and makes the run bit-reproducible.",
    )
    # --- HORNet: replay a recorded selection (no GPU) or run the selector live (GPU) ---
    p.add_argument(
        "--hornet-selections",
        default=None,
        help="Recorded-selections JSON: {video_path: {selected: [pool positions], pool: N}}. "
        "Replays that selection at full resolution: no repo, checkpoint, or GPU needed.",
    )
    p.add_argument("--hornet-pool", type=int, default=32, help="Candidate pool size (standard 32).")
    p.add_argument("--hornet-repo", default=None, help="Live selection: cloned HORNet repo.")
    p.add_argument("--hornet-ckpt", default=None, help="Live selection: checkpoint .pt.")
    p.add_argument("--hornet-load", default="full", choices=["full", "no-encoder"])
    p.add_argument("--hornet-fallback", action="store_true",
                   help="On a missing/failed selection, fall back to uniform-N instead of erroring.")
    # --- F2C (Frames-to-Clips, arXiv:2510.02262) ---
    p.add_argument("--f2c-clip", default=None, help="Relevance encoder (default: SigLIP2 for f2cfull).")
    p.add_argument("--f2c-pool", type=int, default=128, help="Unused; kept so older sbatch env still parses.")
    p.add_argument("--f2c-s-max", type=float, default=2.0, help="f2cfull max resolution scale.")
    p.add_argument("--f2c-lambda-r", type=float, default=0.5, help="f2cfull relevance weight (paper 0.5).")
    p.add_argument("--f2c-lambda-l", type=float, default=0.05, help="f2cfull length weight (paper 0.05).")
    p.add_argument("--log-file", default=None,
                   help="Append the full run log here as well as stdout. Point every cell of a "
                        "sweep at one path to get a single combined log file.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-item blocks; keep only the header and summary.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--check-api",
        action="store_true",
        help="Send one tiny request (text + a 1px image), print the token usage, and exit. "
        "Confirms the key works and the model accepts image input. Costs ~$0.0001.",
    )
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=4, help="Parallel API calls.")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Minimum seconds between API requests, enforced globally across "
                        "all worker threads. Use this on a low-tier key where the 429 "
                        "backoff would otherwise thrash. Wall time becomes roughly "
                        "n_items * sleep, so price it before setting it high.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="max_completion_tokens. Default 32 at effort=none, else 25000. "
                        "Reasoning models spend the budget before emitting text; too small "
                        "a cap returns an empty string.")
    return p.parse_args()


_log_lock = threading.Lock()
_log_fh = None

# --- global request spacer -------------------------------------------------------------
# Paces request starts across every worker thread, not per thread; otherwise
# concurrency multiplies the real rate. The lock is held across the sleep on
# purpose: blocking the other workers is the mechanism. Retries inside
# GPTClient.generate use their own exponential backoff instead.
_rate_lock = threading.Lock()
_next_slot = [0.0]


def rate_gate(min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _rate_lock:
        wait = _next_slot[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_slot[0] = time.monotonic() + min_interval


def log(msg: str = "") -> None:
    """stdout + optional shared log file, atomically (worker threads call this)."""
    with _log_lock:
        print(msg, flush=True)
        if _log_fh:
            _log_fh.write(msg + "\n")
            _log_fh.flush()


def format_item(i: int, total: int, rec: dict, sample: dict, eta_s: float) -> str:
    """The repo's established per-item block (see any eval_motion/eval_eagle .out)."""
    gt = sample.get("answer")
    pred = extract_answer(rec.get("model_output") or "", sample.get("type") or "yes_no")
    truth = extract_answer(str(gt), sample.get("type") or "yes_no")
    mark = "OK" if (pred == truth and pred in (0, 1)) else "X"
    lines = [
        "=" * 70,
        f"[{i}/{total}] {os.path.basename(sample['video_path'])} | "
        f"frames: {rec.get('n_frames_sent', 0)} | GT: {gt} | {mark} | ETA: {eta_s:.0f}s",
        f"Q: {rec.get('question', '')}",
        f"OUT: {(rec.get('model_output') or '').strip()}",
    ]
    if rec.get("error"):
        lines.append(f"ERROR: {rec['error']}")
    return "\n".join(lines) + "\n"


def format_summary(meta: dict, scores: dict, n: int, elapsed_s: float) -> str:
    return "\n".join([
        "",
        "=" * 50,
        f"MODEL: {meta['model_name']} | SAMPLING: {meta['sampling']} | "
        f"FRAMES: {meta['num_frames']} | PROMPT: {meta['prompt_template']}",
        f"Time: {elapsed_s / 60:.1f}min | Samples: {n}",
        "=" * 50,
        f"  Q_Acc: {scores['Q_Acc'] * 100:.1f}%",
        f"  V_Acc: {scores['V_Acc'] * 100:.1f}%",
        f"  Acc: {scores['Acc'] * 100:.1f}%",
        f"  I_Acc: {scores['I_Acc'] * 100:.1f}%",
    ])


from eval_common import make_fix_path, question_of  # noqa: E402


# System messages for the --system-prompt lever. Copied verbatim from eval_motion.py
# (eval_qwen3.py and eval_gemma4.py carry byte-identical copies), so a debias or
# temporal cell measures the same intervention on GPT as on the open models.
# Keep the four strings in sync if any driver's copy changes.
#
# `none` sends no system message at all, which is what every existing OpenAI result
# used; on the open drivers `none` likewise means the model's built-in default. The
# others target the two things that cap I_Acc on TimeBlind/MotionBlind: yes/no
# compliance bias (debias) and failure to read event order/timing (temporal).
SYSTEM_PROMPTS = {
    "none": None,
    "debias": (
        "You are a precise video analyst. Base your answer only on what the video "
        "actually shows. Do not assume a statement is true \u2014 answer No (or the other "
        "option) whenever the video does not clearly support it."
    ),
    "temporal": (
        "You are a precise video analyst. Pay close attention to the order, timing, and "
        "direction of events across frames. Two videos can look similar yet differ in how "
        "the action unfolds over time; judge based on that temporal evidence."
    ),
    "both": (
        "You are a precise video analyst. Pay close attention to the order, timing, and "
        "direction of events across frames \u2014 two videos can look similar yet differ in how "
        "the action unfolds over time. Base your answer only on what the video actually "
        "shows, and do not assume a statement is true: answer No (or the other option) "
        "whenever the video does not clearly support it."
    ),
}


def build_prompt(question: str, task_type: str, template: str) -> str:
    """Templates, and why `default` stays framed.

    `default` == `framed`: prefixed with "The images below are frames from one video, in
    temporal order." Every OpenAI result in this repo was produced that way. Redefining
    `default` to mean something else would mix two prompts inside one resumed cell
    and make archived runs ambiguous.

    `bare` is the preamble-free variant, opted into explicitly. A launcher may splice
    the template name into the output filename, so a bare run lands in its own files and
    is directly comparable rather than blended.

    Removing the preamble is not cosmetic: gpt-4o answers "I can't determine the speed of
    stirring from these images" without it, because it has no reason to read the images as
    one temporal sequence.
    """
    q = question_of({"question": question, "type": task_type})
    if template == "cot":
        return (
            "The images below are frames from one video, in temporal order. Watch them "
            "carefully. Describe any motion or change you notice, then answer the question.\n\n" + q
        )
    if template == "bare":
        return q
    return "The images below are frames from one video, in temporal order. " + q


def jpeg_data_url(pil_img, max_side: int) -> str:
    img = pil_img.convert("RGB")
    if max_side and max(img.size) > max_side:
        from PIL import Image

        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------------------
# frame selection
# --------------------------------------------------------------------------------------
def init_frame_selector(args):
    """Return pick(vpath, key, question, task_type) -> list[PIL] for the chosen sampler.

    Only F2C uses `question`; it is the one query-conditioned sampler here. uniform,
    random and HORNet are query-blind (HORNet's forward never sees the text), which
    is the axis this sweep tests.
    """
    from frame_io import hornet_pool_indices, sample_random_frames, sample_uniform_frames, _read_frames

    if args.selections:
        blob = json.load(open(args.selections))
        meta_sel, items = blob["meta"], blob["items"]
        if meta_sel["sampler"] != args.sampling or meta_sel["num_frames"] != args.num_frames:
            sys.exit(f"--selections is {meta_sel['sampler']}{meta_sel['num_frames']}, "
                     f"but --sampling {args.sampling} --num-frames {args.num_frames} was requested")
        if meta_sel.get("max_side") != args.max_side:
            print(f"[warn] selections priced at max_side={meta_sel.get('max_side')} "
                  f"but sending max_side={args.max_side}", flush=True)
        qcond = meta_sel.get("query_conditioned", False)
        print(f"[replay] {args.selections}: {len(items)} entries, "
              f"query_conditioned={qcond}, frames {meta_sel['frames_sent']['min']}-"
              f"{meta_sel['frames_sent']['max']}", flush=True)

        def pick(vpath, key, question, task_type, _idx=None):
            rec = items.get(str(_idx)) if qcond else items.get(key)
            if rec is None or not rec["indices"]:
                raise KeyError(f"no recorded selection for {'index ' + str(_idx) if qcond else key}")
            frames = _read_frames(vpath, rec["indices"])
            scales = rec.get("scales")
            if scales:                      # f2cfull: resolution is part of the method
                from PIL import Image

                frames = [
                    f if sc <= 1.0 + 1e-6 else
                    f.resize((max(1, round(f.size[0] / sc)), max(1, round(f.size[1] / sc))),
                             Image.BICUBIC)
                    for f, sc in zip(frames, scales)
                ]
            return frames

        return pick

    if args.sampling == "uniform":
        return lambda vpath, key, q, tt, _idx=None: sample_uniform_frames(vpath, args.num_frames)
    if args.sampling == "random":
        return lambda vpath, key, q, tt, _idx=None: sample_random_frames(vpath, args.num_frames, args.seed)

    if args.sampling == "f2cfull":
        from f2c_sampling import load_f2c_full

        clip_id = args.f2c_clip or "google/siglip2-base-patch16-224"
        f2c = load_f2c_full(clip_model=clip_id, s_max=args.f2c_s_max,
                            lambda_r=args.f2c_lambda_r, lambda_l=args.f2c_lambda_l, seed=args.seed)
        print(f"[f2cfull] encoder={clip_id} s_max={args.f2c_s_max} "
              f"lambda_r={args.f2c_lambda_r} lambda_l={args.f2c_lambda_l}", flush=True)
        # No GPT-4o-style warning here: GPT-5.x bills by patch count, so resolution
        # costs tokens at every detail level and F2C-full's scale `s` is meaningful.

        f2c_lock = threading.Lock()      # one encoder, N API worker threads

        def pick(vpath, key, question, task_type, _idx=None):
            with f2c_lock:
                return f2c.select(vpath, question_of({"question": question, "type": task_type}), args.num_frames)[0]

        return pick

    # --- hornet ---
    if args.hornet_selections:
        sel_map = json.load(open(args.hornet_selections))
        n_ok = sum(1 for v in sel_map.values() if "selected" in v)
        print(f"[hornet] replaying {n_ok}/{len(sel_map)} recorded selections "
              f"from {args.hornet_selections} (pool={args.hornet_pool})", flush=True)

        def pick(vpath, key, question, task_type, _idx=None):
            rec = sel_map.get(key) or sel_map.get(os.path.basename(key))
            if not rec or "selected" not in rec:
                if args.hornet_fallback:
                    print(f"[hornet] no selection for {key}; uniform-{args.num_frames}", flush=True)
                    return sample_uniform_frames(vpath, args.num_frames)
                raise KeyError(f"no HORNet selection recorded for {key}")
            pool = int(rec.get("pool") or args.hornet_pool)
            pool_idx = hornet_pool_indices(vpath, pool)
            chosen = sorted(rec["selected"])[: args.num_frames]     # pool positions, temporal order
            return _read_frames(vpath, [pool_idx[i] for i in chosen])

        return pick

    if not (args.hornet_repo and args.hornet_ckpt):
        sys.exit("--sampling hornet needs --hornet-selections (recorded) "
                 "or --hornet-repo + --hornet-ckpt (live, GPU)")

    import torch

    sys.path.insert(0, args.hornet_repo)
    from lmms_eval_utils.hornet import (  # noqa: E402
        VisionGRPOPolicy,
        get_action_by_k as hornet_action,
        load_frames as hornet_load_frames,
    )

    class _StubVLM:
        """The keep_prob forward never touches the answering VLM."""

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = VisionGRPOPolicy(None, 768, 1, _StubVLM(), None).to(dev)
    state = torch.load(args.hornet_ckpt, map_location=dev)
    if args.hornet_load == "no-encoder":
        state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
    res = net.load_state_dict(state, strict=False)
    loaded = len(state) - len(res.unexpected_keys)
    print(f"[hornet] live load={args.hornet_load} matched {loaded}/{len(state)} tensors (device={dev})",
          flush=True)
    if loaded != len(state):
        sys.exit(f"[hornet] key mismatch unexpected[:6]={res.unexpected_keys[:6]}")
    net.eval()
    net_lock = threading.Lock()     # one shared module, N API worker threads

    def pick(vpath, key, question, task_type, _idx=None):
        try:
            videos, _total = hornet_load_frames(vpath)
            with net_lock, torch.no_grad():
                kp = net(videos.unsqueeze(0).to(dev))["keep_prob"][0]
            k = min(args.num_frames, kp.shape[0])
            act = hornet_action(kp.unsqueeze(0), 1, k, random_sample=False)
            chosen = sorted(torch.nonzero(act[0][0]).squeeze(-1).tolist())
            pool_idx = hornet_pool_indices(vpath, int(kp.shape[0]))
            return _read_frames(vpath, [pool_idx[i] for i in chosen])
        except Exception as e:
            if not args.hornet_fallback:
                raise
            print(f"[hornet] fallback uniform-{args.num_frames} ({e})", flush=True)
            return sample_uniform_frames(vpath, args.num_frames)

    return pick


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
class GPTClient:
    def __init__(self, api_key: str, model: str, max_tokens: int, reasoning_effort: str = "none",
                 base_url: str | None = None, temperature: float | None = None,
                 token_param: str = "max_completion_tokens", system: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.temperature = temperature
        self.token_param = token_param
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system = system
        self.empty_responses = 0

    def generate(self, prompt: str, data_urls, detail: str, retries: int):
        """Return (text, prompt_tokens). prompt_tokens is what the sweep normalizes on:
        f2cfull spends a token budget, not a frame budget."""
        content = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        # Omit the system turn entirely when there is none, rather than sending an empty
        # one: an empty system message is not the same request as no system message, and
        # every archived OpenAI result was produced with no system turn.
        messages = ([{"role": "system", "content": self.system}] if self.system else []) \
            + [{"role": "user", "content": content}]
        last = None
        for attempt in range(retries):
            try:
                # Reasoning models take max_completion_tokens (max_tokens is rejected) and
                # generally refuse a custom temperature, so the request sends neither.
                kw = {}
                if self.reasoning_effort != "unset":
                    kw["reasoning_effort"] = self.reasoning_effort
                if self.temperature is not None:
                    kw["temperature"] = self.temperature
                kw[self.token_param] = self.max_tokens
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kw,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                usage = getattr(resp, "usage", None)
                if text:
                    return text, int(getattr(usage, "prompt_tokens", 0) or 0)
                # Empty + finish_reason=length means reasoning spent the whole budget
                # before emitting a token. An identical retry cannot help; fail with the reason.
                self.empty_responses += 1
                fin = getattr(choice, "finish_reason", None)
                rt = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
                if fin == "length":
                    raise RuntimeError(
                        f"empty response: finish_reason=length, reasoning_tokens={rt}, "
                        f"max_completion_tokens={self.max_tokens}. Raise --max-tokens or "
                        f"lower --reasoning-effort (currently {self.reasoning_effort})."
                    )
                last = RuntimeError(f"empty response (finish_reason={fin}, reasoning_tokens={rt})")
            except Exception as e:
                last = e
                msg = redact(repr(e))
                hit = next((c for c in FATAL_API_CODES if c in msg), None)
                if hit:
                    raise RuntimeError(f"fatal API error ({hit}); not retrying: {msg}") from e
                wait = min(60.0, 1.5 * (2**attempt))
                log(f"[warn] attempt {attempt + 1}/{retries}: {msg}; sleep {wait:.1f}s")
                time.sleep(wait)
        raise last


def main():
    load_dotenv(os.path.join(REPO, ".env"))
    args = parse_args()
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "gpt_5_6_luna")
    if args.api_key:
        sys.exit("Refusing --api-key: a secret on the command line is visible to every "
                 "user on the node via `ps`. Export OPENAI_API_KEY, or use --api-key-file.")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if base_url and not api_key:
        api_key = "EMPTY"          # self-hosted vLLM ignores the key but the SDK requires one
    if args.api_key_file:
        with open(os.path.expanduser(args.api_key_file)) as f:
            api_key = f.read().strip()
    if not api_key:
        sys.exit(f"Set ${args.api_key_env} in the environment or a repo .env, or pass "
                 "--api-key-file. (No --api-key flag: `ps` would leak it via argv.)")
    model_name = args.model or os.environ.get("OPENAI_MODEL") or (
        None if base_url else "gpt-5.6-luna")
    if not model_name:
        sys.exit("--model is required with --base-url (use the id the server registered)")
    # Reasoning burns output budget before any visible token. A small cap that suits
    # GPT-4o returns empty on every item and scores the whole run invalid.
    if args.max_new_tokens:
        max_tokens = args.max_new_tokens
    elif args.reasoning_effort == "none":
        max_tokens = 512 if args.prompt_template == "cot" else 32
    else:
        max_tokens = 25000        # OpenAI's recommended reserve for reasoning + output

    if args.check_api:
        import base64 as _b64
        import io as _io

        from PIL import Image as _Image

        buf = _io.BytesIO()
        _Image.new("RGB", (8, 8), (128, 128, 128)).save(buf, format="JPEG")
        url = "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
        client = GPTClient(api_key, model_name, max_tokens, args.reasoning_effort,
                           base_url, args.temperature, args.token_param,
                           SYSTEM_PROMPTS[args.system_prompt])
        t0 = time.time()
        text, ptok = client.generate("Reply with the single word OK.", [url], "low", 2)
        print(f"[check-api] model={model_name} OK in {time.time() - t0:.1f}s | "
              f"reply={text!r} | prompt_tokens={ptok}")
        print("[check-api] key valid and the model accepts image input.")
        return

    missing = [f"--{f}" for f in ("base-path", "data", "out")
               if getattr(args, f.replace("-", "_")) is None]
    if missing:
        sys.exit(f"missing required args: {' '.join(missing)}")
    if args.num_frames is None:
        sys.exit("--num-frames is required")
    if args.ablation in ("shuffled_frames", "reversed_frames") and args.num_frames < 2:
        sys.exit(f"--ablation {args.ablation} needs --num-frames >= 2")

    fix_path = make_fix_path(args.base_path)
    data = _load_json_list(args.data)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    pick_frames = None if args.ablation == "no_video" else init_frame_selector(args)

    client = GPTClient(api_key, model_name, max_tokens, args.reasoning_effort,
                       base_url, args.temperature, args.token_param,
                       SYSTEM_PROMPTS[args.system_prompt])

    global _log_fh
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)) or ".", exist_ok=True)
        _log_fh = open(args.log_file, "a", encoding="utf-8")

    log(f"Model loaded: {model_name}")
    log(f"Endpoint: {base_url or 'https://api.openai.com/v1 (default)'}")
    log(f"Config: sampling={args.sampling} frames={args.num_frames} detail={args.detail} "
        f"max_side={args.max_side} ablation={args.ablation} prompt={args.prompt_template} "
        f"system_prompt={args.system_prompt} "
        f"seed={args.seed} concurrency={args.concurrency} sleep={args.sleep}s "
        f"reasoning_effort={args.reasoning_effort} max_completion_tokens={max_tokens}")
    log(f"Loaded {len(data)} samples")
    p0 = fix_path(data[0]["video_path"])
    log(f"Path check: {p0}")
    log(f"Exists: {os.path.exists(p0)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    log(f"Resuming from {len(done)} existing predictions")

    meta = {
        "model_name": model_name,
        "sampling": args.sampling,
        "num_frames": args.num_frames,
        "detail": args.detail,
        "max_side": args.max_side,
        "seed": args.seed,
        "ablation": args.ablation,
        "prompt_template": args.prompt_template,
        "system_prompt": args.system_prompt,
        "reasoning_effort": args.reasoning_effort,
        "sleep": args.sleep,
        "base_url": base_url,
        "temperature": args.temperature,
        "max_completion_tokens": max_tokens,
        "selections": args.selections,
        "hornet_selections": args.hornet_selections if args.sampling == "hornet" else None,
        "hornet_pool": args.hornet_pool if args.sampling == "hornet" else None,
        "hornet_ckpt": args.hornet_ckpt if args.sampling == "hornet" else None,
        "hornet_load": args.hornet_load if args.sampling == "hornet" else None,
        "f2c_encoder": (
            (args.f2c_clip or "google/siglip2-base-patch16-224") if args.sampling == "f2cfull" else None
        ),
        "f2c_pool": None,
        "f2c_s_max": args.f2c_s_max if args.sampling == "f2cfull" else None,
        "f2c_lambda_r": args.f2c_lambda_r if args.sampling == "f2cfull" else None,
        "f2c_lambda_l": args.f2c_lambda_l if args.sampling == "f2cfull" else None,
    }

    results, lock, t0 = {}, threading.Lock(), time.time()
    # A dead key is fatal for the whole cell, not for one item. Without this guard,
    # a run writes one identical error row per item across every budget. The first
    # fatal code stops the cell instead.
    abort = {"reason": None}
    todo = []
    for sample in data:
        prev = done.get(sample["index"])
        if prev and prev.get("video_path") == sample["video_path"]:
            results[sample["index"]] = prev
        else:
            todo.append(sample)

    if args.sleep > 0 and todo:
        secs = len(todo) * args.sleep
        log(f"[pace] {args.sleep}s between requests x {len(todo)} to do = "
            f"{secs / 60:.0f} min floor ({secs / 3600:.1f} h) before any API latency. "
            f"If the wall clock is shorter than that, the cell resumes on resubmit.")

    by_index = {d["index"]: d for d in data}
    state = {"printed": 0}

    def flush():
        # Atomic: a torn write is unrecoverable, because the resume path parses
        # this file and falls back to `done = {}`, redoing the whole cell.
        tmp = f"{args.out}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump([results[d["index"]] for d in data if d["index"] in results], f, indent=2)
        os.replace(tmp, args.out)

    def drain_log():
        """Emit item blocks in dataset order even though workers finish out of order:
        advance a pointer over `data` and print every contiguous completed item."""
        if args.quiet:
            return
        while state["printed"] < len(data):
            sample = data[state["printed"]]
            rec = results.get(sample["index"])
            if rec is None:
                break
            n_done = state["printed"] + 1
            elapsed = time.time() - t0
            rate = elapsed / max(1, n_done - (len(data) - len(todo)))
            eta = max(0.0, rate * (len(data) - n_done))
            log(format_item(n_done, len(data), rec, sample, eta))
            state["printed"] += 1

    def run_one(sample):
        if abort["reason"]:
            return
        q = build_prompt(sample["question"], sample["type"], args.prompt_template)
        vpath = fix_path(sample["video_path"])
        err, out, prompt_tokens = None, "", 0
        try:
            urls = []
            if args.ablation != "no_video":
                if not os.path.exists(vpath):
                    raise FileNotFoundError(vpath)
                frames = pick_frames(vpath, sample["video_path"], sample["question"], sample["type"], sample["index"])
                if args.ablation == "reversed_frames" and len(frames) > 1:
                    from frame_io import reverse_frames
                    frames = reverse_frames(frames)
                if args.ablation == "shuffled_frames" and len(frames) > 1:
                    from frame_io import shuffle_frames

                    frames = shuffle_frames(frames, vpath, args.seed)
                urls = [jpeg_data_url(fr, args.max_side) for fr in frames]
            rate_gate(args.sleep)
            out, prompt_tokens = client.generate(q, urls, args.detail, args.retries)
        except Exception as e:
            err = redact(repr(e))
            hit = next((c for c in FATAL_API_CODES if c in err), None)
            if hit:
                with lock:
                    if not abort["reason"]:
                        abort["reason"] = hit
                        log(f"[fatal] {hit}; stopping this cell. Remaining items are not "
                            f"attempted and nothing is written for them, so a rerun after "
                            f"the fix resumes cleanly.")
                return          # no record: resume must retry this item, not skip it
            log(f"[error] index={sample['index']}: {redact(e)}")

        rec = {**meta, "index": sample["index"], "video_path": sample["video_path"],
               "question": q, "model_output": out, "error": err,
               "n_frames_sent": len(urls), "prompt_tokens": prompt_tokens,
               "ts": datetime.datetime.now().isoformat()}
        with lock:
            results[sample["index"]] = rec
            drain_log()
            if len(results) % 20 == 0 or len(results) == len(data):
                flush()

    with lock:
        drain_log()          # replay anything already done (resume) before new work
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            list(pool.map(run_one, todo))
    flush()

    predictions = [results[d["index"]] for d in data if d["index"] in results]
    with lock:
        drain_log()
    log(f"Wrote {len(predictions)} predictions -> {args.out}")

    if abort["reason"]:
        # Non-zero on purpose: a grid loop must not continue to the next budget on a
        # dead key. Do not score a partial cell as if it were whole.
        sys.exit(f"aborted after {len(predictions)}/{len(data)} items: {abort['reason']}")

    if any((p.get("model_output") or "").strip() for p in predictions):
        scores = get_scores(build_answers(predictions, data))
        log(format_summary(meta, scores, len(predictions), time.time() - t0))
        cm = confusion_and_f1(predictions, data)
        log("")
        log(format_report(cm))
        sent = [p.get("n_frames_sent", 0) for p in predictions if not p.get("error")]
        toks = [p.get("prompt_tokens", 0) for p in predictions if not p.get("error")]
        realized = {
            "frames_sent_mean": round(sum(sent) / len(sent), 2) if sent else 0,
            "frames_sent_min": min(sent) if sent else 0,
            "frames_sent_max": max(sent) if sent else 0,
            "prompt_tokens_mean": round(sum(toks) / len(toks), 1) if toks else 0,
            "prompt_tokens_total": sum(toks),
        }
        log(f"[realized] frames/item {realized['frames_sent_min']}-{realized['frames_sent_max']} "
              f"(mean {realized['frames_sent_mean']}) | prompt tokens total {realized['prompt_tokens_total']:,} "
              f"(mean {realized['prompt_tokens_mean']})")
        if realized["frames_sent_min"] != realized["frames_sent_max"]:
            log(f"[realized] NOTE: variable frame count; this cell is not frame-matched to "
                  f"uniform/random-{args.num_frames}. Compare on prompt_tokens.")
        sidecar = os.path.splitext(args.out)[0] + "_metrics.json"
        json.dump({**scores, "realized": realized,
                   "F1_yesno_pct": round(cm["F1"] * 100, 2),
                   "yesno_Acc_pct": round(cm["Acc"] * 100, 2),
                   "yes_rate_pct": round(cm["yes_rate"] * 100, 2),
                   "confusion": {k: cm[k] for k in ("TP", "FP", "FN", "TN", "invalid", "n_yesno")},
                   "config": meta},
                  open(sidecar, "w"), indent=2)
        log(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
