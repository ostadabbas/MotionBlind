#!/usr/bin/env python3
"""Gemini 3 Pro eval on MotionBlind / TimeBlind via the official Gemini API.

TimeBlind paper protocol: proprietary models through the official API, videos at 1 FPS,
zero-shot. This script uploads the mp4 (default `--sampling video`) or sends an explicit
frame list (uniform / random / HORNet / paper F2C) for harness-matched sweeps.

  python eval_gemini.py \
      --base-path data \
      --data data/data.jsonl \
      --out results/gemini_3_pro_motionblind_default_video.json

Reads GEMINI_API_KEY / GOOGLE_API_KEY / GEMINI_MODEL (Google backend) or
OPENROUTER_API_KEY / OPENROUTER_MODEL (OpenRouter backend) from the environment or `.env`.
Frame-list mode on OpenRouter bypasses Google's requests-per-day cap.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import time

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
from scoring import _load_json_list, add_question_suffix, build_answers, get_scores  # noqa: E402


PAPER_MODELS = ("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-pro")
RETIRED_MODELS = {"gemini-3-pro-preview": "gemini-3.1-pro-preview"}


class QuotaExhausted(Exception):
    """Raised when the API rejects a call with a rate/quota (429) error that persists.

    The runner treats this as a soft stop: checkpoint what's done and exit 0.
    A later resume (e.g. after the requests-per-day quota resets) picks up the rest.
    """


def _is_quota_error(e: Exception) -> bool:
    s = repr(e).lower()
    return any(t in s for t in ("429", "resource_exhausted", "resourceexhausted", "quota", "rate limit", "ratelimit"))


def _retry_after_seconds(e: Exception):
    """Best-effort parse of a server-suggested retry delay (e.g. 'retryDelay': '31s')."""
    import re

    m = re.search(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", repr(e), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


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
    # Accept either OPENROUTER_API_KEY or the lowercase name used in some .env files.
    if not os.environ.get("OPENROUTER_API_KEY") and os.environ.get("openrouter_api_key"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["openrouter_api_key"].strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Gemini model id (default: $GEMINI_MODEL or gemini-3.1-pro-preview)")
    p.add_argument("--api-key", default=None, help="Overrides GEMINI_API_KEY / GOOGLE_API_KEY")
    p.add_argument(
        "--backend",
        default="google",
        choices=["google", "openrouter"],
        help="google = official Gemini API (supports native video). openrouter = OpenAI-compatible "
        "OpenRouter endpoint (frame-list / no_video only; paid variant bypasses the Gemini RPD cap).",
    )
    p.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high"],
        help="OpenRouter-only. Gemini 3.1 Pro cannot disable thinking (mandatory); "
        "none/minimal are mapped down to low, the lowest supported level.",
    )
    p.add_argument("--base-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument(
        "--sampling",
        default="video",
        choices=["video", "uniform", "random", "hornet", "f2cfull"],
        help="video = upload mp4 (paper-style). Others = explicit N-frame JPEG list "
        "(f2cfull = paper F2C; hornet uses a local VLM encoder only to rank frames).",
    )
    p.add_argument("--fps", type=float, default=1.0, help="Video FPS hint for --sampling video (paper = 1).")
    p.add_argument("--num-frames", type=int, default=None, help="Required for uniform/random/hornet/f2cfull.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", default="none", choices=["none", "no_video", "shuffled_frames", "reversed_frames"])
    p.add_argument("--hornet-ckpt", default=None)
    p.add_argument("--hornet-repo", default=None)
    p.add_argument("--hornet-load", default="full", choices=["full", "no-encoder"])
    p.add_argument("--hornet-fallback", action="store_true")
    p.add_argument("--hornet-min-std", type=float, default=0.0)
    p.add_argument(
        "--hornet-backbone",
        default="allenai/Molmo2-8B",
        help="Local VLM whose vision encoder hosts HORNet (Gemini has no local encoder).",
    )
    p.add_argument("--f2c-clip", default=None)
    p.add_argument("--f2c-pool", type=int, default=128)
    p.add_argument("--f2c-s-max", type=float, default=2.0)
    p.add_argument("--f2c-lambda-r", type=float, default=0.5)
    p.add_argument("--f2c-lambda-l", type=float, default=0.05)
    p.add_argument("--prompt-template", default="default", choices=["default", "cot"])
    p.add_argument(
        "--thinking-level",
        default="low",
        choices=["low", "medium", "high"],
        help="Gemini 3.1 Pro cannot disable thinking; low is the closest to no-CoT "
        "(MotionBlind CoT hurts I_Acc). The API default of high costs extra tokens and reasoning.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--sleep", type=float, default=0.4, help="Seconds between successful calls.")
    p.add_argument("--max-side", type=int, default=768, help="Longest JPEG side for frame-list mode.")
    p.add_argument(
        "--requests-cap",
        type=int,
        default=(int(os.environ["GEMINI_REQUESTS_CAP"]) if os.environ.get("GEMINI_REQUESTS_CAP") else None),
        help="Max new successful generate calls this run, then checkpoint + exit 0 (resume "
        "later). Use to stay under a requests-per-day quota. Env: GEMINI_REQUESTS_CAP.",
    )
    p.add_argument(
        "--quota-retries",
        type=int,
        default=4,
        help="How many times to wait-and-retry a 429/quota error before soft-stopping the run.",
    )
    p.add_argument(
        "--quota-wait-cap",
        type=float,
        default=120.0,
        help="Max seconds to sleep on a single 429/quota backoff (RPM limits recover; RPD won't).",
    )
    return p.parse_args()


from eval_common import make_fix_path, question_of  # noqa: E402


def build_prompt(question: str, task_type: str, template: str) -> str:
    q = question_of({"question": question, "type": task_type})
    if template == "cot":
        return (
            "Watch the video carefully. Describe any motion or change you notice, "
            "then answer the question.\n\n" + q
        )
    return q


def jpeg_bytes(pil_img, max_side: int) -> bytes:
    img = pil_img.convert("RGB")
    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        from PIL import Image

        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class GeminiClient:
    def __init__(self, api_key: str, model: str, thinking_level: str = "low",
                 quota_retries: int = 4, quota_wait_cap: float = 120.0):
        self.model_name = model
        self.thinking_level = thinking_level
        self.quota_retries = quota_retries
        self.quota_wait_cap = quota_wait_cap
        self.uploaded = {}
        self._kind = None
        try:
            from google import genai
            from google.genai import types

            self._kind = "new"
            self._types = types
            self._client = genai.Client(api_key=api_key)
        except Exception:
            import google.generativeai as genai

            self._kind = "old"
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(model)

    def upload_video(self, path: str):
        if path in self.uploaded:
            return self.uploaded[path]
        if self._kind == "new":
            f = self._client.files.upload(file=path)
            for _ in range(60):
                info = self._client.files.get(name=f.name)
                state = getattr(getattr(info, "state", None), "name", None) or str(getattr(info, "state", ""))
                if str(state).endswith("ACTIVE") or str(state) in ("ACTIVE", "2"):
                    self.uploaded[path] = info
                    return info
                if "FAIL" in str(state).upper():
                    raise RuntimeError(f"Gemini file processing failed: {state}")
                time.sleep(2)
            self.uploaded[path] = f
            return f
        f = self._genai.upload_file(path=path)
        for _ in range(60):
            f = self._genai.get_file(f.name)
            if f.state.name == "ACTIVE":
                self.uploaded[path] = f
                return f
            if f.state.name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {f.state.name}")
            time.sleep(2)
        self.uploaded[path] = f
        return f

    def generate(self, parts, retries: int, *, images: bool = False) -> str:
        if self._kind == "new":
            norm = []
            for p in parts:
                if isinstance(p, dict) and "data" in p:
                    norm.append(self._types.Part.from_bytes(data=p["data"], mime_type=p["mime_type"]))
                else:
                    norm.append(p)
            parts = norm
        last = None
        quota_hits = 0
        for attempt in range(retries):
            try:
                if self._kind == "new":
                    think_kw = {}
                    try:
                        think_kw["thinking_config"] = self._types.ThinkingConfig(
                            thinking_level=self.thinking_level,
                            include_thoughts=False,
                        )
                    except TypeError:
                        think_kw["thinking_config"] = self._types.ThinkingConfig(
                            thinking_level=self.thinking_level,
                        )
                    extra = {}
                    if images:
                        try:
                            extra["media_resolution"] = self._types.MediaResolution.MEDIA_RESOLUTION_LOW
                        except Exception:
                            extra["media_resolution"] = "low"
                    cfg = self._types.GenerateContentConfig(temperature=0.0, **think_kw, **extra)
                    resp = self._client.models.generate_content(
                        model=self.model_name, contents=parts, config=cfg
                    )
                    text = (getattr(resp, "text", None) or "").strip()
                    if text:
                        return text
                    last = RuntimeError("empty Gemini response")
                else:
                    resp = self._model.generate_content(
                        parts,
                        generation_config=self._genai.GenerationConfig(temperature=0.0),
                    )
                    text = (getattr(resp, "text", None) or "").strip()
                    if text:
                        return text
                    last = RuntimeError("empty Gemini response")
            except Exception as e:
                last = e
                if _is_quota_error(e):
                    quota_hits += 1
                    # RPM/short-window limits recover after a short wait; RPD won't. Wait a
                    # bounded time a few times, then soft-stop so the run can resume post-reset.
                    if quota_hits >= self.quota_retries:
                        raise QuotaExhausted(e)
                    wait = _retry_after_seconds(e) or min(self.quota_wait_cap, 15.0 * (2 ** (quota_hits - 1)))
                    wait = min(wait, self.quota_wait_cap)
                    print(f"[quota] hit {quota_hits}/{self.quota_retries}: {e!r}; sleep {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                wait = min(60.0, 1.5 * (2**attempt))
                print(f"[warn] generate attempt {attempt + 1}/{retries}: {e!r}; sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
        if last is not None and _is_quota_error(last):
            raise QuotaExhausted(last)
        raise last


class OpenRouterClient:
    """OpenAI-compatible client for OpenRouter (frame-list / text only).

    Mirrors GeminiClient.generate() so the main loop is backend-agnostic. Native video is not
    supported (Gemini via OpenRouter only accepts YouTube links), so upload_video() errors out.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, reasoning_effort: str = "low",
                 quota_retries: int = 6, quota_wait_cap: float = 120.0, max_tokens: int = 2048):
        self.api_key = api_key
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.quota_retries = quota_retries
        self.quota_wait_cap = quota_wait_cap
        self.max_tokens = max_tokens
        self._kind = "openrouter"
        self.uploaded = {}
        self.total_cost = 0.0
        self.n_calls = 0

    def upload_video(self, path: str):
        raise RuntimeError(
            "OpenRouter backend does not support native video upload; use a frame-list sampler."
        )

    def _content(self, parts):
        import base64

        texts, images = [], []
        for p in parts:
            if isinstance(p, dict) and "data" in p:
                b64 = base64.b64encode(p["data"]).decode("ascii")
                images.append(
                    {"type": "image_url", "image_url": {"url": f"data:{p['mime_type']};base64,{b64}"}}
                )
            else:
                texts.append({"type": "text", "text": str(p)})
        # OpenRouter recommends text first, then images.
        return texts + images

    def generate(self, parts, retries: int, *, images: bool = False) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self._content(parts)}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        # Gemini 3.1 Pro: reasoning is mandatory; supported efforts are high/medium/low only.
        # none/minimal would 400, so clamp to low. exclude=true keeps CoT out of model_output.
        effort = self.reasoning_effort or "low"
        if effort in ("none", "minimal"):
            effort = "low"
        payload["reasoning"] = {"effort": effort, "exclude": True}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MotionBlind",
            "X-Title": "MotionBlind eval",
        }

        last = None
        quota_hits = 0
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.BASE_URL, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"openrouter error: {data['error']}")
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content") or ""
                    if isinstance(text, list):  # some providers return content parts
                        text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
                    text = (text or "").strip()
                    if text:
                        usage = data.get("usage") or {}
                        cost = float(usage.get("cost") or 0.0)
                        self.total_cost += cost
                        self.n_calls += 1
                        if self.n_calls == 1 or self.n_calls % 10 == 0:
                            details = usage.get("completion_tokens_details") or {}
                            print(
                                f"[openrouter] calls={self.n_calls} last_cost=${cost:.4f} "
                                f"total=${self.total_cost:.3f} prompt={usage.get('prompt_tokens')} "
                                f"completion={usage.get('completion_tokens')} "
                                f"reasoning={details.get('reasoning_tokens', 0)}",
                                flush=True,
                            )
                        return text
                last = RuntimeError(f"empty OpenRouter response: {str(data)[:200]}")
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", "ignore")
                except Exception:
                    detail = ""
                err = RuntimeError(f"HTTP {e.code}: {detail[:300]}")
                last = err
                if e.code == 402:  # out of credits -> soft stop so a top-up + resume continues
                    raise QuotaExhausted(err)
                if e.code == 429 or _is_quota_error(err):
                    quota_hits += 1
                    if quota_hits >= self.quota_retries:
                        raise QuotaExhausted(err)
                    wait = _retry_after_seconds(err) or min(self.quota_wait_cap, 15.0 * (2 ** (quota_hits - 1)))
                    wait = min(wait, self.quota_wait_cap)
                    print(f"[quota] hit {quota_hits}/{self.quota_retries}: {err}; sleep {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                wait = min(60.0, 1.5 * (2**attempt))
                print(f"[warn] generate attempt {attempt + 1}/{retries}: {err}; sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
            except Exception as e:
                last = e
                if _is_quota_error(e):
                    quota_hits += 1
                    if quota_hits >= self.quota_retries:
                        raise QuotaExhausted(e)
                    wait = min(self.quota_wait_cap, 15.0 * (2 ** (quota_hits - 1)))
                    print(f"[quota] hit {quota_hits}/{self.quota_retries}: {e!r}; sleep {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                wait = min(60.0, 1.5 * (2**attempt))
                print(f"[warn] generate attempt {attempt + 1}/{retries}: {e!r}; sleep {wait:.1f}s", flush=True)
                time.sleep(wait)
        if last is not None and _is_quota_error(last):
            raise QuotaExhausted(last)
        raise last


from eval_common import tensor_pool_to_pil as _tensor_pool_to_pil  # noqa: E402

def init_frame_selector(args):
    """Return pick(vpath, question, task_type) -> list[PIL] for non-video sampling."""
    if args.sampling == "hornet":
        import torch
        sys.path.insert(0, os.path.join(REPO, "eval"))
        from eval_molmo2 import load_model

        if not args.hornet_ckpt or not args.hornet_repo:
            sys.exit("--sampling hornet requires --hornet-ckpt and --hornet-repo")
        model, processor = load_model(args.hornet_backbone)
        sys.path.insert(0, args.hornet_repo)
        from lmms_eval_utils.hornet import (  # noqa: E402
            VisionGRPOPolicy,
            get_action_by_k as hornet_action,
            load_frames as hornet_load_frames,
        )
        from frame_sampling import select_frames as uniform_select_frames

        hornet = VisionGRPOPolicy(None, 768, 1, model, processor).to("cuda")
        state = torch.load(args.hornet_ckpt, map_location="cuda")
        if args.hornet_load == "no-encoder":
            state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        result = hornet.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        print(
            f"[hornet] backbone={args.hornet_backbone} load={args.hornet_load} "
            f"matched {loaded}/{len(state)} tensors",
            flush=True,
        )
        if loaded != len(state):
            sys.exit(f"[hornet] key mismatch unexpected[:6]={result.unexpected_keys[:6]}")
        hornet.eval()
        stats = {"total": 0}

        def pick(vpath, question, task_type):
            stats["total"] += 1
            try:
                videos, _total = hornet_load_frames(vpath)
                videos = videos.to("cuda")
                with torch.no_grad():
                    keep_prob = hornet(videos.unsqueeze(0))["keep_prob"][0]
                k = min(args.num_frames, keep_prob.shape[0])
                actions = hornet_action(keep_prob.unsqueeze(0), 1, k, random_sample=False)
                idx = torch.sort(torch.nonzero(actions[0][0]).squeeze(-1)).values
                return _tensor_pool_to_pil(videos.cpu()[idx.cpu()])
            except Exception as e:
                print(f"[hornet] fallback uniform-{args.num_frames} ({e})", flush=True)
                return uniform_select_frames(vpath, args.num_frames, "uniform", seed=args.seed)

        return pick

    if args.sampling == "f2cfull":
        from f2c_sampling import load_f2c_full
        from scoring import add_question_suffix

        clip_id = args.f2c_clip or "google/siglip2-base-patch16-224"
        f2c = load_f2c_full(
            clip_model=clip_id,
            s_max=args.f2c_s_max,
            lambda_r=args.f2c_lambda_r,
            lambda_l=args.f2c_lambda_l,
            seed=args.seed,
        )
        print(f"[f2cfull] encoder={clip_id}", flush=True)

        def pick(vpath, question, task_type):
            clip_q = question_of({"question": question, "type": task_type})
            selected = f2c.select(vpath, clip_q, args.num_frames)
            return selected[0]

        return pick

    from frame_io import sample_random_frames, sample_uniform_frames

    def pick(vpath, question, task_type):
        if args.sampling == "random":
            return sample_random_frames(vpath, args.num_frames, args.seed)
        return sample_uniform_frames(vpath, args.num_frames)

    return pick


def main():
    load_dotenv(os.path.join(REPO, ".env"))
    args = parse_args()
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "gemini_3_pro")
    if args.backend == "openrouter":
        api_key = (
            args.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("openrouter_api_key")
        )
        if not api_key:
            sys.exit("Set OPENROUTER_API_KEY in .env or the environment for --backend openrouter.")
        api_key = api_key.strip().strip('"').strip("'")
        model_name = args.model or os.environ.get("OPENROUTER_MODEL") or "google/gemini-3.1-pro-preview"
        if args.sampling == "video":
            sys.exit(
                "--backend openrouter cannot use --sampling video (Gemini video via OpenRouter "
                "requires YouTube links). Use a frame-list sampler (uniform/random/hornet/f2cfull)."
            )
    else:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env or the environment.")
        model_name = args.model or os.environ.get("GEMINI_MODEL") or "gemini-3.1-pro-preview"
        if model_name in RETIRED_MODELS:
            nxt = RETIRED_MODELS[model_name]
            print(f"[info] {model_name} retired; using {nxt}", flush=True)
            model_name = nxt
    frame_samplers = ("uniform", "random", "hornet", "f2cfull")
    if args.sampling in frame_samplers and not args.num_frames:
        sys.exit("--num-frames is required for --sampling uniform|random|hornet|f2cfull")
    if args.ablation in ("shuffled_frames", "reversed_frames") and args.sampling == "video":
        sys.exit(f"--ablation {args.ablation} needs a frame-list sampler (not native video)")

    fix_path = make_fix_path(args.base_path)
    data = _load_json_list(args.data)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    pick_frames = None
    if args.sampling in frame_samplers:
        pick_frames = init_frame_selector(args)

    if args.backend == "openrouter":
        client = OpenRouterClient(
            api_key,
            model_name,
            reasoning_effort=args.reasoning_effort,
            quota_retries=args.quota_retries,
            quota_wait_cap=args.quota_wait_cap,
        )
    else:
        client = GeminiClient(
            api_key,
            model_name,
            thinking_level=args.thinking_level,
            quota_retries=args.quota_retries,
            quota_wait_cap=args.quota_wait_cap,
        )
    print(
        f"backend={args.backend} model={model_name} sdk={client._kind} n={len(data)} "
        f"sampling={args.sampling} fps={args.fps} num_frames={args.num_frames} "
        f"ablation={args.ablation} prompt={args.prompt_template} "
        f"thinking={args.thinking_level} reasoning={args.reasoning_effort} "
        f"requests_cap={args.requests_cap}",
        flush=True,
    )
    p0 = fix_path(data[0]["video_path"])
    print(f"Path check: {p0} exists={os.path.exists(p0)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} predictions", flush=True)

    meta = {
        "model_name": model_name,
        "backend": args.backend,
        "sampling": args.sampling,
        "num_frames": args.num_frames,
        "fps": args.fps if args.sampling == "video" else None,
        "seed": args.seed,
        "ablation": args.ablation,
        "prompt_template": args.prompt_template,
        "thinking_level": args.thinking_level,
        "reasoning_effort": args.reasoning_effort if args.backend == "openrouter" else None,
        "hornet_ckpt": args.hornet_ckpt if args.sampling == "hornet" else None,
        "hornet_load": args.hornet_load if args.sampling == "hornet" else None,
        "hornet_backbone": args.hornet_backbone if args.sampling == "hornet" else None,
        "f2c_clip": args.f2c_clip if args.sampling == "f2cfull" else None,
        "f2c_encoder": (
            (args.f2c_clip or "google/siglip2-base-patch16-224")
            if args.sampling == "f2cfull"
            else None
        ),
    }

    predictions = []
    new_success = 0
    t0 = time.time()

    def checkpoint():
        json.dump(predictions, open(args.out, "w"), indent=2)

    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        if prev and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            continue

        q = build_prompt(sample["question"], sample["type"], args.prompt_template)
        vpath = fix_path(sample["video_path"])
        err = None
        out = ""
        try:
            if args.ablation == "no_video":
                out = client.generate([q], args.retries)
            elif args.sampling == "video":
                if not os.path.exists(vpath):
                    raise FileNotFoundError(vpath)
                uploaded = client.upload_video(vpath)
                parts = [uploaded]
                if client._kind == "new":
                    try:
                        uploaded = client._types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=getattr(uploaded, "mime_type", None) or "video/mp4",
                            video_metadata=client._types.VideoMetadata(fps=args.fps),
                        )
                        parts = [uploaded]
                    except Exception:
                        parts = [client.uploaded.get(vpath, uploaded)]
                parts.append(q)
                out = client.generate(parts, args.retries)
            else:
                from frame_io import reverse_frames, shuffle_frames

                frames = pick_frames(vpath, sample["question"], sample["type"])
                if args.ablation == "shuffled_frames" and len(frames) > 1:
                    frames = shuffle_frames(frames, vpath, args.seed)
                if args.ablation == "reversed_frames" and len(frames) > 1:
                    frames = reverse_frames(frames)
                parts = []
                for fr in frames:
                    parts.append({"mime_type": "image/jpeg", "data": jpeg_bytes(fr, args.max_side)})
                parts.append(q)
                out = client.generate(parts, args.retries, images=True)
        except QuotaExhausted as qe:
            # Do not record this item as an error; leave it unanswered so a resume retries it
            # once the quota resets. Checkpoint everything done so far and stop cleanly.
            checkpoint()
            print(
                f"[quota] rate/day quota reached at index={sample['index']} after "
                f"{new_success} new call(s) this run: {qe}. Checkpointed {len(predictions)} "
                f"-> {args.out}; exiting 0 (resume later to continue).",
                flush=True,
            )
            sys.exit(0)
        except Exception as e:
            err = repr(e)
            print(f"[error] index={sample['index']}: {e}", flush=True)

        predictions.append(
            {
                **meta,
                "index": sample["index"],
                "video_path": sample["video_path"],
                "question": q,
                "model_output": out,
                "error": err,
                "ts": datetime.datetime.now().isoformat(),
            }
        )
        if not err and (out or "").strip():
            new_success += 1
            if args.requests_cap and new_success >= args.requests_cap:
                checkpoint()
                print(
                    f"[cap] reached requests-cap={args.requests_cap} new calls this run; "
                    f"checkpointed {len(predictions)} -> {args.out}; exiting 0 (resume later).",
                    flush=True,
                )
                sys.exit(0)
        if args.sleep:
            time.sleep(args.sleep)
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            eta = (time.time() - t0) / (i + 1) * (len(data) - i - 1)
            print(f"  [{i + 1}/{len(data)}] eta {eta / 60:.1f} min", flush=True)
            checkpoint()

    json.dump(predictions, open(args.out, "w"), indent=2)
    print(f"Wrote {len(predictions)} predictions -> {args.out}", flush=True)

    scored = [p for p in predictions if (p.get("model_output") or "").strip()]
    if scored:
        scores = get_scores(build_answers(predictions, data))
        print(f"[TimeBlind metrics] {scores}", flush=True)
        cm = confusion_and_f1(predictions, data)
        print(format_report(cm), flush=True)
        sidecar = os.path.splitext(args.out)[0] + "_metrics.json"
        json.dump(
            {
                **scores,
                "F1_yesno_pct": round(cm["F1"] * 100, 2),
                "yesno_Acc_pct": round(cm["Acc"] * 100, 2),
                "confusion": {k: cm[k] for k in ("TP", "FP", "FN", "TN", "invalid", "n_yesno")},
                "config": meta,
            },
            open(sidecar, "w"),
            indent=2,
        )
        print(f"Wrote {sidecar}", flush=True)


if __name__ == "__main__":
    main()
