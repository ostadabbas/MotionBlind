#!/usr/bin/env python
"""Inkling-Small (Thinking Machines MoE) eval on TimeBlind / MotionBlind.

Inkling takes images, not native video, so the script sends N sampled frames as images.
Inkling-Small (276B total / 12B active MoE) does not fit single-GPU HF generate.
This script is an OpenAI-compatible client for a vLLM (or SGLang) server started separately:

    vllm serve <INKLING_SMALL_REPO> --tensor-parallel-size <N> --port 8000 --trust-remote-code

    python eval_inkling.py --base-url http://<host>:8000/v1 --model <served-name> \
        --base-path <videos_root> --data <data.jsonl> --out <preds.json> \
        --num-frames 8 --sampling uniform

Frames go as OpenAI `image_url` base64 data URIs in temporal order. Mirrors
eval_motion.py: same scoring, --sampling, resume, and checkpointing. Stdlib-only (urllib).
"""

import argparse
import base64
import datetime
import io
import json
import os
import sys
import time
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Served model name (as registered by the vLLM/SGLang server).")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible endpoint base URL of the running server.")
    p.add_argument("--api-key", default="EMPTY", help="Dummy key for a local vLLM server.")
    p.add_argument("--api-key-env", default=None,
                   help="Env var holding the real key (never the key itself in argv).")
    p.add_argument("--ablation", default="none",
                   choices=["none", "no_video", "shuffled_frames", "reversed_frames"])
    p.add_argument("--selections-dir", default=None,
                   help="Frozen selections dir; hornet requires its <sampling><N>.json "
                        "(an API client cannot run the policy live).")
    p.add_argument("--timeblind-repo", default="./TimeBlind",
                   help="Path to the cloned TimeBlind repo (for scoring.py / frame_io.py).")
    p.add_argument("--base-path", required=True, help="Root dir holding the video folders")
    p.add_argument("--data", required=True, help="Path to data.jsonl")
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--num-frames", type=int, default=8,
                   help="Number of frames to sample and send as images.")
    p.add_argument("--sampling", default="uniform",
                   choices=["uniform", "random", "hornet", "f2cfull"],
                   help="Frame selection: uniform (evenly spaced) or random (temporal order).")
    p.add_argument("--seed", type=int, default=0, help="Seed for --sampling random (per-video deterministic).")
    p.add_argument("--max-side", type=int, default=768,
                   help="Downscale each frame so its longest side <= this (bounds image tokens). 0 = no resize.")
    p.add_argument("--prompt-template", default="default", choices=["default", "cot"],
                   help="'default' = bare question (bias-robust); 'cot' = describe-then-answer scaffold.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    add_f2c_args(p)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Only evaluate the first N samples (smoke test).")
    return p.parse_args()


# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))
from eval_common import (add_f2c_args, apply_f2c_scales,  # noqa: E402
                         load_f2c_selector, make_fix_path, question_of, score_item)


def build_prompt(question, task_type, template):
    # `question` already carries the "Please output Yes or No." / "A or B." suffix.
    if template == "default":
        return question
    ending = (
        "End your response with exactly 'Answer: A' or 'Answer: B'."
        if task_type == "multiple_choice"
        else "End your response with exactly 'Answer: Yes' or 'Answer: No'."
    )
    return ("Watch the frames carefully. Describe what changes across them in 2-3 sentences. "
            "Then answer the question.\n\n"
            f"Question: {question}\n\n" + ending)


def encode_frame(pil_img, max_side):
    """PIL image -> data:image/jpeg;base64 URI, optionally downscaled to bound tokens."""
    from PIL import Image
    img = pil_img.convert("RGB")
    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    args = parse_args()
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "inkling_small")

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import sample_random_frames, sample_uniform_frames

    fix_path = make_fix_path(args.base_path)
    data = _load_json_list(args.data)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples", flush=True)
    p0 = fix_path(data[0]["video_path"])
    print(f"Path check: {p0}", flush=True)
    print(f"Exists: {os.path.exists(p0)}", flush=True)
    print(f"Endpoint: {args.base_url} | model: {args.model} | "
          f"sampling: {args.sampling} | frames: {args.num_frames} | prompt: {args.prompt_template}", flush=True)

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + (os.environ[args.api_key_env]
                                            if args.api_key_env else args.api_key)}

    def call_api(messages):
        body = json.dumps({
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }).encode("utf-8")
        last = None
        for attempt in range(args.retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
                return out["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - retry any transient server/HTTP error
                last = e
                time.sleep(2 * (attempt + 1))
        raise last

    f2c_pick = load_f2c_selector(args, REPO)
    sel_cells = None
    if args.sampling == "hornet":
        if not args.selections_dir:
            sys.exit("--sampling hornet needs --selections-dir (frozen picks)")
        import json as _json
        _sf = os.path.join(args.selections_dir, f"hornet{args.num_frames}.json")
        sel_cells = _json.load(open(_sf, encoding="utf-8"))
        print(f"hornet selections: {_sf} ({len(sel_cells)} clips)", flush=True)

    def run_one(video_path, question, task_type):
        full = fix_path(video_path)
        if args.ablation == "no_video":
            frames = []
        elif f2c_pick is not None:
            fr, scales = f2c_pick(full, question, args.num_frames)
            frames = apply_f2c_scales(fr, scales)
        elif args.sampling == "hornet":
            from frame_io import _read_frames
            rec = sel_cells.get(video_path) or sel_cells.get(os.path.basename(video_path))
            if rec is None:
                raise KeyError(f"no frozen hornet cell for {video_path}")
            frames = _read_frames(full, rec["indices"])
        elif args.sampling == "random":
            frames = sample_random_frames(full, args.num_frames, args.seed)
        else:
            frames = sample_uniform_frames(full, args.num_frames)
        if args.ablation == "reversed_frames" and len(frames) > 1:
            from frame_io import reverse_frames
            frames = reverse_frames(frames)
        if args.ablation == "shuffled_frames" and len(frames) > 1:
            from frame_io import shuffle_frames
            frames = shuffle_frames(frames, full, args.seed)
        # frames (temporal order) as image_url parts, then the text question
        content = [{"type": "image_url", "image_url": {"url": encode_frame(f, args.max_side)}}
                   for f in frames]
        content.append({"type": "text", "text": build_prompt(question, task_type, args.prompt_template)})
        messages = [{"role": "user", "content": content}]
        return call_api(messages), len(frames)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} existing predictions", flush=True)

    predictions = []
    start = time.time()
    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        # Reuse a cached prediction only if both index and video_path match (cross-dataset guard).
        if prev is not None and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            continue
        question = question_of(sample)
        error = None
        try:
            model_output, nf = run_one(sample["video_path"], question, sample["type"])
        except Exception as e:  # noqa: BLE001
            model_output, nf, error = "", 0, str(e)

        extracted, gt, correct = score_item(sample, model_output)
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (len(data) - i - 1)

        print(f"\n{'=' * 70}", flush=True)
        print(f"[{i+1}/{len(data)}] {sample['video_path'].split('/')[-1]} | frames: {nf} | "
              f"GT: {sample['answer']} | {'OK' if correct else 'X'} | ETA: {eta:.0f}s", flush=True)
        print(f"Q: {question}", flush=True)
        print(f"OUT: {model_output}", flush=True)
        if error:
            print(f"ERROR: {error}", flush=True)

        predictions.append({
            "index": sample["index"],
            "video_path": sample["video_path"],
            "question": question,
            "model_name": args.model,
            "model_output": model_output,
            "frames": {"strategy": args.sampling, "seed": args.seed, "num_frames": nf},
            "prompt_template": args.prompt_template,
            "error": error,
            "ts": datetime.datetime.now().isoformat(),
        })

        if (i + 1) % 25 == 0:
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(predictions, f, indent=2)

    scores = get_scores(build_answers(predictions, data))
    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | SAMPLING: {args.sampling} | FRAMES: {args.num_frames} | "
          f"PROMPT: {args.prompt_template}", flush=True)
    print(f"Time: {(time.time() - start) / 60:.1f}min | Samples: {len(data)}", flush=True)
    print(f"{'=' * 50}", flush=True)
    for k in ("Q_Acc", "V_Acc", "Acc", "I_Acc"):
        print(f"  {k}: {scores[k] * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
