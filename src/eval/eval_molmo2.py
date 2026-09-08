#!/usr/bin/env python3
"""Molmo2-8B driver with explicit frame sampling.

Feeds a list of PIL images so uniform / random / HORNet sweeps are controlled independently
of the processor's native video reader. Supports integrity ablations: no video (text-only)
and shuffled frame order. The shuffle target is ~50% Acc on balanced yes/no if the
model uses vision.

After a full run, prints I_Acc plus F1 and a confusion matrix via report_run_metrics.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import random
import sys
import time

_T0 = time.time()   # process start; --max-runtime is measured from here

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = os.path.dirname(os.path.abspath(__file__))
# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))

from scoring import _load_json_list, add_question_suffix, build_answers, get_scores  # noqa: E402
from report_run_metrics import confusion_and_f1, format_report  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/Molmo2-8B")
    add_hf_video_args(p)
    p.add_argument("--max-runtime", type=float, default=None, metavar="SECONDS",
                   help="Stop cleanly after this many seconds and save. A chained "
                        "SLURM job then exits before the wall limit kills it mid-write. "
                        "The next chained job resumes from the saved predictions.")
    p.add_argument("--base-path", required=True, help="Root dir for video_path entries in data.jsonl")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "random", "hornet", "f2cfull"],
        help="Frame selection: uniform / random / hornet / "
        "f2cfull (paper F2C: watershed+adaptive res, arXiv:2510.02262).",
    )
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random sampling")
    p.add_argument(
        "--ablation",
        default="none",
        choices=["none", "no_video", "shuffled_frames", "reversed_frames"],
        help="Dataset integrity baselines: text-only, shuffled order, or reversed order.",
    )
    p.add_argument(
        "--f2c-clip",
        default=None,
        help="VL encoder for f2cfull (default: SigLIP).",
    )
    p.add_argument(
        "--f2c-pool",
        type=int,
        default=128,
        help="Unused; kept so older sbatch env still parses.",
    )
    p.add_argument(
        "--f2c-s-max",
        type=float,
        default=2.0,
        help="Paper s_max for --sampling f2cfull (supp. D.3 default 2).",
    )
    p.add_argument(
        "--f2c-lambda-r",
        type=float,
        default=0.5,
        help="Redundancy weight lambda_r for f2cfull (paper 0.5).",
    )
    p.add_argument(
        "--f2c-lambda-l",
        type=float,
        default=0.05,
        help="Length reward lambda_l for f2cfull (paper 0.05).",
    )
    p.add_argument(
        "--f2c-base-max-crops",
        type=int,
        default=8,
        help="Molmo2 max_crops at s=1 for f2cfull; scaled as round(base/s^2) per frame.",
    )
    p.add_argument("--hornet-ckpt", default=None, help="Required when --sampling hornet")
    p.add_argument(
        "--hornet-repo",
        default=None,
        help="Path to the cloned HORNet repo (lmms_eval_utils/hornet.py); for --sampling hornet.",
    )
    p.add_argument(
        "--hornet-load",
        default="full",
        choices=["full", "no-encoder"],
        help="Same as eval_eagle/eval_motion: 'full' loads encoder+policy and asserts every "
        "checkpoint tensor matched; 'no-encoder' drops encoder.* (random vision encoder).",
    )
    p.add_argument(
        "--hornet-fallback",
        action="store_true",
        help="On HORNet read/policy errors, non-finite keep_prob, or std < --hornet-min-std, "
        "use uniform-N frames (same lever as eval_motion.py / eval_qwen3.py).",
    )
    p.add_argument(
        "--hornet-min-std",
        type=float,
        default=0.0,
        help="keep_prob std below which --hornet-fallback uses uniform-N (0 = only hard failures).",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--prompt-template", default="default", choices=["default", "cot"])
    return p.parse_args()


from eval_common import tensor_pool_to_pil as _tensor_pool_to_pil  # noqa: E402

from eval_common import (add_hf_video_args, is_mcq4,  # noqa: E402
                         make_fix_path, resolve_fix_path)

def build_prompt(question: str, task_type: str, template: str) -> str:
    q = add_question_suffix(question, task_type)
    if template == "cot":
        return (
            "Watch the video carefully. Describe any motion or change you notice, "
            "then answer the question.\n\n" + q
        )
    return q


def load_model(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, processor


def infer(
    model,
    processor,
    frames,
    question: str,
    max_new_tokens: int,
    max_crops_per_image=None,
) -> str:
    if frames:
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": question})
    else:
        content = [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]

    restore = None
    if max_crops_per_image is not None and frames:
        from molmo2_budget import patch_image_processor_per_image_crops  # noqa: E402

        restore = patch_image_processor_per_image_crops(
            processor.image_processor, list(max_crops_per_image)
        )
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        if restore is not None:
            restore()

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1] :]
    text = processor.decode(new, skip_special_tokens=True).strip()
    del inputs, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def main():
    args = parse_args()
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "molmo2_8b")
    if args.sampling == "hornet":
        if not args.hornet_ckpt or not args.hornet_repo:
            sys.exit("--sampling hornet requires --hornet-ckpt and --hornet-repo (official HORNet)")

    model, processor = load_model(args.model)

    f2c = None
    if args.sampling == "f2cfull":
        from f2c_sampling import load_f2c_full  # noqa: E402

        clip_id = args.f2c_clip or "google/siglip2-base-patch16-224"
        f2c = load_f2c_full(
            clip_model=clip_id,
            s_max=args.f2c_s_max,
            lambda_r=args.f2c_lambda_r,
            lambda_l=args.f2c_lambda_l,
            seed=args.seed,
        )
        print(
            f"[f2cfull] ready (enc={clip_id}, s_max={args.f2c_s_max}, "
            f"lambda_r={args.f2c_lambda_r}, lambda_l={args.f2c_lambda_l}, "
            f"base_max_crops={args.f2c_base_max_crops})",
            flush=True,
        )

    # The HORNet path mirrors eval_eagle.py and eval_motion.py:
    # VisionGRPOPolicy + get_action_by_k over the uniform-32 pool; --hornet-load full asserts
    # every checkpoint tensor matched. Molmo2 answers on PIL frames converted from that pool.
    hornet = hornet_load_frames = hornet_action = None
    hornet_stats = {"total": 0, "fallback": 0, "std_sum": 0.0, "std_n": 0}
    if args.sampling == "hornet":
        from collections import Counter  # noqa: E402

        sys.path.insert(0, args.hornet_repo)
        from lmms_eval_utils.hornet import (  # noqa: E402
            VisionGRPOPolicy,
            get_action_by_k as hornet_action,
            load_frames as hornet_load_frames,
        )
        from frame_sampling import select_frames as uniform_select_frames  # noqa: E402

        hornet = VisionGRPOPolicy(None, 768, 1, model, processor).to("cuda")
        state = torch.load(args.hornet_ckpt, map_location="cuda")

        def _grp(k):
            return (
                "encoder"
                if k.startswith("encoder.")
                else "policy_head"
                if k.startswith("policy_head.")
                else "policy"
                if k.startswith("policy.")
                else "other"
            )

        print(
            f"[hornet] checkpoint {os.path.basename(args.hornet_ckpt)}: {len(state)} "
            f"tensors by group {dict(Counter(_grp(k) for k in state))}",
            flush=True,
        )
        if args.hornet_load == "no-encoder":
            state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
            print(
                f"[hornet] load=no-encoder -> dropped encoder.* ; vision encoder stays at "
                f"RANDOM init ({len(state)} tensors will load)",
                flush=True,
            )

        _probe = "encoder.patch_embed.proj.weight"
        _before = hornet.state_dict().get(_probe)
        _before = float(_before.norm()) if _before is not None else None
        result = hornet.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        _after = hornet.state_dict().get(_probe)
        _after = float(_after.norm()) if _after is not None else None
        print(
            f"[hornet] load={args.hornet_load}: matched {loaded}/{len(state)} tensors "
            f"(unexpected={len(result.unexpected_keys)}, missing={len(result.missing_keys)})"
            + (
                f" | {_probe} norm {_before:.3f} -> {_after:.3f}"
                if _before is not None and _after is not None
                else ""
            ),
            flush=True,
        )
        if loaded != len(state):
            sys.exit(
                f"[hornet] only {loaded}/{len(state)} checkpoint tensors matched "
                f"HORNet VisionGRPOPolicy (key mismatch). unexpected[:6]="
                f"{result.unexpected_keys[:6]}"
            )
        hornet.eval()
        print(f"HORNet policy loaded: {args.hornet_ckpt}", flush=True)
        if args.hornet_fallback:
            print(
                f"HORNet fallback ON -> uniform-{args.num_frames} on failure / "
                f"non-finite keep_prob / std < {args.hornet_min_std}",
                flush=True,
            )

        def hornet_pick(full_path):
            hornet_stats["total"] += 1
            try:
                videos, _total = hornet_load_frames(full_path)  # [32,H,W,3] @288, /255
                videos = videos.to("cuda")
                with torch.no_grad():
                    keep_prob = hornet(videos.unsqueeze(0))["keep_prob"][0]  # [32]
                std = float(keep_prob.std())
                hornet_stats["std_sum"] += std
                hornet_stats["std_n"] += 1
                if args.hornet_fallback and (
                    (not torch.isfinite(keep_prob).all()) or std < args.hornet_min_std
                ):
                    hornet_stats["fallback"] += 1
                    return uniform_select_frames(
                        full_path, args.num_frames, "uniform", seed=args.seed
                    )
                k = min(args.num_frames, keep_prob.shape[0])
                actions = hornet_action(
                    keep_prob.unsqueeze(0), 1, k, random_sample=False
                )  # [1,1,32]
                idx = torch.sort(torch.nonzero(actions[0][0]).squeeze(-1)).values
                if hornet_stats["total"] <= 20:
                    print(
                        f"[hornet] {os.path.basename(full_path)}: picked {idx.tolist()} "
                        f"of 0..{keep_prob.shape[0]-1} (std={std:.4f})",
                        flush=True,
                    )
                return _tensor_pool_to_pil(videos.cpu()[idx.cpu()])
            except Exception as e:
                hornet_stats["fallback"] += 1
                print(
                    f"[hornet] fallback -> uniform-{args.num_frames} ({e})",
                    flush=True,
                )
                return uniform_select_frames(
                    full_path, args.num_frames, "uniform", seed=args.seed
                )

    fix_path, vsrc = resolve_fix_path(args)
    data = _load_json_list(args.data)
    # Video-MME is 4-way MCQ. The yes/no extractor and the confusion report below
    # do not apply, so it takes a separate path.
    mcq4 = is_mcq4(data)
    if mcq4:
        import videomme
        print(f"[bench] Video-MME: {len(data)} questions, 4-way MCQ, "
              f"accuracy only (chance 25%)", flush=True)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    # F2C reruns must record selected indices; skip resume if they're missing.
                    if args.sampling == "f2cfull" and not prev.get("frame_indices"):
                        continue
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} predictions", flush=True)

    predictions = []
    t0 = time.time()
    meta = {
        "model_name": args.model,
        "sampling": args.sampling,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "ablation": args.ablation,
        "prompt_template": args.prompt_template,
        "hornet_ckpt": args.hornet_ckpt if args.sampling == "hornet" else None,
        "hornet_repo": args.hornet_repo if args.sampling == "hornet" else None,
        "hornet_load": args.hornet_load if args.sampling == "hornet" else None,
        "hornet_loader": "team_full_get_action_by_k" if args.sampling == "hornet" else None,
        "hornet_fallback": args.hornet_fallback if args.sampling == "hornet" else None,
        "hornet_min_std": args.hornet_min_std if args.sampling == "hornet" else None,
        "f2c_clip": args.f2c_clip if args.sampling == "f2cfull" else None,
        "f2c_pool": None,
        "f2c_s_max": args.f2c_s_max if args.sampling == "f2cfull" else None,
        "f2c_lambda_r": args.f2c_lambda_r if args.sampling == "f2cfull" else None,
        "f2c_lambda_l": args.f2c_lambda_l if args.sampling == "f2cfull" else None,
        "f2c_base_max_crops": args.f2c_base_max_crops if args.sampling == "f2cfull" else None,
        "f2c_encoder": (
            (args.f2c_clip or "google/siglip2-base-patch16-224")
            if args.sampling == "f2cfull"
            else None
        ),
    }

    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        if prev and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            continue

        q = (videomme.format_question(sample) if mcq4 else
             build_prompt(sample["question"], sample["type"], args.prompt_template))
        vpath = fix_path(sample["video_path"])
        err = None
        out = ""
        frame_indices = None
        f2c_scales = None
        try:
            scales = None
            if args.ablation == "no_video":
                frames = []
            elif args.sampling == "hornet":
                frames = hornet_pick(vpath)
                if args.ablation == "shuffled_frames" and len(frames) > 1:
                    vseed = hash((os.path.basename(vpath), args.seed, args.num_frames)) & 0xFFFFFFFF
                    rng = random.Random(vseed)
                    frames = frames.copy()
                    rng.shuffle(frames)
                elif args.ablation == "reversed_frames" and len(frames) > 1:
                    frames = list(reversed(frames))
            elif args.sampling == "f2cfull":
                # Query-aware key clips. Include Yes/No options in the text query
                # (paper: MC options help CLIP/SigLIP relevancy).
                clip_q = add_question_suffix(sample["question"], sample["type"])
                selected = f2c.select(vpath, clip_q, args.num_frames)
                if len(selected) == 3:
                    frames, frame_indices, scales = selected
                else:
                    frames, frame_indices = selected
                    scales = None
                f2c_scales = [float(s) for s in scales] if scales is not None else None
                if frame_indices is not None:
                    frame_indices = [int(i) for i in frame_indices]
                if args.ablation == "shuffled_frames" and len(frames) > 1:
                    vseed = hash((os.path.basename(vpath), args.seed, args.num_frames)) & 0xFFFFFFFF
                    rng = random.Random(vseed)
                    order = list(range(len(frames)))
                    rng.shuffle(order)
                    frames = [frames[j] for j in order]
                    if scales is not None:
                        scales = [scales[j] for j in order]
                    if frame_indices is not None:
                        frame_indices = [frame_indices[j] for j in order]
                    f2c_scales = [float(s) for s in scales] if scales is not None else None
                elif args.ablation == "reversed_frames" and len(frames) > 1:
                    frames = list(reversed(frames))
                    if scales is not None:
                        scales = list(reversed(scales))
                    if frame_indices is not None:
                        frame_indices = list(reversed(frame_indices))
                    f2c_scales = [float(s) for s in scales] if scales is not None else None
            else:
                vseed = hash((os.path.basename(vpath), args.seed, args.num_frames)) & 0xFFFFFFFF
                from frame_sampling import select_frames

                frames = select_frames(
                    vpath,
                    args.num_frames,
                    args.sampling,
                    seed=vseed if args.sampling == "random" else args.seed,
                    hornet=None,
                )
                scales = None
                if args.ablation == "shuffled_frames" and len(frames) > 1:
                    rng = random.Random(vseed)
                    frames = frames.copy()
                    rng.shuffle(frames)
                elif args.ablation == "reversed_frames" and len(frames) > 1:
                    frames = list(reversed(frames))

            max_crops = None
            if args.sampling == "f2cfull" and frames and scales is not None:
                from molmo2_budget import scales_to_max_crops  # noqa: E402

                max_crops = scales_to_max_crops(scales, args.f2c_base_max_crops)
            out = infer(
                model, processor, frames, q, args.max_new_tokens,
                max_crops_per_image=max_crops,
            )
        except Exception as e:
            err = repr(e)
            print(f"[error] index={sample['index']}: {e}", flush=True)

        row = {
            **meta,
            "index": sample["index"],
            "video_path": sample["video_path"],
            "question": q,
            "model_output": out,
            "error": err,
            "frame_indices": frame_indices,
            "f2c_scales": f2c_scales,
            "ts": datetime.datetime.now().isoformat(),
        }
        predictions.append(row)
        if (i + 1) % 25 == 0 or i == len(data) - 1:
            eta = (time.time() - t0) / (i + 1) * (len(data) - i - 1)
            print(f"  [{i+1}/{len(data)}] eta {eta/60:.1f} min", flush=True)
        if (i + 1) % 100 == 0:
            json.dump(predictions, open(args.out, "w"), indent=2)

        # See eval_eagle.py: the chained worker needs a clean stop, not a SIGKILL.
        if args.max_runtime and (time.time() - _T0) > args.max_runtime:
            json.dump(predictions, open(args.out, "w"), indent=2)
            print(f"\n[timebox] stopping at {i + 1}/{len(data)} after "
                  f"{(time.time() - _T0) / 60:.1f}min; the next chained job resumes here.",
                  flush=True)
            break

    json.dump(predictions, open(args.out, "w"), indent=2)
    print(f"Wrote {len(predictions)} predictions -> {args.out}", flush=True)

    sidecar = os.path.splitext(args.out)[0] + "_metrics.json"
    if mcq4:
        scores = videomme.score(predictions, data)
        print("\n" + videomme.format_summary(
            scores, args.model, args.sampling, args.num_frames or 0,
            time.time() - t0), flush=True)
        json.dump({**scores, "config": meta}, open(sidecar, "w"), indent=2)
        print(f"Wrote {sidecar}", flush=True)
        if vsrc is not None:
            print(vsrc.summary(), flush=True); vsrc.close()
        return

    scores = get_scores(build_answers(predictions, data))
    print(f"[TimeBlind metrics] {scores}", flush=True)
    cm = confusion_and_f1(predictions, data)
    print(format_report(cm), flush=True)

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

    if args.sampling == "hornet":
        avg_std = hornet_stats["std_sum"] / max(hornet_stats["std_n"], 1)
        print(
            f"[hornet] mean keep_prob std={avg_std:.4f} over {hornet_stats['std_n']} videos "
            f"(load={args.hornet_load})",
            flush=True,
        )
        if args.hornet_fallback:
            print(
                f"HORNet fallback: {hornet_stats['fallback']}/{hornet_stats['total']} videos "
                f"-> uniform-{args.num_frames}",
                flush=True,
            )


if __name__ == "__main__":
    main()
