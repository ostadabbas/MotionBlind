#!/usr/bin/env python
"""Qwen3-VL-4B-Instruct eval on the TimeBlind Challenge subset.

Matches the external baseline configuration (I_Acc 26.2% on the full 2400-item
set): same scorer, same data. The differences are inference-side only:
  model Qwen/Qwen3-VL-4B-Instruct, fps 8 capped at 64 frames, greedy, max_new_tokens 128
  prompt template = "default" (bare question, no chain-of-thought).

The `default` vs `cot` prompt switch is the key lever: the CoT scaffold
("describe what changes, then answer") induces the Yes/A compliance bias that
floors I_Acc on Eagle/Motion. Qwen3-VL is a native Qwen-VL arch (like Motion-o's
Qwen2.5-VL), so it loads without trust_remote_code and honors attn_implementation="sdpa".
"""

import argparse
import datetime
import faulthandler
import gc
import json
import os
import sys
import time

_T0 = time.time()   # process start; --max-runtime is measured from here
import traceback

# Dump a native stack on segfault/abort (decord/ffmpeg, CUDA) instead of a silent .out.
faulthandler.enable()

import torch
from transformers import AutoProcessor

# Prefer the specific class; fall back to the generic image-text-to-text auto class if
# this transformers build registers Qwen3-VL only there.
try:
    from transformers import Qwen3VLForConditionalGeneration as _VLM
    _VLM_NAME = "Qwen3VLForConditionalGeneration"
except Exception:  # pragma: no cover - depends on transformers version
    from transformers import AutoModelForImageTextToText as _VLM
    _VLM_NAME = "AutoModelForImageTextToText"

# Read the video with decord (reliable on these mp4s; torchvision backend reads 0 frames).
# qwen_vl_utils reads this env at import time, so set it before importing.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
from qwen_vl_utils import process_vision_info

# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))
from eval_common import (hornet_pool_frames as _hornet_pool_frames,  # noqa: E402
                         add_f2c_args, load_f2c_selector)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--timeblind-repo", default="./TimeBlind",
                   help="Path to cloned TimeBlind repo (for scoring.py)")
    p.add_argument("--base-path", required=True, help="Root dir holding the video folders")
    # Stream clips from HuggingFace instead of staging the dataset. Video-MME is
    # 101 GB in 20 zips of ~5.2 GB; this range-reads one member at a time.
    p.add_argument("--hf-videos", nargs="?", const="lmms-eval/Video-MME", default=None,
                   metavar="REPO_ID",
                   help="stream missing videos from this HF dataset repo "
                        "(default repo: lmms-eval/Video-MME)")
    p.add_argument("--hf-cache", default=None,
                   help="disk cache for streamed videos (default: <base-path>/_hf_cache)")
    p.add_argument("--hf-cache-gb", type=float, default=20.0,
                   help="LRU cap on the streamed-video cache (default 20 GB)")
    p.add_argument("--hf-cache-policy", default="keep", choices=["keep", "lru"],
                   help="keep = scan-resistant, never evicts a resident file so "
                        "repeated sweep passes hit (default); lru = textbook LRU")
    p.add_argument("--hf-prefetch", type=int, default=2,
                   help="how many upcoming videos to fetch in the background (0 = off)")
    p.add_argument("--data", required=True, help="Path to data.jsonl")
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--dtype", default="auto", help="'auto' (baseline) or a torch dtype like bfloat16")
    # Sampling: the baseline is fps 8 capped at 64 frames. --num-frames forces uniform N.
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--max-frames", type=int, default=64,
                   help="Cap on frames when sampling by fps (baseline: 64).")
    p.add_argument("--num-frames", type=int, default=None,
                   help="If set, sample exactly this many uniform frames (overrides fps).")
    p.add_argument("--sampling", default="fps", choices=["fps", "uniform", "random", "hornet", "f2cfull"],
                   help="Frame selection: fps (fixed rate), uniform (evenly spaced N), "
                        "random (N random frames, temporal order), or hornet (learned top-N via the "
                        "HORNet policy over a uniform-32 pre-sample). uniform/random/hornet need --num-frames.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for --sampling random and the shuffled_frames permutation (per-video deterministic).")
    p.add_argument("--ablation", default="none",
                   choices=["none", "shuffled_frames", "reversed_frames", "no_video"],
                   help="Integrity probe. shuffled_frames scrambles the sampled frame order "
                        "(deterministic per video and --seed); reversed_frames plays the same "
                        "frames backwards; no_video sends the question with no visual input "
                        "(the language-prior baseline). The frame probes require --sampling "
                        "uniform or random.")
    p.add_argument("--hornet-repo", default=None,
                   help="Path to the cloned HORNet repo (holds lmms_eval_utils/hornet.py); for --sampling hornet.")
    p.add_argument("--hornet-ckpt", default=None,
                   help="Path to the HORNet policy checkpoint (.pt); for --sampling hornet.")
    p.add_argument("--hornet-load", default="full", choices=["full", "no-encoder"],
                   help="How to load the HORNet checkpoint. 'full' (default) loads the vision "
                        "encoder and the policy head, and asserts a complete match. "
                        "'no-encoder' drops the encoder tensors, so the vision encoder "
                        "keeps its random init.")
    p.add_argument("--hornet-fallback", action="store_true",
                   help="Graceful fallback: when the HORNet read/policy errors, keep_prob is "
                        "non-finite, or the policy shows no confident frame preference "
                        "(std < --hornet-min-std), use uniform-N instead of a possibly-worse-than-base "
                        "learned pick. Off by default -> byte-identical to the stock selector.")
    p.add_argument("--hornet-min-std", type=float, default=0.0,
                   help="keep_prob std below which the policy is treated as 'no confident preference' "
                        "and --hornet-fallback kicks in. 0.0 = only hard failures / non-finite outputs.")
    p.add_argument("--hornet-pool", type=int, default=32,
                   help="HORNet candidate-pool size: uniformly presample this many frames @288 then keep "
                        "top-(--num-frames). Default 32 (top-32 of 32 == uniform-32, no real selection). "
                        "Set e.g. 64 for a real selection: top-K of a denser pool.")
    add_f2c_args(p)
    p.add_argument("--max-pixels", type=int, default=0,
                   help="Per-frame pixel cap (0 = leave to qwen_vl_utils default).")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = off (baseline is greedy with no penalty).")
    p.add_argument("--prompt-template", default="default", choices=["default", "cot"],
                   help="'default' = bare question (baseline); 'cot' = describe-changes scaffold.")
    p.add_argument("--system-prompt", default="none",
                   choices=["none", "debias", "temporal", "both"],
                   help="System message prepended to the chat. 'none' = model default. "
                        "'debias' = anti-yes-bias, 'temporal' = attend to event order/timing, "
                        "'both' = combined. Separate experiment axis from --prompt-template.")
    p.add_argument("--max-runtime", type=float, default=None, metavar="SECONDS",
                   help="stop cleanly after this many seconds and save, so a chained "
                        "SLURM job exits before the wall limit kills it mid-write. "
                        "The next link resumes from the saved predictions.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Only evaluate the first N samples (smoke test).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also write the sampled frames to disk (for sampling studies).")
    p.add_argument("--frames-dir", default=None,
                   help="Where to write frames (default: <out dir>/frames/<video>/).")
    return p.parse_args()


from eval_common import make_fix_path  # noqa: E402

# System messages for the --system-prompt lever. `none` keeps the model default,
# matching the frame-sampling sweep. The others target the two things that cap I_Acc:
# yes/no compliance bias (debias) and failure to read event order/timing (temporal).
SYSTEM_PROMPTS = {
    "none": None,
    "debias": (
        "You are a precise video analyst. Base your answer only on what the video "
        "actually shows. Do not assume a statement is true — answer No (or the other "
        "option) whenever the video does not clearly support it."
    ),
    "temporal": (
        "You are a precise video analyst. Pay close attention to the order, timing, and "
        "direction of events across frames. Two videos can look similar yet differ in how "
        "the action unfolds over time; judge based on that temporal evidence."
    ),
    "both": (
        "You are a precise video analyst. Pay close attention to the order, timing, and "
        "direction of events across frames — two videos can look similar yet differ in how "
        "the action unfolds over time. Base your answer only on what the video actually "
        "shows, and do not assume a statement is true: answer No (or the other option) "
        "whenever the video does not clearly support it."
    ),
}


def build_prompt(question, task_type, template):
    # `question` already carries the "Please output Yes or No." / "A or B." suffix
    # (add_question_suffix, applied in the loop).
    if template == "cot":
        ending = (
            "End your response with exactly 'Answer: A' or 'Answer: B'."
            if task_type == "multiple_choice"
            else "End your response with exactly 'Answer: Yes' or 'Answer: No'."
        )
        return (
            "Watch the video carefully. Describe what changes between the first and last frame "
            "in 2-3 sentences. Then answer the question.\n\n"
            f"Question: {question}\n\n"
            + ending
        )
    # default: bare question, no scaffold.
    return question


def main():
    args = parse_args()
    # Derive the internal booleans from --ablation.
    args.no_video = args.ablation == "no_video"
    args.shuffle_frames = args.ablation == "shuffled_frames"
    args.reverse_frames = args.ablation == "reversed_frames"
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "qwen3_vl_4b")

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import (save_frames, save_hornet_frames, shuffle_frames, reverse_frames,
                          sample_random_frames, sample_uniform_frames)
    import videomme

    if args.sampling in ("uniform", "random", "hornet", "f2cfull") \
        and not args.num_frames:
        sys.exit(f"--sampling {args.sampling} requires --num-frames N")
    if args.sampling == "hornet" and not (args.hornet_repo and args.hornet_ckpt):
        sys.exit("--sampling hornet requires --hornet-repo and --hornet-ckpt")
    if (args.shuffle_frames or args.reverse_frames) and args.sampling not in ("uniform", "random"):
        sys.exit(f"--ablation {args.ablation} requires --sampling uniform or random "
                 "(needs an explicit frame list)")
    strategy = args.sampling
    if strategy == "fps" and args.num_frames:
        strategy = "uniform"   # backward-compat: --num-frames alone implies uniform

    frames_root = args.frames_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "frames",
        os.path.splitext(os.path.basename(args.out))[0])

    print(torch.cuda.get_device_name(0), flush=True)
    print(f"Model class: {_VLM_NAME} | prompt: {args.prompt_template}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = _VLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    print(f"Model loaded: {args.model}", flush=True)

    # HORNet selects frames only. The model argument is not called during
    # selection, so the answering model is always the model under test.
    hornet = hornet_load_frames = hornet_fit = hornet_action = None
    if args.sampling == "hornet":
        sys.path.insert(0, args.hornet_repo)
        from lmms_eval_utils.hornet import (
            VisionGRPOPolicy, load_frames as hornet_load_frames,
            fit_video_for_qwen as hornet_fit, get_action_by_k as hornet_action,
        )
        hornet = VisionGRPOPolicy(None, 768, 1, model, processor).to("cuda")
        state = torch.load(args.hornet_ckpt, map_location="cuda")

        # Diagnose the checkpoint: log whether the vision encoder tensors are
        # present and whether they load.
        def _grp(k):
            return ("encoder" if k.startswith("encoder.") else
                    "policy_head" if k.startswith("policy_head.") else
                    "policy" if k.startswith("policy.") else "other")
        from collections import Counter
        print(f"[hornet] checkpoint {os.path.basename(args.hornet_ckpt)}: {len(state)} "
              f"tensors by group {dict(Counter(_grp(k) for k in state))}", flush=True)

        # --hornet-load no-encoder drops the encoder tensors so the vision encoder stays at
        # random init (reproduces the silent strict=False bug); `full` loads everything.
        if args.hornet_load == "no-encoder":
            state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
            print(f"[hornet] load=no-encoder -> dropped encoder.* ; vision encoder stays at "
                  f"RANDOM init ({len(state)} tensors will load)", flush=True)

        _probe = "encoder.patch_embed.proj.weight"       # prove whether the encoder changed
        _before = hornet.state_dict().get(_probe)
        _before = float(_before.norm()) if _before is not None else None
        result = hornet.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        _after = hornet.state_dict().get(_probe)
        _after = float(_after.norm()) if _after is not None else None
        print(f"[hornet] load={args.hornet_load}: matched {loaded}/{len(state)} tensors "
              f"(unexpected={len(result.unexpected_keys)}, missing={len(result.missing_keys)})"
              + (f" | {_probe} norm {_before:.3f} -> {_after:.3f}"
                 if _before is not None and _after is not None else ""), flush=True)
        # full must match every tensor; no-encoder matches its stripped subset.
        if loaded != len(state):
            sys.exit(f"[hornet] only {loaded}/{len(state)} checkpoint tensors matched "
                     f"HORNet VisionGRPOPolicy (key mismatch). unexpected[:6]="
                     f"{result.unexpected_keys[:6]}")
        hornet.eval()
        print(f"HORNet policy loaded: {args.hornet_ckpt}", flush=True)
        if args.hornet_fallback:
            print(f"HORNet fallback ON -> uniform-{args.num_frames} on failure / "
                  f"non-finite keep_prob / std < {args.hornet_min_std}", flush=True)

    # f2c needs the question, so it is built here and called per item, not per video.
    f2c_pick = load_f2c_selector(args, args.timeblind_repo)

    hornet_stats = {"total": 0, "fallback": 0, "std_sum": 0.0, "std_n": 0}

    def hornet_pick(full_path):
        """Top-K HORNet selection over the uniform-32 candidate pool, via the repo's own
        get_action_by_k(keep_prob, 1, k, random_sample=False) reduction (deterministic top-k,
        == HORNet select_frames). Always computes the full 32 candidates + kept indices, so
        it can (a) preserve the pool via save_hornet_frames when --save-frames and (b) report
        keep_prob std. Optional --hornet-fallback -> uniform-N when the read/policy errors,
        keep_prob is non-finite, or the policy shows no confident preference (std < min_std),
        so HORNet degrades to the base sampler, not a worse-than-base pick."""
        hornet_stats["total"] += 1
        try:
            videos, _total = (hornet_load_frames(full_path) if args.hornet_pool == 32
                              else _hornet_pool_frames(full_path, args.hornet_pool))   # [pool,H,W,3] @288, /255
            videos = videos.to("cuda")
            with torch.no_grad():
                keep_prob = hornet(videos.unsqueeze(0))["keep_prob"][0]   # [32]
            std = float(keep_prob.std())
            hornet_stats["std_sum"] += std
            hornet_stats["std_n"] += 1
            if args.hornet_fallback and ((not torch.isfinite(keep_prob).all())
                                         or std < args.hornet_min_std):
                hornet_stats["fallback"] += 1
                return sample_uniform_frames(full_path, args.num_frames)
            k = min(args.num_frames, keep_prob.shape[0])
            actions = hornet_action(keep_prob.unsqueeze(0), 1, k, random_sample=False)  # [1,1,32]
            idx = torch.sort(torch.nonzero(actions[0][0]).squeeze(-1)).values           # temporal order
            if hornet_stats["total"] <= 20:                                             # provenance for smokes
                print(f"[hornet] {os.path.basename(full_path)}: picked {idx.tolist()} "
                      f"of 0..{keep_prob.shape[0]-1} (std={std:.4f})", flush=True)
            if args.save_frames:            # preserve the 32-frame pool: selected/ + unselected/
                try:
                    stem = os.path.splitext(os.path.basename(full_path))[0]
                    save_hornet_frames(videos.cpu(), idx, os.path.join(frames_root, stem),
                                       keep_prob=keep_prob)
                except Exception as _e:
                    print(f"[warn] hornet frame save failed: {_e}", flush=True)
            return hornet_fit(videos.cpu()[idx.cpu()])
        except Exception as e:
            hornet_stats["fallback"] += 1
            print(f"[hornet] fallback -> uniform-{args.num_frames} ({e})", flush=True)
            return sample_uniform_frames(full_path, args.num_frames)

    fix_path = make_fix_path(args.base_path)
    vsrc = None
    if args.hf_videos:
        from hf_video_source import HFVideoSource
        vsrc = HFVideoSource(
            repo_id=args.hf_videos,
            cache_dir=args.hf_cache or os.path.join(args.base_path, "_hf_cache"),
            max_gb=args.hf_cache_gb,
            policy=args.hf_cache_policy,
            prefetch=bool(args.hf_prefetch))
        vsrc.build_index()
        # A locally staged copy always wins, so a partial download is still used.
        fix_path = vsrc.make_resolver(fallback=make_fix_path(args.base_path))
        print(f"[hf] streaming from {args.hf_videos} | cache "
              f"{vsrc.cache_dir} (cap {args.hf_cache_gb:.0f} GB)", flush=True)
    data = _load_json_list(args.data)
    # Video-MME is 4-way MCQ with no contrastive quadruples: scoring's 2-way
    # extractor and get_scores() are both wrong for it, so route to videomme.
    is_mcq4 = bool(data) and all(d.get("type") == videomme.TASK_TYPE for d in data)
    if is_mcq4:
        print(f"[bench] Video-MME: {len(data)} questions, 4-way MCQ, "
              f"accuracy only (chance 25%)", flush=True)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples", flush=True)

    p = fix_path(data[0]["video_path"])
    print(f"Path check: {p}", flush=True)
    print(f"Exists: {os.path.exists(p)}", flush=True)

    _fcache = {"key": None, "frames": None, "hits": 0}

    def run_one(video_path, question, task_type):
        full_path = fix_path(video_path)
        prompt = build_prompt(question, task_type, args.prompt_template)

        frames = None                             # explicit frame list (uniform/random/hornet)
        # Benchmarks that ask several questions about the same clip (Video-MME asks
        # 3, consecutively) would otherwise re-decode it once per question. Every
        # selector here is deterministic given (video, sampling, N, seed):
        # sample_random_frames seeds off the basename, not a walking RNG, so a
        # hit returns exactly what the live path would have picked. One entry is
        # enough because the rows are grouped by video.
        # Video-MME asks 3 questions about each clip, so this cache normally saves two
        # decodes out of three. f2c picks frames per question, so keying on the
        # video alone would answer all three items from the first question's frames;
        # the question joins the key exactly when the sampler depends on it.
        ckey = (full_path, args.sampling, strategy, args.num_frames, args.seed,
                question if args.sampling == "f2cfull" else None)
        if _fcache.get("key") == ckey:
            frames = _fcache["frames"]
            _fcache["hits"] += 1
        elif args.sampling == "random":            # N randomly chosen frames (temporal order)
            frames = sample_random_frames(full_path, args.num_frames, args.seed)
        elif args.sampling == "hornet":            # learned top-N frames (HORNet policy over uniform-32)
            frames = hornet_pick(full_path)
        elif f2c_pick is not None:                 # query-conditioned key clips
            frames, _scales = f2c_pick(full_path, question, args.num_frames)
        elif strategy == "uniform":                # exact uniform via explicit list (nframes= breaks at N=1)
            frames = sample_uniform_frames(full_path, args.num_frames)

        if frames is not None and _fcache.get("key") != ckey:
            _fcache["key"], _fcache["frames"] = ckey, frames

        if frames is not None:
            if args.shuffle_frames:               # temporal-blindness probe: scramble the order
                frames = shuffle_frames(frames, full_path, args.seed)
            elif args.reverse_frames:             # direction probe: same frames, played backwards
                frames = reverse_frames(frames)
            video_item = {"type": "video", "video": frames}
        else:                                     # fps sampling, capped (baseline)
            video_item = {"type": "video", "video": full_path, "fps": args.fps}
            if args.max_frames:
                video_item["max_frames"] = args.max_frames
        if args.max_pixels:
            video_item["max_pixels"] = args.max_pixels

        messages = []
        sys_text = SYSTEM_PROMPTS[args.system_prompt]
        if sys_text is not None:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": sys_text}]})
        user_content = ([{"type": "text", "text": prompt}] if args.no_video
                        else [video_item, {"type": "text", "text": prompt}])
        messages.append({"role": "user", "content": user_content})

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs,
        ).to(model.device)

        num_frames = video_inputs[0].shape[0] if video_inputs else 0

        # hornet dumps its own selected/unselected pool in hornet_pick; only the generic
        # (post-processor) save applies to the other samplers.
        if args.save_frames and video_inputs and args.sampling != "hornet":
            try:
                stem = os.path.splitext(os.path.basename(full_path))[0]
                save_frames(video_inputs[0], os.path.join(frames_root, stem))
            except Exception as _e:
                print(f"[warn] frame save failed: {_e}", flush=True)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
            )
        # Native Qwen-VL returns prompt+continuation; trim the prompt. Conditional so it's
        # robust if a build returns only new tokens.
        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, in_len:] if output.shape[1] > in_len else output
        response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        return response, num_frames

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                # Only skip items that actually succeeded; retry empty/errored ones on resume.
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} existing predictions", flush=True)

    def save(preds, upto=None):
        """Write predictions atomically, preserving the un-walked tail.

        Two rules for a resumed run:

        1) `preds` holds rows only up to the current loop position. Re-attach the
           completed tail from `done`, so an early stop keeps that work.
        2) Write to a tmp file and os.replace it. A torn write would make the
           next resume unparsable and redo the whole cell.
        """
        out = list(preds)
        if upto is not None:
            seen = {p["index"] for p in out}
            for s in data[upto:]:
                prev = done.get(s["index"])
                if (prev is not None and s["index"] not in seen
                        and prev.get("video_path") == s["video_path"]):
                    out.append(prev)
        tmp = f"{args.out}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, args.out)
        return len(out)

    predictions = []
    _err_dbg = {"n": 0}
    start_time = time.time()

    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        # Reuse a cached row only when index and video_path both match. This guards
        # against resuming a file that was built for a different dataset.
        if prev is not None and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            continue
        if vsrc is not None and args.hf_prefetch:
            # Rows are grouped by video, so the next few distinct ids are the
            # next few fetches. Queue them while this one is on the GPU.
            nxt, seen_ids = [], set()
            for s2 in data[i + 1:]:
                vid = os.path.splitext(os.path.basename(s2["video_path"]))[0]
                if vid not in seen_ids:
                    seen_ids.add(vid)
                    nxt.append(vid)
                if len(nxt) >= args.hf_prefetch:
                    break
            vsrc.queue(nxt)
        if is_mcq4:
            question = videomme.format_question(sample)
        else:
            question = add_question_suffix(sample["question"], sample["type"])
        error = None
        try:
            model_output, num_frames = run_one(sample["video_path"], question, sample["type"])
        except Exception as e:
            model_output, num_frames = "", 0
            error = str(e)
            if _err_dbg["n"] < 2:
                _err_dbg["n"] += 1
                traceback.print_exc()

        if is_mcq4:
            extracted = videomme.extract_choice(model_output, sample.get("options"))
            gt = videomme.gt_index(sample["answer"])
            correct = extracted == gt and extracted >= 0
        else:
            extracted = extract_answer(model_output, sample["type"])
            gt = extract_answer(sample["answer"], sample["type"])
            correct = extracted == gt

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(data) - i - 1)

        print(f"\n{'=' * 70}", flush=True)
        print(f"[{i+1}/{len(data)}] {sample['video_path'].split('/')[-1]} | "
              # Qwen-VL merges frames in temporal pairs, so a 1-frame request comes back
              # as a 2-deep tensor (the frame duplicated). Print both when they differ:
              # "frames: 2" in a sweep whose budgets are 1/4/8/16/24 reads like a
              # misconfiguration otherwise.
              f"frames: {num_frames}"
              f"{f' (N={args.num_frames} requested)' if args.num_frames and num_frames != args.num_frames else ''}"
              f" | GT: {sample['answer']} | "
              f"{'OK' if correct else 'X'} | ETA: {eta:.0f}s", flush=True)
        print(f"Q: {question}", flush=True)
        print(f"REASONING: {model_output}", flush=True)
        if error:
            print(f"ERROR: {error}", flush=True)

        predictions.append({
            "index": sample["index"],
            "video_path": sample["video_path"],
            "question": question,
            "model_name": args.model,
            "model_output": model_output,
            "frames": {"strategy": strategy, "seed": args.seed,
                       "shuffle": args.shuffle_frames,
                       "reverse": args.reverse_frames,
                       "hornet_load": (args.hornet_load if args.sampling == "hornet" else None),
                       "fps": args.fps, "max_frames": args.max_frames,
                       "num_frames": num_frames,
                       "saved_dir": (os.path.join(
                           frames_root,
                           os.path.splitext(os.path.basename(sample["video_path"]))[0])
                           if args.save_frames else None)},
            "prompt_template": args.prompt_template,
            "system_prompt": args.system_prompt,
            "error": error,
            "ts": datetime.datetime.now().isoformat(),
        })

        if (i + 1) % 25 == 0:
            save(predictions, upto=i + 1)

        if args.max_runtime and (time.time() - _T0) > args.max_runtime:
            kept = save(predictions, upto=i + 1)
            print(f"\n[timebox] stopping at {i + 1}/{len(data)} after "
                  f"{(time.time() - _T0) / 60:.1f}min; {kept}/{len(data)} rows on "
                  f"disk. The next chained job resumes here.", flush=True)
            if vsrc is not None:
                print(vsrc.summary(), flush=True)
                vsrc.close()
            return

    save(predictions)

    total_time = time.time() - start_time

    if is_mcq4:
        scores = videomme.score(predictions, data)
        nf = args.num_frames if args.num_frames else f"fps {args.fps} (<= {args.max_frames})"
        print("\n\n" + videomme.format_summary(scores, args.model, strategy, nf,
                                                total_time), flush=True)
        meta = {"model_name": args.model, "benchmark": "videomme",
                "sampling": strategy, "num_frames": args.num_frames,
                "seed": args.seed, "prompt_template": args.prompt_template,
                "system_prompt": args.system_prompt, "fps": args.fps,
                "max_frames": args.max_frames, "shuffle": args.shuffle_frames,
                "reverse": args.reverse_frames,
                # The checkpoint is part of the result: two hornet cells with the same
                # load mode and different .pt files are different experiments, not
                # re-runs. Record the ckpt path here: the SLURM log rotates and is
                # not pulled with results.
                "hornet_load": (args.hornet_load if args.sampling == "hornet" else None),
                "hornet_ckpt": (args.hornet_ckpt if args.sampling == "hornet" else None),
                "hornet_repo": (args.hornet_repo if args.sampling == "hornet" else None),
                "hornet_pool": (args.hornet_pool if args.sampling == "hornet" else None),
                "f2c_clip": (args.f2c_clip if args.sampling in ("f2c", "f2cfull") else None),
                "f2c_pool": (args.f2c_pool if args.sampling == "f2c" else None),
                "f2c_s_max": (args.f2c_s_max if args.sampling == "f2cfull" else None),
                "f2c_lambda_r": (args.f2c_lambda_r if args.sampling == "f2cfull" else None),
                "f2c_lambda_l": (args.f2c_lambda_l if args.sampling == "f2cfull" else None)}
        side = os.path.splitext(args.out)[0] + "_metrics.json"
        with open(side, "w") as f:
            json.dump({**scores, "config": meta}, f, indent=2)
        if _fcache["hits"]:
            print(f"frame-decode cache: {_fcache['hits']} hits of {len(data)} items "
                  f"({100 * _fcache['hits'] / max(len(data), 1):.0f}% of decodes skipped)",
                  flush=True)
        if vsrc is not None:
            print(vsrc.summary(), flush=True)
            vsrc.close()
        print(f"metrics -> {side}", flush=True)
        return

    answers = build_answers(predictions, data)
    scores = get_scores(answers)

    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | SAMPLING: {strategy}"
          f"{' +shuffle' if args.shuffle_frames else ''}"
          f"{f' load={args.hornet_load}' if args.sampling == 'hornet' else ''} | FRAMES: "
          f"{args.num_frames if args.num_frames else f'fps {args.fps} (<= {args.max_frames})'} | "
          f"PROMPT: {args.prompt_template} | SYS: {args.system_prompt}", flush=True)
    print(f"Time: {total_time / 60:.1f}min | Samples: {len(data)}", flush=True)
    if args.sampling == "hornet" and args.hornet_fallback:
        avg_std = hornet_stats["std_sum"] / max(hornet_stats["std_n"], 1)
        print(f"HORNet fallback: {hornet_stats['fallback']}/{hornet_stats['total']} videos "
              f"-> uniform-{args.num_frames} | mean keep_prob std={avg_std:.4f}", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  Q_Acc: {scores['Q_Acc'] * 100:.1f}%", flush=True)
    print(f"  V_Acc: {scores['V_Acc'] * 100:.1f}%", flush=True)
    print(f"  Acc:   {scores['Acc'] * 100:.1f}%", flush=True)
    print(f"  I_Acc: {scores['I_Acc'] * 100:.1f}%", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  External baseline (Qwen3-VL-4B, default prompt): I_Acc 26.2%", flush=True)
    print(f"  Paper Qwen3-VL-4B:                               I_Acc 17.7%", flush=True)


if __name__ == "__main__":
    main()
