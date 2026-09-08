#!/usr/bin/env python
"""Eagle2.5-8B driver: runs a benchmark and scores the run.

    python src/eval/eval_eagle.py --base-path data --data data/data.jsonl \
        --sampling uniform --num-frames 16 --out results/eagle_mb_uniform16.json

Works on MotionBlind, TimeBlind, and Video-MME; mcq4 rows route automatically.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time

_T0 = time.time()   # process start; --max-runtime is measured from here

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nvidia/Eagle2.5-8B")
    p.add_argument("--timeblind-repo", default="./TimeBlind",
                   help="Path to cloned TimeBlind repo (for scoring.py)")
    add_hf_video_args(p)
    p.add_argument("--max-runtime", type=float, default=None, metavar="SECONDS",
                   help="Stop cleanly and save after this many seconds. A rerun "
                        "resumes from the saved predictions.")
    p.add_argument("--base-path", required=True,
                   help="Root dir holding the video folders")
    p.add_argument("--data", required=True,
                   help="Path to data.jsonl")
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--max-pixels", type=int, default=360 * 420)
    p.add_argument("--num-frames", type=int, default=None,
                   help="Sample exactly N uniform frames across the whole video. "
                        "Overrides --fps.")
    p.add_argument("--sampling", default="fps",
                   choices=["fps", "uniform", "random", "hornet", "f2cfull"],
                   help="Frame selection: fps (fixed rate), uniform (evenly spaced N), "
                        "random (N random frames, temporal order), hornet (learned top-N "
                        "selector), or f2cfull (paper Frames-to-Clips). "
                        "uniform/random/hornet/f2cfull need --num-frames; hornet also needs "
                        "--hornet-repo/--hornet-ckpt.")
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
                   help="How to load the HORNet checkpoint. full loads every tensor and "
                        "asserts a complete match. no-encoder drops the encoder tensors, "
                        "so the vision encoder keeps its random init.")
    p.add_argument("--hornet-fallback", action="store_true",
                   help="On a HORNet error, a non-finite keep_prob, or std below "
                        "--hornet-min-std, fall back to uniform-N.")
    p.add_argument("--hornet-min-std", type=float, default=0.0,
                   help="keep_prob std threshold for --hornet-fallback. 0.0 falls back "
                        "only on hard failures and non-finite outputs.")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--prompt-template", default="cot", choices=["default", "cot"],
                   help="cot adds the describe-then-answer scaffold (the paper default). "
                        "default sends the bare question.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Evaluate the first N samples only, for smoke tests.")
    p.add_argument("--save-frames", action="store_true",
                   help="Write the sampled frames to disk, for sampling studies.")
    p.add_argument("--frames-dir", default=None,
                   help="Where to write frames (default: <out dir>/frames/<video>/).")
    add_f2c_args(p)
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
from eval_common import (add_hf_video_args, apply_f2c_scales, is_mcq4,  # noqa: E402
                         make_fix_path, resolve_fix_path,
                         add_f2c_args, load_f2c_selector)

def build_prompt(question, task_type, template):
    # The question already carries the answer-format suffix; the loop adds it.
    if template == "default":
        return question  # bare question, no scaffold
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




def main():
    args = parse_args()
    args.no_video = args.ablation == "no_video"
    args.shuffle_frames = args.ablation == "shuffled_frames"
    args.reverse_frames = args.ablation == "reversed_frames"
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "eagle2_5_8b")

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import (save_frames, shuffle_frames, reverse_frames, save_hornet_frames,
                          sample_random_frames, sample_uniform_frames)

    if args.sampling in ("uniform", "random", "hornet", "f2cfull") and not args.num_frames:
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

    gc.collect()
    torch.cuda.empty_cache()

    # Eagle's remote code hardcodes flash_attention_2 on the vision tower, and
    # transformers rebuilds that module on load. No config value can override it.
    # Force every sub-model's attention to SDPA here.
    import transformers.modeling_utils as _tf_mu
    _tf_mu.PreTrainedModel._check_and_adjust_attn_implementation = (
        lambda self, *a, **k: "sdpa"
    )

    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
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

        # Report the checkpoint tensor groups, so the load is verifiable.
        def _grp(k):
            return ("encoder" if k.startswith("encoder.") else
                    "policy_head" if k.startswith("policy_head.") else
                    "policy" if k.startswith("policy.") else "other")
        from collections import Counter
        print(f"[hornet] checkpoint {os.path.basename(args.hornet_ckpt)}: {len(state)} "
              f"tensors by group {dict(Counter(_grp(k) for k in state))}", flush=True)

        # no-encoder drops the encoder tensors; the vision encoder keeps its
        # random init.
        if args.hornet_load == "no-encoder":
            state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
            print(f"[hornet] load=no-encoder -> dropped encoder.* ; vision encoder stays at "
                  f"RANDOM init ({len(state)} tensors will load)", flush=True)

        _probe = "encoder.patch_embed.proj.weight"       # tracks an encoder tensor
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

    # Query-conditioned; called per item (not per video). None unless --sampling f2cfull.
    f2c_pick = load_f2c_selector(args, args.timeblind_repo)

    hornet_stats = {"total": 0, "fallback": 0, "std_sum": 0.0, "std_n": 0}

    def hornet_pick(full_path):
        """Top-K HORNet selection over the uniform-32 candidate pool.

        Uses the repo's get_action_by_k reduction (deterministic top-k). Saves the
        full pool with --save-frames. Falls back per --hornet-fallback."""
        hornet_stats["total"] += 1
        try:
            videos, _total = hornet_load_frames(full_path)          # [32,H,W,3] @288, /255
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

    fix_path, vsrc = resolve_fix_path(args)
    data = _load_json_list(args.data)
    # Video-MME is 4-way MCQ with no contrastive quadruples. scoring's 2-way extractor and
    # get_scores() are both wrong for it, so route to videomme.
    mcq4 = is_mcq4(data)
    if mcq4:
        import videomme
        print(f"[bench] Video-MME: {len(data)} questions, 4-way MCQ, "
              f"accuracy only (chance 25%)", flush=True)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples", flush=True)

    p = fix_path(data[0]["video_path"])
    print(f"Path check: {p}", flush=True)
    print(f"Exists: {os.path.exists(p)}", flush=True)

    _dbg = {"n": 0}  # one-shot shape/decode debug on the first sample

    def run_one(video_path, question, task_type):
        full_path = fix_path(video_path)
        prompt = build_prompt(question, task_type, args.prompt_template)
        frames = None                          # explicit frame list (uniform/random/hornet/f2cfull)
        f2c_scales = None
        if args.sampling == "random":          # N randomly chosen frames (temporal order)
            frames = sample_random_frames(full_path, args.num_frames, args.seed)
        elif args.sampling == "hornet":        # learned top-N frames (HORNet policy over uniform-32)
            frames = hornet_pick(full_path)
        elif f2c_pick is not None:             # paper F2C: query-conditioned clips + scale s*
            frames, f2c_scales = f2c_pick(full_path, question, args.num_frames)
            frames = apply_f2c_scales(frames, f2c_scales)
        elif strategy == "uniform":            # exactly N evenly-spaced frames via explicit list
            frames = sample_uniform_frames(full_path, args.num_frames)  # nframes= breaks at N=1

        if frames is not None:
            if args.reverse_frames:            # direction probe: same frames, played backwards
                frames = reverse_frames(frames)
            elif args.shuffle_frames:          # temporal-blindness probe: scramble the order
                frames = shuffle_frames(frames, full_path, args.seed)
            video_item = {"type": "video", "video": frames, "max_pixels": args.max_pixels}
        else:                                  # fixed-rate fps
            video_item = {"type": "video", "video": full_path,
                          "fps": args.fps, "max_pixels": args.max_pixels}
        user_content = ([{"type": "text", "text": prompt}] if args.no_video
                        else [video_item, {"type": "text", "text": prompt}])
        messages = [{"role": "user", "content": user_content}]

        text_list = [processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs, video_kwargs = processor.process_vision_info(
            messages, return_video_kwargs=True)

        inputs = processor(
            text=text_list, images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True, videos_kwargs=video_kwargs,
        ).to("cuda")

        # Eagle's processor returns an explicit PIL-frame video as a list (Qwen
        # stacks it into a tensor). video_inputs[0] may not have .shape; count either way.
        _vi0 = video_inputs[0] if video_inputs else None
        num_frames = (_vi0.shape[0] if hasattr(_vi0, "shape")
                      else len(_vi0) if _vi0 is not None else 0)

        if args.save_frames and video_inputs and args.sampling != "hornet":  # hornet saves its own pool
            try:
                stem = os.path.splitext(os.path.basename(full_path))[0]
                save_frames(video_inputs[0], os.path.join(frames_root, stem))
            except Exception as _e:
                print(f"[warn] frame save failed: {_e}", flush=True)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
        # Some remote-code generate() implementations return only the new tokens;
        # others return prompt plus continuation. Trim only when the prompt is present.
        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, in_len:] if output.shape[1] > in_len else output
        if _dbg["n"] < 1:
            _dbg["n"] += 1
            full = processor.batch_decode(output, skip_special_tokens=False)[0]
            print(f"[debug] out_shape={tuple(output.shape)} in_len={in_len} "
                  f"gen_shape={tuple(gen_ids.shape)}", flush=True)
            print(f"[debug] full_decode(no-skip)[:500]={full[:500]!r}", flush=True)
        response = processor.batch_decode(
            gen_ids, skip_special_tokens=True,
        )[0]
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        return response, num_frames, f2c_scales

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                # Keep only rows that succeeded; a resumed run retries the rest.
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} existing predictions", flush=True)

    predictions = []
    start_time = time.time()

    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        # Reuse a cached row only when index and video_path both match. This guards
        # against resuming a file that was built for a different dataset.
        if prev is not None and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            continue
        question = (videomme.format_question(sample) if mcq4 else
                    add_question_suffix(sample["question"], sample["type"]))
        error = None
        try:
            model_output, num_frames, f2c_scales = run_one(
                sample["video_path"], question, sample["type"])
        except Exception as e:
            model_output, num_frames, f2c_scales = "", 0, None
            error = str(e)

        if mcq4:
            extracted = videomme.extract_choice(model_output, sample.get("options"))
            gt = videomme.gt_index(sample["answer"])
        else:
            extracted = extract_answer(model_output, sample["type"])
            gt = extract_answer(sample["answer"], sample["type"])
        correct = extracted == gt

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(data) - i - 1)

        print(f"\n{'=' * 70}", flush=True)
        print(f"[{i+1}/{len(data)}] {sample['video_path'].split('/')[-1]} | "
              # Qwen-VL merges frames in temporal pairs, so a 1-frame request reports 2.
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
            # Frame provenance for this row: strategy, seed, probe state, and count.
            "frames": {"strategy": strategy, "seed": args.seed,
                       "shuffle": args.shuffle_frames,
                       "reverse": args.reverse_frames,
                       "hornet_load": (args.hornet_load if args.sampling == "hornet" else None),
                       "f2c_scales": ([float(s) for s in f2c_scales]
                                      if f2c_scales is not None else None),
                       "fps": args.fps, "num_frames": num_frames,
                       "saved_dir": (os.path.join(
                           frames_root,
                           os.path.splitext(os.path.basename(sample["video_path"]))[0])
                           if args.save_frames else None)},
            "prompt_template": args.prompt_template,
            "error": error,
            "ts": datetime.datetime.now().isoformat(),
        })

        if (i + 1) % 25 == 0:
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)

        # Save and exit inside the runtime budget, so an external scheduler never
        # kills the run mid-write.
        if args.max_runtime and (time.time() - _T0) > args.max_runtime:
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)
            print(f"\n[timebox] stopping at {i + 1}/{len(data)} after "
                  f"{(time.time() - _T0) / 60:.1f}min; a rerun resumes here.",
                  flush=True)
            break

    with open(args.out, "w") as f:
        json.dump(predictions, f, indent=2)

    total_time = time.time() - start_time
    if mcq4:
        scores = videomme.score(predictions, data)
        print("\n\n" + videomme.format_summary(
            scores, args.model, strategy,
            args.num_frames if args.num_frames else 0, total_time), flush=True)
        if vsrc is not None:
            print(vsrc.summary(), flush=True); vsrc.close()
        return

    answers = build_answers(predictions, data)
    scores = get_scores(answers)

    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | SAMPLING: {strategy}"
          f"{' +shuffle' if args.shuffle_frames else ''} | "
          f"FRAMES: {args.num_frames if args.num_frames else f'fps {args.fps}'} | "
          f"PROMPT: {args.prompt_template}", flush=True)
    print(f"Time: {total_time / 60:.1f}min | Samples: {len(data)}", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  Q_Acc: {scores['Q_Acc'] * 100:.1f}%", flush=True)
    print(f"  V_Acc: {scores['V_Acc'] * 100:.1f}%", flush=True)
    print(f"  Acc:   {scores['Acc'] * 100:.1f}%", flush=True)
    print(f"  I_Acc: {scores['I_Acc'] * 100:.1f}%", flush=True)
    print(f"{'=' * 50}", flush=True)
    if vsrc is not None:
        print(vsrc.summary(), flush=True); vsrc.close()


if __name__ == "__main__":
    main()
