#!/usr/bin/env python
"""Motion-o (Qwen2.5-VL-7B-Instruct fine-tune) eval on the TimeBlind Challenge subset.

Motion-o is a native Qwen2.5-VL architecture (no trust_remote_code), so it loads
with `Qwen2_5_VLForConditionalGeneration` and honors `attn_implementation="sdpa"`
directly; the Eagle attention workaround is not needed. Native video input via
qwen_vl_utils.process_vision_info.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time

_T0 = time.time()   # process start; --max-runtime is measured from here
import traceback

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig,
)

# Read the video with decord (installed, reliable on these mp4s). qwen_vl_utils reads
# this env at import time, so set it before the import.
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
    p.add_argument("--model", default="bishoygaloaa/motion-o")
    p.add_argument("--subfolder", default="motion-o",
                   help="HF subfolder holding the weights (repo has 3 variants: "
                        "'motion-o' [main], 'open-o3-mcot', 'open-o3-mcot-no-vg').")
    p.add_argument("--timeblind-repo", default="./TimeBlind",
                   help="Path to cloned TimeBlind repo (for scoring.py)")
    add_hf_video_args(p)
    p.add_argument("--max-runtime", type=float, default=None, metavar="SECONDS",
                   help="stop cleanly after this many seconds and save, so a chained "
                        "SLURM job exits before the wall limit kills it mid-write. "
                        "The next link resumes from the saved predictions.")
    p.add_argument("--base-path", required=True,
                   help="Root dir holding the video folders")
    p.add_argument("--data", required=True, help="Path to data.jsonl")
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max-pixels", type=int, default=360 * 420)
    p.add_argument("--num-frames", type=int, default=None,
                   help="If set, sample exactly N uniform frames across the whole video "
                        "(nframes; overrides --fps).")
    p.add_argument("--sampling", default="fps", choices=["fps", "uniform", "random", "hornet", "f2cfull"],
                   help="Frame selection: fps (fixed rate), uniform (evenly spaced N), "
                        "random (N random frames), or hornet (learned top-N via HORNet policy). "
                        "uniform/random/hornet need --num-frames.")
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
                        "(std < --hornet-min-std), use the base model's default uniform-N "
                        "sampling instead of a possibly-worse-than-base learned pick. "
                        "Off by default -> byte-identical to the stock selector.")
    p.add_argument("--hornet-min-std", type=float, default=0.0,
                   help="keep_prob std below which the policy is treated as 'no confident "
                        "preference' and --hornet-fallback kicks in. 0.0 = only hard failures / "
                        "non-finite outputs. Raise (e.g. 0.02) to floor HORNet at uniform when "
                        "the selector is near-flat.")
    p.add_argument("--hornet-pool", type=int, default=32,
                   help="HORNet candidate-pool size: uniformly presample this many frames @288 then keep "
                        "top-(--num-frames). Default 32 (top-32 of 32 == uniform-32, no real selection). "
                        "Set e.g. 64 for a real selection: top-K of a denser pool.")
    add_f2c_args(p)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--prompt-template", default="cot", choices=["default", "cot"],
                   help="'cot' = describe-changes scaffold (baseline); 'default' = bare question "
                        "(the +17.5 I_Acc debias lever on Eagle).")
    p.add_argument("--system-prompt", default="none",
                   choices=["none", "debias", "temporal", "both"],
                   help="System message prepended to the chat. 'none' = model default "
                        "(matches the frame-sampling sweep). 'debias' = anti-yes-bias, "
                        "'temporal' = attend to event order/timing, 'both' = combined. "
                        "A separate axis from --prompt-template.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Only evaluate the first N samples (smoke test).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also write the sampled frames to disk (for sampling studies).")
    p.add_argument("--frames-dir", default=None,
                   help="Where to write frames (default: <out dir>/frames/<video>/).")
    return p.parse_args()


from eval_common import (add_hf_video_args, is_mcq4,  # noqa: E402
                         make_fix_path, resolve_fix_path)

# System messages for the --system-prompt lever. `none` keeps the model default
# (Qwen2.5-VL "You are a helpful assistant."), matching the frame-sampling sweep.
# The others target the two things that cap I_Acc on TimeBlind/MotionBlind: yes/no
# compliance bias (debias) and failure to read event order/timing (temporal).
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
    # `question` already carries the "Please output Yes or No." / "A or B." suffix.
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
    # Derive the internal booleans from --ablation.
    args.no_video = args.ablation == "no_video"
    args.shuffle_frames = args.ablation == "shuffled_frames"
    args.reverse_frames = args.ablation == "reversed_frames"
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "motion_o")

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import (save_frames, save_hornet_frames, shuffle_frames, reverse_frames,
                          sample_random_frames, sample_uniform_frames)

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

    gc.collect()
    torch.cuda.empty_cache()

    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        # Native Qwen2.5-VL honors sdpa; no extra attention build is needed.
        attn_implementation="sdpa",
    )
    proc_kwargs = {}
    if args.subfolder:
        load_kwargs["subfolder"] = args.subfolder
        proc_kwargs["subfolder"] = args.subfolder

    # Motion-o's fine-tuned config.json dropped `rope_scaling`, but Qwen2.5-VL's
    # attention forward does `self.rope_scaling["mrope_section"]` -> at generate time it
    # crashes with "'NoneType' object is not subscriptable". Restore the stock
    # Qwen2.5-VL-7B M-RoPE section so the config is complete before the model is built
    # (the attention modules capture rope_scaling at init, so this must precede load).
    cfg = AutoConfig.from_pretrained(
        args.model, **({"subfolder": args.subfolder} if args.subfolder else {}))
    rs = getattr(cfg, "rope_scaling", None)
    if not isinstance(rs, dict) or "mrope_section" not in rs:
        cfg.rope_scaling = {"type": "default", "mrope_section": [16, 24, 24]}
        print("Patched config.rope_scaling -> mrope_section [16,24,24]", flush=True)
    load_kwargs["config"] = cfg

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
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

    # Aggregate diagnostics for the fallback lever (how often the policy had no signal).
    # f2c needs the question, so it is called per item rather than per video.
    # This driver has no frame cache (unlike eval_qwen3.py).
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

    fix_path, vsrc = resolve_fix_path(args)
    data = _load_json_list(args.data)
    # Video-MME is 4-way MCQ: scoring's 2-way extractor and get_scores() are both wrong here.
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

    def run_one(video_path, question, task_type):
        full_path = fix_path(video_path)
        prompt = build_prompt(question, task_type, args.prompt_template)
        frames = None                          # explicit frame list (uniform/random/hornet)
        if args.sampling == "random":          # N randomly chosen frames (temporal order)
            frames = sample_random_frames(full_path, args.num_frames, args.seed)
        elif args.sampling == "hornet":        # learned top-N frames (HORNet policy over a uniform-32 pre-sample)
            frames = hornet_pick(full_path)    # optional graceful fallback to uniform-N (--hornet-fallback)
        elif f2c_pick is not None:             # query-conditioned key clips
            frames, _scales = f2c_pick(full_path, question, args.num_frames)
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

        # hornet dumps its own selected/unselected pool in hornet_pick; the generic
        # (post-processor) save applies only to the other samplers.
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
                repetition_penalty=args.repetition_penalty,
            )
        # Native Qwen2.5-VL returns prompt+continuation, so trim the prompt; conditional
        # so it's robust if a build returns only new tokens (Eagle's failure mode).
        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, in_len:] if output.shape[1] > in_len else output
        response = processor.batch_decode(
            gen_ids, skip_special_tokens=True,
        )[0]
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        return response, num_frames

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                # Only skip items that actually succeeded; retry empty/errored ones on
                # resume (else a failed run cements its failures and re-scores garbage).
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} existing predictions", flush=True)

    predictions = []
    _err_dbg = {"n": 0}  # one-shot full traceback on the first failures
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
            model_output, num_frames = run_one(sample["video_path"], question, sample["type"])
        except Exception as e:
            model_output, num_frames = "", 0
            error = str(e)
            if _err_dbg["n"] < 2:  # full traceback for the first couple of failures
                _err_dbg["n"] += 1
                traceback.print_exc()

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
            # Frame provenance: fps (uniform-in-time) or uniform-N across the whole clip
            # (--num-frames). Selection is internal to the processor; count recorded here.
            "frames": {"strategy": strategy, "seed": args.seed,
                       "shuffle": args.shuffle_frames,
                       "reverse": args.reverse_frames,
                       "hornet_load": (args.hornet_load if args.sampling == "hornet" else None),
                       "fps": args.fps, "num_frames": num_frames,
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
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)

        # See eval_eagle.py: the chained worker needs a clean stop, not a SIGKILL.
        if args.max_runtime and (time.time() - _T0) > args.max_runtime:
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)
            print(f"\n[timebox] stopping at {i + 1}/{len(data)} after "
                  f"{(time.time() - _T0) / 60:.1f}min; the next chained job resumes here.",
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
        # src/analysis/make_tables.py reads the _metrics.json sidecar, not the SLURM
        # log; without the sidecar the leaderboard's Video-MME column stays empty.
        # Selector settings are recorded too: two hornet cells with the same load mode
        # and different checkpoints are different experiments, not re-runs.
        meta = {"model_name": args.model, "benchmark": "videomme",
                "sampling": strategy, "num_frames": args.num_frames,
                "seed": args.seed, "prompt_template": args.prompt_template,
                "fps": args.fps,
                "shuffle": args.shuffle_frames, "reverse": args.reverse_frames,
                "hornet_load": (args.hornet_load if args.sampling == "hornet" else None),
                "hornet_ckpt": (args.hornet_ckpt if args.sampling == "hornet" else None),
                "hornet_repo": (args.hornet_repo if args.sampling == "hornet" else None),
                "f2c_clip": (args.f2c_clip if args.sampling in ("f2c", "f2cfull") else None),
                "f2c_pool": (args.f2c_pool if args.sampling == "f2c" else None),
                "f2c_s_max": (args.f2c_s_max if args.sampling == "f2cfull" else None),
                "f2c_lambda_r": (args.f2c_lambda_r if args.sampling == "f2cfull" else None),
                "f2c_lambda_l": (args.f2c_lambda_l if args.sampling == "f2cfull" else None)}
        side = os.path.splitext(args.out)[0] + "_metrics.json"
        with open(side, "w") as f:
            json.dump({**scores, "config": meta}, f, indent=2)
        print(f"metrics -> {side}", flush=True)
        if vsrc is not None:
            print(vsrc.summary(), flush=True); vsrc.close()
        return

    answers = build_answers(predictions, data)
    scores = get_scores(answers)

    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | SAMPLING: {strategy}"
          f"{' +shuffle' if args.shuffle_frames else ''}"
          f"{f' load={args.hornet_load}' if args.sampling == 'hornet' else ''} | "
          f"FRAMES: {args.num_frames if args.num_frames else f'fps {args.fps}'} | "
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
    print(f"  Qwen3-VL-4B (8 FPS, no CoT): I-Acc 10.3%", flush=True)
    print(f"  Paper Qwen3-VL-4B:           I-Acc 17.7%", flush=True)
    if vsrc is not None:
        print(vsrc.summary(), flush=True); vsrc.close()


if __name__ == "__main__":
    main()
