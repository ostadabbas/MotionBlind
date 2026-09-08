#!/usr/bin/env python
"""Gemma-4-12B-it (google/gemma-4-12B-it) eval on the TimeBlind / MotionBlind subset.

Sibling of eval_qwen3.py: same scorer (scoring.py), same data, same axes
(--sampling / --num-frames / --prompt-template / --system-prompt / --ablation),
and the same cache-backed frame source (--uniform-h5 / --random-h5 /
--hornet-json). Runs are comparable to the Qwen sweep at matched frames.

Constraints: model_type `gemma4_unified` has a native video path
(Gemma4UnifiedVideoProcessor, num_frames=32, 70 soft tokens/frame vs 280 for a
still image); frames go in as one `{"type": "video"}` item, not N image items.
Requires transformers >= 5.10 (the model_type is unknown to 4.57.x), so it runs
in its own conda env (`gemma4`); the pinned `mllm2` env stays on 4.57.5 for
Qwen/Eagle. Use the `-it` repo: the base `google/gemma-4-12B` ships no chat
template, so apply_chat_template raises. Per the model card: `enable_thinking=False`,
decode with special tokens, then `processor.parse_response()`. Greedy, 128 new tokens.
"""

import argparse
import datetime
import faulthandler
import gc
import json
import os
import sys
import time
import traceback

# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))
from eval_common import add_f2c_args, apply_f2c_scales, load_f2c_selector  # noqa: E402

from tqdm import tqdm

# Dump a native stack on segfault/abort (decord/ffmpeg, CUDA) instead of a silent .out.
faulthandler.enable()

import numpy as np
import torch
from transformers import AutoProcessor

# The model card loads Gemma-4 via AutoModelForMultimodalLM; fall back to the concrete class
# and then to the image-text-to-text auto class depending on the transformers build.
_VLM = _VLM_NAME = None
for _name in ("AutoModelForMultimodalLM", "Gemma4UnifiedForConditionalGeneration",
              "AutoModelForImageTextToText"):
    try:
        import transformers as _tf
        _VLM = getattr(_tf, _name)
        _VLM_NAME = _name
        break
    except Exception:  # pragma: no cover - depends on transformers version
        continue
if _VLM is None:
    raise ImportError("no usable Gemma-4 model class in this transformers build "
                      "(need >= 5.10 for model_type `gemma4_unified`)")


def parse_args():
    p = argparse.ArgumentParser()
    # Use the -it repo. The base google/gemma-4-12B ships no chat template, so
    # apply_chat_template fails.
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--timeblind-repo", default="./TimeBlind",
                   help="Path to cloned TimeBlind repo (for scoring.py)")
    p.add_argument("--base-path", required=True, help="Root dir holding the video folders")
    p.add_argument("--data", required=True, help="Path to data.jsonl")
    p.add_argument("--out", default=None,
                   help="Predictions JSON path (default: the repo naming convention under results/)")
    p.add_argument("--dtype", default="bfloat16",
                   help="'auto' or a torch dtype like bfloat16 (Gemma runs in bf16).")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Number of frames to feed. Required for uniform/random/hornet; "
                        "omit only with --sampling native (processor picks 32) or --ablation no_video.")
    p.add_argument("--sampling", default="uniform",
                   choices=["native", "uniform", "random", "hornet", "f2cfull"],
                   help="Frame selection: native (let Gemma's own video processor sample the "
                        "clip; its default is 32 frames), uniform (evenly spaced N), random "
                        "(N random frames, temporal order), or hornet (top-N via HORNet "
                        "keep_prob, from the cache only). The last three need --num-frames.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for --sampling random and the shuffled_frames permutation.")
    add_f2c_args(p)
    p.add_argument("--ablation", default="none",
                   choices=["none", "shuffled_frames", "reversed_frames", "no_video"],
                   help="Integrity probe. shuffled_frames scrambles the sampled frame order "
                        "(deterministic per video and --seed); reversed_frames plays the same "
                        "frames backwards; no_video sends the question with no visual input "
                        "(the language-prior baseline). The frame probes require --sampling "
                        "uniform or random.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = off (baseline is greedy with no penalty).")
    p.add_argument("--prompt-template", default="default", choices=["default", "cot"],
                   help="'default' = bare question (baseline); 'cot' = describe-changes scaffold.")
    p.add_argument("--system-prompt", default="none",
                   choices=["none", "debias", "temporal", "both"],
                   help="System message prepended to the chat (see SYSTEM_PROMPTS).")
    p.add_argument("--no-metadata", action="store_true",
                   help="Do not pass video_metadata (fps + original frame indices) to the "
                        "processor. Gemma-4 then assumes fps=24 over the bare frame list, so K "
                        "frames spanning a whole clip are presented as K consecutive frames. "
                        "Off by default (metadata on); this flag measures how much "
                        "that timestamp distortion moves the score.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Only evaluate the first N samples (smoke test).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also write the sampled frames to disk (for sampling studies).")
    p.add_argument("--frames-dir", default=None,
                   help="Where to write frames (default: <out dir>/frames/<video>/).")
    # Cache mode: source frames from the pre-extracted caches (keyed by clip stem) instead of
    # decoding mp4s. Selection is done once and reused byte-identically across models.
    p.add_argument("--uniform-h5", default=None,
                   help="Cache mode (--sampling uniform): read K frames from this uniform-32 HDF5 pool.")
    p.add_argument("--random-h5", default=None,
                   help="Cache mode (--sampling random): read K frames from this HDF5 (group k{N}).")
    p.add_argument("--hornet-json", default=None,
                   help="Cache mode (--sampling hornet): read the top-N HORNet frames from this "
                        "selections JSON, using --uniform-h5 for the actual pixels.")
    # Selection mode: the standard path. Frames come from the frozen selections/<cell>.json that
    # every other model in the study was scored on, so cells are comparable. Overrides cache mode.
    # The --uniform-h5/--random-h5 pool path does not reproduce these (uniform sub-slices
    # a 32-frame pool; random uses a different RNG).
    p.add_argument("--selections-dir", default=None,
                   help="Frozen-selection dir, e.g. data/mbhuman/selections. Enables selection mode.")
    p.add_argument("--cell", default=None,
                   help="Selection cell name (default: <sampling><num-frames>, e.g. uniform8).")
    p.add_argument("--cache-h5", default=None,
                   help="Selection-backed frame cache (data/mbhuman/videos_mbhuman.h5). Omit to "
                        "decode from mp4 at the selection's indices: same pixels, no 100 GB cache.")
    return p.parse_args()


def make_fix_path(base_path):
    def fix_path(video_path):
        path = os.path.join(base_path, video_path)
        if os.path.exists(path):
            return path
        parts = video_path.split("/")
        for i, part in enumerate(parts):
            candidate = os.path.join(base_path, *parts[:i], part + " ", *parts[i + 1:])
            if os.path.exists(candidate):
                return candidate
        return path
    return fix_path


# System messages for the --system-prompt lever (identical text to eval_qwen3.py so the axis
# is comparable across models). `none` keeps the model default.
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
    if template == "cot":
        if task_type == "multiple_choice":
            ending = "End your response with exactly 'Answer: A' or 'Answer: B'."
        elif task_type == "multiple_choice_4":
            ending = "End your response with exactly 'Answer: A', 'Answer: B', 'Answer: C', or 'Answer: D'."
        else:
            ending = "End your response with exactly 'Answer: Yes' or 'Answer: No'."
        return (
            "Watch the video carefully. Describe what changes between the first and last frame "
            "in 2-3 sentences. Then answer the question.\n\n"
            f"Question: {question}\n\n"
            + ending
        )
    return question


def main():
    args = parse_args()
    # Derive the internal booleans from --ablation.
    args.no_video = args.ablation == "no_video"
    args.shuffle_frames = args.ablation == "shuffled_frames"
    args.reverse_frames = args.ablation == "reversed_frames"
    if not args.out:
        from eval_common import default_out
        args.out = default_out(args, "gemma4_12b_it")

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import (reverse_frames, save_frames, shuffle_frames,
                          sample_random_frames, sample_uniform_frames)
    from cache_frames import _fps_of        # needed for video_metadata in both cache and live modes

    explicit = args.sampling in ("uniform", "random", "hornet")
    if explicit and not args.no_video and not args.num_frames:
        sys.exit(f"--sampling {args.sampling} requires --num-frames N")
    # Cache mode: the cache matching --sampling was supplied -> pull frames from it (no live decode).
    use_sel = bool(args.selections_dir)
    # Paper F2C: query-conditioned live selection (deterministic given encoder+params);
    # None for every other sampler. Frozen selection/cache paths are untouched.
    f2c_pick = load_f2c_selector(args, args.timeblind_repo)
    cell = args.cell or (f"{args.sampling}{args.num_frames}" if explicit else None)
    use_cache = (not use_sel) and {"uniform": bool(args.uniform_h5), "random": bool(args.random_h5),
                 "hornet": bool(args.hornet_json)}.get(args.sampling, False)
    if args.sampling == "hornet" and not use_sel and not args.hornet_json:
        sys.exit("--sampling hornet on Gemma-4 is CACHE-ONLY: pass --hornet-json (+ --uniform-h5 pixels)")
    if args.sampling == "hornet" and not use_sel and args.hornet_json and not args.uniform_h5:
        sys.exit("--hornet-json needs --uniform-h5 (the pool holding the actual frame pixels)")
    if (args.shuffle_frames or args.reverse_frames) and args.sampling not in ("uniform", "random"):
        sys.exit(f"--ablation {args.ablation} requires --sampling uniform or random "
                 "(needs an explicit frame list)")

    # Selection mode: load the cell's frozen frame list once; open the union cache if given.
    sel_items = sel_h5 = None
    if use_sel:
        import h5py
        from cache_frames import (load_selection, frames_from_selection,
                                  frames_from_video_selection, stem_of)
        sel_items = load_selection(args.selections_dir, cell)
        if args.cache_h5:
            sel_h5 = h5py.File(args.cache_h5, "r")
        print(f"[selection] cell={cell} n={len(sel_items)} src="
              f"{args.cache_h5 or 'mp4 (live decode)'}", flush=True)

    # Open the caches once (keyed by clip stem). uni_h5 backs both uniform sampling and hornet pixels.
    uni_h5 = rand_h5 = hornet_sel = None
    if use_cache:
        import h5py
        from cache_frames import (uniform_from_pool_h5, random_from_h5,
                                  hornet_from_selection, stem_of,
                                  uniform_indices_from_pool_h5, random_indices_from_h5,
                                  hornet_indices_from_selection)
        if args.uniform_h5:
            uni_h5 = h5py.File(args.uniform_h5, "r")
        if args.random_h5:
            rand_h5 = h5py.File(args.random_h5, "r")
        if args.hornet_json:
            hornet_sel = json.load(open(args.hornet_json))
        print(f"[cache] {args.sampling}: uniform_h5={bool(uni_h5)} random_h5={bool(rand_h5)} "
              f"hornet_json={bool(hornet_sel)}", flush=True)

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
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Model loaded: {args.model} (dtype={model.dtype})", flush=True)

    fix_path = make_fix_path(args.base_path)
    data = _load_json_list(args.data)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples", flush=True)

    p = fix_path(data[0]["video_path"])
    print(f"Path check: {p}", flush=True)
    print(f"Exists: {os.path.exists(p)}", flush=True)

    class _BoundedMemo(dict):
        """(key) -> (frames, idxs, total), capped. Consecutive items share a clip, so a
        tiny cache captures the reuse. The cache must stay bounded: an unbounded memo
        holds about 80 GB of PIL frames at K=24 on TimeBlind."""

        def __init__(self, cap=8):
            super().__init__()
            self.cap = cap
            self._order = []

        def __setitem__(self, k, v):
            if k in self:
                self._order.remove(k)
            elif len(self._order) >= self.cap:
                super().__delitem__(self._order.pop(0))
            super().__setitem__(k, v)
            self._order.append(k)

    _frame_memo = _BoundedMemo()   # cache mode: reused across a video's items, capped for TB

    def get_frames(video_path, question=None):
        """(frames, meta) for uniform/random/hornet; (None, None) for native or the no_video ablation.

        `meta` carries the original frame indices + source fps so Gemma-4 can compute true
        timestamps (frames_indices / fps). Without it the processor assumes fps=24 over the
        bare frame list, i.e. presents a whole-clip sample as consecutive frames."""
        if args.no_video or not explicit:
            return None, None
        full_path = fix_path(video_path)
        if f2c_pick is not None:
            fr, f2c_scales = f2c_pick(full_path, question, args.num_frames)
            return apply_f2c_scales(fr, f2c_scales), None
        idxs = total = None
        if use_sel:
            memo_key = (video_path, cell)
            if memo_key not in _frame_memo:
                if sel_h5 is not None:
                    fr, idxs, total, _fps = frames_from_selection(sel_h5, sel_items, video_path)
                else:
                    fr, idxs, total, _fps = frames_from_video_selection(
                        sel_items, video_path, full_path)
                _frame_memo[memo_key] = (fr, idxs, total)
            frames, idxs, total = _frame_memo[memo_key]
            frames = list(frames)
        elif use_cache:
            stem = stem_of(video_path)
            memo_key = (stem, args.num_frames)
            if memo_key not in _frame_memo:
                if args.sampling == "random":
                    frames = random_from_h5(rand_h5, stem, args.num_frames)
                    idxs, total = random_indices_from_h5(rand_h5, stem, args.num_frames)
                elif args.sampling == "hornet":
                    frames = hornet_from_selection(hornet_sel, uni_h5, stem, args.num_frames)
                    idxs, total = hornet_indices_from_selection(hornet_sel, uni_h5, stem,
                                                                args.num_frames)
                else:                             # uniform
                    frames = uniform_from_pool_h5(uni_h5, stem, args.num_frames)
                    idxs, total = uniform_indices_from_pool_h5(uni_h5, stem, args.num_frames)
                _frame_memo[memo_key] = (frames, idxs, total)
            frames, idxs, total = _frame_memo[memo_key]
            frames = list(frames)
        elif args.sampling == "random":
            frames = sample_random_frames(full_path, args.num_frames, args.seed)
        else:                                     # uniform (live decode)
            frames = sample_uniform_frames(full_path, args.num_frames)

        meta = None
        if not args.no_metadata:
            try:
                fps = _fps_of(full_path)
                if idxs is None:                  # live decode: mirror the sampler's own choice
                    import decord
                    total = len(decord.VideoReader(full_path))
                    k = max(1, min(int(args.num_frames), total))
                    idxs = np.linspace(0, total - 1, k).round().astype(int).tolist()
                meta = {"fps": fps, "frames_indices": list(idxs),
                        "total_num_frames": int(total)}
            except Exception as _e:
                print(f"[warn] no video_metadata for {video_path} ({_e}); "
                      f"processor will assume fps=24", flush=True)

        if args.shuffle_frames:
            frames = shuffle_frames(frames, full_path, args.seed)
        if args.reverse_frames:
            frames = reverse_frames(frames)
            if meta is not None:                  # order is the probe; timestamps must follow it
                meta = dict(meta, frames_indices=sorted(meta["frames_indices"]))
        return frames, meta

    def run_one(video_path, question, task_type):
        full_path = fix_path(video_path)
        prompt = build_prompt(question, task_type, args.prompt_template)
        frames, vmeta = get_frames(video_path, question)

        if args.save_frames and frames is not None:
            try:
                stem = os.path.splitext(os.path.basename(video_path))[0]
                save_frames(frames, os.path.join(frames_root, stem))
            except Exception as _e:
                print(f"[warn] frame save failed: {_e}", flush=True)

        messages = []
        sys_text = SYSTEM_PROMPTS[args.system_prompt]
        if sys_text is not None:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": sys_text}]})
        # Gemma-4 has a native video path: hand the frame list (or the mp4 path, for
        # --sampling native) to the video processor as one video item.
        user_content = []
        if not args.no_video:
            user_content.append({"type": "video",
                                 "video": frames if frames is not None else full_path})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # enable_thinking=False: the baseline is a bare greedy answer, not a reasoning trace
        # (Gemma-4 emits <|think|> content otherwise, which would blow past max_new_tokens
        # before reaching the Yes/No and mirror the CoT bias that floors I_Acc).
        # do_sample_frames=False: the sampler above already chose the frames (the
        # frame choice is the experiment).
        # Left on, Gemma's video processor re-samples to its own num_frames=32 and hard-fails for
        # K<32 ("num_frames=32 exceeds total_num_frames=K").
        tmpl_kwargs = {"enable_thinking": False}
        if frames is not None:
            tmpl_kwargs["do_sample_frames"] = False
            if vmeta is not None:                 # real timestamps (frames_indices / fps)
                tmpl_kwargs["video_metadata"] = [vmeta]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **tmpl_kwargs,
        ).to(model.device)
        if "pixel_values" in inputs:               # cast pixels to the model dtype, leave ids int
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        num_frames = (len(frames) if frames is not None
                      else (0 if args.no_video else args.num_frames or 32))
        in_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
            )
        gen_ids = output[:, in_len:] if output.shape[1] > in_len else output
        # Gemma-4 wraps its reply in channel/turn markers, so decode with special tokens and
        # let processor.parse_response strip them (the model card's own recipe). Fall back to a
        # plain decode if this build has no parse_response.
        raw = processor.decode(gen_ids[0], skip_special_tokens=False)
        response = raw
        if hasattr(processor, "parse_response"):
            try:
                # prefix= is required: the chat template can pre-open part of the assistant
                # message (e.g. a <think> tag), and the parser needs to see it to split
                # thinking from content correctly.
                prompt_text = processor.decode(inputs["input_ids"][0],
                                               skip_special_tokens=False)
                parsed = processor.parse_response(raw, prefix=prompt_text)
                if isinstance(parsed, dict):       # {'content': ..., 'thinking': ...}
                    response = parsed.get("content") or parsed.get("text") or raw
                elif isinstance(parsed, str):
                    response = parsed
            except Exception as _e:
                if _err_dbg["n"] < 3:
                    print(f"[warn] parse_response failed ({_e}); using raw decode", flush=True)
        if response is raw:                        # no parser -> at least drop the special tokens
            response = processor.decode(gen_ids[0], skip_special_tokens=True)
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

    predictions = []
    _err_dbg = {"n": 0}
    start_time = time.time()

    pbar = tqdm(total=len(data), desc=f"{args.sampling} K={args.num_frames}",
                dynamic_ncols=True, mininterval=2.0)
    _seen = {"n": 0, "correct": 0}
    for i, sample in enumerate(data):
        prev = done.get(sample["index"])
        if prev is not None and prev.get("video_path") == sample["video_path"]:
            predictions.append(prev)
            pbar.update(1)
            continue
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

        extracted = extract_answer(model_output, sample["type"])
        gt = extract_answer(sample["answer"], sample["type"])
        correct = extracted == gt

        _seen["n"] += 1
        _seen["correct"] += int(bool(correct))
        pbar.update(1)
        pbar.set_postfix(acc=f"{100 * _seen['correct'] / _seen['n']:.1f}%")

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(data) - i - 1)

        print(f"\n{'=' * 70}", flush=True)
        print(f"[{i+1}/{len(data)}] {sample['video_path'].split('/')[-1]} | "
              f"frames: {num_frames} | GT: {sample['answer']} | "
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
            "frames": {"strategy": args.sampling, "seed": args.seed,
                       "shuffle": args.shuffle_frames,
                       "reverse": args.reverse_frames,
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
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)

    pbar.close()
    with open(args.out, "w") as f:
        json.dump(predictions, f, indent=2)

    # Winoground scoring assumes contrastive 4-item groups; Video-MME (multiple_choice_4)
    # is non-contrastive and scored by plain accuracy only.
    contrastive = all(s["type"] != "multiple_choice_4" for s in data)
    if contrastive:
        answers = build_answers(predictions, data)
        scores = get_scores(answers)

    total_time = time.time() - start_time
    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | SAMPLING: {args.sampling}"
          f"{' +shuffle' if args.shuffle_frames else ''}"
          f"{' +reverse' if args.reverse_frames else ''} | FRAMES: "
          f"{args.num_frames if not args.no_video else 'none (text-only)'} | "
          f"PROMPT: {args.prompt_template} | SYS: {args.system_prompt}", flush=True)
    print(f"Time: {total_time / 60:.1f}min | Samples: {len(data)}", flush=True)
    print(f"{'=' * 50}", flush=True)
    if contrastive:
        print(f"  Q_Acc: {scores['Q_Acc'] * 100:.1f}%", flush=True)
        print(f"  V_Acc: {scores['V_Acc'] * 100:.1f}%", flush=True)
        print(f"  Acc:   {scores['Acc'] * 100:.1f}%", flush=True)
        print(f"  I_Acc: {scores['I_Acc'] * 100:.1f}%", flush=True)
    else:
        n_ok = sum(1 for s, p in zip(data, predictions)
                   if extract_answer(p["model_output"], s["type"])
                   == extract_answer(s["answer"], s["type"]))
        print(f"  Acc (plain, 4-way MC): {100 * n_ok / max(1, len(data)):.1f}%", flush=True)
    print(f"{'=' * 50}", flush=True)


if __name__ == "__main__":
    main()
