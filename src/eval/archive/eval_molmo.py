#!/usr/bin/env python
"""Molmo-7B-D evaluation on the TimeBlind Challenge subset.

Molmo has no native video input. The script samples frames and passes them
as a list of images in chronological order.
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

# Native crashes (decord/ffmpeg at import, CUDA init, flash-attn) die below Python and
# leave the .err silent. Arm faulthandler so a segfault/abort dumps a C+Python stack.
faulthandler.enable()

# Import torch before decord. Both link ffmpeg/OpenMP; importing decord first
# segfaults at load on this build.
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import decord


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/Molmo-7B-D-0924")
    p.add_argument("--timeblind-repo", default="./TimeBlind")
    p.add_argument("--base-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Only evaluate the first N samples (smoke test).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also write the sampled frames to disk (for sampling studies).")
    p.add_argument("--frames-dir", default=None,
                   help="Where to write frames (default: <out dir>/frames/<video>/).")
    return p.parse_args()


from eval_common import make_fix_path  # noqa: E402

def sample_frames(video_path, fps, max_frames, _dbg=[0]):
    # decord, not OpenCV: cv2's build here can't decode these mp4s (returns 0 frames);
    # decord reads them reliably (same reader Eagle/Motion-o use). Returns RGB already.
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps() or 24.0
    total_frames = len(vr)
    step = max(1, round(video_fps / fps))
    indices = list(range(0, total_frames, step))[:max_frames]
    if not indices:
        indices = list(range(min(total_frames, max_frames)))
    if _dbg[0] < 1:  # one-shot: confirm decord read a nonzero frame count
        _dbg[0] += 1
        print(f"[debug] sample_frames: total_frames={total_frames} avg_fps={video_fps} "
              f"step={step} n_indices={len(indices)}", flush=True)
    batch = vr.get_batch(indices).asnumpy()  # (T, H, W, C), RGB
    return [Image.fromarray(f) for f in batch], indices


def main():
    args = parse_args()

    sys.path.insert(0, args.timeblind_repo)
    from scoring import (
        _load_json_list, build_answers, get_scores,
        add_question_suffix, extract_answer,
    )
    from frame_io import save_frames

    frames_root = args.frames_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "frames")

    print(torch.cuda.get_device_name(0), flush=True)

    # Molmo's remote image-preprocessing imports tensorflow only for type detection
    # (isinstance(x, tf.Tensor)); the PyTorch path never computes with it. transformers'
    # check_imports demands the import exist, and an empty stub trips "module
    # 'tensorflow' has no attribute 'Tensor'". Stub dummy classes that PIL/numpy inputs
    # never match; real tensorflow would conflict with the pinned torch/numpy/transformers env.
    import types as _types
    _tf_stub = _types.ModuleType("tensorflow")
    _tf_stub.__version__ = "2.0.0"
    _tf_stub.Tensor = type("Tensor", (), {})
    _tf_stub.__getattr__ = lambda name: type(name, (), {})  # any other tf.<X> -> dummy class
    sys.modules["tensorflow"] = _tf_stub

    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # Molmo runs in the 4.45 env, which uses `torch_dtype` (the `dtype` alias is 4.56+).
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Model loaded: {args.model}", flush=True)

    fix_path = make_fix_path(args.base_path)
    data = _load_json_list(args.data)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} samples", flush=True)

    p = fix_path(data[0]["video_path"])
    print(f"Path check: {p}", flush=True)
    print(f"Exists: {os.path.exists(p)}", flush=True)

    _dbg = {"n": 0}  # one-shot shape/decode debug on the first sample

    def run_one(video_path, question, task_type):
        full_path = fix_path(video_path)
        frames, frame_indices = sample_frames(full_path, args.fps, args.max_frames)
        num_frames = len(frames)

        if args.save_frames:  # non-invasive: record the exact sampled frames
            try:
                stem = os.path.splitext(os.path.basename(full_path))[0]
                save_frames(frames, os.path.join(frames_root, stem))
            except Exception as _e:
                print(f"[warn] frame save failed: {_e}", flush=True)

        ending = (
            "End your response with exactly 'Answer: A' or 'Answer: B'."
            if task_type == "multiple_choice"
            else "End your response with exactly 'Answer: Yes' or 'Answer: No'."
        )
        prompt = (
            "The following images are frames sampled from a video in chronological order. "
            "Describe what changes between the first and last frame in 2-3 sentences. "
            "Then answer the question.\n\n"
            f"Question: {question}\n\n"
            + ending
        )

        inputs = processor.process(images=frames, text=prompt)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        # Model is bf16 but the processor emits float32 pixel tensors; the bf16 patch
        # embedding then errors ("mat1 and mat2 dtype: float != BFloat16"). Cast the
        # floating-point inputs to the model dtype; leave int tensors (ids/indices) alone.
        inputs = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v)
                  for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    stop_strings="<|endoftext|>",
                ),
                tokenizer=processor.tokenizer,
            )
        # generate_from_batch returns prompt+continuation (per Molmo's usage
        # example). A full decode would echo the instruction text; extract_answer
        # would then match the echoed "Answer: Yes", not the model's answer.
        # Drop the prompt tokens. Conditional so it stays robust if a version
        # returns only the new tokens.
        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[0, in_len:] if output.shape[1] > in_len else output[0]
        if _dbg["n"] < 1:
            _dbg["n"] += 1
            full = processor.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"[debug] out_shape={tuple(output.shape)} in_len={in_len} "
                  f"gen_len={gen_ids.shape[0]}", flush=True)
            print(f"[debug] full_decode[:300]={full[:300]!r}", flush=True)
        response = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True).strip()
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        return response, num_frames, frame_indices

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out)):
                # Only skip items that succeeded; retry empty/errored ones on resume
                # (else a failed run keeps its failures and re-scores them).
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
        except Exception:
            done = {}
    print(f"Resuming from {len(done)} existing predictions", flush=True)

    predictions = []
    _err_dbg = {"n": 0}  # one-shot full traceback on the first failures
    start_time = time.time()

    for i, sample in enumerate(data):
        if sample["index"] in done:
            predictions.append(done[sample["index"]])
            continue
        question = add_question_suffix(sample["question"], sample["type"])
        error = None
        frame_indices = []
        try:
            model_output, num_frames, frame_indices = run_one(sample["video_path"], question, sample["type"])
        except Exception as e:
            model_output, num_frames = "", 0
            error = str(e)
            if _err_dbg["n"] < 2:  # full traceback for the first couple of failures
                _err_dbg["n"] += 1
                traceback.print_exc()

        extracted = extract_answer(model_output, sample["type"])
        gt = extract_answer(sample["answer"], sample["type"])
        correct = extracted == gt

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
            # This script does the frame sampling (decord), so record the exact
            # frame indices for later sampling studies.
            "frames": {"strategy": "fps", "fps": args.fps,
                       "max_frames": args.max_frames, "num_frames": num_frames,
                       "indices": frame_indices,
                       "saved_dir": (os.path.join(
                           frames_root,
                           os.path.splitext(os.path.basename(sample["video_path"]))[0])
                           if args.save_frames else None)},
            "error": error,
            "ts": datetime.datetime.now().isoformat(),
        })

        if (i + 1) % 25 == 0:
            with open(args.out, "w") as f:
                json.dump(predictions, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(predictions, f, indent=2)

    answers = build_answers(predictions, data)
    scores = get_scores(answers)

    total_time = time.time() - start_time
    print(f"\n\n{'=' * 50}", flush=True)
    print(f"MODEL: {args.model} | FPS: {args.fps} | MAX_FRAMES: {args.max_frames}", flush=True)
    print(f"Time: {total_time / 60:.1f}min | Samples: {len(data)}", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  Q_Acc: {scores['Q_Acc'] * 100:.1f}%", flush=True)
    print(f"  V_Acc: {scores['V_Acc'] * 100:.1f}%", flush=True)
    print(f"  Acc:   {scores['Acc'] * 100:.1f}%", flush=True)
    print(f"  I_Acc: {scores['I_Acc'] * 100:.1f}%", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"  Qwen3-VL-4B (8 FPS, no CoT): I-Acc 10.3%", flush=True)
    print(f"  Paper Qwen3-VL-4B:           I-Acc 17.7%", flush=True)


if __name__ == "__main__":
    main()
