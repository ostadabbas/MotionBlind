#!/usr/bin/env python3
"""Evaluate your model on MotionBlind.

    python src/eval/examples/evaluate_your_model.py                  # stub model
    python src/eval/examples/evaluate_your_model.py --max-samples 8  # quick smoke

The script has four parts. Edit part 2 only.
  1. THE BENCHMARK   load the questions from data.jsonl
  2. YOUR MODEL      load_model() and predict(); replace the stub
  3. THE RUN LOOP    ask every question, checkpoint, resume on rerun
  4. SCORING         the paper's scorer; prints I_Acc and Acc, writes the JSON

The stub answers "Yes" to every question and scores at chance: Acc near 50%,
I_Acc near 0%. Works unchanged on TimeBlind (point --data and --base-path at its
HuggingFace release) and on Video-MME (build with src/eval/lib/prep_videomme.py).
"""
import argparse
import json
import os
import sys
import time

# Locate the shared scorer: eval/lib in this repo, or flat beside this file on a cluster.
REPO = os.path.dirname(os.path.abspath(__file__))
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))

from eval_common import default_out, question_of, score_item  # noqa: E402
from scoring import _load_json_list, build_answers, get_scores  # noqa: E402

# The repo root, for the in-repo benchmark defaults.
ROOT = REPO
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "data", "data.jsonl")):
    ROOT = os.path.dirname(ROOT)


# =====================================================================================
# 1. THE BENCHMARK
#
# One JSON object per line: {"index": 0, "video_path": "videos/02_00_0.mp4",
# "question": "...", "answer": "yes", "type": "yes_no"}. Four consecutive items form
# one contrastive instance: two clips that differ only in motion, two complementary
# questions each.
# =====================================================================================
def load_benchmark(data_path):
    rows = _load_json_list(data_path)
    print(f"{len(rows)} questions from {data_path}")
    return rows


# =====================================================================================
# 2. YOUR MODEL. Edit this part only.
#
# load_model() runs once. predict() runs once per question and returns the model's
# raw text answer. The question arrives fully formatted ("... Please output Yes or
# No."; Video-MME items carry their lettered options). Sample frames however your
# model expects; src/eval/lib/frame_io.py has the paper's uniform and random samplers.
# =====================================================================================
def load_model():
    return None                     # the stub has nothing to load


def predict(model, video_path, question):
    return "Yes"                    # stub: answers Yes to every question


# A complete reference implementation (Eagle 2.5 8B, https://huggingface.co/nvidia/Eagle2.5-8B).
# Molmo2 follows the same pattern with AutoModelForImageTextToText (see src/eval/eval_molmo2.py).
# Delete the stub above, uncomment, and run.
#
# def load_model():
#     from transformers import AutoModel, AutoProcessor
#     import torch
#     model = AutoModel.from_pretrained("nvidia/Eagle2.5-8B", trust_remote_code=True,
#                                       torch_dtype=torch.bfloat16, device_map="auto")
#     processor = AutoProcessor.from_pretrained("nvidia/Eagle2.5-8B",
#                                               trust_remote_code=True, use_fast=True)
#     processor.tokenizer.padding_side = "left"
#     model.eval()
#     return processor, model
#
#
# def predict(bundle, video_path, question):
#     import torch
#     processor, model = bundle
#     messages = [{"role": "user", "content": [
#         {"type": "video", "video": video_path},
#         {"type": "text", "text": question},
#     ]}]
#     text = [processor.apply_chat_template(messages, tokenize=False,
#                                           add_generation_prompt=True)]
#     images, videos, kw = processor.process_vision_info(messages, return_video_kwargs=True)
#     inputs = processor(text=text, images=images, videos=videos,
#                        return_tensors="pt", padding=True, videos_kwargs=kw).to(model.device)
#     with torch.inference_mode():
#         out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
#     out = out[:, inputs["input_ids"].shape[1]:]
#     return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# =====================================================================================
# 3. THE RUN LOOP
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate your own model on MotionBlind.")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "data.jsonl"),
                    help="Benchmark rows (default: the in-repo MotionBlind set).")
    ap.add_argument("--base-path", default=os.path.join(ROOT, "data"),
                    help="Root that video_path values join onto.")
    ap.add_argument("--out", default=None,
                    help="Predictions JSON (default: the repo naming convention).")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--sampling", default="custom",
                    help="Recorded in the output name only; sample frames as you wish.")
    ap.add_argument("--num-frames", type=int, default=None,
                    help="Recorded in the output name only.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.data = os.path.normpath(args.data)
    args.base_path = os.path.normpath(args.base_path)
    if not args.out:
        args.out = default_out(args, "my_model")

    data = load_benchmark(args.data)
    if args.max_samples:
        data = data[:args.max_samples]

    # Resume: rows already answered in a previous run are kept, not re-asked.
    done = {}
    if os.path.exists(args.out):
        try:
            for prev in json.load(open(args.out, encoding="utf-8")):
                if (prev.get("model_output") or "").strip() and not prev.get("error"):
                    done[prev["index"]] = prev
            print(f"resuming past {len(done)} finished items")
        except Exception:
            done = {}

    model = load_model()
    predictions, start = [], time.time()
    for i, sample in enumerate(data):
        if sample["index"] in done:
            predictions.append(done[sample["index"]])
            continue
        question = question_of(sample)
        video = os.path.join(args.base_path, sample["video_path"])
        error = None
        try:
            model_output = predict(model, video, question)
        except Exception as e:  # noqa: BLE001: record the failure and continue
            model_output, error = "", str(e)
        extracted, gt, correct = score_item(sample, model_output)
        predictions.append({"index": sample["index"], "video_path": sample["video_path"],
                            "question": question, "model_output": model_output,
                            "extracted": extracted, "gt": gt,
                            "correct": bool(correct), "error": error})
        if (i + 1) % 20 == 0 or i + 1 == len(data):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(predictions, open(args.out, "w", encoding="utf-8"), indent=1)
            eta = (time.time() - start) / (i + 1) * (len(data) - i - 1)
            print(f"[{i + 1}/{len(data)}] saved -> {args.out} (ETA {eta:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(predictions, open(args.out, "w", encoding="utf-8"), indent=1)

    # =================================================================================
    # 4. SCORING. The paper's scorer: an instance counts only when all four of its
    #    answers are correct.
    # =================================================================================
    answers = build_answers(predictions, data)
    scores = get_scores(answers)
    print(json.dumps(scores, indent=2))
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
