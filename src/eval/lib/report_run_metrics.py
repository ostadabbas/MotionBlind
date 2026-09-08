#!/usr/bin/env python3
"""F1 score + confusion matrix for TimeBlind / MotionBlind prediction JSONs.

Report per-item binary metrics (yes = positive) alongside I_Acc
to surface yes/no bias. Balanced yes/no sets should sit near 50% Acc; flag runs
outside 40 to 60% Acc for review.

  python report_run_metrics.py results/molmo2_8b_mbhuman_uniform16.json
  python report_run_metrics.py --data data/data.jsonl results/eagle2_5_8b_mbhuman_uniform16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from scoring import build_answers, extract_answer, get_scores, _load_json_list  # noqa: E402


def confusion_and_f1(predictions: list[dict], dataset: list[dict]) -> dict:
    by_index = {d["index"]: d for d in dataset}
    TP = FP = FN = TN = inv = 0
    for p in predictions:
        idx = p.get("index")
        d = by_index.get(idx)
        if not d or d.get("type") != "yes_no":
            continue
        gt = extract_answer(d["answer"], "yes_no")
        pr = extract_answer(p.get("model_output") or "", "yes_no")
        if pr not in (0, 1):
            inv += 1
            continue
        if gt == 1 and pr == 1:
            TP += 1
        elif gt == 0 and pr == 1:
            FP += 1
        elif gt == 1 and pr == 0:
            FN += 1
        else:
            TN += 1

    tot = TP + FP + FN + TN
    acc = (TP + TN) / tot if tot else 0.0
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_rate = (TP + FP) / tot if tot else 0.0

    tb = get_scores(build_answers(predictions, dataset))
    return {
        "n_yesno": tot,
        "invalid": inv,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Acc": acc,
        "Prec": prec,
        "Rec": rec,
        "F1": f1,
        "yes_rate": yes_rate,
        "I_Acc": tb["I_Acc"],
        "Q_Acc": tb["Q_Acc"],
        "V_Acc": tb["V_Acc"],
        "TB_Acc": tb["Acc"],
    }


def format_report(metrics: dict) -> str:
    m = metrics
    lines = [
        f"yes/no items scored: {m['n_yesno']}  (invalid parse: {m['invalid']})",
        f"confusion (rows=truth, cols=pred)  Yes/No:",
        f"           pred Yes  pred No",
        f"  true Yes    {m['TP']:4d}     {m['FN']:4d}",
        f"  true No     {m['FP']:4d}     {m['TN']:4d}",
        f"Acc={m['Acc']*100:.1f}%  F1={m['F1']*100:.1f}%  yes_rate={m['yes_rate']*100:.1f}%",
        f"I_Acc={m['I_Acc']*100:.1f}%  Q_Acc={m['Q_Acc']*100:.1f}%  "
        f"V_Acc={m['V_Acc']*100:.1f}%  Acc(all types)={m['TB_Acc']*100:.1f}%",
    ]
    acc_pct = m["Acc"] * 100
    if m["n_yesno"] and (acc_pct < 40 or acc_pct > 60):
        lines.append(
            f"NOTE: yes/no Acc {acc_pct:.1f}% is outside 40 to 60%. Check for a label flip or strong bias."
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="F1 + confusion matrix for a predictions JSON")
    p.add_argument("predictions", help="Path to predictions JSON")
    p.add_argument(
        "--data",
        default=None,
        help="data.jsonl ground truth (default: probe fixed paths beside the script)",
    )
    p.add_argument("--json-out", default=None, help="Optional path to write metrics JSON")
    args = p.parse_args()

    pred_path = os.path.abspath(args.predictions)
    preds = json.load(open(pred_path))
    if not preds:
        raise SystemExit("Empty predictions file")

    data_path = args.data
    if not data_path:
        for candidate in (
            os.path.join(REPO, "dataset", "data.jsonl"),
            os.path.join(REPO, "TimeBlind", "TimeBlind", "data.jsonl"),
            os.path.join(REPO, "TimeBlind", "data.jsonl"),
        ):
            if os.path.isfile(candidate):
                data_path = candidate
                break
    if not data_path or not os.path.isfile(data_path):
        raise SystemExit("Could not find data.jsonl; pass --data")

    dataset = _load_json_list(data_path)
    metrics = confusion_and_f1(preds, dataset)
    print(format_report(metrics))

    if args.json_out:
        out = {k: (round(v * 100, 2) if k in (
            "Acc", "Prec", "Rec", "F1", "yes_rate", "I_Acc", "Q_Acc", "V_Acc", "TB_Acc"
        ) else v) for k, v in metrics.items()}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
