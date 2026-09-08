#!/usr/bin/env python3
"""The analysis pipeline: raw prediction JSONs -> the paper's tables, one command.

    python src/analysis/make_tables.py results/eagle2_5_8b_mbhuman_uniform16.json

infers the model and dataset from the filename, then runs the three stages for it:

  ingest   raw prediction JSONs -> per-question CSVs + _derived_metrics/ sidecars
           under results/models/<Model>/<benchmark>/
  rollup   sidecars -> <subject>_<benchmark>_sweep.csv at the model root
  paper    per-question CSVs -> results/paper/*.csv    (Table 1, Table 2, site tables)

With no arguments it rebuilds every rollup and paper table from the stored CSVs.
Each stage is also a subcommand (`make_tables.py ingest|rollup|paper --help`) with the
explicit flags for unusual cases. Cells that were never run are absent, never
zero-filled. A rollup that would drop committed rows is skipped with a warning.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
# Locate shared modules: lib/ in this repo, or flat beside this file on the cluster.
while REPO != os.path.dirname(REPO) and not any(
        os.path.exists(os.path.join(REPO, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "lib"))
sys.path.insert(0, os.path.join(REPO, "eval", "lib"))

from scoring import build_answers, extract_answer, get_scores  # noqa: E402

# The repo root, for the in-repo benchmark default (REPO stops at src/, or at the flat
# stage dir on the cluster, where this default is inert and --data is explicit).
ROOT = REPO
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "data", "data.jsonl")):
    ROOT = os.path.dirname(ROOT)

# ------------------------------------------------------------- shared cell naming
# `f2cfull` must precede `f2c`: the alternation is first-match. `f2c` is the
# simplified variant (CLIP ViT-B/32 relevance, no DP allocation); every Motion-o and
# Qwen3 f2c cell in this repo uses it. The optional trailing variant appears on HORNet
# cells in some runs: the checkpoint load (`_full`, `_noencoder`) and/or the candidate
# pool size (`_pool64`).
CELL_RE = re.compile(
    r"_(uniform|random|hornet|burst|f2cfull|f2c)(\d+)"
    r"(?:_full|_noencoder)?(?:_pool\d+)?\.json$")
SAMPLER_ORDER = ["uniform", "random", "hornet", "burst", "f2cfull", "f2c"]

# =================================================================================
# stage 1: ingest (raw prediction JSONs -> per-question CSVs + metrics sidecars)
# =================================================================================

# The item sets that share results/. Pooling any two of these produces a number that
# is wrong in a way no downstream tool can detect. The loaded predictions are checked
# against the registry.
ING_DATASETS = {
    "mbhuman": {"n_items": 240, "n_instances": 60, "n_videos": 82,
                "data": os.path.join(ROOT, "data", "data.jsonl"),
                "aka": ("mbhuman", "motionblindv2", "mb_default", "mb")},
    "motionblind388": {"n_items": 388, "n_instances": 97, "n_videos": None,
                       "data": None, "aka": ("motionblind", "cv4s")},
    "timeblind": {"n_items": 2400, "n_instances": 600, "n_videos": None,
                  "data": None, "aka": ("timeblind", "tb")},
    # Video-MME: 4-way multiple choice, no quadruples, so only Acc is defined.
    "videomme": {"n_items": 2700, "n_instances": None, "n_videos": 900,
                 "data": None, "aka": ("videomme", "vmme")},
}

FIELDS = ["strategy", "nframes", "ablation", "index", "instance_id", "video_path",
          "question", "gold", "model_output", "extracted", "correct",
          "frame_indices", "f2c_scales", "error", "pred_file"]

EXTRACTED = {1: "yes", 0: "no", -1: ""}
# Labels per item type. TimeBlind multiple-choice items have gold "a" or "b".
LABELS = {"yes_no": EXTRACTED, "multiple_choice": {1: "a", 0: "b", -1: ""},
          "mcq4": {0: "a", 1: "b", 2: "c", 3: "d", -1: ""}}
# Full label rows of mcq4 items (options, duration, task_type), keyed by index.
MCQ4_ROWS = {}
MCQ4_COLUMNS = ["duration", "task_type"]


def load_truth(path):
    """index -> (video_path, question, gold, type). The single source of `gold`."""
    truth = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("type") == "multiple_choice_4":
                s["type"] = "mcq4"
            truth[s["index"]] = (s["video_path"], s["question"],
                                 s["answer"].strip().lower(), s.get("type") or "yes_no")
            if (s.get("type") or "") == "mcq4":
                MCQ4_ROWS[s["index"]] = s
    return truth


def load_selections(sel_dir):
    """(sampler, N) -> {key: {'indices': [...], 'scales': [...]}}.

    Frozen selections are keyed by video_path for the query-independent samplers and by
    item index for the query-conditioned ones (f2c/f2cfull pick frames per question).
    Both key shapes are kept and resolved per row.
    """
    out = {}
    if not sel_dir or not os.path.isdir(sel_dir):
        return out
    for p in sorted(glob.glob(os.path.join(sel_dir, "*.json"))):
        m = re.fullmatch(r"(uniform|random|hornet|f2cfull|f2c)(\d+)",
                         os.path.splitext(os.path.basename(p))[0])
        if not m:
            continue
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        items = d.get("items")
        if isinstance(items, dict):
            out[(m.group(1), int(m.group(2)))] = items
    return out


ABL_SUFFIXES = (("_shuf0", "shuffled_frames"), ("_rev", "reversed_frames"),
                ("_novid", "no_video"), ("_no_video", "no_video"),
                ("_shuffled_frames", "shuffled_frames"))


def cells_from_predictions(results, prefix, truth):
    """(sampler, N, ablation) -> list of output rows, straight from the prediction jsons."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(results, f"{prefix}_*.json"))):
        base = os.path.basename(path)
        if base.startswith("_") or "_metrics" in base:
            continue
        # Smoke dumps are a few items, not the full set; Video-MME writes the marker in
        # the middle of the name, and letting one through costs the real cell.
        if "_smoke_" in base or base.endswith("_smoke.json"):
            continue
        # An ablation arm answers a different question from a sweep cell, so it is
        # tagged rather than mixed into the clean sweep.
        abl = "none"
        stem = base[:-len(".json")]
        for suffix, name in ABL_SUFFIXES:
            if stem.endswith(suffix):
                abl, stem = name, stem[:-len(suffix)]
                break
        m = CELL_RE.search(stem + ".json")
        if not m:
            # The no-video arm sends no frames, so its filename carries neither sampler
            # nor budget and CELL_RE cannot match it. ("none", 0) matches the frames=0
            # convention the ablation CSVs already use.
            if abl == "no_video":
                sampler, n = "none", 0
            else:
                continue
        else:
            sampler, n = m.group(1), int(m.group(2))
        try:
            rows = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rows, list):
            continue
        cells[(sampler, n, abl)] = (rows, base)
    return cells


def rows_from_prediction_cell(sampler, n, abl, preds, pred_file, truth, sel):
    out = []
    picks = sel.get((sampler, n), {})
    for r in preds:
        idx = r.get("index")
        if idx is None or idx not in truth:
            continue
        vp, question, gold, qtype = truth[idx]
        raw = r.get("model_output") or ""
        if qtype == "mcq4":
            import videomme
            ex = videomme.extract_choice(raw, MCQ4_ROWS[idx].get("options"))
        else:
            ex = extract_answer(raw, qtype)
        labels = LABELS[qtype]
        pick = picks.get(str(idx)) or picks.get(r.get("video_path") or vp) or {}
        out.append({
            "strategy": sampler, "nframes": n, "ablation": abl,
            "index": idx, "instance_id": ("" if qtype == "mcq4" else idx // 4),
            "video_path": r.get("video_path") or vp,
            # Prediction files store the wrapped prompt; the benchmark question is the
            # stable identifier, so use the truth copy.
            "question": question, "gold": gold,
            "model_output": raw, "extracted": labels.get(ex, ""),
            "correct": int(ex != -1 and labels[ex] == gold),
            "frame_indices": " ".join(str(i) for i in pick.get("indices", [])),
            "f2c_scales": " ".join(str(x) for x in pick.get("scales", [])),
            "error": r.get("error") or "",
            "pred_file": pred_file,
        })
        if qtype == "mcq4":
            for c in MCQ4_COLUMNS:
                out[-1][c] = MCQ4_ROWS[idx].get(c, "")
    out.sort(key=lambda r: r["index"])
    return out


def cells_from_csv(path, truth):
    """Re-split a combined per-question export, re-deriving `gold` and `correct`.

    The gold column is not trusted: it is checked against the benchmark file. A
    CSV exported against a different item set cannot quietly become this one.
    """
    cells, mism = defaultdict(list), 0
    for r in csv.DictReader(open(path)):
        try:
            n = int(r["nframes"])
        except (KeyError, ValueError):
            # The no-video arm legitimately has no budget; keep it as frames=0. Anything
            # else without a budget (e.g. the native-video arm) is still skipped.
            if (r.get("ablation") or "") != "no_video":
                continue
            n = 0
        idx = int(r["index"])
        row = {k: r.get(k, "") for k in FIELDS}
        row["index"], row["nframes"] = idx, n
        row["instance_id"] = idx // 4
        row["ablation"] = r.get("ablation") or "none"
        if truth and idx in truth:
            vp, question, gold, qtype = truth[idx]
            if (r.get("gold") or "").strip().lower() != gold or r.get("video_path") != vp:
                mism += 1
            row["video_path"], row["question"], row["gold"] = vp, question, gold
            if qtype == "mcq4":
                import videomme
                ex = videomme.extract_choice(row["model_output"] or "", MCQ4_ROWS[idx].get("options"))
                for c in MCQ4_COLUMNS:
                    row[c] = MCQ4_ROWS[idx].get(c, "")
            else:
                ex = extract_answer(row["model_output"] or "", qtype)
            row["extracted"] = LABELS[qtype].get(ex, "")
            row["correct"] = int(ex != -1 and LABELS[qtype][ex] == gold)
        cells[(r["strategy"], n, row["ablation"])].append(row)
    for v in cells.values():
        v.sort(key=lambda r: r["index"])
    return dict(cells), mism


def metrics_for(rows, truth, config):
    """Standard `*_metrics.json` sidecar, so the rollup stage reads it unchanged.

    Scores come from scoring.get_scores/build_answers: the same functions the eval
    drivers call, rather than a second implementation of I_Acc that could drift.
    """
    if any(truth[r["index"]][3] == "mcq4" for r in rows):
        import videomme
        preds = [{"index": r["index"], "model_output": r["model_output"]} for r in rows]
        sc = videomme.score(preds, [MCQ4_ROWS[r["index"]] for r in rows])
        sc.update({"config": config, "_derived": True})
        return sc
    dataset = [{"index": i, "answer": g, "type": qt, "video_path": v,
                "question": q} for i, (v, q, g, qt) in sorted(truth.items())]
    preds = [{"index": r["index"], "model_output": r["model_output"]} for r in rows]
    sc = get_scores(build_answers(preds, dataset))

    tp = fp = fn = tn = inv = 0
    for r in rows:
        if truth[r["index"]][3] != "yes_no":
            continue            # F1, yes-rate and the confusion block use yes_no items only
        ex = r["extracted"]
        if ex not in ("yes", "no"):
            inv += 1
            continue
        gold = r["gold"]
        if gold == "yes":
            tp += ex == "yes"
            fn += ex == "no"
        else:
            fp += ex == "yes"
            tn += ex == "no"
    tot = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "Q_Acc": sc["Q_Acc"], "V_Acc": sc["V_Acc"], "Acc": sc["Acc"], "I_Acc": sc["I_Acc"],
        "F1_yesno_pct": round(100 * f1, 2),
        "yesno_Acc_pct": round(100 * (tp + tn) / tot, 2) if tot else 0.0,
        "yes_rate_pct": round(100 * (tp + fp) / tot, 2) if tot else 0.0,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                      "invalid": inv, "n_yesno": tot + inv},
        "config": config,
        # Provenance is part of the artifact: a derived sidecar must never be mistaken
        # for one an eval driver wrote.
        "_derived": True,
    }


def cmd_ingest(a):
    if not a.prefix and not a.from_per_question:
        sys.exit("need --prefix or --from-per-question")
    subject = a.subject or a.prefix
    label = a.model or subject
    spec = ING_DATASETS[a.dataset]
    data_path = a.data or spec["data"]
    if not data_path or not os.path.isfile(data_path):
        sys.exit(f"no ground truth for dataset {a.dataset!r}: gold/correct cannot be "
                 f"derived, so no per-question file can be written. Expected {data_path!r}.")
    truth = load_truth(data_path)
    if len(truth) != spec["n_items"]:
        sys.exit(f"{data_path} has {len(truth)} items, expected {spec['n_items']} "
                 f"for {a.dataset}")

    sel = load_selections(a.selections)

    if a.from_per_question:
        cells, mism = cells_from_csv(a.from_per_question, truth)
        source = os.path.relpath(a.from_per_question, REPO)
        rowsets = {k: v for k, v in cells.items()}
        if mism:
            print(f"  !! {mism} rows disagreed with {os.path.basename(data_path)} on "
                  f"gold/video_path; re-derived from the benchmark file")
    else:
        raw = cells_from_predictions(a.results, a.prefix, truth)
        source = f"{a.results}/{a.prefix}_*.json"
        rowsets = {}
        for (s, n, abl), (preds, pred_file) in raw.items():
            if len(preds) != spec["n_items"] and not (a.allow_partial and 0 < len(preds) < spec["n_items"]):
                print(f"  !! SKIP {pred_file}: {len(preds)} rows, not {spec['n_items']} "
                      f"-- wrong item set for --dataset {a.dataset}")
                continue
            rowsets[(s, n, abl)] = rows_from_prediction_cell(
                s, n, abl, preds, pred_file, truth, sel)

    if not rowsets:
        sys.exit(f"no cells found for {subject!r} ({source})")

    budgets = [int(b) for b in a.budgets.split(",") if b.strip()]
    samplers = [s.strip() for s in a.samplers.split(",") if s.strip()]

    os.makedirs(a.out_dir, exist_ok=True)
    mdir = os.path.join(a.out_dir, "_derived_metrics")
    if a.emit_metrics:
        os.makedirs(mdir, exist_ok=True)

    written, skipped, present = [], [], set()
    for (s, n, abl) in sorted(rowsets, key=lambda k: (
            SAMPLER_ORDER.index(k[0]) if k[0] in SAMPLER_ORDER else 99, k[1], k[2])):
        rows = rowsets[(s, n, abl)]
        if abl != "none" and not a.include_ablations:
            skipped.append(f"{s}{n}[{abl}]")
            continue
        # A budgetless arm names itself after the ablation alone; there is exactly one
        # no-video cell per model per dataset.
        tag = abl if (s == "none" and n == 0) else \
            f"{s}{n}" + ("" if abl == "none" else f"_{abl}")
        out = os.path.join(a.out_dir, f"{subject}_{a.dataset}_per_question_{tag}.csv")
        fields = FIELDS + (MCQ4_COLUMNS if a.dataset == "videomme" else [])
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        written.append(os.path.basename(out))
        if abl == "none":
            present.add((s, n))
        if a.emit_metrics:
            cfg = {"model_name": label, "sampling": s, "num_frames": n,
                   "ablation": abl, "prompt_template": "default",
                   "source": source, "dataset": a.dataset}
            json.dump(metrics_for(rows, truth, cfg),
                      open(os.path.join(mdir, f"{subject}_{tag}_metrics.json"), "w"),
                      indent=2)

    missing = [f"{s}{n}" for s in samplers for n in budgets if (s, n) not in present]
    print(f"{label}: wrote {len(written)} per-question files to {a.out_dir}")
    if skipped:
        print(f"  ablation arms not exported (pass --include-ablations): {', '.join(skipped)}")
    print(f"  cells present {len(present)}/{len(samplers) * len(budgets)}; "
          f"missing: {', '.join(missing) if missing else '(none)'}")


# =================================================================================
# stage 2: rollup (metrics sidecars -> the per-model sweep tables)
# =================================================================================

# Fixed column order for the quadruple benchmarks. Only samplers found on disk are
# kept; budgets are discovered from disk, because a hardcoded ladder drops any
# cell that was actually run outside it.
CMP_SAMPLERS = ["uniform", "random", "hornet", "f2cfull", "f2c"]
CMP_BUDGETS = [1, 4, 8, 16, 24]
# The sweep tables show the paper grid. Larger budgets stay in the per-question
# record and the sidecars; they do not get table rows.
TABLE_BUDGETS = (1, 4, 8, 16, 24)

# Three different item sets live side by side and they are not comparable. The confusion
# block only counts yes/no items. `n_item/4` is the right instance count only when
# every item is yes/no: true for the MotionBlind sets, false for TimeBlind. Use the
# registry, not a guess.   name -> (n_items_total, n_instances)
CMP_DATASETS = {"mbhuman": (240, 60), "motionblind388": (388, 97), "timeblind": (2400, 600)}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def z_vs(k, n, p0):
    return (k / n - p0) / math.sqrt(p0 * (1 - p0) / n) if n else 0.0


def load_quad_sidecars(results, prefix, dataset):
    """cells[(sampler, N)] -> dict of metrics, or absent if the cell was not run."""
    cells = {}
    for s in CMP_SAMPLERS:
        for p in sorted(glob.glob(os.path.join(results, f"{prefix}_{s}*_metrics.json"))):
            # Anchored so `..._uniform8_metrics.json` cannot also match a longer prefix.
            # Optional trailing variant: some runs record the HORNet checkpoint load as
            # `hornet16_full` / `hornet16_noencoder`.
            m = re.fullmatch(
                rf"{re.escape(prefix)}_{s}(\d+)(?:_full|_noencoder)?(?:_pool\d+)?_metrics\.json",
                os.path.basename(p))
            if not m:
                continue
            n = int(m.group(1))
            d = json.load(open(p))
            c = d["confusion"]
            tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
            pos, neg = tp + fn, fp + tn
            sens = 100 * tp / pos if pos else 0.0
            spec = 100 * tn / neg if neg else 0.0
            se = 100 * math.sqrt((sens / 100 * (1 - sens / 100) / pos if pos else 0)
                                 + (spec / 100 * (1 - spec / 100) / neg if neg else 0))
            n_inst = (CMP_DATASETS[dataset][1] if dataset in CMP_DATASETS
                      else round((tp + fp + fn + tn + c["invalid"]) / 4))
            cells[(s, n)] = {
                "I_Acc": d["I_Acc"] * 100, "Acc": d["Acc"] * 100,
                "Q_Acc": d["Q_Acc"] * 100, "V_Acc": d["V_Acc"] * 100,
                "yes_rate": d.get("yes_rate_pct",
                                  100 * (tp + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0),
                "invalid": c["invalid"],
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "sens": sens, "spec": spec, "J": sens + spec - 100.0,
                "J_lo": sens + spec - 100 - 1.96 * se, "J_hi": sens + spec - 100 + 1.96 * se,
                "n_inst": n_inst, "n_item": tp + fp + fn + tn + c["invalid"],
                # Items the Acc figure is computed over. For TimeBlind that is the
                # full 2400, not the 1200 yes/no rows the confusion table counts.
                "n_acc_item": (CMP_DATASETS[dataset][0] if dataset in CMP_DATASETS
                               else tp + fp + fn + tn + c["invalid"]),
            }
    return cells


def rollup_quad(results, dataset, model_specs, csv_path):
    """The quadruple-benchmark sweep table: every number, machine-readable."""
    models = []
    for spec in model_specs:
        prefix, _, label = spec.partition(":")
        models.append((label or prefix, load_quad_sidecars(results, prefix, dataset)))

    budgets = sorted({n for _, cells in models for (_, n) in cells
                      if n in TABLE_BUDGETS}) or CMP_BUDGETS
    seen = {s for _, cells in models for (s, _) in cells}
    samplers = [s for s in CMP_SAMPLERS if s in seen] or CMP_SAMPLERS

    cols = ["model", "dataset", "sampler", "num_frames", "n_items", "n_instances",
            "I_Acc", "I_Acc_lo", "I_Acc_hi", "Q_Acc", "V_Acc", "Acc",
            "yes_rate", "TP", "FP", "FN", "TN", "invalid",
            "sensitivity", "specificity", "youden_J", "J_lo", "J_hi",
            "z_I_Acc_vs_chance", "z_Acc_vs_chance"]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for label, cells in models:
            for s_ in samplers:
                for n in budgets:
                    c = cells.get((s_, n))
                    if not c:
                        continue
                    has_inst = bool(c.get("n_inst"))
                    k = round(c["I_Acc"] / 100 * c["n_inst"]) if has_inst else 0
                    lo, hi = wilson(k, c["n_inst"]) if has_inst else (0.0, 0.0)

                    def _r(v, nd=2):
                        """None -> empty cell. Never 0; a gap must not read as a result."""
                        return "" if v is None else round(v, nd)

                    w.writerow({
                        "model": label, "dataset": dataset, "sampler": s_,
                        "num_frames": n,
                        "n_items": "" if c.get("n_item") is None else c["n_item"],
                        "n_instances": "" if c.get("n_inst") is None else c["n_inst"],
                        "I_Acc": _r(c["I_Acc"]),
                        "I_Acc_lo": _r(lo) if has_inst else "",
                        "I_Acc_hi": _r(hi) if has_inst else "",
                        "Q_Acc": _r(c["Q_Acc"]), "V_Acc": _r(c["V_Acc"]),
                        "Acc": _r(c["Acc"]), "yes_rate": _r(c.get("yes_rate")),
                        "TP": "" if c["TP"] is None else c["TP"],
                        "FP": "" if c["FP"] is None else c["FP"],
                        "FN": "" if c["FN"] is None else c["FN"],
                        "TN": "" if c["TN"] is None else c["TN"],
                        "invalid": "" if c.get("invalid") is None else c["invalid"],
                        "sensitivity": _r(c.get("sens")),
                        "specificity": _r(c.get("spec")),
                        "youden_J": _r(c.get("J")),
                        "J_lo": _r(c.get("J_lo")), "J_hi": _r(c.get("J_hi")),
                        "z_I_Acc_vs_chance": (round(z_vs(k, c["n_inst"], 0.0625), 2)
                                              if has_inst else ""),
                        "z_Acc_vs_chance": (
                            round(z_vs(round(c["Acc"] / 100 * c["n_acc_item"]),
                                       c["n_acc_item"], 0.5), 2)
                            if c.get("n_acc_item") else ""),
                    })
                    n_rows += 1
    return n_rows


# ---- the Video-MME branch: no quadruples, so only Acc (plus Wilson CIs) ----------
VMME_CELL_RE = re.compile(
    r"_(uniform|random|hornet|burst|f2cfull|f2c)(\d+)(?:_full|_noencoder)?(?:_pool\d+)?_metrics\.json$")
DURATIONS = ("short", "medium", "long")
VMME_FIELDS = ["model", "dataset", "sampler", "num_frames", "n", "Acc", "Acc_lo", "Acc_hi",
               "invalid", "short_Acc", "medium_Acc", "long_Acc", "majority_baseline", "chance"]


def load_vmme_cells(results_dir, subject):
    cells = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{subject}_*_metrics.json"))):
        m = VMME_CELL_RE.search(os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path))
        if (d.get("config") or {}).get("ablation", "none") not in ("none", None):
            continue
        cells[(m.group(1), int(m.group(2)))] = d
    return cells


def vmme_row_for(label, sampler, n, d):
    lo, hi = d.get("Acc_ci95") or (None, None)
    dur = d.get("by_duration") or {}
    return {"model": label, "dataset": "videomme", "sampler": sampler, "num_frames": n,
            "n": d.get("n"), "Acc": round(d["Acc_pct"], 2),
            "Acc_lo": round(lo, 2) if lo is not None else "",
            "Acc_hi": round(hi, 2) if hi is not None else "",
            "invalid": d.get("invalid"),
            **{f"{k}_Acc": (round(dur[k]["Acc_pct"], 2) if k in dur else "") for k in DURATIONS},
            "majority_baseline": round(d.get("majority_baseline_pct", 0.0), 2),
            "chance": 25.0}


def rollup_vmme(results_dir, model_spec, csv_path):
    subject, _, label = model_spec.partition(":")
    label = label or subject
    cells = load_vmme_cells(results_dir, subject)
    if not cells:
        return 0
    rows = [vmme_row_for(label, s, n, cells[(s, n)]) for (s, n) in
            sorted(cells, key=lambda k: (SAMPLER_ORDER.index(k[0]) if k[0] in SAMPLER_ORDER else 99, k[1]))
            if n in TABLE_BUDGETS]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=VMME_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# =================================================================================
# stage 3: paper (per-question CSVs -> results/paper/*.csv)
# =================================================================================

OUT = "results/paper"
DATA = "data/data.jsonl"

# The MotionBlind taxonomy id is the leading field of the clip filename: `15_05_1.mp4`
# -> 15 -> direction/rotational. Ids 1-7 are speed, 8-13 magnitude, 14/15 the two
# direction subtypes.
CHARS = ["speed", "magnitude", "direction (translational)", "direction (rotational)"]
CATID = {**{i: "speed" for i in range(1, 8)},
         **{i: "magnitude" for i in range(8, 14)},
         14: "direction (translational)", 15: "direction (rotational)"}

# Display name -> (results dir, table group). Order is the paper's Table 1.
PAPER_MODELS = [
    ("Qwen3-VL-4B",   "Qwen3-VL-4B",           "open"),
    ("Eagle2.5-8B",   "Eagle-2.5-8B",          "open"),
    ("Inkling-Small", "Inkling-Small",         "open"),
    ("Motion-o (7B)", "Motion-o",              "open"),
    ("Molmo2-8B",     "Molmo2-8B",             "open"),
    ("Gemini 3.1 Pro", "Gemini-3.1-Pro",       "frontier"),
    ("GPT-5.6",       "GPT-5.6-Luna-720x1280", "frontier"),
]
SAMPLER_RANK = {"uniform": 0, "random": 1, "hornet": 2, "f2cfull": 3, "f2c": 4}
TB_SWEEP = {"Qwen3-VL-4B": "results/models/Qwen3-VL-4B/qwen3_vl_4b_timeblind_sweep.csv",
            "Eagle2.5-8B": "results/models/Eagle-2.5-8B/eagle2_5_8b_timeblind_sweep.csv",
            "Motion-o (7B)": "results/models/Motion-o/motion_o_timeblind_sweep.csv",
            "Molmo2-8B": "results/models/Molmo2-8B/molmo2_8b_timeblind_sweep.csv"}


def cat_of(video_path):
    m = re.search(r"(\d+)_\d+_\d+\.mp4$", os.path.basename(video_path))
    return CATID.get(int(m.group(1))) if m else None


CELL_SOURCES = {}   # (model_dir, cell key) -> repo-relative per-question CSV path


def load_mb_cells(model_dir):
    """-> {(sampler, n, ablation): {instance_id: [(correct, category), ...]}}"""
    cells = defaultdict(lambda: defaultdict(list))
    # MotionBlind lives in its own per-benchmark folder; without the subdir this glob
    # would also sweep up the other two benchmarks' cells and score them against the
    # MotionBlind answer key.
    for f in glob.glob(os.path.join("results/models", model_dir, "motionblind",
                                    "*_per_question_*.csv")):
        rel = f.replace(os.sep, "/")
        for r in csv.DictReader(open(f, encoding="utf-8")):
            key = (r["strategy"], int(r["nframes"]), r["ablation"])
            CELL_SOURCES[(model_dir, key)] = rel
            cells[key][r["instance_id"]].append((int(r["correct"]),
                                                 cat_of(r["video_path"])))
    return cells


def score(inst, only_cat=None):
    """I_Acc, Acc, n. An instance counts only if all four of its items are correct.
    That strictness defines the metric, so it is never averaged."""
    sel = {k: v for k, v in inst.items()
           if only_cat is None or v[0][1] == only_cat}
    if not sel:
        return None, None, 0
    items = [c for v in sel.values() for c, _ in v]
    ok = sum(1 for v in sel.values() if all(c for c, _ in v))
    return 100 * ok / len(sel), 100 * sum(items) / len(items), len(sel)


def _tb_rows(model):
    """Normalised TimeBlind cells for a model, read from the committed sweep tables
    under results/models/. Ablation rows are excluded; older uniform/random rows leave
    the confusion-matrix columns empty, but every field read here is present."""
    p = TB_SWEEP.get(model)
    if not p or not os.path.exists(p):
        return []
    return [{"sampling": r["sampler"], "N": r["num_frames"], "I_Acc": float(r["I_Acc"]),
             "Acc": float(r["Acc"]), "src": p}
            for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["dataset"] == "timeblind" and r["I_Acc"]
            and (r.get("ablation") or "none") in ("none", "")]


def vmme_best(model_dir):
    """-> (Acc, 'sampler-N [sweep CSV]') for the best committed Video-MME cell, or (None, '')."""
    for f in glob.glob(os.path.join("results/models", model_dir, "*_videomme_sweep.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        cells = [r for r in rows if (r.get("Acc") or "").strip()]
        if not cells:
            continue
        b = max(cells, key=lambda r: float(r["Acc"]))
        rel = f.replace(os.sep, "/")
        return round(float(b["Acc"]), 1), f'{b["sampler"]}-{b["num_frames"]} [{rel}]'
    return None, ""


def tb_best(model):
    """Best TimeBlind I_Acc over the sampler x budget grid, Acc maximised independently."""
    rows = _tb_rows(model)
    if not rows:
        return None, None, ""
    b = max(rows, key=lambda r: r["I_Acc"])
    ba = max((r for r in rows if r.get("Acc") is not None), key=lambda r: r["Acc"],
             default=None)
    return b["I_Acc"], (ba["Acc"] if ba else b["Acc"]), f'{b["sampling"]}-{b["N"]} [{b["src"]}]'


def tb_cell(model, sampler, n):
    for r in _tb_rows(model):
        if r["sampling"] == sampler and int(r["N"]) == n:
            return r["I_Acc"]
    return None


def write_paper_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  {name:26} {len(rows)} rows")


def cmd_paper(a):
    scored = {name: load_mb_cells(d) for name, d, _ in PAPER_MODELS}

    # ---- Table 1: leaderboard at each model's best MotionBlind configuration --------
    rows = []
    for name, mdir, group in PAPER_MODELS:
        cells = scored[name]
        real = {k: v for k, v in cells.items() if k[2] == "none"}
        # Ties on I_Acc are common at this floor, so the tie-break is fixed: prefer the
        # plainest sampler, then the largest budget. That configuration gave the model
        # the most to work with, so a tie never favours the benchmark.
        best = max(real, key=lambda k: (score(real[k])[0],
                                        -SAMPLER_RANK.get(k[0], 9), k[1]))
        i, _, n = score(real[best])
        # Acc is maximised over the sweep independently of I_Acc, matching the paper's
        # "best combination of frame budget and sampling method per dataset". At this
        # floor the I_Acc argmax is often a different cell from the Acc argmax.
        acc_k = max(real, key=lambda k: score(real[k])[1])
        acc = score(real[acc_k])[1]
        tbi, tba, tbsrc = tb_best(name)
        if name == "Gemini 3.1 Pro" and tbi is None:
            tbi, tbsrc = 48.2, "cited: TimeBlind [12], best of 20+ MLLMs"
        va, vcfg = vmme_best(mdir)
        mb_src = CELL_SOURCES.get((mdir, best), "")
        rows.append([name, group, f"{i:.1f}", f"{acc:.1f}",
                     "" if tbi is None else f"{tbi:.1f}",
                     "" if tba is None else f"{tba:.1f}",
                     "" if va is None else f"{va:.1f}",
                     n, f"{best[0]}-{best[1]}"
                     + ("" if acc_k == best else f" (Acc {acc_k[0]}-{acc_k[1]})"),
                     mb_src, tbsrc, vcfg])
    # Gemma-4-12B-it was run on a 56-instance subset and the human study is not a model
    # run at all. Neither has per-question predictions here; both are carried with
    # their source named. The human row is the ceiling the whole page is measured
    # against. Omitting it would make 60.0 look like a good score.
    rows.append(["Gemma-4-12B-it", "open", "3.6", "52.2", "", "", "70.3", 56, "", "",
                 "cited: paper Table 1 (56-instance subset, not 1:1 comparable)",
                 "cited: paper Table 1"])
    rows.append(["Human", "human", "91.3", "97.8", "98.2", "99.3", "87.9", 60, "", "",
                 "cited: paper Table 1 (mean of 5 annotators, range 85-97)",
                 "cited: paper Table 1"])

    # Cells not run in this repo. Every one is marked cited so no reader mistakes it
    # for a repo run.
    CITED = {
        "Gemini 3.1 Pro": {"TB_Acc": "76.2", "VMME": "78.2"},
        "GPT-5.6":        {"TB_IAcc": "46.3", "TB_Acc": "77.3", "VMME": "74.8"},
    }
    for r in rows:
        c = CITED.get(r[0])
        if not c:
            continue
        if c.get("TB_IAcc") and not r[4]:
            r[4] = c["TB_IAcc"]
            r[10] = r[10] or "cited: paper Table 1"
        if c.get("TB_Acc") and not r[5]:
            r[5] = c["TB_Acc"]
            r[10] = r[10] or "cited: paper Table 1"
        if c.get("VMME") and not r[6]:
            r[6] = c["VMME"]
            r[11] = "cited: paper Table 1"
    rows.sort(key=lambda r: -float(r[2]))
    # The best cell is recorded per benchmark: a model's strongest MotionBlind
    # configuration is often not its strongest TimeBlind or Video-MME one.
    write_paper_csv("table1_leaderboard.csv",
                    ["model", "group", "MB_IAcc", "MB_Acc", "TB_IAcc", "TB_Acc", "VMME_Acc",
                     "n_instances", "MB_config", "MB_source", "TB_source", "VMME_config"], rows)

    # ---- Table 2: the order ablation at a matched 16-frame budget -------------------
    ARMS = [("ordered", ("uniform", 16, "none")),
            ("shuffled", ("uniform", 16, "shuffled_frames")),
            ("reversed", ("uniform", 16, "reversed_frames")),
            ("no-video", None)]
    rows = []
    for name, mdir, _ in PAPER_MODELS:
        cells = scored[name]
        for arm, key in ARMS:
            # The no-video arm sends no frames; its cell has no sampler or budget.
            if key is None:
                hit = [k for k in cells if k[2] == "no_video"]
                key = hit[0] if hit else None
            if key is None or key not in cells:
                continue
            i, acc, n = score(cells[key])
            rows.append([name, arm, f"{i:.1f}", f"{acc:.1f}", n,
                         CELL_SOURCES.get((mdir, key), "")])
    order = {a_: k for k, a_ in enumerate(["ordered", "shuffled", "reversed", "no-video"])}
    rows.sort(key=lambda r: ([m for m, _, _ in PAPER_MODELS].index(r[0]), order[r[1]]))
    write_paper_csv("table2_probes.csv",
                    ["model", "arm", "I_Acc", "Acc", "n_instances", "source"], rows)

    # ---- Figure 4: per-category I_Acc at uniform-16 ---------------------------------
    rows = []
    for name, _, _ in PAPER_MODELS:
        key = ("uniform", 16, "none")
        if key not in scored[name]:
            continue
        for c in CHARS:
            i, acc, n = score(scored[name][key], only_cat=c)
            if n:
                rows.append([name, c, f"{i:.1f}", f"{acc:.1f}", n])
    write_paper_csv("by_category.csv", ["model", "category", "I_Acc", "Acc", "n_instances"], rows)

    # ---- appendix grids: every model's sweep rows, paper naming, one per suite ------
    disp = {d: n for n, d, _ in PAPER_MODELS}
    morder = [n for n, _, _ in PAPER_MODELS]
    for token, fname in (("mbhuman", "sweep_motionblind.csv"),
                         ("timeblind", "sweep_timeblind.csv"),
                         ("videomme", "sweep_videomme.csv")):
        rows, header = [], None
        for tab in sorted(glob.glob(f"results/models/*/[a-z]*_{token}_sweep.csv")):
            with open(tab, encoding="utf-8") as fh:
                rd = csv.reader(fh)
                h = next(rd)
                header = header or h
                for r in rd:
                    r[0] = disp.get(r[0], r[0])
                    rows.append(r)
        if header:
            rows.sort(key=lambda r: (morder.index(r[0]) if r[0] in morder else 99,
                                     SAMPLER_ORDER.index(r[2]) if r[2] in SAMPLER_ORDER else 99,
                                     int(r[3])))
            write_paper_csv(fname, header, rows)


# =================================================================================
# filename inference + orchestration
# =================================================================================

# subject token -> results/models directory. Longest tokens first, so
# `gpt_5_6_luna_hires_*` never half-matches `gpt_5_6_luna`.
SUBJECT_DIRS = [
    ("gpt_5_6_luna_hires", "GPT-5.6-Luna-720x1280"),
    ("gpt_5_6_luna",       "GPT-5.6-Luna-288x512"),
    ("gemma4_12b_it",      "Gemma-4-12B-it"),
    ("inkling_small",      "Inkling-Small"),
    ("gemini_3_pro",       "Gemini-3.1-Pro"),
    ("eagle2_5_8b",        "Eagle-2.5-8B"),
    ("qwen3_vl_4b",        "Qwen3-VL-4B"),
    ("molmo2_8b",          "Molmo2-8B"),
    ("motion_o",           "Motion-o"),
]
DS_DIR = {"mbhuman": "motionblind", "timeblind": "timeblind", "videomme": "videomme"}


def infer(path):
    """prediction JSON path -> (subject, dataset, model_dir, prefix) or None."""
    stem = os.path.basename(path)
    if stem.endswith(".json"):
        stem = stem[:-len(".json")]
    subject = model_dir = None
    for tok, d in SUBJECT_DIRS:
        if stem == tok or stem.startswith(tok + "_"):
            subject, model_dir = tok, d
            break
    if not subject:
        return None
    rest = stem[len(subject):].strip("_").split("_")
    dataset = None
    if rest:
        for ds, spec in ING_DATASETS.items():
            if rest[0] in spec["aka"]:
                dataset = ds
                break
    if not dataset:
        return None
    # The ingest prefix is everything before the cell suffix: strip a probe suffix,
    # then the sampler+budget token.
    cell = stem
    for suffix, _ in ABL_SUFFIXES:
        if cell.endswith(suffix):
            cell = cell[:-len(suffix)]
            break
    m = CELL_RE.search(cell + ".json")
    prefix = cell[:m.start()] if m else cell
    return subject, dataset, model_dir, prefix


def _rollup_one(model_dir, dsdir):
    """Rebuild one model's sweep table for one benchmark, from its sidecars.

    Refuses to shrink: a committed table can carry cells whose sidecars exist only in
    the raw archive. Regenerating from the repo alone must not delete them.
    """
    subject = next((tok for tok, d in SUBJECT_DIRS if d == model_dir), None)
    if not subject:
        return
    sdir = os.path.join("results/models", model_dir, dsdir, "_derived_metrics")
    if not os.path.isdir(sdir) or not glob.glob(os.path.join(sdir, "*_metrics.json")):
        return
    token = {"motionblind": "mbhuman", "timeblind": "timeblind", "videomme": "videomme"}[dsdir]
    out = os.path.join("results/models", model_dir, f"{subject}_{token}_sweep.csv")
    old_rows = (sum(1 for _ in open(out, encoding="utf-8")) - 1) if os.path.exists(out) else 0
    tmp = out + ".tmp"
    if dsdir == "videomme":
        n = rollup_vmme(sdir, f"{subject}:{model_dir}", tmp)
    else:
        n = rollup_quad(sdir, token, [f"{subject}:{model_dir}"], tmp)
    if n == 0:
        os.remove(tmp)
        return
    if n < old_rows:
        os.remove(tmp)
        print(f"  keep  {os.path.basename(out)}: committed table has {old_rows} cells, "
              f"repo sidecars only {n} (the rest live in the raw archive)")
        return
    os.replace(tmp, out)
    print(f"  wrote {os.path.basename(out)} ({n} cells)")


def cmd_rollup(a):
    pairs = a.pairs or [(d, ds) for d in sorted(os.listdir("results/models"))
                        if os.path.isdir(os.path.join("results/models", d))
                        for ds in ("motionblind", "timeblind", "videomme")
                        if os.path.isdir(os.path.join("results/models", d, ds))]
    for model_dir, dsdir in pairs:
        _rollup_one(model_dir, dsdir)


def cmd_auto(files, data_override=None):
    touched = set()
    groups = {}
    for f in files:
        got = infer(f)
        if not got:
            sys.exit(f"cannot infer model/dataset from {os.path.basename(f)!r}; use "
                     f"`make_tables.py ingest` with explicit flags")
        subject, dataset, model_dir, prefix = got
        groups[(os.path.dirname(os.path.abspath(f)) or ".", prefix, dataset, model_dir)] = subject
    for (results_dir, prefix, dataset, model_dir), subject in groups.items():
        print(f"== ingest {prefix} ({dataset}) -> results/models/{model_dir}/{DS_DIR[dataset]}")
        ns = argparse.Namespace(
            results=results_dir, prefix=prefix, from_per_question=None,
            subject=subject, model=model_dir, dataset=dataset, data=data_override,
            selections=os.path.join(REPO, "eval", "selections", "mbhuman"),
            out_dir=os.path.join("results/models", model_dir, DS_DIR[dataset]),
            emit_metrics=True, include_ablations=True,
            budgets="1,4,8,16,24,32", samplers="uniform,random,hornet,f2cfull",
            allow_partial=False)
        cmd_ingest(ns)
        touched.add((model_dir, DS_DIR[dataset]))
    print("== rollup")
    for model_dir, dsdir in sorted(touched) or []:
        _rollup_one(model_dir, dsdir)
    if not touched:
        cmd_rollup(argparse.Namespace(pairs=None))
    print("== paper")
    cmd_paper(argparse.Namespace())


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("ingest", "rollup", "paper"):
        # auto mode: every argument is a raw prediction JSON (or --data override)
        ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        ap.add_argument("files", nargs="*",
                        help="raw prediction JSONs: ingest each (model/dataset inferred "
                             "from the filename), then refresh the affected rollups and "
                             "the paper tables. With nothing given, rebuild every rollup "
                             "and table.")
        ap.add_argument("--data", default=None,
                        help="benchmark jsonl for inferred ingests (needed for "
                             "timeblind/videomme)")
        a = ap.parse_args(argv)
        cmd_auto(a.files, a.data)
        return

    ap = argparse.ArgumentParser(prog=f"make_tables.py {argv[0]}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    gi = sub.add_parser("ingest", help="raw prediction JSONs -> per-question CSVs + sidecars")
    gi.add_argument("--results", default="results")
    gi.add_argument("--prefix", default=None, help="run prefix, e.g. inkling_small_mbhuman")
    gi.add_argument("--from-per-question", default=None,
                    help="combined per-question CSV to re-split instead of reading jsons")
    gi.add_argument("--subject", default=None, help="filename subject token; defaults to --prefix")
    gi.add_argument("--model", default=None, help="display label")
    gi.add_argument("--dataset", default="mbhuman", choices=sorted(ING_DATASETS))
    gi.add_argument("--data", default=None, help="override the benchmark jsonl")
    gi.add_argument("--selections", default=os.path.join(REPO, "eval", "selections", "mbhuman"),
                    help="frozen selection dir, to fill frame_indices/f2c_scales")
    gi.add_argument("--out-dir", required=True)
    gi.add_argument("--emit-metrics", action="store_true")
    gi.add_argument("--include-ablations", action="store_true",
                    help="also export the shuffled / reversed / no-video arms")
    gi.add_argument("--budgets", default="1,4,8,16,24,32")
    gi.add_argument("--samplers", default="uniform,random,hornet,f2cfull")
    gi.add_argument("--allow-partial", action="store_true",
                    help="export cells with fewer rows than the item set (smoke tests)")
    gi.set_defaults(func=cmd_ingest)

    gr = sub.add_parser("rollup", help="sidecars -> the per-model sweep tables")
    gr.add_argument("pairs", nargs="*", metavar="Model/benchmark",
                    help="e.g. Eagle-2.5-8B/motionblind; default: every pair on disk")
    gr.set_defaults(func=lambda a: cmd_rollup(argparse.Namespace(
        pairs=[tuple(p.split("/", 1)) for p in a.pairs] or None)))

    gp = sub.add_parser("paper", help="per-question CSVs -> results/paper/*.csv")
    gp.set_defaults(func=cmd_paper)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
