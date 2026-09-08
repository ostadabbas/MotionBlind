#!/usr/bin/env python3
"""Video-MME (4-way multiple choice) support.

Video-MME is 4-way MCQ, so it gets its own extractor and its own scorer here.
Nothing in this module touches the TimeBlind contrastive
path; Video-MME has no quadruple structure, so I_Acc/Q_Acc/V_Acc are undefined
for it and only Acc is reported.

Task type string used in data.jsonl: "mcq4".
"""
import json
import math
import re
import unicodedata

LETTERS = "ABCD"
TASK_TYPE = "mcq4"

# The lmms-eval Video-MME prompt. Kept verbatim so reported numbers sit on the
# same scale as published ones.
PROMPT_PREFIX = (
    "Select the best answer to the following multiple-choice question based on "
    "the video. Respond with only the letter (A, B, C, or D) of the correct option."
)
PROMPT_SUFFIX = "The best answer is:"


def _norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()


def format_question(sample):
    """Build the full user-visible question text for one Video-MME row."""
    opts = sample.get("options") or []
    lines = [PROMPT_PREFIX, "", _norm(sample["question"])]
    for i, o in enumerate(opts):
        o = _norm(o)
        # Options usually already ship as "A. text"; don't double the letter.
        if not re.match(r"^[A-D][\.\)]", o):
            o = f"{LETTERS[i]}. {o}"
        lines.append(o)
    lines += ["", PROMPT_SUFFIX]
    return "\n".join(lines)


def extract_choice(output, options=None):
    """Model text -> 0..3, or -1 if nothing parses.

    Ordered most-explicit-first. The bare-letter rule is last resort. It takes the
    final standalone A-D in the reply (models put the answer at the end). It skips
    a capitalised English article "A " so "A person walks" doesn't score as A.
    """
    if not output or not str(output).strip():
        return -1
    text = _norm(output)

    for pat in (
        r"(?i)\b(?:the\s+)?best\s+answer\s+is\s*[:\.]?\s*[\(\[]?([A-D])[\)\]]?",
        r"(?i)(?:final(?:\s+answer)?|answer|prediction|option|choice)\s*[:：]\s*[\(\[]?([A-D])[\)\]]?",
        r"^[\(\[]?([A-D])[\)\]]?\s*[\.\):]",
        r"^\s*([A-D])\s*$",
        r"[\(\[\{]\s*([A-D])\s*[\)\]\}]",
    ):
        m = re.search(pat, text)
        if m:
            return LETTERS.index(m.group(1).upper())

    cands = [(m.start(1), m.group(1))
             for m in re.finditer(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])", text)]
    cands = [(p, c) for p, c in cands
             if not (c == "A" and re.match(r"\s+[a-z]", text[p + 1:p + 3]))]
    if cands:
        return LETTERS.index(cands[-1][1])

    # Last chance: the model paraphrased instead of answering with a letter.
    if options:
        low = text.lower()
        hits = []
        for i, o in enumerate(options):
            body = _norm(re.sub(r"^[A-D][\.\)]\s*", "", o)).lower().rstrip(".")
            if len(body) >= 4 and body in low:
                hits.append(i)
        if len(hits) == 1:
            return hits[0]
    return -1


def gt_index(answer):
    """Ground-truth letter -> 0..3."""
    a = _norm(str(answer)).upper()
    m = re.search(r"[A-D]", a)
    return LETTERS.index(m.group(0)) if m else -1


def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def score(predictions, data):
    """Accuracy overall and split by duration / task_type.

    An unparseable reply counts as wrong, not dropped; that is how the
    Video-MME leaderboard scores it, and dropping would let a model inflate its
    number by refusing the hard items. `invalid` is reported separately so the
    two failure modes stay distinguishable.
    """
    by_index = {d["index"]: d for d in data}
    n = correct = invalid = 0
    groups, tasks, chosen = {}, {}, [0, 0, 0, 0]
    gt_counts = [0, 0, 0, 0]

    for p in predictions:
        d = by_index.get(p["index"])
        if d is None:
            continue
        gt = gt_index(d["answer"])
        if gt >= 0:
            gt_counts[gt] += 1
        pr = extract_choice(p.get("model_output") or "", d.get("options"))
        ok = (pr == gt and pr >= 0)
        n += 1
        correct += int(ok)
        if pr < 0:
            invalid += 1
        else:
            chosen[pr] += 1
        for key, bucket in (("duration", groups), ("task_type", tasks)):
            v = d.get(key)
            if v:
                b = bucket.setdefault(v, [0, 0])
                b[0] += int(ok)
                b[1] += 1

    lo, hi = _wilson(correct, n)
    out = {
        "Acc": correct / n if n else 0.0,
        "Acc_pct": 100 * correct / n if n else 0.0,
        "Acc_ci95": [lo, hi],
        "n": n,
        "correct": correct,
        "invalid": invalid,
        "chance_pct": 25.0,
        # Video-MME's answer key is not uniform across A-D, so "always guess the
        # most common letter" beats 25%. Headline Acc must clear this baseline, not
        # 25%, before it is evidence the model watched anything.
        "majority_baseline_pct": (100 * max(gt_counts) / n) if n else 0.0,
        "majority_letter": LETTERS[gt_counts.index(max(gt_counts))] if n else None,
        "gt_distribution": {LETTERS[i]: gt_counts[i] for i in range(4)},
        "choice_distribution": {LETTERS[i]: chosen[i] for i in range(4)},
    }
    for name, bucket in (("by_duration", groups), ("by_task_type", tasks)):
        out[name] = {k: {"Acc_pct": 100 * v[0] / v[1] if v[1] else 0.0,
                         "correct": v[0], "n": v[1]}
                     for k, v in sorted(bucket.items())}
    return out


def format_summary(scores, model, sampling, num_frames, elapsed_s):
    L = ["=" * 50,
         f"MODEL: {model} | SAMPLING: {sampling} | FRAMES: {num_frames} | BENCH: Video-MME",
         f"Time: {elapsed_s / 60:.1f}min | Samples: {scores['n']}",
         "=" * 50,
         f"  Acc:     {scores['Acc_pct']:.1f}%  "
         f"(95% CI {scores['Acc_ci95'][0]:.1f}-{scores['Acc_ci95'][1]:.1f}, chance 25.0%)",
         f"  invalid: {scores['invalid']}",
         f"  majority-class baseline: {scores.get('majority_baseline_pct', 0):.1f}% "
         f"(always answer {scores.get('majority_letter')})"]
    if scores.get("by_duration"):
        L.append("  " + "-" * 46)
        for k, v in scores["by_duration"].items():
            L.append(f"  {k:<10} {v['Acc_pct']:5.1f}%   ({v['correct']}/{v['n']})")
    d = scores.get("choice_distribution") or {}
    if d:
        L.append("  " + "-" * 46)
        L.append("  answer spread: " + "  ".join(f"{k}={v}" for k, v in d.items()))
    L.append("=" * 50)
    return "\n".join(L)


def load(path):
    """Read a Video-MME data.jsonl, preserving `options` and the metadata columns."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("type", TASK_TYPE)
            rows.append(r)
    return rows
