#!/usr/bin/env python3
"""Convert Video-MME into this repo's data.jsonl schema.

Video-MME ships QA rows on the Hub (`lmms-lab/Video-MME`) and the mp4s as
separate chunked zips. This writes the QA side into the schema every eval_*.py
driver already reads. Rows are tagged `type: "mcq4"`, so eval_qwen3 routes them
to videomme.py instead of the 2-way TimeBlind scorer.

  # full set (900 videos / 2700 questions)
  python3 prep_videomme.py --videos-root /scratch/$USER/videomme/data \
      --out videomme/data.jsonl

  # only rows whose video actually landed, so a partial download still runs
  python3 prep_videomme.py ... --require-video
"""
import argparse
import json
import os
import sys

import os, sys  # noqa: E402
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not any(
        os.path.exists(os.path.join(_R, *p, "scoring.py"))
        for p in ((), ("lib",), ("eval", "lib"))):
    _R = os.path.dirname(_R)
sys.path.insert(0, _R)  # flat cluster stage: shared modules sit beside this file
sys.path.insert(0, os.path.join(_R, "eval", "lib"))  # repo checkout

DURATIONS = ("short", "medium", "long")


def _rows_from_parquet(path):
    """Read the QA parquet to plain dicts. pyarrow only; no pandas, no numpy."""
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _rows_from_hub(repo_id):
    """Pull only the QA parquet (405 KB) and read it.

    Deliberately avoids `datasets`, which imports pandas, which imports numexpr
    and bottleneck. When those are built against NumPy 1.x under a NumPy 2.x
    runtime, pandas prints a full traceback for each and carries on. The noise
    is non-fatal but buries the real output. pandas is not needed here.
    """
    from huggingface_hub import hf_hub_download
    for cand in ("videomme/test-00000-of-00001.parquet",
                 "videomme/test-00000-of-00001-*.parquet"):
        try:
            p = hf_hub_download(repo_id, cand, repo_type="dataset")
            return _rows_from_parquet(p)
        except Exception:
            continue
    raise FileNotFoundError(f"no QA parquet in {repo_id}")


def _rows_from_datasets_server(repo_id, config="videomme", split="test"):
    """Last-resort: the HF datasets-server REST API. Pure stdlib."""
    import urllib.parse
    import urllib.request
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": repo_id, "config": config,
                                    "split": split, "offset": offset, "length": 100})
        req = urllib.request.Request(
            f"https://datasets-server.huggingface.co/rows?{q}",
            headers={"User-Agent": "prepare_videomme"})
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        batch = [x["row"] for x in page.get("rows", [])]
        if not batch:
            break
        rows += batch
        offset += len(batch)
        if offset >= page.get("num_rows_total", 0):
            break
    if not rows:
        raise RuntimeError("datasets-server returned no rows")
    return rows


def load_rows(source):
    if source and os.path.exists(source):
        if source.endswith(".parquet"):
            return _rows_from_parquet(source)
        with open(source, encoding="utf-8") as f:
            txt = f.read().strip()
        return json.loads(txt) if txt.startswith("[") else [
            json.loads(l) for l in txt.splitlines() if l.strip()]

    repo = source or "lmms-eval/Video-MME"
    errs = []
    for name, fn in (("hub parquet", lambda: _rows_from_hub(repo)),
                     ("datasets-server", lambda: _rows_from_datasets_server(repo))):
        try:
            rows = fn()
            print(f"  QA rows via {name}")
            return rows
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {e}")
    try:
        from datasets import load_dataset
        return list(load_dataset(repo, split="test"))
    except Exception as e:
        errs.append(f"datasets: {type(e).__name__}: {e}")
    raise SystemExit("could not load Video-MME QA rows:\n  " + "\n  ".join(errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="lmms-eval/Video-MME",
                   help="HF dataset id, or a local .parquet/.json/.jsonl QA dump")
    p.add_argument("--videos-root", default=None,
                   help="dir that contains the videos/ folder of mp4s "
                        "(omit when using --hf-videos)")
    p.add_argument("--hf-videos", nargs="?", const="lmms-eval/Video-MME", default=None,
                   metavar="REPO_ID",
                   help="validate against the HF repo index instead of local files, "
                        "for streaming runs that never stage the 101 GB dataset")
    p.add_argument("--hf-cache", default=None, help="where to keep the HF index/cache")
    p.add_argument("--video-subdir", default="videos")
    p.add_argument("--out", default="videomme/data.jsonl")
    p.add_argument("--durations", default="short,medium,long")
    p.add_argument("--require-video", action="store_true",
                   help="drop rows whose mp4 is missing instead of keeping them")
    p.add_argument("--max-videos", type=int, default=None,
                   help="cap the number of distinct videos (debug)")
    a = p.parse_args()

    keep = {d.strip() for d in a.durations.split(",") if d.strip()}
    bad = keep - set(DURATIONS)
    if bad:
        sys.exit(f"unknown duration(s): {sorted(bad)}; expected {DURATIONS}")

    if not a.videos_root and not a.hf_videos:
        sys.exit("pass --videos-root (staged dataset) or --hf-videos (stream from the Hub)")

    hf_index = None
    if a.hf_videos:
        from hf_video_source import HFVideoSource
        src = HFVideoSource(repo_id=a.hf_videos,
                            cache_dir=a.hf_cache or os.path.join(
                                os.path.dirname(os.path.abspath(a.out)) or ".", "_hf_cache"))
        hf_index = src.build_index()

    rows = load_rows(a.source)
    print(f"loaded {len(rows)} QA rows from {a.source}")

    seen_videos, out, missing, skipped_dur = [], [], 0, 0
    seen = set()
    for r in rows:
        dur = str(r.get("duration") or "").strip().lower()
        if dur and dur not in keep:
            skipped_dur += 1
            continue
        vid = r.get("videoID") or r.get("video_id")
        if not vid:
            continue
        if a.max_videos is not None:
            if vid not in seen and len(seen) >= a.max_videos:
                continue
        seen.add(vid)
        rel = os.path.join(a.video_subdir, f"{vid}.mp4")
        if hf_index is not None:
            present = vid in hf_index
        else:
            present = os.path.exists(os.path.join(a.videos_root, rel))
        if not present:
            missing += 1
            if a.require_video:
                continue
        opts = r.get("options")
        if isinstance(opts, str):
            opts = json.loads(opts)
        opts = [str(o) for o in (opts or [])]
        if len(opts) != 4:
            sys.exit(f"expected 4 options, got {len(opts)} for {r.get('question_id')}")
        out.append({
            "index": len(out),
            "video_path": rel,
            "question": r.get("question"),
            "options": opts,
            "answer": str(r.get("answer") or "").strip(),
            "type": "mcq4",
            "duration": dur,
            "domain": r.get("domain"),
            "sub_category": r.get("sub_category"),
            "task_type": r.get("task_type"),
            "video_id": vid,
            "question_id": r.get("question_id"),
        })
        seen_videos.append(vid)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nvid = len(set(seen_videos))
    print(f"wrote {a.out}: {len(out)} questions over {nvid} videos")
    if skipped_dur:
        print(f"  filtered out by --durations: {skipped_dur}")
    if missing:
        verb = "DROPPED" if a.require_video else "KEPT (will error at eval time)"
        where = (f"the {a.hf_videos} index" if hf_index is not None
                 else os.path.join(a.videos_root or "", a.video_subdir))
        print(f"  !! {missing} rows have no mp4 in {where} -> {verb}")
    counts = {}
    for r in out:
        counts[r["duration"]] = counts.get(r["duration"], 0) + 1
    print("  by duration: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if out and len(out) % 3:
        print("  note: Video-MME is 3 questions per video; count is not a multiple of 3")


if __name__ == "__main__":
    main()
