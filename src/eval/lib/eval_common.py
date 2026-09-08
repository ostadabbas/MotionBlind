#!/usr/bin/env python3
"""Helpers shared by every eval_*.py driver.

build_prompt stays per driver on purpose. Its wording differs: eval_openai sends
discrete images and prepends "The images below are frames from one video, in
temporal order.", which the video-native drivers must not say. Unifying it would
change prompts.
"""
import base64
import io
import os


def make_fix_path(base_path):
    """Resolve a dataset-relative video path against `base_path`.

    Some folders in the release carry a trailing space in their name. On a
    miss, the resolver retries each path component with one appended.
    """
    def fix_path(video_path):
        direct = os.path.join(base_path, video_path)
        if os.path.exists(direct):
            return direct
        parts = video_path.split(os.sep)
        for i, part in enumerate(parts):
            candidate = os.path.join(base_path, *parts[:i], part + " ", *parts[i + 1:])
            if os.path.exists(candidate):
                return candidate
        return direct
    return fix_path


def tensor_pool_to_pil(videos_khwc):
    """HORNet pool tensors are float RGB in [0,1] -> list[PIL.Image]."""
    import numpy as np
    from PIL import Image
    arr = (videos_khwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(frame) for frame in arr]


def _resize(pil_img, max_side):
    from PIL import Image
    img = pil_img.convert("RGB")
    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return img


def jpeg_bytes(pil_img, max_side, quality=90):
    """PIL image -> JPEG bytes, optionally downscaled to bound token cost."""
    buf = io.BytesIO()
    _resize(pil_img, max_side).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_data_url(pil_img, max_side, quality=90):
    """PIL image -> data:image/jpeg;base64 URI."""
    b = jpeg_bytes(pil_img, max_side, quality)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


def hornet_pool_frames(video_path, n, size=288):
    """Uniformly presample `n` frames at `size`, HORNet's load_frames recipe.

    linspace + astype(int) is truncation, not rounding, and short clips pad by
    repeating the last frame. Must match frame_io.hornet_pool_indices exactly or
    a replayed selection lands on different frames than the live one did.
    """
    import numpy as np
    import torch
    import decord
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total >= n:
        indices = np.linspace(0, total - 1, n).astype(int)
    else:
        indices = np.array(list(range(total)) + [total - 1] * (n - total))
    fr = torch.from_numpy(vr.get_batch(indices).asnumpy()).float()
    fr = torch.nn.functional.interpolate(fr.permute(0, 3, 1, 2), size=(size, size),
                                         mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return fr.float() / 255.0, total


# --------------------------------------------------------------------------- Video-MME
# Video-MME needs two things no yes/no benchmark does: its clips are streamed (101 GB in
# 20 zips, no per-video files), and its questions are 4-way MCQ rather than yes/no. Both
# live here so a driver opts in with three small edits instead of a
# copied block that then drifts.

def add_hf_video_args(p):
    """Register the --hf-videos family on an eval driver's parser."""
    p.add_argument("--hf-videos", nargs="?", const="lmms-eval/Video-MME", default=None,
                   metavar="REPO_ID",
                   help="stream missing videos from this HF dataset repo "
                        "(default repo: lmms-eval/Video-MME)")
    p.add_argument("--hf-cache", default=None,
                   help="disk cache for streamed videos (default: <base-path>/_hf_cache)")
    p.add_argument("--hf-cache-gb", type=float, default=20.0,
                   help="cap on the streamed-video cache (default 20 GB)")
    p.add_argument("--hf-cache-policy", default="keep", choices=("keep", "lru"),
                   help="keep = admission control (default); lru is pathological for a "
                        "repeated sequential scan over the same 900 videos")
    p.add_argument("--hf-prefetch", type=int, default=0,
                   help="fetch N videos ahead of the cursor")


def resolve_fix_path(args):
    """-> (fix_path, vsrc). vsrc is None unless --hf-videos was given.

    A locally staged copy always wins over the stream, so a partial download is still
    used and re-running after staging costs no network.
    """
    fix_path = make_fix_path(args.base_path)
    if not getattr(args, "hf_videos", None):
        return fix_path, None
    from hf_video_source import HFVideoSource

    vsrc = HFVideoSource(
        repo_id=args.hf_videos,
        cache_dir=args.hf_cache or os.path.join(args.base_path, "_hf_cache"),
        max_gb=args.hf_cache_gb,
        policy=args.hf_cache_policy,
        prefetch=bool(args.hf_prefetch))
    vsrc.build_index()
    print(f"[hf] streaming from {args.hf_videos} | cache "
          f"{vsrc.cache_dir} (cap {args.hf_cache_gb:.0f} GB)", flush=True)
    return vsrc.make_resolver(fallback=fix_path), vsrc


def is_mcq4(data):
    """True when every row is a Video-MME style 4-way MCQ.

    All-or-nothing on purpose: a mixed file would score half its rows with the wrong
    extractor, and that failure is invisible in the output.
    """
    import videomme

    return bool(data) and all(d.get("type") == videomme.TASK_TYPE for d in data)

# ------------------------------------------------------------------- Frames-to-Clips
# f2cfull is query-conditioned: unlike uniform/random/hornet it needs the question, so the
# same clip yields different frames for different questions. That has a consequence every
# caller has to respect; see the note on the frame cache in the drivers.

def add_f2c_args(p):
    """Register the --f2c-* family on an eval driver's parser."""
    p.add_argument("--f2c-clip", default=None,
                   help="Relevance encoder (default: SigLIP2 for f2cfull).")
    p.add_argument("--f2c-pool", type=int, default=128,
                   help="Unused; kept so older sbatch env still parses.")
    p.add_argument("--f2c-s-max", type=float, default=2.0,
                   help="f2cfull max resolution scale.")
    p.add_argument("--f2c-lambda-r", type=float, default=0.5,
                   help="f2cfull relevance weight (paper 0.5).")
    p.add_argument("--f2c-lambda-l", type=float, default=0.05,
                   help="f2cfull length weight (paper 0.05).")
    p.add_argument("--f2c-embed-cache", default=None,
                   help="Dir of frozen per-video frame embeddings (freeze-once "
                        "replay, like frozen selections). Off by default.")


def load_f2c_selector(args, repo):
    """-> pick(video_path, query, n) -> (frames, scales|None), or None if not selected.

    `f2cfull` grows each anchor into a contiguous clip and pays for the extra frames by
    dropping their resolution. It returns per-frame scales and can return more than n
    frames; n is a token budget, not a frame count.
    """
    import os
    import sys
    if args.sampling != "f2cfull":
        return None
    sys.path.insert(0, os.path.join(repo, "molmo2"))
    from f2c_sampling import load_f2c_full

    clip_id = args.f2c_clip or "google/siglip2-base-patch16-224"
    sel = load_f2c_full(clip_model=clip_id, s_max=args.f2c_s_max,
                        lambda_r=args.f2c_lambda_r, lambda_l=args.f2c_lambda_l,
                        seed=args.seed,
                        embed_cache_dir=getattr(args, 'f2c_embed_cache', None))
    print(f"[f2cfull] encoder={clip_id} s_max={args.f2c_s_max} "
          f"lambda_r={args.f2c_lambda_r} lambda_l={args.f2c_lambda_l}", flush=True)

    def pick(vpath, query, n):
        frames, _idx, scales = sel.select(vpath, query, n)
        return frames, scales
    return pick


def question_of(sample):
    """mcq4 rows (Video-MME) carry options and need the lettered prompt; every
    other type keeps the yes/no suffix convention."""
    import videomme
    from scoring import add_question_suffix
    if sample.get("type") == videomme.TASK_TYPE:
        return videomme.format_question(sample)
    return add_question_suffix(sample["question"], sample["type"])


def apply_f2c_scales(frames, scales):
    """Downscale frames by paper s* (s>1 -> fewer pixels, same token budget as K)."""
    if not frames or not scales:
        return frames
    from PIL import Image
    out = []
    for f, sc in zip(frames, scales):
        if sc <= 1.0 + 1e-6:
            out.append(f)
            continue
        w, h = f.size
        out.append(f.resize((max(1, round(w / sc)), max(1, round(h / sc))), Image.BICUBIC))
    return out


def score_item(sample, model_output):
    """-> (extracted, gt, correct) for any row type, mcq4 included."""
    import videomme
    from scoring import extract_answer
    if sample.get("type") == videomme.TASK_TYPE:
        extracted = videomme.extract_choice(model_output, sample.get("options"))
        gt = videomme.gt_index(sample["answer"])
        return extracted, gt, (extracted == gt and extracted >= 0)
    extracted = extract_answer(model_output, sample["type"])
    gt = extract_answer(sample["answer"], sample["type"])
    return extracted, gt, extracted == gt


def default_out(args, subject):
    """Derive the conventional predictions path when --out is omitted.

    results/<subject>_<dataset>_<sampler><N>[_novid|_rev|_shuf<seed>].json, with
    dataset from the data file's parent directory name. Explicit --out always
    wins; the cluster paths pass it, so this only serves direct driver runs.
    """
    import os as _os
    ds = _os.path.basename(_os.path.dirname(_os.path.abspath(args.data))) or "data"
    if ds in ("data", "."):
        # the in-repo benchmark ships at data/; name it what it is
        ds = "motionblind"
    samp = getattr(args, "sampling", "uniform")
    n = getattr(args, "num_frames", None)
    cell = f"{samp}{n}" if n else samp
    arm = ""
    abl = getattr(args, "ablation", None)
    if abl == "no_video" or getattr(args, "no_video", False):
        arm = "_novid"
    elif abl == "reversed_frames" or getattr(args, "reverse_frames", False):
        arm = "_rev"
    elif abl == "shuffled_frames" or getattr(args, "shuffle_frames", False):
        arm = f"_shuf{getattr(args, 'seed', 0)}"
    name = f"{subject}_{ds}_{cell}{arm}.json"
    # Land inside the organized tree (results/models/<Model>/<dataset>/) when the
    # model folder and dataset are unambiguous; else fall back to results/.
    model_dir = {
        "eagle2_5_8b": "Eagle-2.5-8B",
        "motion_o": "Motion-o",
        "qwen3_vl_4b": "Qwen3-VL-4B",
        "molmo2_8b": "Molmo2-8B",
        "gemma4_12b_it": "Gemma-4-12B-it",
        "gemini_3_pro": "Gemini-3.1-Pro",
        "inkling_small": "Inkling-Small",
    }.get(subject)
    if subject == "gpt_5_6_luna" and (getattr(args, "max_side", 0) or 0) >= 1280:
        model_dir = "GPT-5.6-Luna-720x1280"   # the low-res folder is resolution-keyed too;
                                               # ambiguous runs stay at results/ root
    dsdir = {"mbhuman": "motionblind", "motionblind": "motionblind",
             "timeblind": "timeblind", "videomme": "videomme"}.get(ds)
    if model_dir and dsdir:
        out = _os.path.join("results", "models", model_dir, dsdir, name)
    else:
        out = _os.path.join("results", name)
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    print(f"[out] --out not given; using the convention: {out}", flush=True)
    return out
