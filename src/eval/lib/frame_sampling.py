"""Decode a video and return up to N frames as PIL images, chosen by a selection
strategy. Selection is decoupled from the model, so strategy and n_frames are
sweep parameters.

Strategies:
  - uniform : N evenly spaced frames across the whole clip
  - random  : N random frames (sorted, so temporal order is preserved)
  - hornet  : learned selection via the HORNet policy

Requires: decord (fast) or opencv as a fallback, numpy, pillow.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

# decord is much faster than cv2 for random frame access; fall back if absent.
try:
    from decord import VideoReader, cpu
    _HAVE_DECORD = True
except Exception:
    import cv2
    _HAVE_DECORD = False


def _read_all_indices(video_path: str):
    """Return (VideoReader-or-cap, total_frame_count)."""
    if _HAVE_DECORD:
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr, len(vr)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, total


def _grab(reader, idxs):
    """Grab frames at integer indices -> list[PIL.Image]."""
    idxs = [int(i) for i in idxs]
    if _HAVE_DECORD:
        batch = reader.get_batch(idxs).asnumpy()  # (N,H,W,3) RGB
        return [Image.fromarray(f) for f in batch]
    frames = []
    for i in idxs:
        reader.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = reader.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    reader.release()
    return frames


def _uniform_idx(total: int, n: int):
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def _random_idx(total: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)  # fixed seed => reproducible
    if total <= n:
        return list(range(total))
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def select_frames(video_path: str, n_frames: int, strategy: str = "uniform",
                  seed: int = 0, hornet=None):
    """Return a list of exactly <= n_frames PIL images."""
    reader, total = _read_all_indices(video_path)
    if total == 0:
        return []

    if strategy == "uniform":
        idx = _uniform_idx(total, n_frames)
    elif strategy == "random":
        idx = _random_idx(total, n_frames, seed)
    elif strategy == "hornet":
        # HORNet was trained to pick top-N from a dense uniform pool (default 256).
        if hornet is None:
            raise ValueError("strategy='hornet' requires a loaded HORNet policy")
        pool = getattr(hornet, "pool", 256)
        cand = _uniform_idx(total, min(total, pool))
        cand_frames = _grab(reader, cand)
        keep = hornet.top_k(cand_frames, n_frames)   # positions into cand_frames
        return [cand_frames[i] for i in keep]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return _grab(reader, idx)
