"""Frame sampling, ablation reorderings, and frame-dump helpers shared by the drivers.

sample_uniform_frames / sample_random_frames decode a clip and return PIL.Images.
shuffle_frames / reverse_frames reorder an already-sampled list for the ablations.
save_frames / save_hornet_frames only record what was sampled; callers guard them
so a save failure can never break inference. They accept a list of
PIL.Images/ndarrays or a stacked (T,H,W,C)/(T,C,H,W) uint8-or-float tensor/array.
"""

import os

import numpy as np
from PIL import Image


def _to_uint8_hwc(frame):
    arr = frame
    if hasattr(arr, "detach"):          # torch tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))          # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = arr * 255.0 if arr.max() <= 1.5 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def _read_frames(video_path, idxs):
    """Decode the given frame indices as a list of PIL.Images.

    decord is the cluster path (fast, exact seeking). It has no Apple-Silicon wheel, so
    fall back to OpenCV: sequential decode + grab/retrieve, which is exact on these mp4s
    where CAP_PROP_POS_FRAMES seeking is unreliable. Both paths return identical RGB.
    """
    try:
        from decord import VideoReader  # torch already imported by the caller (segfault order)

        vr = VideoReader(video_path)
        return [Image.fromarray(f) for f in vr.get_batch(list(idxs)).asnumpy()]
    except ImportError:
        pass

    import cv2

    want = sorted(set(int(i) for i in idxs))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    got, pos, target = {}, 0, 0
    while target < len(want):
        ok = cap.grab()
        if not ok:
            break
        if pos == want[target]:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            got[pos] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            target += 1
        pos += 1
    cap.release()
    if not got:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    last = got[max(got)]
    return [got.get(int(i), last) for i in idxs]   # short reads repeat the final frame


def frame_count(video_path):
    """Total frames, via decord when present and OpenCV otherwise."""
    try:
        from decord import VideoReader

        return len(VideoReader(video_path))
    except ImportError:
        pass

    import cv2

    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n > 0:
        return n
    cap = cv2.VideoCapture(video_path)          # broken/absent header -> count by decoding
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def sample_random_frames(video_path, n, seed=0):
    """Pick n random (not evenly-spaced) frame indices, kept in temporal order,
    and return them as a list of PIL.Images to feed as the video's frames.

    Deterministic per (video basename, seed, n) so a run is reproducible and a
    resume re-picks the same frames. Contrast with uniform (evenly spaced) and
    fps (fixed rate) sampling, both of which the processor does internally.
    """
    import random

    total = frame_count(video_path)
    k = max(1, min(int(n), total))
    rng = random.Random(f"{os.path.basename(video_path)}|{seed}|{n}")
    idxs = sorted(rng.sample(range(total), k))
    return _read_frames(video_path, idxs)


def sample_uniform_frames(video_path, n):
    """Pick n evenly-spaced frame indices (temporal order) and return them as a list of
    PIL.Images to feed as the video's frames.

    Use this instead of the processor's `nframes=` sampling: Qwen2.5-VL/Qwen3-VL merge
    frames in temporal pairs, so `nframes=1` breaks the processor (degenerate 0-frame
    result). An explicit frame list works at any N. It mirrors
    sample_random_frames' path, so uniform and random differ only in index selection
    (evenly spaced vs random). This keeps the two directly comparable.
    """
    total = frame_count(video_path)
    k = max(1, min(int(n), total))
    idxs = np.linspace(0, total - 1, k).round().astype(int).tolist()
    return _read_frames(video_path, idxs)


def shuffle_frames(frames, video_path, seed=0):
    """Return `frames` (a list of PIL.Images) in a random order: the temporal-blindness
    probe. Deterministic per (video basename, seed) so a run is reproducible and a resume
    re-uses the same permutation. The frame content is identical to the ordered
    uniform/random sample; only the order changes. Any I_Acc drop vs the ordered run is
    then attributable to temporal scrambling (a model that ignores order won't move)."""
    import random

    seq = list(frames)
    rng = random.Random(f"shuffle|{os.path.basename(video_path)}|{seed}")
    rng.shuffle(seq)
    return seq


def reverse_frames(frames):
    """Return `frames` in reversed temporal order: the direction probe.

    Shuffling destroys order; reversing inverts it while preserving adjacency and
    continuity, so the clip still looks like continuous motion, run backwards.

    Consequence for scoring: on a question of the form "faster the FIRST time?" or
    "spun clockwise?", reversing the clip flips the correct answer. Against the original
    gold labels, a direction-reading model therefore scores below chance
    (perfectly order-sensitive => ~0%), while an order-blind model scores whatever it
    scored ordered. Low is the positive result here. Report both the raw score and the
    score against flipped gold.
    """
    return list(frames)[::-1]


def save_frames(frames, out_dir):
    """Write frame_000.jpg ... into out_dir; return the list of file paths."""
    os.makedirs(out_dir, exist_ok=True)
    seq = frames
    if hasattr(seq, "ndim") and seq.ndim == 4:
        seq = list(seq)                              # iterate the T axis of a stacked tensor
    paths = []
    for i, f in enumerate(seq):
        img = f if isinstance(f, Image.Image) else Image.fromarray(_to_uint8_hwc(f))
        p = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        img.save(p, quality=90)
        paths.append(p)
    return paths


def save_hornet_frames(candidates, selected_idx, out_dir, keep_prob=None):
    """Preserve the HORNet candidate pool for one clip. Write all presampled frames (the
    uniform-32 pool) into `selected/` (the top-K HORNet kept) and `unselected/`
    subfolders. Each file is named frame_<i:02d>.jpg by its original pool position, so
    the temporal order and the picked frames stay legible. Also writes selection.json
    (selected indices, keep_prob, its std).

    `candidates` is the [T,H,W,3] float-in-[0,1] tensor HORNet's load_frames returns;
    `selected_idx` a list/tensor of the kept indices. Returns the two subfolder paths.
    """
    import json

    sel = {int(i) for i in (selected_idx.tolist() if hasattr(selected_idx, "tolist") else selected_idx)}
    sel_dir = os.path.join(out_dir, "selected")
    uns_dir = os.path.join(out_dir, "unselected")
    os.makedirs(sel_dir, exist_ok=True)
    os.makedirs(uns_dir, exist_ok=True)

    seq = candidates
    if hasattr(seq, "ndim") and seq.ndim == 4:
        seq = list(seq)
    for i, f in enumerate(seq):
        img = Image.fromarray(_to_uint8_hwc(f))
        img.save(os.path.join(sel_dir if i in sel else uns_dir, f"frame_{i:02d}.jpg"),
                 quality=90)

    meta = {"selected": sorted(sel), "n_candidates": len(seq)}
    if keep_prob is not None:
        kp = keep_prob.detach().cpu().tolist() if hasattr(keep_prob, "detach") else list(keep_prob)
        meta["keep_prob"] = [round(float(x), 5) for x in kp]
        meta["keep_prob_std"] = round(float(np.std(kp)), 6)
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return sel_dir, uns_dir


def hornet_pool_indices(video_path, pool=32):
    """The original frame indices behind HORNet's candidate pool.

    HORNet's `load_frames` presamples `pool` frames with
    `np.linspace(0, total - 1, pool).astype(int)`: truncation, not rounding. It pads
    by repeating the last frame on short clips. A selector run records pool positions
    (0..pool-1). Mapping them back through this list replays that exact selection later
    with no GPU, no checkpoint, and at full resolution instead of HORNet's 288x288.
    """
    total = frame_count(video_path)
    if total >= pool:
        return np.linspace(0, total - 1, pool).astype(int).tolist()
    return list(range(total)) + [total - 1] * (pool - total)
