"""Cache-backed frame loaders for the frozen-selection replay path (the Gemma-4 driver).

Each loader returns a list of PIL.Image (RGB, native res); the same type as
frame_io.sample_uniform_frames, so it drops into a driver's existing frame dispatch with no
downstream change. Frames come from the pre-extracted caches instead of re-decoding mp4s. Frame
selection is done once and reused byte-identically across every model / prompt variant.

  uniform : sub-slice K evenly-spaced positions from the native-res 32-frame pool in
            videos_uniform32.h5 (K=32 -> all 32). Mirrors sample_uniform_frames, but over the
            cached pool rather than the raw video.
  random  : videos_random_samples.h5 group k{K} already holds the K random frames per clip.
  hornet  : top-K of keep_prob in videos_hornet_selections.json (scores over the uniform-32 pool),
            kept in temporal order, gathered from the same uniform-32 pool for pixels.

The clip `stem` (e.g. videos/02_00_0.mp4 -> 02_00_0) is the join key to every cache.
"""
import json
import os

import numpy as np
from PIL import Image


def stem_of(video_path):
    """videos/02_00_0.mp4 -> 02_00_0 (the cache key)."""
    return os.path.splitext(os.path.basename(video_path))[0]


# --- frame index lookups (for models that need real timestamps) ------------------------
# Gemma-4 writes frame timestamps into the prompt and computes them as frames_indices / fps.
# Handed a bare list of pre-sampled frames, it cannot infer fps and assumes 24. K frames
# that span a whole clip then present as K consecutive frames: a systematic temporal
# distortion on exactly the speed/duration/magnitude categories MotionBlind targets.
# Both HDF5 caches already record the original frame indices per clip, so the true timestamps
# are recoverable. These return the same indices the *_from_* loaders below select, in the
# same order.

def _fps_of(video_path, _memo={}):
    """Average fps of the source clip (memoized per path; one decord open per clip)."""
    if video_path not in _memo:
        from decord import VideoReader          # torch already imported by the caller
        _memo[video_path] = float(VideoReader(video_path).get_avg_fps())
    return _memo[video_path]


def uniform_indices_from_pool_h5(h5, stem, k):
    """Original frame indices for uniform_from_pool_h5's K frames -> (indices, total_frames)."""
    d = h5[stem]
    pool_idx = np.asarray(d.attrs["sampled_indices"])       # 32 original indices
    n = pool_idx.shape[0]
    k = max(1, min(int(k), n))
    sub = np.linspace(0, n - 1, k).round().astype(int)      # same sub-slice as the loader
    return pool_idx[sub].tolist(), int(d.attrs["total_frames"])


def random_indices_from_h5(h5rand, stem, k):
    """Original frame indices for random_from_h5's K frames -> (indices, total_frames)."""
    d = h5rand[f"k{int(k)}"][stem]
    return np.asarray(d.attrs["sampled_indices"]).tolist(), int(d.attrs["total_frames"])


def hornet_indices_from_selection(sel, pool_h5, stem, k):
    """Original frame indices for hornet_from_selection's top-K -> (indices, total_frames)."""
    entry = sel.get(stem + ".mp4") or sel.get(stem)
    if entry is None:
        raise KeyError(f"no HORNet selection for {stem}")
    kp = np.asarray(entry["keep_prob"], dtype=float)
    d = pool_h5[stem]
    pool_idx = np.asarray(d.attrs["sampled_indices"])
    k = max(1, min(int(k), len(kp)))
    top = np.argsort(-kp, kind="stable")[:k]                # same selection as the loader
    return pool_idx[np.sort(top)].tolist(), int(d.attrs["total_frames"])


def _pil_list(arr):
    """uint8 [T,H,W,3] -> list[PIL.Image]."""
    return [Image.fromarray(np.asarray(f)) for f in np.asarray(arr)]


def uniform_from_pool_h5(h5, stem, k):
    """K evenly-spaced frames from the native-res uniform-32 pool (K>=32 -> all 32)."""
    pool = h5[stem][:]                                  # uint8 [32,H,W,3]
    n = pool.shape[0]
    k = max(1, min(int(k), n))
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return _pil_list(pool[idx])


def random_from_h5(h5rand, stem, k):
    """The K pre-drawn random frames for this clip (HDF5 group k{K}, already temporal order)."""
    return _pil_list(h5rand[f"k{int(k)}"][stem][:])


def hornet_from_selection(sel, pool_h5, stem, k):
    """Top-K HORNet frames: the K highest keep_prob pool positions, sorted to temporal order,
    gathered from the uniform-32 pool. keep_prob[i] scores pool position i (the selection was
    computed over videos_uniform32.h5), so the pool here must be that same cache."""
    entry = sel.get(stem + ".mp4") or sel.get(stem)
    if entry is None:
        raise KeyError(f"no HORNet selection for {stem}")
    kp = np.asarray(entry["keep_prob"], dtype=float)
    pool = pool_h5[stem][:]                              # uint8 [32,H,W,3]
    k = max(1, min(int(k), len(kp)))
    top = np.argsort(-kp, kind="stable")[:k]            # K highest-scoring pool positions
    idx = np.sort(top)                                  # temporal order
    return _pil_list(pool[idx])


# --- frozen-selection loader (canonical mbhuman path) ----------------------------------
# The loaders above derive frames from a pool (uniform sub-slices a 32-frame pool; random reads
# a name-seeded draw). Neither reproduces selections/mbhuman/<cell>.json, so results built on
# them are not cell-comparable with the rest of the study. These read the frozen selection
# directly: the JSON names the frame indices, videos_mbhuman.h5 holds those exact frames.

def load_selection(sel_dir, cell):
    """src/eval/selections/mbhuman/<cell>.json -> {video_path: {"indices": [...]}}"""
    with open(os.path.join(sel_dir, f"{cell}.json")) as fh:
        return json.load(fh)["items"]


def selection_indices(items, video_path):
    """The canonical frame indices this cell wants for this clip."""
    entry = items.get(video_path)
    if entry is None:                                  # tolerate leading ./ or a bare basename
        base = os.path.basename(video_path)
        entry = next((v for k, v in items.items() if os.path.basename(k) == base), None)
    if entry is None:
        raise KeyError(f"no selection entry for {video_path}")
    return [int(x) for x in entry["indices"]]


def frames_from_selection(h5, items, video_path):
    """The cell's exact frames for this clip -> (list[PIL.Image], indices, total_frames, fps).

    Frames are gathered out of the per-clip union store by original frame number. The
    returned pixels are byte-identical to decoding the clip at the selection's indices.
    """
    stem = stem_of(video_path)
    want = selection_indices(items, video_path)
    d = h5[stem]
    have = np.asarray(d.attrs["indices"])
    pos = np.searchsorted(have, want)
    if np.any(pos >= have.shape[0]) or not np.array_equal(have[pos], np.asarray(want)):
        missing = sorted(set(want) - set(have.tolist()))
        raise KeyError(f"{stem}: frames {missing[:5]} not in cache; rebuild it")
    return _pil_list(d[:][pos]), want, int(d.attrs["total_frames"]), float(d.attrs["fps"])


def frames_from_video_selection(items, video_path, full_path):
    """The cell's exact frames, decoded straight from the mp4: the no-h5 twin of
    frames_from_selection, for sets too large to cache (TimeBlind: 1200 clips, ~100 GB).

    `video_path` is the dataset-relative key used to look the selection up; `full_path` is where
    the file actually lives. Pixels are identical to the h5 path; the cache builder uses
    exactly this decode.
    """
    import decord
    want = selection_indices(items, video_path)
    vr = decord.VideoReader(full_path)
    total = len(vr)
    bad = [i for i in want if i >= total]
    if bad:
        raise ValueError(f"{full_path}: selection asks for frames {bad[:5]} but clip has {total}")
    frames = vr.get_batch(want).asnumpy().astype(np.uint8)
    return _pil_list(frames), want, int(total), float(vr.get_avg_fps())
