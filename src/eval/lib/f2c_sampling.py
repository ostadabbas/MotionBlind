"""
Frames-to-Clips (F2C) selection for Molmo2.

Paper: Sun et al., "From Frames to Clips" (arXiv:2510.02262).

F2CFullSelector is the paper algorithm (--sampling f2cfull):
  1 FPS candidates, CLIP/SigLIP relevance, watershed + K-means anchors
  (Alg. 1), adaptive per-clip length (Eq. 8) + resolution scale s,
  merge overlapping same-res clips, emit resized PIL frames.

Paper defaults (supp. D.3): lambda_r=0.5, lambda_l=0.05, s_max=2, K_anchor=K.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from decord import VideoReader, cpu

    _HAVE_DECORD = True
except Exception:
    import cv2

    _HAVE_DECORD = False


def _read(video_path: str):
    if _HAVE_DECORD:
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr, len(vr)
    cap = cv2.VideoCapture(video_path)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def _fps(reader) -> float:
    if _HAVE_DECORD:
        try:
            f = float(reader.get_avg_fps())
            return f if f > 1e-3 else 30.0
        except Exception:
            return 30.0
    f = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
    return f if f > 1e-3 else 30.0


def _grab(reader, idxs, release: bool = False) -> List[Image.Image]:
    idxs = [int(i) for i in idxs]
    if not idxs:
        if release and not _HAVE_DECORD:
            reader.release()
        return []
    if _HAVE_DECORD:
        batch = reader.get_batch(idxs).asnumpy()
        frames = [Image.fromarray(f) for f in batch]
    else:
        frames = []
        for i in idxs:
            reader.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = reader.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if release:
            reader.release()
        return frames
    if release and not _HAVE_DECORD:
        reader.release()
    return frames


def _uniform_idx(total: int, n: int) -> List[int]:
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def _one_fps_idx(total: int, fps: float) -> List[int]:
    """Paper: load / score at 1 FPS."""
    if total <= 0:
        return []
    fps = max(float(fps), 1e-3)
    duration = total / fps
    n = max(1, int(np.floor(duration)) + 1)
    idxs = []
    for t in range(n):
        i = int(round(t * fps))
        if i >= total:
            i = total - 1
        idxs.append(i)
    out = sorted(set(idxs))
    return out if out else [0]


def _kmeans_1d(positions: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means on 1-D temporal indices; returns cluster id per point."""
    n = len(positions)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    # init: evenly spaced quantiles
    centers = np.quantile(positions.astype(np.float64), np.linspace(0, 1, k))
    labels = np.zeros(n, dtype=int)
    for _ in range(25):
        dist = np.abs(positions.astype(np.float64)[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = positions[mask].mean()
            else:
                new_centers[c] = positions[int(rng.integers(0, n))]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def watershed_anchors(
    scores: np.ndarray,
    positions: np.ndarray,
    k_anchor: int,
    seed: int = 0,
) -> List[int]:
    """
    Algorithm 1 (supp. D.2): valleys -> basin peaks -> optional K-means on time.

    Returns indices into the candidate arrays (not raw video indices).
    """
    n = len(scores)
    if n == 0:
        return []
    k_anchor = max(1, min(k_anchor, n))
    if n == 1:
        return [0]

    # Local minima (valleys) as basin boundaries.
    valleys = [0]
    for i in range(1, n - 1):
        if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
            valleys.append(i)
    valleys.append(n - 1)
    valleys = sorted(set(valleys))

    # Peak (highest similarity) in each basin between consecutive valleys.
    cand: List[int] = []
    for a, b in zip(valleys[:-1], valleys[1:]):
        if b <= a:
            continue
        # basin includes both ends; peak in [a, b]
        seg = scores[a : b + 1]
        cand.append(a + int(np.argmax(seg)))
    cand = sorted(set(cand))
    if not cand:
        cand = [int(np.argmax(scores))]

    if len(cand) <= k_anchor:
        # Keep the highest-score candidates if the count exceeds k_anchor (not expected).
        cand = sorted(cand, key=lambda i: scores[i], reverse=True)[:k_anchor]
        return sorted(cand)

    # Cluster candidate temporal positions into k_anchor groups; pick best per cluster.
    pos = positions[np.array(cand, dtype=int)]
    rng = np.random.default_rng(seed)
    labels = _kmeans_1d(pos, k_anchor, rng)
    chosen = []
    for c in range(labels.max() + 1):
        members = [cand[j] for j in range(len(cand)) if labels[j] == c]
        if not members:
            continue
        chosen.append(max(members, key=lambda i: scores[i]))
    # If k-means under-filled, pad with next-best unused candidates.
    if len(chosen) < k_anchor:
        unused = [i for i in sorted(cand, key=lambda i: scores[i], reverse=True) if i not in chosen]
        chosen.extend(unused[: k_anchor - len(chosen)])
    return sorted(chosen)[:k_anchor]


class _VLEncoder:
    """CLIP / SigLIP / SigLIP2 wrapper with batched image/text features."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        print(f"[f2c] VL encoder: {model_id} on {device}", flush=True)

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        # SigLIP2 often wants padding="max_length"; CLIP is happy with padding=True.
        try:
            kw = self.processor(
                text=[text], return_tensors="pt", padding="max_length", truncation=True
            )
        except Exception:
            kw = self.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
        kw = {k: v.to(self.device) for k, v in kw.items() if torch.is_tensor(v)}
        if hasattr(self.model, "get_text_features"):
            feat = self.model.get_text_features(**kw)
        else:
            out = self.model(**kw)
            feat = out.text_embeds if hasattr(out, "text_embeds") else out[1]
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def encode_images(self, frames: Sequence[Image.Image], bs: int = 32) -> torch.Tensor:
        if not frames:
            return torch.zeros(0, 1, device=self.device)
        feats = []
        for i in range(0, len(frames), bs):
            batch = list(frames[i : i + bs])
            kw = self.processor(images=batch, return_tensors="pt")
            kw = {k: v.to(self.device) for k, v in kw.items() if torch.is_tensor(v)}
            if hasattr(self.model, "get_image_features"):
                f = self.model.get_image_features(**kw)
            else:
                out = self.model(**kw)
                f = out.image_embeds if hasattr(out, "image_embeds") else out[0]
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0)


class F2CFullSelector:
    """Paper-faithful F2C (arXiv:2510.02262)."""

    def __init__(
        self,
        clip_model: str = "google/siglip2-base-patch16-224",
        device: Optional[str] = None,
        k_anchor_ratio: float = 1.0,  # paper: K_anchor = K
        s_max: float = 2.0,
        lambda_r: float = 0.5,
        lambda_l: float = 0.05,
        seed: int = 0,
        embed_cache_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = _VLEncoder(clip_model, self.device)
        self.k_anchor_ratio = k_anchor_ratio
        self.s_max = s_max
        self.lambda_r = lambda_r
        self.lambda_l = lambda_l
        self.seed = seed
        self.clip_model_id = clip_model
        self.embed_cache_dir = embed_cache_dir

    # ---- freeze-once embedding cache (question-independent frame features) ----
    def _cache_file(self, video_path: str):
        import hashlib
        stem = os.path.splitext(os.path.basename(video_path))[0]
        h = hashlib.sha1(os.path.abspath(video_path).encode()).hexdigest()[:8]
        return os.path.join(self.embed_cache_dir, f"{stem}.{h}.npz")

    def _cache_load(self, video_path: str, total: int):
        """-> (img_feat tensor, cand list) or (None, None)."""
        f = self._cache_file(video_path)
        if not os.path.isfile(f):
            return None, None
        try:
            z = np.load(f, allow_pickle=False)
            if str(z["model"]) != self.clip_model_id or int(z["total"]) != int(total):
                return None, None
            feat = torch.from_numpy(z["features"]).to(self.device)
            return feat, [int(i) for i in z["cand"]]
        except Exception as e:  # corrupt cache: recompute rather than fail the run
            print(f"[f2c] embed cache unreadable ({e}); recomputing", flush=True)
            return None, None

    def _cache_save(self, video_path: str, total: int, cand, img_feat):
        os.makedirs(self.embed_cache_dir, exist_ok=True)
        f = self._cache_file(video_path)
        np.savez(f, features=img_feat.float().cpu().numpy(),
                 cand=np.asarray(cand, dtype=np.int64),
                 total=np.int64(total), model=np.str_(self.clip_model_id))

    def prime(self, video_path: str) -> bool:
        """Build (or verify) this video's frozen frame embeddings; True if written."""
        assert self.embed_cache_dir, "prime() needs embed_cache_dir"
        reader, total = _read(video_path)
        if total <= 0:
            if not _HAVE_DECORD:
                reader.release()
            return False
        feat, cand = self._cache_load(video_path, total)
        if feat is not None:
            if not _HAVE_DECORD:
                reader.release()
            return False
        fps = _fps(reader)
        cand = _one_fps_idx(total, fps)
        if len(cand) > 512:
            cand = _uniform_idx(total, 512)
        frames = _grab(reader, cand, release=True)
        if not frames:
            return False
        self._cache_save(video_path, total, cand, self.enc.encode_images(frames))
        return True

    def select(
        self, video_path: str, query: str, n_frames: int
    ) -> Tuple[List[Image.Image], List[int], List[float]]:
        """
        Return (PIL frames, video indices, per-frame scale s*).

        n_frames is the paper budget K (equivalent full-res frame count).
        Output length may exceed K because clips expand temporally while
        spatial scale s keeps the approximate token budget ~K.
        """
        K = int(n_frames)
        reader, total = _read(video_path)
        if total <= 0 or K <= 0:
            if not _HAVE_DECORD:
                reader.release()
            return [], [], []

        img_feat = None
        if self.embed_cache_dir:
            img_feat, cached = self._cache_load(video_path, total)
            if img_feat is not None:
                cand = cached          # the frozen pool is the cache's pool
        if img_feat is None:
            fps = _fps(reader)
            cand = _one_fps_idx(total, fps)
            # Cap very long videos for practicality (paper scores at 1 FPS; MB/TB are short).
            if len(cand) > 512:
                cand = _uniform_idx(total, 512)

            cand_frames = _grab(reader, cand, release=False)
            if not cand_frames:
                if not _HAVE_DECORD:
                    reader.release()
                return [], [], []

            img_feat = self.enc.encode_images(cand_frames)  # [M, D]
            if self.embed_cache_dir:
                self._cache_save(video_path, total, cand, img_feat)
        txt_feat = self.enc.encode_text(query)  # [1, D]
        scores = (img_feat @ txt_feat.T).squeeze(-1).float().cpu().numpy()
        positions = np.asarray(cand, dtype=np.int64)

        k_anchor = max(1, int(round(self.k_anchor_ratio * K)))
        k_anchor = min(k_anchor, len(cand))
        anchor_local = watershed_anchors(scores, positions, k_anchor, seed=self.seed)
        if not anchor_local:
            idx = _uniform_idx(total, K)
            frames = _grab(reader, idx, release=True)
            return frames, idx, [1.0] * len(idx)

        # Paper: l_max = s_max^2 * K / K_anchor
        k_anchor_eff = len(anchor_local)
        l_max = max(1, int(round((self.s_max ** 2) * K / k_anchor_eff)))

        # Optimize each clip length (Eq. 8); store (start, end, s, scale_key)
        clips = []
        feat_np = img_feat.float().cpu().numpy()
        for a_local in anchor_local:
            l_star = self._optimize_clip_length(a_local, scores, feat_np, l_max)
            # s* = sqrt(K_anchor * l* / K)
            s_star = float(np.sqrt(k_anchor_eff * l_star / max(K, 1)))
            s_star = float(np.clip(s_star, 1.0, self.s_max))
            half = (l_star - 1) // 2
            # clip on candidate timeline, then map to video indices
            lo = max(0, a_local - half)
            hi = min(len(cand) - 1, a_local + (l_star - 1 - half))
            # expand to exact l_star if possible
            while (hi - lo + 1) < l_star and lo > 0:
                lo -= 1
            while (hi - lo + 1) < l_star and hi < len(cand) - 1:
                hi += 1
            v_idxs = [cand[j] for j in range(lo, hi + 1)]
            # quantize s for merge key (paper: identical resolutions)
            s_key = round(s_star, 3)
            clips.append({"idxs": v_idxs, "s": s_star, "s_key": s_key})

        clips = self._merge_overlaps(clips)

        # Materialize resized frames in temporal order (dedupe index+scale).
        # Prefer higher-res (smaller s) if same frame appears twice.
        frame_plan: dict[int, float] = {}
        for c in clips:
            for vi in c["idxs"]:
                prev = frame_plan.get(vi)
                if prev is None or c["s"] < prev:
                    frame_plan[vi] = c["s"]
        ordered = sorted(frame_plan.keys())
        if not ordered:
            idx = _uniform_idx(total, K)
            frames = _grab(reader, idx, release=True)
            return frames, idx, [1.0] * len(idx)

        raw = _grab(reader, ordered, release=True)
        out_frames: List[Image.Image] = []
        out_scales: List[float] = []
        for img, vi in zip(raw, ordered):
            s = frame_plan[vi]
            if s > 1.0 + 1e-6:
                fw, fh = img.size
                w = max(1, int(round(fw / s)))
                h = max(1, int(round(fh / s)))
                img = img.resize((w, h), Image.Resampling.BICUBIC)
            out_frames.append(img)
            out_scales.append(float(s))
        return out_frames, ordered, out_scales

    def _optimize_clip_length(
        self,
        center: int,
        scores: np.ndarray,
        feats: np.ndarray,
        l_max: int,
    ) -> int:
        best_l, best_obj = 1, -1e18
        n = len(scores)
        for l in range(1, l_max + 1):
            half = (l - 1) // 2
            lo = max(0, center - half)
            hi = min(n - 1, center + (l - 1 - half))
            # adjust window to length l when possible
            while (hi - lo + 1) < l and lo > 0:
                lo -= 1
            while (hi - lo + 1) < l and hi < n - 1:
                hi += 1
            idxs = list(range(lo, hi + 1))
            if not idxs:
                continue
            s_c = float(scores[idxs].mean())  # Eq. 6
            if len(idxs) <= 1:
                r_c = 0.0
            else:
                f = feats[idxs]
                sim = f @ f.T
                # average off-diagonal (Eq. 7)
                m = len(idxs)
                r_c = float((sim.sum() - np.trace(sim)) / (m * (m - 1)))
            obj = s_c - self.lambda_r * r_c + self.lambda_l * (l / l_max)
            if obj > best_obj:
                best_obj, best_l = obj, l
        return best_l

    @staticmethod
    def _merge_overlaps(clips: List[dict]) -> List[dict]:
        """Merge overlapping clips that share the same resolution key (1-FPS idxs)."""
        by_key: dict[float, List[dict]] = {}
        for c in clips:
            by_key.setdefault(c["s_key"], []).append(c)
        merged: List[dict] = []
        for s_key, group in by_key.items():
            group = sorted(
                [c for c in group if c["idxs"]],
                key=lambda c: min(c["idxs"]),
            )
            if not group:
                continue
            cur_idxs = set(group[0]["idxs"])
            cur_s = group[0]["s"]
            for c in group[1:]:
                a0, a1 = min(cur_idxs), max(cur_idxs)
                b0, b1 = min(c["idxs"]), max(c["idxs"])
                if b0 <= a1:  # overlap on the selected-index timeline
                    cur_idxs.update(c["idxs"])
                    cur_s = min(cur_s, c["s"])
                else:
                    merged.append(
                        {"idxs": sorted(cur_idxs), "s": cur_s, "s_key": s_key}
                    )
                    cur_idxs = set(c["idxs"])
                    cur_s = c["s"]
            merged.append({"idxs": sorted(cur_idxs), "s": cur_s, "s_key": s_key})
        return merged


def load_f2c_full(
    clip_model: str = "google/siglip2-base-patch16-224",
    device: Optional[str] = None,
    k_anchor_ratio: float = 1.0,
    s_max: float = 2.0,
    lambda_r: float = 0.5,
    lambda_l: float = 0.05,
    seed: int = 0,
    embed_cache_dir: Optional[str] = None,
) -> F2CFullSelector:
    return F2CFullSelector(
        clip_model=clip_model,
        device=device,
        k_anchor_ratio=k_anchor_ratio,
        s_max=s_max,
        lambda_r=lambda_r,
        lambda_l=lambda_l,
        seed=seed,
        embed_cache_dir=embed_cache_dir,
    )
