"""Map F2C resolution scale s* to the Molmo2 multi-crop token budget.

Molmo2ImageProcessor tiles each image into at most max_crops overlapping patches
(plus one global low-res crop). Larger max_crops or larger pixels give more vision tokens.

Paper F2C keeps token budget B proportional to K*H*W by downsampling with s
(tokens proportional to 1/s^2). The Molmo2 approximation is:
  max_crops(s) = max(1, round(base_max_crops / s^2))
and (already) resizing PIL frames by 1/s before the processor runs.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def scale_to_max_crops(s: float, base_max_crops: int = 8) -> int:
    s = max(float(s), 1.0)
    return max(1, int(round(base_max_crops / (s * s))))


def scales_to_max_crops(
    scales: Sequence[float], base_max_crops: int = 8
) -> List[int]:
    return [scale_to_max_crops(s, base_max_crops) for s in scales]


def patch_image_processor_per_image_crops(image_processor, max_crops_list: List[int]):
    """
    Temporarily make image_processor.preprocess use a per-image max_crops list.

    Returns a restore() callable. Molmo2's stock preprocess takes a single
    max_crops for the whole batch; the patched version runs one image at a
    time and concatenates.
    """
    orig = image_processor.preprocess
    crops = list(max_crops_list)

    def preprocess(images, max_crops=None, return_tensors=None, **kwargs):
        # Normalize to a flat list without going through the full orig path twice.
        if images is None:
            return orig(images, max_crops=max_crops, return_tensors=return_tensors, **kwargs)

        # Let orig fetch/validate via single-image calls.
        feats = []
        # images may be a single PIL or a list
        if not isinstance(images, (list, tuple)):
            img_list = [images]
        else:
            img_list = list(images)

        if len(img_list) != len(crops):
            # Fall back to a shared ceiling rather than crashing mid-eval.
            mc = max(crops) if crops else max_crops
            return orig(images, max_crops=mc, return_tensors=return_tensors, **kwargs)

        for img, mc in zip(img_list, crops):
            feats.append(
                orig([img], max_crops=int(mc), return_tensors=None, **kwargs)
            )

        import numpy as np
        from transformers.feature_extraction_utils import BatchFeature

        data = {
            "pixel_values": np.concatenate([f["pixel_values"] for f in feats], 0),
            "image_token_pooling": np.concatenate(
                [f["image_token_pooling"] for f in feats], 0
            ),
            "image_grids": np.concatenate([f["image_grids"] for f in feats], 0),
            "image_num_crops": np.concatenate(
                [f["image_num_crops"] for f in feats], 0
            ),
        }
        return BatchFeature(data, tensor_type=return_tensors)

    image_processor.preprocess = preprocess

    def restore():
        image_processor.preprocess = orig

    return restore
