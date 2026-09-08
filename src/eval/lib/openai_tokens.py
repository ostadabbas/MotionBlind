"""Exact image-token accounting for GPT-5.x (patch-based).

GPT-5.x does not use the old GPT-4o "85 base + 170 per 512x512 tile" rule. It
tokenizes images into 32x32 patches and applies a per-model multiplier:

    patch_count     = ceil(width / 32) * ceil(height / 32)
    billable_tokens = ceil(patch_count * MULTIPLIER)

`detail` bounds how large the image may be before that count is taken; unlike
GPT-4o's flat-rate low detail, resolution costs tokens at every level:

    low       fits within 512 x 512
    high      fits within 2048 x 2048, capped at 2500 patches
    original  up to 65535 x 65535, no patch budget
    auto      same bounds as original

Ref: developers.openai.com/api/docs/guides/images-vision
"""
from __future__ import annotations

import math

PATCH = 32
# gpt-5.6-{sol,terra,luna}, gpt-5.5, gpt-5.4* all bill at 1.2x the patch count.
MULTIPLIERS = {"gpt-5.6": 1.2, "gpt-5.5": 1.2, "gpt-5.4": 1.2}
DEFAULT_MULTIPLIER = 1.2

# (max_w, max_h, max_patches or None)
DETAIL_BOUNDS = {
    "low": (512, 512, None),
    "high": (2048, 2048, 2500),
    "original": (65535, 65535, None),
    "auto": (65535, 65535, None),
}


def multiplier_for(model: str) -> float:
    for prefix, mult in MULTIPLIERS.items():
        if (model or "").startswith(prefix):
            return mult
    return DEFAULT_MULTIPLIER


def _shrink_to_fit(w: int, h: int, max_w: int, max_h: int):
    """Scale down preserving aspect so the image fits the box. Never upscales."""
    if w <= max_w and h <= max_h:
        return w, h
    s = min(max_w / w, max_h / h)
    return max(1, int(w * s)), max(1, int(h * s))


def image_tokens(width: int, height: int, detail: str = "low",
                 model: str = "gpt-5.6-luna") -> int:
    """Tokens billed for one image of the given pixel size."""
    if detail not in DETAIL_BOUNDS:
        raise ValueError(f"detail must be one of {sorted(DETAIL_BOUNDS)}, got {detail!r}")
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return 0
    max_w, max_h, max_patches = DETAIL_BOUNDS[detail]
    w, h = _shrink_to_fit(w, h, max_w, max_h)
    patches = math.ceil(w / PATCH) * math.ceil(h / PATCH)
    if max_patches and patches > max_patches:
        # Shrink further until the patch budget is met (the API does this for you).
        s = math.sqrt(max_patches / patches)
        w, h = max(1, int(w * s)), max(1, int(h * s))
        patches = math.ceil(w / PATCH) * math.ceil(h / PATCH)
        while patches > max_patches and (w > PATCH or h > PATCH):
            w, h = max(1, int(w * 0.98)), max(1, int(h * 0.98))
            patches = math.ceil(w / PATCH) * math.ceil(h / PATCH)
    return math.ceil(patches * multiplier_for(model))


def request_tokens(sizes, detail: str = "low", text_tokens: int = 40,
                   model: str = "gpt-5.6-luna") -> int:
    """Tokens for one request: the prompt plus every frame in `sizes` [(w,h), ...]."""
    return text_tokens + sum(image_tokens(w, h, detail, model) for w, h in sizes)


def fit_max_side(width: int, height: int, max_side: int):
    """The (w,h) the eval will actually send after its --max-side downscale."""
    w, h = int(width), int(height)
    if max_side and max(w, h) > max_side:
        s = max_side / float(max(w, h))
        return max(1, int(w * s)), max(1, int(h * s))
    return w, h
