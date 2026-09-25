"""Crop confirmation: the patch -> parent matcher applied to embedding candidates.

A crop keeps only part of its parent, so neither perceptual hash nor the SSCD cosine
reliably clears the copy bar. For a candidate pair that is not yet confirmed, not
low-texture, has ``cos >= cos_crop`` and a clear area mismatch, the smaller image is
located inside the larger by multi-scale NCC (:func:`embed.match_patch`); a peak of
``ncc_crop`` or more confirms it as ``crop``.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np

from .embed import match_patch

THUMB_SIDE = 512


def thumb_gray(data: bytes, side: int = THUMB_SIDE) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        if img.format == "JPEG":
            img.draft("L", (side, side))
        gray = img.convert("L")
        gray.thumbnail((side, side))
        return gray


def crop_candidates(
    scores: dict[str, np.ndarray],
    ok: np.ndarray,
    area_a: np.ndarray,
    area_b: np.ndarray,
    rules: Any,
) -> np.ndarray:
    cos = np.nan_to_num(scores["cos"], nan=-1.0)
    ratio = np.minimum(area_a, area_b) / np.maximum(np.maximum(area_a, area_b), 1)
    return np.nonzero(
        ~ok & ~scores["lowtex"] & (cos >= rules.cos_crop) & (ratio <= rules.crop_area_max)
    )[0]


def crop_refine(
    scores: dict[str, np.ndarray],
    ok: np.ndarray,
    kind: np.ndarray,
    area_a: np.ndarray,
    area_b: np.ndarray,
    get_a: Callable[[int], Any],
    get_b: Callable[[int], Any],
    rules: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ok, kind, ncc)`` with crop pairs confirmed; ``get_a(k)``/``get_b(k)`` return the
    grayscale thumbnails of pair ``k``'s two images."""
    ok, kind = ok.copy(), kind.copy()
    ncc = np.full(len(ok), np.nan, np.float32)
    for k in crop_candidates(scores, ok, area_a, area_b, rules).tolist():
        img_a, img_b = get_a(k), get_b(k)
        if img_a is None or img_b is None:
            continue
        patch, parent = (img_a, img_b) if area_a[k] <= area_b[k] else (img_b, img_a)
        m = match_patch(patch, parent)
        ncc[k] = m.score
        if m.score >= rules.ncc_crop:
            ok[k], kind[k] = True, "crop"
    return ok, kind, ncc
