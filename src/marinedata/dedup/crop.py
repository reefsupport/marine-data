"""Crop confirmation: the patch -> parent matcher applied to embedding candidates.

A crop keeps only part of its parent, so neither perceptual hash nor the SSCD cosine
reliably clears the copy bar. For a candidate pair that is not yet confirmed, not
low-texture, has ``cos >= cos_crop`` and a clear area mismatch, the smaller image is
located inside the larger by multi-scale NCC (:func:`embed.match_patch`), restricted to
scales at or above ``crop_scale_min`` (below that floor the template is background
gradient, not content). A peak of ``ncc_crop`` or more is only a location, not a
verification: with ``box_cos`` supplied, the matched box is cropped out of the larger
image and SSCD-re-embedded; the pair confirms as ``crop`` only if that box's cosine to
the smaller image's own embedding also clears ``cos_box_crop``. Without ``box_cos``
(e.g. no embedder available) the NCC peak alone confirms, matching the pre-fix
behaviour.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embed import DEFAULT_SCALES, match_patch

THUMB_SIDE = 512


@dataclass(frozen=True)
class Thumb:
    gray: Any
    rgb: Any


def thumb_pair(data: bytes, side: int = THUMB_SIDE) -> Thumb:
    """A ``(gray, rgb)`` thumbnail pair decoded once: gray for NCC, rgb for box re-embed."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        if img.format == "JPEG":
            img.draft("RGB", (side, side))
        rgb = img.convert("RGB")
        rgb.thumbnail((side, side))
        return Thumb(gray=rgb.convert("L"), rgb=rgb)


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
    box_cos: Callable[[int, Any, tuple[int, int, int, int], bool], float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ok, kind, ncc)`` with crop pairs confirmed; ``get_a(k)``/``get_b(k)`` return
    :class:`Thumb` pairs for pair ``k``'s two images. ``box_cos(k, parent_rgb, box,
    a_is_patch)``, if given, re-verifies the NCC box by SSCD cosine to the smaller
    image; a ``None`` return (e.g. a degenerate box) is treated as a reject."""
    ok, kind = ok.copy(), kind.copy()
    ncc = np.full(len(ok), np.nan, np.float32)
    scales = [s for s in DEFAULT_SCALES if s >= rules.crop_scale_min] or None
    for k in crop_candidates(scores, ok, area_a, area_b, rules).tolist():
        img_a, img_b = get_a(k), get_b(k)
        if img_a is None or img_b is None:
            continue
        a_is_patch = area_a[k] <= area_b[k]
        patch, parent = (img_a, img_b) if a_is_patch else (img_b, img_a)
        m = match_patch(patch.gray, parent.gray, scales=scales)
        ncc[k] = m.score
        if m.score < rules.ncc_crop:
            continue
        if box_cos is None:
            ok[k], kind[k] = True, "crop"
            continue
        cos_box = box_cos(k, parent.rgb, m.box, a_is_patch)
        if cos_box is not None and cos_box >= rules.cos_box_crop:
            ok[k], kind[k] = True, "crop"
    return ok, kind, ncc
