"""Format-neutral box geometry shared by the ``boxes`` readers (WP-U6).

Every reader (:mod:`boxes_coco`, :mod:`boxes_yolo`, :mod:`boxes_fathomnet`) turns its native box
into a :class:`RawBox`: normalised ``xyxy`` in [0, 1] (the U1 ``boxes`` columns), clipped onto the
image, plus the native label and per-box extras. Nothing here knows a source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

EPS = 1e-6
UNLABELLED = "__unlabelled"  # native label of a box with no category (held-out competition labels)
PIXEL_KEYS = ("x_min_px", "y_min_px", "x_max_px", "y_max_px")


class BoxFormatError(ValueError):
    """A label file or record that cannot be read as the format it claims to be."""


@dataclass(frozen=True)
class RawBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    img_w: int | None = None
    img_h: int | None = None
    native: str | None = None
    native_id: str | None = None
    is_crowd: bool = False
    confidence: float | None = None
    licence: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BoxCounts:
    """What a reader saw: ``read`` boxes, of which ``kept``; ``clipped`` boxes were moved onto the
    image, ``degenerate`` ones had no area left and were dropped; ``orphan`` annotations pointed
    at no image."""

    read: int = 0
    kept: int = 0
    clipped: int = 0
    degenerate: int = 0
    crowd: int = 0
    ignored: int = 0
    orphan: int = 0

    def __add__(self, other: BoxCounts) -> BoxCounts:
        return BoxCounts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


class Tally:
    """Mutable accumulator local to one reader call; :meth:`freeze` returns the immutable counts."""

    def __init__(self) -> None:
        self.n = dict.fromkeys(f.name for f in fields(BoxCounts))

    def bump(self, name: str, by: int = 1) -> None:
        self.n[name] = (self.n[name] or 0) + by

    def freeze(self) -> BoxCounts:
        return BoxCounts(**{k: v or 0 for k, v in self.n.items()})


def normalise_xyxy(
    x0: float, y0: float, x1: float, y1: float, *, img_w=None, img_h=None, unit: bool = False
) -> tuple[tuple[float, float, float, float] | None, bool]:
    """``(clipped box | None, was_clipped)``. ``unit`` coordinates are already in [0, 1]; pixel
    ones need the image size. ``None`` means no area is left (zero/negative extent, or fully
    outside); coordinates within :data:`EPS` of the range are snapped without counting."""
    vals = [float(v) for v in (x0, y0, x1, y1)]
    if not all(math.isfinite(v) for v in vals):
        raise BoxFormatError("non-finite box coordinate")
    if not unit:
        if not img_w or not img_h or img_w <= 0 or img_h <= 0:
            raise BoxFormatError("a pixel box needs the image size")
        vals = [vals[0] / img_w, vals[1] / img_h, vals[2] / img_w, vals[3] / img_h]
    outside = any(v < -EPS or v > 1 + EPS for v in vals)
    cl = [min(max(v, 0.0), 1.0) for v in vals]
    if cl[2] - cl[0] <= EPS or cl[3] - cl[1] <= EPS:
        return None, outside
    return (cl[0], cl[1], cl[2], cl[3]), outside


def make_box(
    tally: Tally,
    coords: tuple[float, float, float, float],
    *,
    img_w: int | None,
    img_h: int | None,
    unit: bool,
    is_crowd: bool = False,
    ignore: bool = False,
    attrs: Mapping[str, object] | None = None,
    **named,
) -> RawBox | None:
    """One counted box, or ``None`` (dropped as degenerate)."""
    tally.bump("read")
    got, outside = normalise_xyxy(*coords, img_w=img_w, img_h=img_h, unit=unit)
    if got is None:
        tally.bump("degenerate")
        return None
    tally.bump("kept")
    extra = dict(attrs or {})
    if outside:
        tally.bump("clipped")
        extra["clipped"] = True
    if is_crowd:
        tally.bump("crowd")
    if ignore:
        tally.bump("ignored")
        extra["ignore"] = True
    return RawBox(*got, img_w=img_w, img_h=img_h, is_crowd=is_crowd, attrs=extra, **named)


def pixel_columns(box: RawBox) -> dict[str, int | None]:
    """``x_min_px..y_max_px`` (floor the minima, ceil the maxima, so a sub-pixel box keeps an
    area); all null when the image size is unknown or the rounded box collapses."""
    if not box.img_w or not box.img_h:
        return dict.fromkeys(PIXEL_KEYS)
    w, h = box.img_w, box.img_h

    def lo(v: float, dim: int) -> int:
        return max(0, math.floor(v * dim + EPS))

    def hi(v: float, dim: int) -> int:
        return min(dim, math.ceil(v * dim - EPS))

    px = (lo(box.x_min, w), lo(box.y_min, h), hi(box.x_max, w), hi(box.y_max, h))
    if px[0] >= px[2] or px[1] >= px[3]:
        return dict.fromkeys(PIXEL_KEYS)
    return dict(zip(PIXEL_KEYS, px, strict=True))
