"""D-Y point/mask -> image rollup rules (WP-8, charter decision D-Y, manager 2026-09-25).

Both points-per-image (SEAVIEW/Reefolution/IBF style CoralNet exports) and pixels-per-
image (Coralscapes/Coralseg/rs_labelled masks) reduce to the same shape once the raw
labels are counted per coarse class: a `Mapping[str, int]` of class -> count for one
image. This module implements the one rollup those counts get, so points and masks
share it rather than each task re-deriving it (D-Y: "the same three fields").

Rules (D-Y, verbatim):
- `benthic_cover` = the fraction of points/pixels per coarse class.
- `benthic_dominant` = a class with >= 50% of points/pixels, else "mixed".
- `benthic_present` = the multi-label set of classes with >= 10% of points/pixels.
- Images with < 10 points get cover only (`benthic_dominant`/`benthic_present` are
  withheld, not guessed from too little evidence).
- Unknown/unlabelled points or pixels (and ignore/void pixels, for masks) are excluded
  from the denominator.
- An image is excluded entirely when > 50% of it is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

UNKNOWN = "unknown"
MIXED = "mixed"

MIN_KNOWN_FOR_DOMINANT = 10
"""D-Y: "Images with < 10 points get cover only." Applied identically to mask pixel
counts, which are always far above this floor in practice."""

DOMINANT_THRESHOLD = 0.5
PRESENT_THRESHOLD = 0.10
UNKNOWN_EXCLUSION_THRESHOLD = 0.5
"""D-Y: "An image is excluded when > 50% of it is unknown." Strictly greater-than: a
class split exactly 50/50 known/unknown is not excluded."""


@dataclass(frozen=True)
class RollupResult:
    """One image's D-Y rollup, from either point counts or pixel counts."""

    cover: dict[str, float]
    """`benthic_cover`: fraction of *known* points/pixels per class. Empty when the
    image is excluded or carries no known points/pixels at all."""

    dominant: str | None
    """`benthic_dominant`: a class name, `"mixed"`, or `None` when withheld (excluded,
    or fewer than :data:`MIN_KNOWN_FOR_DOMINANT` known points/pixels)."""

    present: frozenset[str]
    """`benthic_present`: classes at or above :data:`PRESENT_THRESHOLD`. Empty under
    the same withholding conditions as `dominant`."""

    n_known: int
    n_total: int
    excluded: bool
    """True when more than half of the image's points/pixels are unknown (D-Y)."""

    cover_only: bool
    """True when the image has cover but too few known points/pixels for `dominant`/
    `present` to be meaningful (D-Y's "< 10 points" rule)."""


def rollup_counts(
    class_counts: Mapping[str, int],
    *,
    unknown_label: str = UNKNOWN,
) -> RollupResult:
    """Apply the D-Y rollup to one image's per-class point or pixel counts.

    `class_counts` is class label -> count for ONE image; callers must already have
    excluded ignore/void pixels for masks (D-Y: "ignore/void pixels excluded" is a
    decode-time concern, distinct from the `unknown_label` semantic class handled
    here). Negative counts are rejected; a class is dropped if its count is 0.
    """
    for cls, n in class_counts.items():
        if n < 0:
            raise ValueError(f"rollup_counts: negative count for class {cls!r}: {n}")

    total = sum(class_counts.values())
    unknown_n = class_counts.get(unknown_label, 0)
    known = {cls: n for cls, n in class_counts.items() if cls != unknown_label and n > 0}
    known_n = sum(known.values())

    if total > 0 and (unknown_n / total) > UNKNOWN_EXCLUSION_THRESHOLD:
        return RollupResult(
            cover={},
            dominant=None,
            present=frozenset(),
            n_known=known_n,
            n_total=total,
            excluded=True,
            cover_only=False,
        )

    cover = {cls: n / known_n for cls, n in known.items()} if known_n else {}

    if known_n < MIN_KNOWN_FOR_DOMINANT:
        return RollupResult(
            cover=cover,
            dominant=None,
            present=frozenset(),
            n_known=known_n,
            n_total=total,
            excluded=False,
            cover_only=True,
        )

    dominant = next((cls for cls, frac in cover.items() if frac >= DOMINANT_THRESHOLD), MIXED)
    present = frozenset(cls for cls, frac in cover.items() if frac >= PRESENT_THRESHOLD)
    return RollupResult(
        cover=cover,
        dominant=dominant,
        present=present,
        n_known=known_n,
        n_total=total,
        excluded=False,
        cover_only=False,
    )
