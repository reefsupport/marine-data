"""The canonical sample — what every loader yields, regardless of source.

Design constraints that shaped this:

**Lazy by default.** ``image`` is a reference (path or URI), not decoded pixels. A
registry spanning millions of images cannot materialise them, and most pipeline stages
(filtering, splitting, auditing) never need the bytes.

**Supervision is explicit.** ``supervised`` records which axes this sample actually
carries labels for. Absence of a label and a label of "unknown" are different facts, and
conflating them is what makes multi-source training silently wrong — a source that never
labelled growth form must not push the growth-form head toward any value.

**Provenance rides along.** ``source_id`` and ``licence_tier`` travel with every sample
so a lineage report can be produced from the data actually consumed, not from the
intent declared at build time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enums import Tier
from .schema import Axis, Fidelity


@dataclass(frozen=True)
class LabelValue:
    """One axis value, with the fidelity of the mapping that produced it."""

    node_id: str
    fidelity: Fidelity = Fidelity.EXACT
    confidence: float | None = None

    def is_reliable(self) -> bool:
        return self.fidelity in (Fidelity.EXACT, Fidelity.COARSENED)


@dataclass(frozen=True)
class Sample:
    """A single harmonised item from any source."""

    source_id: str
    key: str
    """Stable identifier within the source — usually a relative path."""

    image: Path | str | None = None
    """Reference, not bytes. ``None`` for non-image modalities."""

    labels: dict[Axis, LabelValue] = field(default_factory=dict)
    supervised: frozenset[Axis] = frozenset()
    """Axes this source actually supervises. Drives the masked hierarchical loss.

    Invariant: an axis may be supervised with no label present (the annotator saw it and
    recorded nothing), but a label must never appear on an unsupervised axis.
    """

    mask: Path | str | None = None
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    points: tuple[tuple[float, float], ...] = ()
    audio: Path | str | None = None

    licence_tier: Tier | None = None
    split: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stray = set(self.labels) - set(self.supervised)
        if stray:
            raise ValueError(
                f"{self.source_id}/{self.key}: labels present on unsupervised axes "
                f"{sorted(a.value for a in stray)}. Declare them in `supervised` or drop them."
            )

    def reliable_labels(self) -> dict[Axis, LabelValue]:
        """Labels safe to train on. Approximate mappings are excluded by default."""
        return {axis: value for axis, value in self.labels.items() if value.is_reliable()}
