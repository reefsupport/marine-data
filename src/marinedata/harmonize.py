"""Applying crosswalks — turning native labels into canonical ones.

The single rule this module exists to enforce: **when a source label has no canonical
equivalent, the axis becomes unsupervised rather than being forced into the nearest
class.**

That is not a detail. Forcing unmappable labels into a nearest neighbour is precisely
how a Red-Sea schema with no soft-coral class produces "50% unknown hard substrate" on
Caribbean reefs. The model learns the wrong thing confidently, and the training set
records no trace of the substitution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sample import LabelValue
from .schema import Axis, Crosswalk, Fidelity, LabelSchema


class HarmonizationError(Exception):
    """A crosswalk is inconsistent with its declared target schema."""


@dataclass(frozen=True)
class Harmonized:
    """Result of mapping one native label."""

    labels: dict[Axis, LabelValue]
    supervised: frozenset[Axis]
    dropped: bool = False
    """True when the label was unmappable and produced no supervision."""


class Harmonizer:
    """Maps native labels into a canonical schema via a crosswalk.

    Args:
        crosswalk: The mapping to apply.
        target: The canonical schema, used to validate that every mapped target exists.
        supervised_axes: Axes this *source* supervises at all. A crosswalk edge can only
            supervise an axis the source actually annotates — a point-label dataset that
            records taxon must not be credited with condition supervision just because
            one of its labels happens to mention bleaching.
        strict: If True, an unknown source label raises. If False (default) it is
            treated as unmappable, which is the safer behaviour for messy real data.
    """

    def __init__(
        self,
        crosswalk: Crosswalk,
        target: LabelSchema,
        *,
        supervised_axes: frozenset[Axis] | None = None,
        strict: bool = False,
    ) -> None:
        if crosswalk.target_schema != target.id:
            raise HarmonizationError(
                f"crosswalk '{crosswalk.id}' targets '{crosswalk.target_schema}' "
                f"but was given schema '{target.id}'"
            )
        self._validate_targets(crosswalk, target)
        self.crosswalk = crosswalk
        self.target = target
        self.supervised_axes = supervised_axes
        self.strict = strict

    @staticmethod
    def _validate_targets(crosswalk: Crosswalk, target: LabelSchema) -> None:
        """Fail at construction, not mid-epoch, if an edge points at a missing node."""
        missing: list[str] = []
        for edge in crosswalk.edges:
            for axis, node_id in edge.targets.items():
                if target.node(node_id) is None:
                    missing.append(f"{edge.source_label} → {axis.value}:{node_id}")
        if missing:
            raise HarmonizationError(
                f"crosswalk '{crosswalk.id}' maps to nodes absent from schema "
                f"'{target.id}':\n  - " + "\n  - ".join(missing)
            )

    def map_label(self, native_label: str) -> Harmonized:
        """Map one native label onto canonical axes."""
        edge = self.crosswalk.edge(native_label)

        if edge is None:
            if self.strict:
                raise HarmonizationError(
                    f"crosswalk '{self.crosswalk.id}' has no edge for '{native_label}'"
                )
            return Harmonized(labels={}, supervised=frozenset(), dropped=True)

        if edge.fidelity is Fidelity.UNMAPPABLE:
            return Harmonized(labels={}, supervised=frozenset(), dropped=True)

        labels: dict[Axis, LabelValue] = {}
        for axis, node_id in edge.targets.items():
            if self.supervised_axes is not None and axis not in self.supervised_axes:
                continue  # the source does not annotate this axis; do not invent it
            labels[axis] = LabelValue(node_id=node_id, fidelity=edge.fidelity)

        return Harmonized(labels=labels, supervised=frozenset(labels), dropped=not labels)

    def coverage_report(self) -> str:
        """Human-readable summary of how lossy this crosswalk is.

        Worth reading before trusting a multi-source training set: a crosswalk that is
        mostly ``approximate`` is a crosswalk that will quietly degrade a benchmark.
        """
        counts = self.crosswalk.coverage
        total = sum(counts.values()) or 1
        lines = [f"crosswalk {self.crosswalk.id} → {self.target.id}  ({total} edges)"]
        for fidelity, count in counts.items():
            if count:
                lines.append(f"  {fidelity.value:<13} {count:>4}  ({count / total:.0%})")
        return "\n".join(lines)
