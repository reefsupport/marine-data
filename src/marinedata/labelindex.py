"""Label indexing — canonical node ids to integer class indices.

Models need contiguous integers; the registry speaks in node ids. This maps between
them, per axis, and does one thing that matters more than it looks:

**Unsupervised positions get ``-100``**, which is PyTorch's default
``CrossEntropyLoss(ignore_index=-100)``. A source that never annotated growth form
therefore contributes exactly zero gradient to the growth-form head *by default*, with
no special-casing in the training loop. That is the masked hierarchical loss the
multi-source strategy depends on, and getting it wrong is silent — the head trains on
fabricated negatives and nobody sees it in a loss curve.

TensorFlow has no equivalent convention, so the adapter emits an explicit boolean mask
alongside; see ``integrations.tensorflow``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .sample import Sample
from .schema import Axis, LabelSchema

IGNORE_INDEX = -100
"""PyTorch's ``CrossEntropyLoss`` default. Positions with this value are skipped."""


@dataclass(frozen=True)
class AxisIndex:
    """Bidirectional map between node ids and class indices for one axis."""

    axis: Axis
    classes: tuple[str, ...]
    _to_index: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, axis: Axis, classes: list[str]) -> AxisIndex:
        ordered = tuple(sorted(set(classes)))
        return cls(axis, ordered, {name: i for i, name in enumerate(ordered)})

    def __len__(self) -> int:
        return len(self.classes)

    def index_of(self, node_id: str | None) -> int:
        """Class index, or ``IGNORE_INDEX`` when unlabelled or out of vocabulary.

        An unknown node id returns the ignore value rather than raising: harmonisation
        may legitimately produce a label outside a restricted training vocabulary, and
        skipping it is correct where inventing a class is not.
        """
        if node_id is None:
            return IGNORE_INDEX
        return self._to_index.get(node_id, IGNORE_INDEX)

    def name_of(self, index: int) -> str | None:
        if index == IGNORE_INDEX or not (0 <= index < len(self.classes)):
            return None
        return self.classes[index]


@dataclass(frozen=True)
class LabelIndex:
    """Per-axis indices for a training set."""

    axes: dict[Axis, AxisIndex]
    schema_id: str

    @classmethod
    def from_samples(
        cls,
        samples: list[Sample],
        schema: LabelSchema,
        *,
        axes: tuple[Axis, ...] | None = None,
        min_count: int = 1,
    ) -> LabelIndex:
        """Build indices from the labels actually present.

        Deriving the vocabulary from the data rather than the full schema keeps head
        widths honest: RS-Benthic declares Indo-Pacific genera that simply never occur
        in a Caribbean training set, and allocating logits for them wastes capacity and
        flatters accuracy.

        Args:
            min_count: drop classes rarer than this. Rare tail classes cannot be learned
                from a handful of examples and destabilise macro metrics.
        """
        counts: dict[Axis, dict[str, int]] = {}
        for sample in samples:
            for axis, value in sample.labels.items():
                counts.setdefault(axis, {})
                counts[axis][value.node_id] = counts[axis].get(value.node_id, 0) + 1

        wanted = axes or tuple(counts)
        return cls(
            axes={
                axis: AxisIndex.build(
                    axis, [n for n, c in counts.get(axis, {}).items() if c >= min_count]
                )
                for axis in wanted
            },
            schema_id=schema.id,
        )

    @classmethod
    def from_counts(
        cls,
        counts: dict[Axis, Counter[str]],
        schema: LabelSchema,
        *,
        min_count: int = 1,
    ) -> LabelIndex:
        """Build indices from pre-aggregated counts rather than from samples.

        The streaming path needs this: it has counters from a scan and must not
        materialise the corpus to derive a vocabulary. Equivalent to
        :meth:`from_samples` given the same data.
        """
        return cls(
            axes={
                axis: AxisIndex.build(axis, [n for n, c in counter.items() if c >= min_count])
                for axis, counter in counts.items()
            },
            schema_id=schema.id,
        )

    def encode(self, sample: Sample) -> dict[Axis, int]:
        """Encode one sample. Unsupervised axes yield ``IGNORE_INDEX``."""
        out: dict[Axis, int] = {}
        for axis, index in self.axes.items():
            if axis not in sample.supervised:
                out[axis] = IGNORE_INDEX
                continue
            value = sample.labels.get(axis)
            out[axis] = index.index_of(value.node_id if value else None)
        return out

    def num_classes(self) -> dict[Axis, int]:
        """Head widths, keyed by axis."""
        return {axis: len(index) for axis, index in self.axes.items()}

    def class_weights(self, samples: list[Sample]) -> dict[Axis, list[float]]:
        """Inverse-frequency weights, normalised to mean 1.

        Reef data is severely long-tailed — a handful of classes carry most pixels while
        the ecologically interesting ones are rare. Unweighted training optimises for
        sand.
        """
        weights: dict[Axis, list[float]] = {}
        for axis, index in self.axes.items():
            counts = [0] * len(index)
            for sample in samples:
                encoded = self.encode(sample).get(axis, IGNORE_INDEX)
                if encoded != IGNORE_INDEX:
                    counts[encoded] += 1
            total = sum(counts)
            if not total:
                weights[axis] = [1.0] * len(index)
                continue
            raw = [(total / (len(index) * c)) if c else 0.0 for c in counts]
            present = [w for w in raw if w > 0]
            mean = sum(present) / len(present) if present else 1.0
            weights[axis] = [w / mean if w else 0.0 for w in raw]
        return weights

    def summary(self) -> str:
        lines = [f"LabelIndex({self.schema_id})"]
        for axis, index in self.axes.items():
            head = ", ".join(index.classes[:6])
            more = f" … +{len(index) - 6}" if len(index) > 6 else ""
            lines.append(f"  {axis.value:<10} {len(index):>3} classes: {head}{more}")
        return "\n".join(lines)
