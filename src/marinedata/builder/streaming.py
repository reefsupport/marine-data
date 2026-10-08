"""``StreamingDataset`` — a corpus too large for memory, streamed on demand from a plan."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..labelindex import LabelIndex
from ..lineage import Lineage
from ..sample import Sample
from ..scan import CorpusScan, group_key
from ..task import TaskKind
from .defaults import SplitName
from .exclusions import PartialAbstainExclusion

if TYPE_CHECKING:
    from .dataset_builder import DatasetBuilder


@dataclass(frozen=True)
class StreamingDataset:
    """A corpus too large to hold in memory, described by a plan and streamed on demand.

    ``build()`` returns a :class:`Dataset` holding every sample. That is correct and
    simple, and it stops working at roughly one million samples — measured at 7,772 bytes
    per Sample, BenthicNet-1M alone needs 7.8 GB resident just for the index, and it is
    already in the registry under CC-BY.

    This holds only the *plan*: a label index, a group-to-split assignment, and the
    scan counters. All are O(groups + classes), never O(samples). Samples are produced by
    re-reading the sources, so memory is flat regardless of corpus size.

    The trade is a second pass over the data. That is the right trade: the first pass
    reads metadata only (loaders yield references, not pixels), and the alternative is
    not working at all.
    """

    builder: DatasetBuilder
    label_index: LabelIndex
    lineage: Lineage
    corpus: CorpusScan
    assignment: dict[str, SplitName]
    by: str = "site"
    task_kind: TaskKind | None = None
    partial_abstain_excluded: tuple[PartialAbstainExclusion, ...] = ()
    """Sources dropped entirely from this task (WS-D S26) — see
    ``Dataset.partial_abstain_excluded``."""

    def __iter__(self) -> Iterator[Sample]:
        yield from self.builder.stream_samples()

    def split_stream(self, name: SplitName) -> Iterator[Sample]:
        """Stream only the samples belonging to one split.

        Filtering happens during the stream, so a held-out test set never costs the
        memory of the training set it was separated from.
        """
        if name not in set(self.assignment.values()):
            known = ", ".join(sorted(set(self.assignment.values())))
            raise KeyError(f"Unknown split '{name}'. Known: {known}")
        for sample in self.builder.stream_samples():
            if self.assignment.get(group_key(sample, self.by)) == name:
                yield sample

    def split_sizes(self) -> dict[SplitName, int]:
        """Sample counts per split, from the scan — no second pass needed."""
        sizes: dict[SplitName, int] = {}
        for key, count in self.corpus.groups.items():
            name = self.assignment.get(key)
            if name is not None:
                sizes[name] = sizes.get(name, 0) + count
        return sizes

    def summary(self) -> str:
        sizes = self.split_sizes()
        total = max(self.corpus.total, 1)
        parts = "  ".join(f"{n}={c:,} ({c / total:.0%})" for n, c in sorted(sizes.items()))
        return (
            f"StreamingDataset  {self.corpus.total:,} samples  "
            f"from {len(self.corpus.sources)} source(s)  profile={self.lineage.profile}\n"
            f"  splits   {parts}\n"
            f"  {self.label_index.summary()}"
        )
