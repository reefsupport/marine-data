"""``Dataset`` — the in-memory training set and its group-wise splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..labelindex import LabelIndex
from ..lineage import Lineage
from ..sample import Sample
from ..scan import assign_splits, check_ratios, group_key
from ..schema import Axis
from ..splitmap import resolve_splits
from ..task import TaskKind
from .defaults import SplitName, _default_ratios, _to_pretrain_vocabulary
from .exclusions import PartialAbstainExclusion


@dataclass
class Dataset:
    """A built, licence-cleared, harmonised training set."""

    samples: list[Sample]
    label_index: LabelIndex
    lineage: Lineage
    splits: dict[SplitName, list[int]] = field(default_factory=dict)
    """Split name to sample positions. Positions, not copies, so splits stay cheap."""

    skipped: dict[str, str] = field(default_factory=dict)
    """Source id to reason, for sources that were permitted but could not be read."""

    projector: object | None = None
    """Task projector, when the dataset was built for a supervised task. Framework
    adapters pass it to ``LabelIndex.encode`` so genus labels roll up and coarser ones
    abstain."""

    task_kind: TaskKind | None = None
    """The task's kind, when built with ``task_id``. Drives ``split()``'s default
    ratios — a self-supervised corpus doesn't want the supervised 70/15/15."""

    partial_abstain_excluded: tuple[PartialAbstainExclusion, ...] = ()
    """Sources dropped entirely from this task (WS-D S26) because their own native
    labels on the task's axis were a mix of resolved and coarser-abstaining."""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, position: int) -> Sample:
        return self.samples[position]

    def split_samples(self, name: SplitName) -> list[Sample]:
        if name not in self.splits:
            known = ", ".join(sorted(self.splits)) or "(none — call split() first)"
            raise KeyError(f"Unknown split '{name}'. Known: {known}")
        return [self.samples[i] for i in self.splits[name]]

    # ── splitting ─────────────────────────────────────────────────────────

    def split(
        self,
        *,
        by: str = "site",
        ratios: dict[SplitName, float] | None = None,
        seed: int = 0,
        tolerance: float | None = 0.10,
        split_map: str | Path | None = None,
        frozen: bool = False,
    ) -> Dataset:
        """Assign splits. Returns self so it chains.

        Args:
            by: ``"site"`` groups by partition then source — the default, and the only
                one that gives a trustworthy generalisation estimate for transect data.
                ``"source"`` holds out whole datasets, which measures cross-dataset
                transfer. ``"group"`` uses ``sample.meta["split_group"]``, the unit that
                must land in one split across every source and task (see
                :mod:`marinedata.scan`). ``"random"`` is available but leaks; use it
                knowingly.
            ratios: split name to fraction. Defaults to 70/15/15 for a supervised
                dataset (or one built with no task at all); to 95/5 (``train``/``probe``)
                for a self-supervised one — see ``SELF_SUPERVISED_DEFAULT_RATIOS``. When
                ``split_map`` already exists on disk and ``ratios`` is omitted, the
                map's own ratios are used instead of the task default, so a
                self-supervised build does not raise against a shared 70/15/15 map.
            seed: groups are hashed with this, so the assignment is deterministic and
                stable when new samples arrive in an existing group.
            tolerance: raise if any achieved split deviates from its requested ratio by
                more than this. Set ``None`` to accept whatever the group sizes allow.
            split_map: path to a ``SPLIT_MAP.json`` to read and extend. Omitted (the
                default), splits are computed fresh every call, as before. Given, a
                group already recorded there keeps its split forever — see
                :mod:`marinedata.splitmap` — shared by every task that passes the same
                path, since the map is keyed by group only.
            frozen: require every group to already be in ``split_map`` — raise instead
                of allocating and appending a new one. A release build passes this so
                it can never grow the shared map; nothing is written to disk either way.

        A self-supervised task reading a shared ``split_map`` never inherits its
        ``test`` split: those groups are excluded from ``self.splits`` outright, and a
        ``val`` group is renamed ``probe`` — see :func:`_to_pretrain_vocabulary`. A
        labelled task's held-out evaluation images must never enter a pretraining
        corpus.
        """
        ratios = _default_ratios(
            ratios, self_supervised=self.task_kind is TaskKind.SELF_SUPERVISED, split_map=split_map
        )
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")

        keys = [group_key(sample, by) for sample in self.samples]

        counts: dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

        # Shared with the streaming path so the two cannot drift. A split that differed
        # between them would be near-impossible to notice and would invalidate every
        # comparison between runs.
        if split_map is not None:
            assignment = resolve_splits(split_map, counts, ratios, seed=seed, by=by, frozen=frozen)
        else:
            assignment = assign_splits(counts, ratios, seed=seed)

        if self.task_kind is TaskKind.SELF_SUPERVISED and split_map is not None:
            # A shared map speaks train/val/test — the labelled-task vocabulary, not
            # this task's own. See ``_to_pretrain_vocabulary`` for the rule this
            # translation enforces: `val` demotes to `probe`, `test` is excluded
            # outright, never appended to ``self.splits`` at all.
            assignment, ratios = _to_pretrain_vocabulary(assignment, ratios)

        splits: dict[SplitName, list[int]] = {name: [] for name in ratios}
        for position, key in enumerate(keys):
            name = assignment.get(key)
            if name is None:
                continue  # a supervised-vocabulary `test` group, excluded from pretraining
            splits[name].append(position)
        self.splits = splits

        empty = [name for name, positions in splits.items() if not positions]
        if empty and by != "random":
            raise ValueError(
                f"split(by={by!r}) produced empty splits {empty} — only "
                f"{len(counts)} group(s) available. Use more sources/sites, adjust "
                f"ratios, or pass by='random' knowingly."
            )

        check_ratios(
            {name: len(positions) for name, positions in splits.items()},
            ratios,
            len(keys),
            tolerance=tolerance,
            by=by,
            largest_group=max(counts.values(), default=0),
        )
        return self

    # ── statistics ────────────────────────────────────────────────────────

    def class_counts(self, axis: Axis = Axis.TAXON) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            value = sample.labels.get(axis)
            if value is not None:
                counts[value.node_id] = counts.get(value.node_id, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def supervision_coverage(self) -> dict[str, float]:
        """Fraction of samples supervising each axis.

        Low coverage is not a bug — it is the ragged reality of combining sources — but
        it should be known before wondering why a head underperforms.
        """
        if not self.samples:
            return {}
        counts: dict[str, int] = {}
        for sample in self.samples:
            for axis in sample.supervised:
                counts[axis.value] = counts.get(axis.value, 0) + 1
        return {k: v / len(self.samples) for k, v in sorted(counts.items())}

    def summary(self) -> str:
        lines = [
            f"Dataset  {len(self.samples):,} samples  "
            f"from {len({s.source_id for s in self.samples})} source(s)  "
            f"profile={self.lineage.profile}"
        ]
        if self.splits:
            parts = "  ".join(f"{n}={len(v):,}" for n, v in sorted(self.splits.items()))
            lines.append(f"  splits   {parts}")
        coverage = self.supervision_coverage()
        if coverage:
            lines.append("  axes     " + "  ".join(f"{k}={v:.0%}" for k, v in coverage.items()))
        lines.append("  " + self.label_index.summary().replace("\n", "\n  "))
        if self.skipped:
            lines.append("  skipped  " + "; ".join(f"{k}: {v}" for k, v in self.skipped.items()))
        return "\n".join(lines)

    # ── framework adapters ────────────────────────────────────────────────

    def to_pandas(self, split: SplitName | None = None):
        from ..integrations.pandas import to_dataframe

        return to_dataframe(self, split=split)

    def to_torch(self, split: SplitName | None = None, **kwargs):
        from ..integrations.torch import to_torch_dataset

        return to_torch_dataset(self, split=split, **kwargs)

    def to_tf(self, split: SplitName | None = None, **kwargs):
        from ..integrations.tensorflow import to_tf_dataset

        return to_tf_dataset(self, split=split, **kwargs)
