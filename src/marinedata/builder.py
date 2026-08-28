"""Building a training set from many sources.

Two decisions are baked in because getting them wrong is both easy and invisible:

**The licence gate runs before anything is read.** You cannot accidentally build a
shippable dataset containing research-only data, because the sources never enter.

**Splits are group-wise by default, never random.** Consecutive frames from a transect
overlap heavily; the same colony appears in dozens of them. A random split puts near
duplicates on both sides and inflates every metric — the JBG060 groups did exactly this
and reported numbers nobody could reproduce. ``split(by="site")`` is the honest default,
and ``by="random"`` exists only so that choosing it is deliberate and visible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .enums import AnnotationKind
from .gate import Decision, evaluate
from .labelindex import LabelIndex
from .lineage import Lineage, build_lineage
from .loaders import DataNotAvailable, LoaderError, build_loader
from .models import Source
from .registry import Registry
from .sample import Sample
from .scan import (
    CorpusScan,
    assign_splits,
    check_ratios,
    group_key,
    scan,
)
from .schema import Axis
from .task import TaskKind

SUPERVISED_DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

SELF_SUPERVISED_DEFAULT_RATIOS = {"train": 0.95, "probe": 0.05}
"""A self-supervised corpus has no ground truth to score a "val"/"test" split against —
the 70/15/15 supervised default would carve off 30% of a pretraining corpus for nothing.
The held-out slice here is named ``probe`` rather than ``val``/``test`` on purpose: its
job is monitoring representation quality (e.g. a linear probe or k-NN check), not a task
accuracy metric, and reusing supervised split names would suggest otherwise. Final
evaluation of a self-supervised encoder happens on a separate, labelled downstream task
— not inside the pretraining corpus itself."""

SplitName = str

_SCALAR_LABEL_KINDS = frozenset(
    {
        AnnotationKind.IMAGE_LABEL,
        AnnotationKind.MULTILABEL,
        AnnotationKind.POINT_LABEL,
        AnnotationKind.BBOX,
        AnnotationKind.TRACK,
        AnnotationKind.AUDIO_EVENT,
    }
)
"""Annotation kinds that yield a per-sample scalar label, and can therefore leak an
unmapped vocabulary into the label index. Dense masks and point clouds carry classes in
the raster instead, which the index never touches."""


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
    ) -> Dataset:
        """Assign splits. Returns self so it chains.

        Args:
            by: ``"site"`` groups by partition then source — the default, and the only
                one that gives a trustworthy generalisation estimate for transect data.
                ``"source"`` holds out whole datasets, which measures cross-dataset
                transfer. ``"random"`` is available but leaks; use it knowingly.
            ratios: split name to fraction. Defaults to 70/15/15 for a supervised
                dataset (or one built with no task at all); to 95/5 (``train``/``probe``)
                for a self-supervised one — see ``SELF_SUPERVISED_DEFAULT_RATIOS``.
            seed: groups are hashed with this, so the assignment is deterministic and
                stable when new samples arrive in an existing group.
            tolerance: raise if any achieved split deviates from its requested ratio by
                more than this. Set ``None`` to accept whatever the group sizes allow.
        """
        if ratios is None:
            ratios = (
                SELF_SUPERVISED_DEFAULT_RATIOS
                if self.task_kind is TaskKind.SELF_SUPERVISED
                else SUPERVISED_DEFAULT_RATIOS
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
        assignment = assign_splits(counts, ratios, seed=seed)

        splits: dict[SplitName, list[int]] = {name: [] for name in ratios}
        for position, key in enumerate(keys):
            splits[assignment[key]].append(position)
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
        from .integrations.pandas import to_dataframe

        return to_dataframe(self, split=split)

    def to_torch(self, split: SplitName | None = None, **kwargs):
        from .integrations.torch import to_torch_dataset

        return to_torch_dataset(self, split=split, **kwargs)

    def to_tf(self, split: SplitName | None = None, **kwargs):
        from .integrations.tensorflow import to_tf_dataset

        return to_tf_dataset(self, split=split, **kwargs)


class DatasetBuilder:
    """Assemble a training set from registry sources under a release profile."""

    def __init__(
        self,
        registry: Registry,
        *,
        profile: str,
        roots: dict[str, str | Path],
        task: str | None = None,
        schema_id: str = "rs-benthic-v1",
        legal_opinion_ref: str | None = None,
        strict: bool = False,
        allow_unmapped: bool = False,
        task_id: str | None = None,
    ) -> None:
        """
        Args:
            roots: source id to local path. Only sources present here are read; the
                registry says what is *permitted*, not what you have on disk.
            strict: if True, a source that fails to load raises instead of being
                recorded in ``Dataset.skipped``.
            allow_unmapped: permit sources with no crosswalk into ``schema_id``. Their
                native labels then enter the label index verbatim, mixed in with
                canonical node ids. That is almost always a mistake — a class list of
                ``["HC", "SC", "18", "47"]`` is two vocabularies pretending to be one —
                so it is off by default and must be chosen deliberately.
        """
        self.registry = registry
        self.profile = registry.profile(profile)
        self.roots = {k: Path(v) for k, v in roots.items()}
        self.task = task
        self.schema_id = schema_id
        self.legal_opinion_ref = legal_opinion_ref
        self.strict = strict
        self.allow_unmapped = allow_unmapped
        self.task_id = task_id
        self.task_spec = registry.task(task_id) if task_id else None

        # A self-supervised task fixes no vocabulary, so there is nothing to build a
        # projector for — the label index falls back to whatever supervision the roots
        # actually carry, exactly as it does with no task_id at all. That is deliberate:
        # see TaskSpec's docstring for why this is one type rather than two.
        self.projector = (
            registry.projector_for(task_id)
            if self.task_spec is not None and self.task_spec.kind is TaskKind.SUPERVISED
            else None
        )
        if self.projector is not None:
            # The task fixes the schema; a mismatch would silently project onto the
            # wrong vocabulary.
            task_schema = self.projector.task.schema_id
            if task_schema != schema_id:
                raise ValueError(
                    f"task '{task_id}' targets schema '{task_schema}' but the builder was "
                    f"given '{schema_id}'"
                )

    def _check_mappable(self, sources: list[Source]) -> list[str]:
        """Sources that produce labels but cannot map them into the target schema."""
        unmapped: list[str] = []
        for source in sources:
            spec = source.loader
            if spec is None:
                continue
            walk_id = spec.crosswalk_id
            if walk_id and self.registry.crosswalk(walk_id).target_schema == self.schema_id:
                continue
            # Only *scalar* labels can pollute the label index. A dense mask or a
            # labelled point cloud carries its classes in the raster, which the index
            # never sees, so those sources need no crosswalk to be safe here. (They
            # still need one to be *useful* — that is a modelling question, not a
            # correctness one.)
            if not any(a.supervises and a.kind in _SCALAR_LABEL_KINDS for a in source.annotations):
                continue
            unmapped.append(source.id)
        return unmapped

    def _permitted(self) -> tuple[list[Source], list[Decision]]:
        allowed: list[Source] = []
        denied: list[Decision] = []
        for source_id in self.roots:
            source = self.registry.source(source_id)
            if self.task and self.task not in {c.value for c in source.capabilities}:
                continue
            decision = evaluate(source, self.profile, legal_opinion_ref=self.legal_opinion_ref)
            (allowed if decision.allowed else denied).append(
                source if decision.allowed else decision
            )
        return allowed, denied

    def stream_samples(self) -> Iterator[Sample]:
        """Yield every permitted sample without materialising the corpus.

        The licence gate and the crosswalk check both run BEFORE the first read, exactly
        as in :meth:`build` — a disallowed source cannot reach the stream, so there is no
        path by which a research-only sample arrives in a training batch.
        """
        allowed, denied = self._permitted()
        if not allowed:
            reasons = "\n  - ".join(d.reason for d in denied) or "no sources matched the task"
            raise ValueError(
                f"No permitted sources for profile '{self.profile.id}':\n  - {reasons}"
            )
        unmapped = self._check_mappable(allowed)
        if unmapped and not self.allow_unmapped:
            raise ValueError(
                f"These sources emit labels but have no crosswalk into "
                f"'{self.schema_id}': {', '.join(unmapped)}."
            )

        for source in allowed:
            try:
                loader = build_loader(source, self.roots[source.id])
                harmonizer = self.registry.harmonizer_for(source.id)
                if harmonizer is not None and hasattr(loader, "bind_harmonizer"):
                    loader.bind_harmonizer(harmonizer)
                yield from loader
            except (LoaderError, DataNotAvailable):
                if self.strict:
                    raise
                continue

    def build_streaming(
        self,
        *,
        by: str = "site",
        ratios: dict[SplitName, float] | None = None,
        seed: int = 0,
        tolerance: float | None = 0.10,
        min_count: int = 1,
    ) -> StreamingDataset:
        """Plan a corpus in constant memory, then stream it.

        Makes one metadata pass to gather counters, computes the label index and split
        assignment from those, and returns a plan. Nothing proportional to the corpus is
        retained.

        Use this when the corpus exceeds ~1M samples; :meth:`build` stays the simpler
        choice below that. Defaults to 70/15/15 for a supervised (or task-less) corpus,
        95/5 for a self-supervised one — see ``SELF_SUPERVISED_DEFAULT_RATIOS``.
        """
        if ratios is None:
            ratios = (
                SELF_SUPERVISED_DEFAULT_RATIOS
                if self.task_spec is not None and self.task_spec.kind is TaskKind.SELF_SUPERVISED
                else SUPERVISED_DEFAULT_RATIOS
            )
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

        corpus = scan(self.stream_samples(), by=by)
        if not corpus.total:
            raise ValueError("No samples found. Check `roots` point at fetched data.")

        assignment = assign_splits(dict(corpus.groups), ratios, seed=seed)

        achieved: dict[SplitName, int] = {}
        for key, count in corpus.groups.items():
            achieved[assignment[key]] = achieved.get(assignment[key], 0) + count
        check_ratios(
            achieved,
            ratios,
            corpus.total,
            tolerance=tolerance,
            by=by,
            largest_group=max(corpus.groups.values(), default=0),
        )

        schema = self.registry.label_schema(self.schema_id)
        index = (
            LabelIndex.for_task(self.projector.task, schema)
            if self.projector is not None
            else LabelIndex.from_counts(corpus.labels, schema, min_count=min_count)
        )
        allowed, denied = self._permitted()
        return StreamingDataset(
            builder=self,
            label_index=index,
            lineage=build_lineage(
                allowed,
                self.profile,
                excluded=denied,
                legal_opinion_ref=self.legal_opinion_ref,
                registry_commit=self.registry.commit,
                items_consumed=dict(corpus.sources),
            ),
            corpus=corpus,
            assignment=assignment,
            by=by,
            task_kind=self.task_spec.kind if self.task_spec is not None else None,
        )

    def build(self, *, min_count: int = 1) -> Dataset:
        """Gate, load, harmonise and index. Raises before reading anything disallowed."""
        allowed, denied = self._permitted()
        if not allowed:
            reasons = "\n  - ".join(d.reason for d in denied) or "no sources matched the task"
            raise ValueError(
                f"No permitted sources for profile '{self.profile.id}':\n  - {reasons}"
            )

        unmapped = self._check_mappable(allowed)
        if unmapped and not self.allow_unmapped:
            raise ValueError(
                f"These sources emit labels but have no crosswalk into "
                f"'{self.schema_id}': {', '.join(unmapped)}.\n"
                f"Their native labels would enter the label index verbatim and be "
                f"indistinguishable from canonical node ids — a class list mixing "
                f"'HC' with '47' is two vocabularies pretending to be one.\n"
                f"Add a crosswalk (registry/crosswalks/), drop the source, or pass "
                f"allow_unmapped=True if you genuinely want native labels."
            )

        samples: list[Sample] = []
        skipped: dict[str, str] = {}
        used: list[Source] = []

        for source in allowed:
            try:
                loader = build_loader(source, self.roots[source.id])
                harmonizer = self.registry.harmonizer_for(source.id)
                if harmonizer is not None and hasattr(loader, "bind_harmonizer"):
                    loader.bind_harmonizer(harmonizer)
                loaded = list(loader)
            except (LoaderError, DataNotAvailable) as exc:
                if self.strict:
                    raise
                skipped[source.id] = str(exc)[:200]
                continue
            samples.extend(loaded)
            used.append(source)

        if not samples:
            detail = "; ".join(f"{k}: {v}" for k, v in skipped.items())
            raise ValueError(f"No samples loaded. {detail}")

        schema = self.registry.label_schema(self.schema_id)
        index = (
            LabelIndex.for_task(self.projector.task, schema)
            if self.projector is not None
            else LabelIndex.from_samples(samples, schema, min_count=min_count)
        )
        lineage = build_lineage(
            used,
            self.profile,
            excluded=denied,
            legal_opinion_ref=self.legal_opinion_ref,
            registry_commit=self.registry.commit,
            items_consumed=Counter(sample.source_id for sample in samples),
        )
        return Dataset(
            samples=samples,
            label_index=index,
            lineage=lineage,
            skipped=skipped,
            projector=self.projector,
            task_kind=self.task_spec.kind if self.task_spec is not None else None,
        )


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
