"""``DatasetBuilder`` — turns registry sources into a ``Dataset`` or ``StreamingDataset``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..gate import Decision, evaluate
from ..labelindex import LabelIndex
from ..lineage import build_lineage
from ..loaders import DataNotAvailable, LoaderError, build_loader
from ..models import Source
from ..registry import Registry
from ..sample import Sample
from ..scan import assign_splits, check_ratios, scan
from ..splitmap import resolve_splits
from ..task import SourceFit, TaskKind, fit_source
from .dataset import Dataset
from .defaults import _SCALAR_LABEL_KINDS, SplitName, _default_ratios
from .exclusions import PartialAbstainExclusion, SourcesDroppedError
from .streaming import StreamingDataset


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
        exclude_unmapped: bool = False,
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
            exclude_unmapped: drop a source with no crosswalk into ``schema_id`` from this
                build (reason in :attr:`unmapped_excluded`) instead of raising for the whole
                task — the release builder's per-source task exclusion (WP-R6).
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
        self.exclude_unmapped = exclude_unmapped
        self.unmapped_excluded: tuple[tuple[str, str], ...] = ()
        """``(source_id, reason)`` per source dropped for lacking a crosswalk into the task's
        schema (only with ``exclude_unmapped``); set by :meth:`build`/:meth:`stream_samples`."""
        self.partial_abstain_excluded: tuple[PartialAbstainExclusion, ...] = ()
        """Set by :meth:`stream_samples`/:meth:`build_streaming` once run — see
        ``Dataset.partial_abstain_excluded`` for the eager path's equivalent."""

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

    def _drop_unmapped(self, allowed: list[Source]) -> list[Source]:
        """With ``exclude_unmapped``: ``allowed`` minus the sources that cannot map into
        ``schema_id`` (recorded with the reason); otherwise ``allowed`` unchanged."""
        if not self.exclude_unmapped or self.allow_unmapped:
            return allowed
        unmapped = set(self._check_mappable(allowed))
        self.unmapped_excluded = tuple(
            (sid, f"no crosswalk into schema {self.schema_id!r} for task {self.task_id!r}")
            for sid in sorted(unmapped)
        )
        kept = [source for source in allowed if source.id not in unmapped]
        if not kept:
            raise ValueError(
                "every permitted source lacks a crosswalk into "
                f"'{self.schema_id}': {', '.join(sorted(unmapped))}."
            )
        return kept

    def _partial_abstain_fits(self, sources: list[Source]) -> dict[str, SourceFit]:
        """Sources among ``sources`` whose native labels on the task's axis are a mix of
        resolved and coarser-abstaining (WS-D S26).

        Supervised tasks only — a self-supervised task fixes no vocabulary to abstain
        against. A static crosswalk analysis, same as ``fit_source`` everywhere else in
        this codebase, so it costs nothing to run before any data is read.
        """
        if self.projector is None:
            return {}
        fits: dict[str, SourceFit] = {}
        for source in sources:
            spec = source.loader
            if spec is None or not spec.crosswalk_id:
                continue
            crosswalk = self.registry.crosswalk(spec.crosswalk_id)
            if crosswalk.target_schema != self.schema_id:
                continue
            fit = fit_source(self.projector, crosswalk, source.id)
            if fit.partial_abstain:
                fits[source.id] = fit
        return fits

    def _drop_partial_abstain(
        self, sources: list[Source]
    ) -> tuple[list[Source], tuple[PartialAbstainExclusion, ...]]:
        """Remove partial-abstain sources from ``sources``, reporting what was dropped."""
        fits = self._partial_abstain_fits(sources)
        if not fits:
            return sources, ()
        kept = [s for s in sources if s.id not in fits]
        excluded = tuple(
            PartialAbstainExclusion(
                source_id=source_id,
                task_id=self.task_id or "",
                abstaining_labels=fit.coarser,
                rows_dropped=self.registry.source(source_id).primary_count or 0,
            )
            for source_id, fit in sorted(fits.items())
        )
        return kept, excluded

    def stream_samples(self, *, dropped: dict[str, str] | None = None) -> Iterator[Sample]:
        """Yield every permitted sample without materialising the corpus.

        The licence gate and the crosswalk check both run BEFORE the first read, exactly
        as in :meth:`build` — a disallowed source cannot reach the stream, so there is no
        path by which a research-only sample arrives in a training batch.

        Training-only, never a compliance record: this yields bare samples with **no
        lineage object**, so a denied source's exclusion reason is never surfaced here.
        Iterating this directly is for feeding a training loop, not for producing (or
        standing in for) an audit artifact. Callers who need the recorded, reasoned
        exclusion list — e.g. before shipping anything derived from this stream — must
        go through :meth:`build_streaming`, whose returned ``lineage.excluded`` carries
        it, exactly like :meth:`build`.

        Args:
            dropped: if given, a source that is permitted but fails to load (and
                ``strict`` is False) records its id and exception here rather than
                vanishing silently — the caller (:meth:`build_streaming`) reads this
                back once the stream is exhausted to build an honest lineage record.
        """
        allowed, denied = self._permitted()
        if not allowed:
            reasons = "\n  - ".join(d.reason for d in denied) or "no sources matched the task"
            raise ValueError(
                f"No permitted sources for profile '{self.profile.id}':\n  - {reasons}"
            )
        allowed = self._drop_unmapped(allowed)
        unmapped = self._check_mappable(allowed)
        if unmapped and not self.allow_unmapped:
            raise ValueError(
                f"These sources emit labels but have no crosswalk into "
                f"'{self.schema_id}': {', '.join(unmapped)}."
            )

        allowed, self.partial_abstain_excluded = self._drop_partial_abstain(allowed)

        for source in allowed:
            try:
                loader = build_loader(source, self.roots[source.id])
                harmonizer = self.registry.harmonizer_for(source.id)
                if harmonizer is not None and hasattr(loader, "bind_harmonizer"):
                    loader.bind_harmonizer(harmonizer)
                yield from loader
            except (LoaderError, DataNotAvailable) as exc:
                if self.strict:
                    raise
                if dropped is not None:
                    dropped[source.id] = str(exc)[:200]
                continue

    def build_streaming(
        self,
        *,
        by: str = "site",
        ratios: dict[SplitName, float] | None = None,
        seed: int = 0,
        tolerance: float | None = 0.10,
        min_count: int = 1,
        split_map: str | Path | None = None,
        frozen: bool = False,
    ) -> StreamingDataset:
        """Plan a corpus in constant memory, then stream it.

        Makes one metadata pass to gather counters, computes the label index and split
        assignment from those, and returns a plan. Nothing proportional to the corpus is
        retained.

        Use this when the corpus exceeds ~1M samples; :meth:`build` stays the simpler
        choice below that. Defaults to 70/15/15 for a supervised (or task-less) corpus,
        95/5 for a self-supervised one — see ``SELF_SUPERVISED_DEFAULT_RATIOS`` — unless
        ``split_map`` already has a map on disk and ``ratios`` is omitted, in which case
        the map's own ratios are adopted (see :meth:`Dataset.split`).

        ``split_map``: as in :meth:`Dataset.split` — a path to a ``SPLIT_MAP.json`` to
        read and extend, shared with the eager path and every task that passes it.
        ``frozen``: as in :meth:`Dataset.split` — raise on a group absent from the map
        instead of allocating and appending it; nothing is written either way.
        """
        ratios = _default_ratios(
            ratios,
            self_supervised=self.task_spec is not None
            and self.task_spec.kind is TaskKind.SELF_SUPERVISED,
            split_map=split_map,
        )
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

        dropped: dict[str, str] = {}
        corpus = scan(self.stream_samples(dropped=dropped), by=by)
        if not corpus.total:
            raise ValueError("No samples found. Check `roots` point at fetched data.")

        if split_map is not None:
            assignment = resolve_splits(
                split_map, dict(corpus.groups), ratios, seed=seed, by=by, frozen=frozen
            )
        else:
            assignment = assign_splits(dict(corpus.groups), ratios, seed=seed)

        achieved: dict[SplitName, int] = {}
        for key, count in corpus.groups.items():
            if key not in assignment:  # a group the frozen map withholds: no split, no samples
                continue
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
        excluded_ids = {e.source_id for e in self.partial_abstain_excluded}
        allowed = [s for s in allowed if s.id not in excluded_ids]
        # Only sources that actually yielded a sample belong in the lineage's `datasets`
        # — `allowed` is what the gate let through, not what contributed. A source that
        # raised while loading is in `dropped`, never in `used`, so the audit record
        # cannot claim it as a contributor.
        used = [s for s in allowed if s.id not in dropped]
        streaming_dataset = StreamingDataset(
            builder=self,
            label_index=index,
            lineage=build_lineage(
                used,
                self.profile,
                excluded=denied,
                dropped=dropped,
                legal_opinion_ref=self.legal_opinion_ref,
                registry_commit=self.registry.commit,
                items_consumed=dict(corpus.sources),
            ),
            corpus=corpus,
            assignment=assignment,
            by=by,
            task_kind=self.task_spec.kind if self.task_spec is not None else None,
            partial_abstain_excluded=self.partial_abstain_excluded,
        )
        if dropped:
            # D2: complete the run, report every drop by name, then fail loud — a
            # caller must not be able to read a clean return as "all sources included".
            # The dataset just built is attached so a caller who deliberately wants the
            # partial result can still recover it.
            raise SourcesDroppedError(dropped, streaming_dataset)
        return streaming_dataset

    def build(self, *, min_count: int = 1) -> Dataset:
        """Gate, load, harmonise and index. Raises before reading anything disallowed."""
        allowed, denied = self._permitted()
        if not allowed:
            reasons = "\n  - ".join(d.reason for d in denied) or "no sources matched the task"
            raise ValueError(
                f"No permitted sources for profile '{self.profile.id}':\n  - {reasons}"
            )

        allowed = self._drop_unmapped(allowed)
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

        allowed, partial_abstain_excluded = self._drop_partial_abstain(allowed)

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
            dropped=skipped,
            legal_opinion_ref=self.legal_opinion_ref,
            registry_commit=self.registry.commit,
            items_consumed=Counter(sample.source_id for sample in samples),
        )
        dataset = Dataset(
            samples=samples,
            label_index=index,
            lineage=lineage,
            skipped=skipped,
            projector=self.projector,
            task_kind=self.task_spec.kind if self.task_spec is not None else None,
            partial_abstain_excluded=partial_abstain_excluded,
        )
        if skipped:
            # D2: the run completed and every source was attempted — this is the
            # loud, non-zero-exit report of what got dropped, not a fail-fast abort.
            # `dataset` is attached so a caller who deliberately wants the partial
            # result anyway can catch this and pull it back out.
            raise SourcesDroppedError(skipped, dataset)
        return dataset
