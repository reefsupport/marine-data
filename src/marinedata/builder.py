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

import hashlib
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
from .schema import Axis

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
            ratios: split name to fraction. Defaults to 70/15/15.
            seed: groups are hashed with this, so the assignment is deterministic and
                stable when new samples arrive in an existing group.
            tolerance: raise if any achieved split deviates from its requested ratio by
                more than this. Set ``None`` to accept whatever the group sizes allow.
        """
        ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")

        if by == "random":
            keys = [str(i) for i in range(len(self.samples))]
        elif by == "source":
            keys = [s.source_id for s in self.samples]
        elif by == "site":
            keys = [f"{s.source_id}/{s.meta.get('partition', '')}" for s in self.samples]
        else:
            raise ValueError(f"unknown split strategy '{by}' (site | source | random)")

        # Assign whole groups to splits, weighted by SAMPLE COUNT.
        #
        # An earlier version sliced the group *list* proportionally, which silently
        # produced unusable splits whenever groups differed in size — and they always
        # do. Measured with a realistic corpus (one 60k public set, six own sites of
        # 105-2,075), a requested 70/15/15 returned 95.8/1.0/3.2, with test containing
        # only Coralscapes and val only Tayrona: an "evaluation" over one site each.
        # The empty-split guard below did not fire, because no split was empty.
        #
        # Groups are ordered by a seeded hash so assignment is deterministic and stable,
        # then each is given to whichever split is furthest below its target share of
        # samples. Groups are never divided — that is the whole point of grouping.
        counts: dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

        groups = sorted(set(keys), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        # Largest first within the deterministic order: a greedy packer that places the
        # big groups while every split is still empty gets far closer to target than one
        # that meets them last with no room left.
        groups.sort(key=lambda k: (-counts[k], hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()))

        total_samples = len(keys)
        targets = {name: fraction * total_samples for name, fraction in ratios.items()}
        filled = dict.fromkeys(ratios, 0)
        assignment: dict[str, SplitName] = {}

        for key in groups:
            # Deficit relative to target, so a split needing 15% of a large corpus is
            # not starved by one needing 70%.
            name = max(ratios, key=lambda n: (targets[n] - filled[n], n))
            assignment[key] = name
            filled[name] += counts[key]

        splits: dict[SplitName, list[int]] = {name: [] for name in ratios}
        for position, key in enumerate(keys):
            splits[assignment[key]].append(position)
        self.splits = splits

        empty = [name for name, positions in splits.items() if not positions]
        if empty and by != "random":
            # Grouped splitting on few groups can leave a split empty. Better to say so
            # than to hand back a silently unusable test set.
            raise ValueError(
                f"split(by={by!r}) produced empty splits {empty} — only "
                f"{len(set(keys))} group(s) available. Use more sources/sites, adjust "
                f"ratios, or pass by='random' knowingly."
            )

        # Whole groups cannot be divided, so a corpus dominated by one huge group may be
        # unable to hit the requested ratios however well it is packed. That is a real
        # constraint, not a bug — but it must be reported, because a test split that is
        # 3% instead of 15% produces confident metrics over a fraction of the data.
        if by != "random" and tolerance is not None:
            skewed = {
                name: len(positions) / total_samples
                for name, positions in splits.items()
                if abs(len(positions) / total_samples - ratios[name]) > tolerance
            }
            if skewed:
                achieved = "  ".join(f"{n}={v:.1%}" for n, v in sorted(skewed.items()))
                wanted = "  ".join(f"{n}={ratios[n]:.1%}" for n in sorted(skewed))
                raise ValueError(
                    f"split(by={by!r}) could not hit the requested ratios within "
                    f"{tolerance:.0%}: wanted {wanted}, got {achieved}. Group sizes are "
                    f"too uneven to divide this way — the largest group holds "
                    f"{max(counts.values()):,} of {total_samples:,} samples. Rebalance "
                    f"the corpus, relax `tolerance`, or split by a finer unit."
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
        index = LabelIndex.from_samples(samples, schema, min_count=min_count)
        lineage = build_lineage(
            used, self.profile, excluded=denied, legal_opinion_ref=self.legal_opinion_ref
        )
        return Dataset(samples=samples, label_index=index, lineage=lineage, skipped=skipped)
