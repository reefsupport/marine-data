"""Label auditing — do crosswalk edges match the labels the data actually contains?

The gap this closes is the exact analogue of layout verification, one level up.

A crosswalk lists ``source_label`` strings that somebody transcribed from a paper, a
config file or a dataset card. Nothing checks them against the data. If Coralscapes emits
``"massive/meandering  alive"`` with two spaces, or CoralNet's group is ``"Hard Coral"``
rather than ``"Hard coral"``, the edge matches nothing — and :class:`Harmonizer` treats an
unknown label as unmappable by design, so the pixels quietly become unsupervised. The
training run succeeds. The metrics are just worse than they should be, for a reason no
loss curve reveals.

Two failure modes, opposite in shape and both invisible without real data:

**Silent drops** — a label in the data with no edge. Supervision is discarded.
**Dead edges** — an edge whose label never appears. A typo, a renamed class, or a
crosswalk written against a different version.

Extraction reuses what the loaders already record in ``Sample.meta``, so this works for
every layout that yields scalar labels without a second parser to keep in sync.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .loaders import DataNotAvailable, LoaderError, build_loader
from .registry import Registry
from .sample import Sample


def observed_labels(sample: Sample) -> list[str]:
    """Native labels a loader recorded for one sample.

    Loaders put the source's own strings in ``meta`` before harmonisation; reading them
    back is how the audit sees what the data really says.
    """
    out: list[str] = []
    single = sample.meta.get("native_label")
    if isinstance(single, str):
        out.append(single)
    many = sample.meta.get("native_labels")
    if isinstance(many, list):
        out.extend(x for x in many if isinstance(x, str))
    return out


@dataclass(frozen=True)
class LabelAudit:
    """What a source's labels look like against its declared crosswalk."""

    source_id: str
    crosswalk_id: str | None
    samples: int
    observed: Counter[str] = field(default_factory=Counter)
    unmapped: tuple[str, ...] = ()
    dead_edges: tuple[str, ...] = ()
    status: str = "audited"
    detail: str = ""

    @property
    def instances(self) -> int:
        return sum(self.observed.values())

    @property
    def coverage(self) -> float:
        """Fraction of observed label INSTANCES that a crosswalk edge maps.

        Instance-weighted, not type-weighted: one unmapped label on 40% of pixels matters
        far more than ten unmapped labels appearing once each, and a type count hides that.
        """
        if not self.instances:
            return 0.0
        dropped = sum(self.observed[label] for label in self.unmapped)
        return 1.0 - dropped / self.instances

    def line(self) -> str:
        if self.status != "audited":
            mark = {"skipped": "-", "unavailable": "·", "failed": "✗"}[self.status]
            return f"{mark} {self.source_id:<30} {self.status:<12} {self.detail[:60]}"
        flag = "✓" if not self.unmapped else "!"
        return (
            f"{flag} {self.source_id:<30} {self.coverage:>6.1%} covered  "
            f"{len(self.observed):>3} labels  "
            f"{len(self.unmapped)} unmapped  {len(self.dead_edges)} dead"
        )

    def report(self) -> str:
        lines = [self.line()]
        for label in self.unmapped:
            share = self.observed[label] / max(self.instances, 1)
            lines.append(f"    SILENT DROP  {label!r} ({self.observed[label]}x, {share:.1%})")
        for label in self.dead_edges:
            lines.append(f"    unseen edge  {label!r} — declared but absent from this sample")
        if self.dead_edges:
            lines.append(
                f"    (a {self.samples}-sample audit sees a partial vocabulary; "
                f"unseen edges are a prompt to look, not proof of a typo)"
            )
        return "\n".join(lines)


def audit_source(
    registry: Registry,
    source_id: str,
    root: str | Path,
    *,
    limit: int = 500,
) -> LabelAudit:
    """Read a local sample and compare its labels against the declared crosswalk."""
    source = registry.source(source_id)
    spec = source.loader
    if spec is None or spec.layout == "metadata-only":
        return LabelAudit(source_id, None, 0, status="skipped", detail="no sample-level loader")

    try:
        loader = build_loader(source, root)
        observed: Counter[str] = Counter()
        seen = 0
        for sample in loader:
            observed.update(observed_labels(sample))
            seen += 1
            if seen >= limit:
                break
    except DataNotAvailable as exc:
        return LabelAudit(source_id, spec.crosswalk_id, 0, status="unavailable", detail=str(exc))
    except LoaderError as exc:
        return LabelAudit(source_id, spec.crosswalk_id, 0, status="failed", detail=str(exc))

    if not observed:
        return LabelAudit(
            source_id,
            spec.crosswalk_id,
            seen,
            status="skipped",
            detail="loader yields no scalar labels (dense masks carry classes in the raster)",
        )

    if not spec.crosswalk_id:
        # Every observed label is unmapped, because there is nothing to map with.
        return LabelAudit(
            source_id,
            None,
            seen,
            observed,
            unmapped=tuple(sorted(observed)),
            detail="no crosswalk declared",
        )

    walk = registry.crosswalk(spec.crosswalk_id)
    declared = {edge.source_label for edge in walk.edges}
    return LabelAudit(
        source_id,
        spec.crosswalk_id,
        seen,
        observed,
        unmapped=tuple(sorted(set(observed) - declared)),
        dead_edges=tuple(sorted(declared - set(observed))),
    )


def summarise(audits: list[LabelAudit]) -> str:
    audited = [a for a in audits if a.status == "audited"]
    drops = sum(len(a.unmapped) for a in audited)
    dead = sum(len(a.dead_edges) for a in audited)
    worst = min((a.coverage for a in audited), default=1.0)
    return (
        f"{len(audited)} audited · {drops} silent drop(s) · {dead} dead edge(s) · "
        f"worst coverage {worst:.1%}"
    )
