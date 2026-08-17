"""Faceted query over the registry.

The primary entry point. Callers ask for what they need by *capability* and *profile*;
everything else is optional narrowing:

    >>> import marinedata as md
    >>> sources = md.find(task="benthic-segmentation", profile="ship-commercial")

The profile is deliberately a required argument. There is no "give me everything"
default, because the most likely mistake in this problem domain is training on data
you did not check.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import AnnotationKind, Capability, Modality, Provenance, Region
from .gate import Decision, evaluate
from .models import Source
from .registry import Registry


@dataclass(frozen=True)
class QueryResult:
    """Sources that matched, plus an audit trail of what was excluded and why.

    Exclusions are first-class. "Which datasets did the licence gate remove from my
    benthic training set?" is a question people need answered, and answering it after
    the fact from logs is how mistakes survive.
    """

    sources: tuple[Source, ...]
    excluded: tuple[Decision, ...]
    profile_id: str

    @property
    def total_items(self) -> int:
        return sum(s.items or 0 for s in self.sources)

    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.sources)

    def summary(self) -> str:
        lines = [
            f"profile={self.profile_id}  matched={len(self.sources)}  "
            f"excluded={len(self.excluded)}  items≈{self.total_items:,}",
        ]
        for s in self.sources:
            lines.append(f"  ✓ {s.id:<28} {s.licence.tier.value:<18} {s.items or '—':>10}")
        for d in self.excluded:
            lines.append(f"  ✗ {d.source_id:<28} {d.reason}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self):
        return iter(self.sources)


def _matches(
    source: Source,
    *,
    task: Capability | None,
    annotation: AnnotationKind | None,
    modality: Modality | None,
    regions: tuple[Region, ...] | None,
    provenance: Provenance | None,
    schema_id: str | None,
    min_items: int | None,
) -> bool:
    if task is not None and task not in source.capabilities:
        return False
    if modality is not None and modality not in source.modalities:
        return False
    if provenance is not None and source.provenance is not provenance:
        return False
    if annotation is not None and not any(a.kind is annotation for a in source.annotations):
        return False
    if schema_id is not None and not any(a.schema_id == schema_id for a in source.annotations):
        return False
    if regions:
        covered = set(source.coverage.regions)
        # Global-coverage sources satisfy any regional filter.
        if Region.GLOBAL not in covered and not (set(regions) & covered):
            return False
    return min_items is None or (source.items or 0) >= min_items


def find(
    *,
    profile: str,
    task: Capability | str | None = None,
    annotation: AnnotationKind | str | None = None,
    modality: Modality | str | None = None,
    region: Region | str | list[Region | str] | None = None,
    provenance: Provenance | str | None = None,
    schema_id: str | None = None,
    min_items: int | None = None,
    registry: Registry | None = None,
    legal_opinion_ref: str | None = None,
) -> QueryResult:
    """Find sources matching the facets *and* permitted by the profile.

    Args:
        profile: Required. The declared purpose of the build — see ``registry/profiles.yaml``.
        task: Capability the source must support (e.g. ``"benthic-segmentation"``).
        annotation: Required supervision kind (e.g. ``"dense-mask"``).
        modality: Required modality (e.g. ``"point-cloud"``).
        region: Region or regions of interest. Global-coverage sources always match.
        provenance: Restrict to ``own`` / ``partner`` / ``public`` / ``auto``.
        schema_id: Require a specific label schema.
        min_items: Minimum item count.
        registry: Registry to query; loads the packaged one if omitted.
        legal_opinion_ref: Counsel opinion identifier, required for TDM-based profiles.

    Returns:
        A :class:`QueryResult` carrying both matches and the reasons for exclusions.

    Raises:
        RegistryError: if the profile is unknown.
        ValueError: if a facet value is not a member of its vocabulary.
    """
    reg = registry if registry is not None else Registry.load()
    prof = reg.profile(profile)

    task_e = Capability(task) if task is not None else None
    ann_e = AnnotationKind(annotation) if annotation is not None else None
    mod_e = Modality(modality) if modality is not None else None
    prov_e = Provenance(provenance) if provenance is not None else None

    regions_e: tuple[Region, ...] | None = None
    if region is not None:
        raw = region if isinstance(region, list) else [region]
        regions_e = tuple(Region(r) for r in raw)

    matched: list[Source] = []
    excluded: list[Decision] = []

    for src in reg:
        if not _matches(
            src,
            task=task_e,
            annotation=ann_e,
            modality=mod_e,
            regions=regions_e,
            provenance=prov_e,
            schema_id=schema_id,
            min_items=min_items,
        ):
            continue  # facet mismatch is not a licence exclusion; don't report it as one

        decision = evaluate(src, prof, legal_opinion_ref=legal_opinion_ref)
        if decision.allowed:
            matched.append(src)
        else:
            excluded.append(decision)

    return QueryResult(tuple(matched), tuple(excluded), prof.id)
