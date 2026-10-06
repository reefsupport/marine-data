"""Registry schema.

Every model here is frozen. A registry is a set of assertions about legal and
scientific fact; code that mutates it in place is code that can silently disagree
with the lineage report it emitted five minutes earlier.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    AccessClass,
    AccessMethod,
    AnnotationKind,
    Capability,
    Domain,
    Habitat,
    LegalBasis,
    Modality,
    Provenance,
    Redistribution,
    Region,
    Tier,
)
from .schema import Axis


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LicenceFlags(_Frozen):
    """Specific acts barred, independent of tier.

    Tier alone is insufficient. ``CC-BY-NC`` and ``CC-BY-NC-ND`` share a tier, but ND
    forbids derivative works outright — you cannot generate a mask, a crop, or an
    augmentation from an ND source. That distinction has to reach the loader.
    """

    attribution_required: bool = False
    share_alike: bool = False
    """Derivatives must carry the same licence. Compatible with commercial use."""

    no_derivatives: bool = False
    """Blocks masks, crops, resizes, augmentation — i.e. blocks training entirely."""

    contract_gated: bool = False
    """Access required assent to terms (form, click-through). A contract can override
    the EU commercial TDM exception, because DSM Art. 7 protects Arts. 3/5/6 but not 4."""

    provenance_defective: bool = False
    """The licensor appears to grant rights it does not hold (e.g. scraped stock imagery).
    No downstream permission can cure this."""


class Licence(_Frozen):
    id: str = Field(description="SPDX identifier where one exists, else a stable local id")
    name: str
    tier: Tier
    flags: LicenceFlags = LicenceFlags()
    url: str | None = None
    notes: str | None = None


VERIFICATION_METHODS = frozenset(
    {
        "licence-file",
        "dataset-card",
        "written-permission",
        "site-terms",
        "absent",
        "secondary",
    }
)

PRIMARY_METHODS = VERIFICATION_METHODS - {"secondary"}


class Verification(_Frozen):
    """How we know the licence claim is true.

    ``verified_by`` must cite a *primary* source — a licence file we opened, a dataset
    card, written permission. Our own prior catalog recorded MarineInst20M as
    "Mixed/Open" on the strength of a paper's phrasing; the LICENSE.txt in the repo
    said CC-BY-NC-SA. Secondary sources are how that happens.

    ``method: secondary`` records an entry transcribed from another catalog rather than
    checked at source. Such entries are usable for research but blocked from shipping
    profiles, so the backlog of unverified claims is visible instead of assumed away.
    """

    verified_on: date
    verified_by: str = Field(min_length=8, description="Source, cited specifically")
    method: str = Field(
        description="licence-file | dataset-card | written-permission | site-terms "
        "| absent | secondary"
    )
    disputed: bool = False
    """Two authoritative sources disagree. Blocks use on shipping profiles."""

    dispute_note: str | None = None

    @model_validator(mode="after")
    def _dispute_needs_note(self) -> Verification:
        if self.disputed and not self.dispute_note:
            raise ValueError("disputed=true requires dispute_note explaining the conflict")
        return self

    @model_validator(mode="after")
    def _method_is_known(self) -> Verification:
        if self.method not in VERIFICATION_METHODS:
            raise ValueError(
                f"unknown verification method '{self.method}'. "
                f"Expected one of: {', '.join(sorted(VERIFICATION_METHODS))}"
            )
        return self

    @property
    def is_primary(self) -> bool:
        return self.method in PRIMARY_METHODS


class LoaderSpec(_Frozen):
    """How to read this source off disk.

    Sources declare a layout rather than shipping bespoke code, so that one tested
    reader serves every source sharing the convention. See ``loaders/base.py``.
    """

    layout: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    schema_id: str | None = Field(
        default=None, description="Native label schema id, for harmonisation"
    )
    crosswalk_id: str | None = Field(
        default=None, description="Crosswalk mapping schema_id into a canonical schema"
    )

    verified_on: date | None = None
    """Date the declared layout was run against a real fetched sample.

    Unset means the layout is a *guess* from documentation. That distinction matters:
    Coralscapes was declared here as image/mask directories and is in fact HuggingFace
    parquet with ``image``/``label`` columns. Synthetic fixtures cannot catch that class
    of error — only real data can. ``marinedata verify`` sets this honestly.
    """

    verified_note: str | None = None

    @property
    def is_verified(self) -> bool:
        return self.verified_on is not None


class Access(_Frozen):
    method: AccessMethod
    uri: str | None = None
    gated: bool = False
    size_bytes: int | None = None
    params: dict[str, str | int | bool] = Field(default_factory=dict)
    """Fetcher parameters — e.g. ``hf_id``, ``image_column``, ``sample_url``.

    The fetcher normalises a source into the local layout the loader declares, so
    source-specific unpacking lives here rather than leaking into loaders.
    """

    notes: str | None = None

    @model_validator(mode="after")
    def _gated_needs_note(self) -> Access:
        if self.gated and not self.notes:
            raise ValueError("gated=true requires notes describing the gate")
        return self


class Annotation(_Frozen):
    """One supervision stream. A source may carry several of different kinds."""

    kind: AnnotationKind
    count: int | None = None
    schema_id: str | None = Field(
        default=None, description="Label schema id, e.g. 'coralscapes-39', 'catami-1.4'"
    )
    classes: int | None = None
    supervises: tuple[str, ...] = ()
    """RS-Benthic axes this stream supervises, e.g. ('taxon', 'condition').

    Drives the masked hierarchical loss: a source that only labels taxon must not
    contribute gradient to the growth-form or condition heads.
    """

    depth: str | None = Field(default=None, description="Deepest hierarchy level, e.g. 'L2'")
    notes: str | None = None


class Coverage(_Frozen):
    regions: tuple[Region, ...]
    ecoregions: tuple[str, ...] = ()
    """MEOW ecoregion names, where known."""

    depth_range_m: tuple[float, float] | None = None
    years: tuple[int, int] | None = None
    sites: int | None = None


class KnownDrop(_Frozen):
    """A measured cross-domain performance drop, with a citation."""

    to_region: Region
    metric: str
    delta: float = Field(description="Signed change, e.g. -30.0 for a 30-point drop")
    source: str = Field(description="Citation or URL for the measurement")


class DomainShift(_Frozen):
    """Generalisation risk, encoded as data.

    Cross-region collapse is the most common deployment failure in marine CV and no
    existing catalog records it. A source can be excellent and still be the wrong
    choice for a given deployment region — that belongs in metadata, not folklore.
    """

    trained_regions: tuple[Region, ...] = ()
    known_drops: tuple[KnownDrop, ...] = ()
    missing_classes: tuple[str, ...] = ()
    """Taxa or categories absent from the schema that are abundant elsewhere.

    Coralscapes-39 has no soft-coral class at all; on Caribbean reefs, where octocorals
    are ~24% of annotations, the model cannot express what it is looking at and collapses
    into the nearest substrate class.
    """

    caveats: tuple[str, ...] = ()


class Checksums(_Frozen):
    """What a stored version's ``CHECKSUMS.sha256`` covers, pinned by one digest.

    Set only once a version has actually been ingested into our storage — it records a
    fact about bytes we hold, not a claim about upstream, so it stays ``None`` for every
    source until then. ``root_digest`` is the sha256 of the ``CHECKSUMS.sha256`` file
    itself (see :mod:`marinedata.checksums`): one short value that transitively pins
    every file in the version, cheap to compare and cheap to store here.
    """

    version: str
    """The version this covers. Carried explicitly so that bumping ``Source.version``
    without re-ingesting fails loudly instead of leaving a digest that silently
    describes the previous version's bytes."""

    root_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: int = Field(gt=0)
    size_bytes: int = Field(gt=0)


class SplitGroupRule(_Frozen):
    """How to derive an image's ``split_group`` for ``resolve_splits(by="group")``.

    Required on every :class:`Source` — a source with no source-specific pattern
    still resolves to the explicit fallback ``<source_id>/<partition>``, the same
    grouping ``by="site"`` already uses, so ``by="group"`` never raises for a source
    nobody has written a rule for yet.

    ``pattern`` is a regex tested against ``stem`` or ``upstream_path`` (per
    ``match_field``) with exactly one capturing group; the captured text fills
    ``{group}`` in ``template``. ``pattern=None`` means "no source-specific
    extraction" and ``template`` is filled from ``{source_id}``/``{partition}``
    instead.
    """

    pattern: str | None = None
    match_field: str = "stem"
    template: str = "{source_id}/{partition}"

    @model_validator(mode="after")
    def _pattern_is_well_formed(self) -> SplitGroupRule:
        if self.match_field not in ("stem", "upstream_path"):
            raise ValueError(
                f"split_group match_field must be 'stem' or 'upstream_path', "
                f"got {self.match_field!r}"
            )
        if self.pattern is not None:
            try:
                compiled = re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"split_group pattern {self.pattern!r} is invalid: {exc}") from exc
            if compiled.groups < 1:
                raise ValueError(
                    f"split_group pattern {self.pattern!r} needs exactly one capturing group"
                )
        return self

    def resolve(self, *, source_id: str, stem: str, upstream_path: str, partition: str) -> str:
        """Derive the split_group for one image. Raises if a declared pattern misses —
        a silent fallback would let one malformed stem quietly leak across splits."""
        if self.pattern is None:
            return self.template.format(source_id=source_id, partition=partition)
        value = stem if self.match_field == "stem" else upstream_path
        match = re.search(self.pattern, value)
        if not match:
            raise ValueError(
                f"{source_id}: split_group pattern {self.pattern!r} did not match "
                f"{self.match_field} {value!r}"
            )
        return self.template.format(group=match.group(1), source_id=source_id, partition=partition)


class Source(_Frozen):
    """A single dataset entry."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-_]*$")
    name: str
    version: str = "unversioned"
    description: str

    licence: Licence
    verification: Verification
    legal_basis: LegalBasis
    redistribution: Redistribution = Redistribution.UNKNOWN
    """Recorded position on redistributing a verbatim copy. Never a storage filter —
    see :class:`~marinedata.enums.Redistribution`."""
    provenance: Provenance

    access_class: AccessClass = AccessClass.UNKNOWN
    """WP-L1a: what this source may be released as (open / restricted-nc / restricted-nd /
    internal-only / unknown). Orthogonal to ``licence.tier``; the flavour filter reads this."""
    licence_per_row: bool = False
    """Every item carries its own licence string (FathomNet, iNat, ...): the row class wins
    over ``access_class``, which is then only the strictest-member bound."""

    access: Access
    modalities: tuple[Modality, ...]
    capabilities: tuple[Capability, ...]
    annotations: tuple[Annotation, ...] = ()
    coverage: Coverage
    domain_shift: DomainShift | None = None
    loader: LoaderSpec | None = None

    checksums: Checksums | None = None
    """Set once this source's declared version is stored. ``None`` means not ingested
    yet, which is every source today — never "ingested but unverified"."""

    default_platform: str | None = None
    default_habitat: str | None = None
    default_instrument: str | None = None
    split_group: SplitGroupRule = Field(default_factory=SplitGroupRule)
    """Required per-source rule for ``resolve_splits(by="group")``. Defaults to the
    explicit fallback (``<source_id>/<partition>``) so every source has one without
    needing a per-entry YAML edit."""

    images_from: tuple[str, ...] = ()
    """Other source ids whose images this source's annotations sit on top of, with no
    pixel copy of its own — a second label layer over an image pool it does not own.
    Referenced ids are validated to exist at registry load (see
    ``Registry._check_references``)."""

    items: int | None = Field(default=None, description="Primary unit count (images/clips)")
    items_note: str | None = None
    citation: str | None = None
    homepage: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    release_skip_reason: str | None = None
    """Why this source is deliberately left out of a release (WP-R2e). Set, the split map skips
    it and ``release build`` records it under ``skipped_sources`` with no ``--allow-skip``."""

    domain: Domain | None = Field(default=None)
    """HK-4a: one subject-matter bucket (:class:`~marinedata.enums.Domain`) for selecting a
    hackathon track's sources. ``None`` only on a hand-built ``Source``; every registry entry
    sets it (``tests/test_registry_domain.py``)."""

    habitat: tuple[Habitat, ...] | None = Field(default=None)
    """WP-2b: controlled-vocabulary physical setting(s) this source's imagery was
    captured in. ``None`` means not yet classified (never "no habitat") — most v1
    sources are backfilled by inference from their own description/notes; see the
    ``# habitat inferred from ...`` comment above each entry in ``registry/sources/*.yaml``.
    A tuple (not a single value) because some sources genuinely mix habitats."""

    location_sensitive: bool = False
    """WP-2b: first-class flag for a source whose true coordinates must never be
    published at full precision (e.g. a poaching-risk species site) — ``is_location_sensitive()``
    treats this as authoritative. Defaults to ``False``; the historical
    ``"location-sensitive"`` tag is still honoured as a fallback for sources that predate
    this field, so no existing YAML needs an edit to keep working."""

    def split_group_for(self, *, stem: str, upstream_path: str, partition: str) -> str:
        """Apply this source's :class:`SplitGroupRule` to one staged image."""
        return self.split_group.resolve(
            source_id=self.id, stem=stem, upstream_path=upstream_path, partition=partition
        )

    def declared_supervision(self) -> frozenset[Axis]:
        """Axes this source's registry entry actually claims, across all annotations.

        The one honest source of truth for a dense mask or point cloud, whose real
        classes live in the raster/geometry and are otherwise invisible to a loader —
        there is no native label string to fall back on the way a scalar-label loader
        has. Shared by ``ImageMaskPairLoader``/``DualConditionMaskLoader`` (what a
        `Sample` should claim) and ``marinedata doctor`` (what the registry claims),
        so the two cannot silently drift apart.
        """
        axes: set[Axis] = set()
        for annotation in self.annotations:
            for name in annotation.supervises:
                try:
                    axes.add(Axis(name))
                except ValueError:
                    continue
        return frozenset(axes)

    @model_validator(mode="after")
    def _checksums_pin_the_declared_version(self) -> Source:
        """A digest that names a different version is worse than no digest at all: it
        would pass every automated check while describing bytes nobody is serving."""
        if self.checksums is not None and self.checksums.version != self.version:
            raise ValueError(
                f"{self.id}: checksums cover version {self.checksums.version!r} but the "
                f"source declares {self.version!r} — re-ingest and re-checksum, or drop "
                "the stale checksums block."
            )
        return self

    @model_validator(mode="after")
    def _prohibited_needs_reason(self) -> Source:
        if self.licence.tier is Tier.PROHIBITED and not self.notes:
            raise ValueError(f"{self.id}: PROHIBITED tier requires notes explaining why")
        return self

    @model_validator(mode="after")
    def _unknown_basis_is_not_usable(self) -> Source:
        """An unknown legal basis must not masquerade as a usable tier."""
        if self.legal_basis is LegalBasis.UNKNOWN and self.licence.tier in (
            Tier.OWN,
            Tier.PERMISSIVE,
            Tier.COPYLEFT,
        ):
            raise ValueError(
                f"{self.id}: legal_basis=unknown is inconsistent with tier "
                f"{self.licence.tier.value}. Verify the licence or use TDM_ONLY."
            )
        return self


class Profile(_Frozen):
    """A declared purpose for a build. The gate enforces it."""

    id: str
    description: str
    allow_tiers: tuple[Tier, ...]
    deny_flags: tuple[str, ...] = ()
    require_legal_opinion: bool = False
    """TDM-based profiles cannot be instantiated without a counsel opinion reference."""

    allow_access_classes: tuple[str, ...] = ()
    """WP-L1b: the ``access_class`` values this profile admits (empty = no class bar). A
    shipping profile sets it so the tier gate and the release flavour never disagree."""

    public_release: bool = False
    """WP-L1b: the output is published even though ``T3_NONCOMMERCIAL`` is allowed (the NC
    flavour), so the disputed / secondary-verification bars apply as on any shipping profile."""

    retention_days: int | None = None
    weights_licence: str | None = None
    """If set, derivative weights must be released under this licence for the profile
    to be honoured. Recorded in lineage so the obligation is not merely intended."""
