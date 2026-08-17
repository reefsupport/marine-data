"""Mirroring — copying source data into our own storage, legally.

The project's headline is "registry, not a data lake", and that remains true for anything
*published*. But training needs data local to compute, and refusing to copy anything
would make the registry useless for its actual purpose. The resolution is that **there
are two different acts here, and conflating them is the mistake**:

**A private working cache** is internal copying. It is not distribution. For permissive
and copyleft sources the licence plainly permits it; for non-commercial sources it is
defensible under a research profile, since no commercial exploitation and no
redistribution occurs. It is what makes training throughput possible.

**A public mirror** is redistribution. It requires the licence to actually permit
redistribution, which NC and especially ND do not. Publishing an ND sample would be
precisely the breach this registry exists to prevent.

So the tier does not decide "may we copy?" — it decides **where the copy may live**. This
module makes that decision mechanical, because it is the one place in the project where a
mistake is a licence breach rather than a bad metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .enums import Tier
from .models import Source


class MirrorTarget(str, Enum):
    """Where a copy is going. The legal question is entirely about this."""

    PRIVATE_CACHE = "private-cache"
    """Our own storage, not served to anyone. Internal copying, not distribution.

    Admits everything except prohibited and no-derivatives sources. ND is excluded
    because caching is harmless but *using* the cache means creating derivatives, and a
    cache nobody may train on is storage cost with no purpose.
    """

    PUBLIC_MIRROR = "public-mirror"
    """Published for others — e.g. a HuggingFace sample repo. This is redistribution.

    Admits only sources whose licence permits redistribution: our own, permissive, and
    copyleft (with share-alike honoured on the mirror itself).
    """

    TRAINING_SHARD = "training-shard"
    """Derived shards (WebDataset/parquet) in our own storage for throughput.

    Same admission as the private cache, but recorded separately because a shard is a
    *derivative* of the source, not a copy of it — which is why ND is excluded outright
    and share-alike must be tracked through to any released weights.
    """


_ADMITTED: dict[MirrorTarget, frozenset[Tier]] = {
    MirrorTarget.PRIVATE_CACHE: frozenset(
        {Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL}
    ),
    MirrorTarget.TRAINING_SHARD: frozenset(
        {Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL}
    ),
    MirrorTarget.PUBLIC_MIRROR: frozenset({Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT}),
}


class MirrorViolation(Exception):
    """Copying this source to this target is not permitted."""


@dataclass(frozen=True)
class MirrorDecision:
    source_id: str
    target: MirrorTarget
    allowed: bool
    reason: str
    obligations: tuple[str, ...] = ()

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise MirrorViolation(self.reason)


def evaluate_mirror(source: Source, target: MirrorTarget) -> MirrorDecision:
    """May this source be copied to this target? Pure; no I/O.

    Ordered most-absolute-first so the reported reason is the fundamental one.
    """
    lic = source.licence
    flags = lic.flags

    if lic.tier is Tier.PROHIBITED:
        return MirrorDecision(
            source.id, target, False, f"{source.id}: TX_PROHIBITED — never copy anywhere."
        )

    if flags.provenance_defective:
        return MirrorDecision(
            source.id,
            target,
            False,
            f"{source.id}: provenance_defective — copying propagates a grant the "
            f"licensor did not hold. Storing it makes us a redistributor of it.",
        )

    if flags.no_derivatives:
        return MirrorDecision(
            source.id,
            target,
            False,
            f"{source.id}: no_derivatives — masks, crops, resizes and shards are all "
            f"derivative acts, so a cache would be unusable and a mirror unlawful.",
        )

    if source.legal_basis.value == "unknown":
        return MirrorDecision(
            source.id,
            target,
            False,
            f"{source.id}: legal_basis unknown. Resolve provenance before copying — "
            f"storage makes an unresolved question permanent.",
        )

    if source.legal_basis.value == "tdm":
        # Art. 4(2) permits retention only as long as necessary for the mining. A
        # permanent lake is the opposite of that, and a public mirror of material we
        # only ever had an exception for is plainly redistribution.
        return MirrorDecision(
            source.id,
            target,
            False,
            f"{source.id}: held under a TDM exception, which permits retention only as "
            f"long as the mining requires. Mirror the derived features or weights, "
            f"never the corpus.",
        )

    if lic.tier not in _ADMITTED[target]:
        extra = (
            " Non-commercial material may be cached for research but never republished."
            if target is MirrorTarget.PUBLIC_MIRROR and lic.tier is Tier.NONCOMMERCIAL
            else ""
        )
        return MirrorDecision(
            source.id,
            target,
            False,
            f"{source.id}: tier {lic.tier.value} not admitted to '{target.value}'.{extra}",
        )

    obligations: list[str] = []
    if flags.attribution_required:
        obligations.append(
            f"{source.id}: carry attribution with every copy — see ATTRIBUTION.md in the target."
        )
    if flags.share_alike and target is MirrorTarget.PUBLIC_MIRROR:
        obligations.append(
            f"{source.id}: share-alike — the mirror must itself be published under "
            f"{lic.id} or a compatible licence."
        )
    if flags.share_alike and target is MirrorTarget.TRAINING_SHARD:
        obligations.append(
            f"{source.id}: share-alike propagates to derivatives — if shards feed "
            f"released weights, those weights inherit the obligation."
        )
    if lic.tier is Tier.NONCOMMERCIAL:
        obligations.append(
            f"{source.id}: non-commercial — this copy may not feed a shipped model. "
            f"Keep it out of ship-* profile builds."
        )

    return MirrorDecision(
        source.id,
        target,
        True,
        f"{source.id}: may be copied to '{target.value}'.",
        tuple(obligations),
    )


@dataclass(frozen=True)
class MirrorPlan:
    """What a mirror operation would do, before it does it."""

    target: MirrorTarget
    included: tuple[MirrorDecision, ...]
    excluded: tuple[MirrorDecision, ...]

    @property
    def obligations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for decision in self.included:
            for obligation in decision.obligations:
                if obligation not in seen:
                    seen.append(obligation)
        return tuple(seen)

    def summary(self) -> str:
        lines = [
            f"mirror plan → {self.target.value}   "
            f"include={len(self.included)}  exclude={len(self.excluded)}"
        ]
        for decision in self.included:
            lines.append(f"  ✓ {decision.source_id}")
        for decision in self.excluded:
            lines.append(f"  ✗ {' '.join(decision.reason.split())}")
        if self.obligations:
            lines.append("  obligations:")
            lines += [f"    - {' '.join(o.split())}" for o in self.obligations]
        return "\n".join(lines)


def plan_mirror(sources: list[Source], target: MirrorTarget) -> MirrorPlan:
    """Evaluate a whole set. Always inspect this before copying anything."""
    decisions = [evaluate_mirror(s, target) for s in sources]
    return MirrorPlan(
        target=target,
        included=tuple(d for d in decisions if d.allowed),
        excluded=tuple(d for d in decisions if not d.allowed),
    )


def attribution_document(plan: MirrorPlan, sources: list[Source]) -> str:
    """ATTRIBUTION.md for a mirror target.

    Generated rather than hand-maintained, because an attribution file that drifts from
    what is actually in the bucket is worse than none — it asserts compliance falsely.
    """
    by_id = {s.id: s for s in sources}
    lines = [
        "# Attribution",
        "",
        "This location contains data from the sources below. Each remains under its own",
        "licence; inclusion here is not a relicensing.",
        "",
    ]
    for decision in plan.included:
        source = by_id[decision.source_id]
        citation = source.citation or source.name
        lines.append(f"## {source.name}")
        lines.append("")
        lines.append(f"- Licence: `{source.licence.id}` ({source.licence.tier.value})")
        lines.append(f"- Citation: {citation}")
        if source.homepage or source.access.uri:
            lines.append(f"- Source: {source.homepage or source.access.uri}")
        lines.append("")
    if plan.obligations:
        lines += ["## Obligations carried", ""]
        lines += [f"- {' '.join(o.split())}" for o in plan.obligations]
        lines.append("")
    return "\n".join(lines)
