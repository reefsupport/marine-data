"""Lineage reports — the audit artifact.

Every build emits a record of exactly which sources contributed, under which tier and
legal basis, and which were excluded and why. Attach it to the model card.

This is the file that answers *"prove you had the right to train on this."* It is also
what lets a stranger check the claim, which is the point of publishing the registry at
all — an unverifiable compliance claim is a marketing statement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .gate import Decision
from .models import Profile, Source


@dataclass(frozen=True)
class LineageEntry:
    id: str
    name: str
    version: str
    tier: str
    licence: str
    legal_basis: str
    provenance: str
    items: int | None
    items_source: str
    """``"consumed"`` when ``items`` counts samples this build actually read (from
    ``build()``/``build_streaming()``), ``"declared"`` when it falls back to the
    registry's own estimate because nothing was read yet (e.g. a CLI ``lineage`` report
    built from ``find()`` alone). Without this, the field silently overstates what the
    audit record proves — a source declared at 90k items but sampled to 100 for a
    verification run would report 90k, unchanged."""
    licence_flags: dict[str, bool]
    """The specific acts barred (``no_derivatives``, ``share_alike``, ...), not just the
    tier — without this the record can state a decision but not re-derive it: two tier-3
    sources can carry different obligations, and the tier string alone cannot tell you
    which."""
    attribution: str | None


@dataclass(frozen=True)
class Lineage:
    """A complete, hashable record of one dataset build."""

    profile: str
    built_at: str
    datasets: list[LineageEntry]
    excluded: list[dict[str, str]]
    obligations: list[str] = field(default_factory=list)
    legal_opinion_ref: str | None = None
    registry_commit: str | None = None
    content_hash: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def attribution_text(self) -> str:
        """Attribution block for a model card or NOTICE file.

        Sources requiring attribution must be credited wherever the model is
        distributed. Generating this mechanically is the only way it stays correct
        as the training set changes.
        """
        lines = ["This model was trained on the following datasets:", ""]
        for d in self.datasets:
            cite = d.attribution or d.name
            lines.append(f"  - {cite}  [{d.licence}]")
        if self.obligations:
            lines += ["", "Licence obligations carried by this model:", ""]
            lines += [f"  - {o}" for o in self.obligations]
        return "\n".join(lines)


def _obligations(sources: list[Source], profile: Profile) -> list[str]:
    """Derive the obligations this build imposes on whoever ships the result."""
    out: list[str] = []

    if any(s.licence.flags.attribution_required for s in sources):
        out.append("Attribution required — distribute the attribution block with the model.")

    share_alike = [s.id for s in sources if s.licence.flags.share_alike]
    if share_alike:
        target = profile.weights_licence or "the same licence as the source"
        out.append(
            f"ShareAlike: {', '.join(share_alike)} require derivatives under {target}. "
            f"Releasing the weights discharges this."
        )

    if profile.retention_days is not None:
        out.append(
            f"Retention: mined copies must be deleted within {profile.retention_days} days "
            f"of the build (statutory TDM limit)."
        )

    tdm = [s.id for s in sources if s.legal_basis.value == "tdm"]
    if tdm:
        out.append(
            f"TDM basis relied upon for: {', '.join(tdm)}. This is an affirmative defence "
            f"under EU law, not a licence. Deployment outside the EU needs separate review."
        )

    return out


def build_lineage(
    sources: list[Source],
    profile: Profile,
    *,
    excluded: list[Decision] | None = None,
    legal_opinion_ref: str | None = None,
    registry_commit: str | None = None,
    items_consumed: dict[str, int] | None = None,
    now: datetime | None = None,
) -> Lineage:
    """Build a lineage record for a completed dataset build.

    Args:
        sources: The sources that actually contributed.
        profile: The profile the build ran under.
        excluded: Gate decisions for sources that were considered and rejected.
        legal_opinion_ref: Counsel opinion id, where a TDM profile was used.
        registry_commit: Git SHA of the registry, for reproducibility.
        items_consumed: source id to the number of samples this build actually read.
            Omit when nothing was read yet (a registry-only report) — each entry then
            falls back to the registry's declared estimate, marked as such.
        now: Timestamp override, for deterministic tests.
    """
    items_consumed = items_consumed or {}
    entries = [
        LineageEntry(
            id=s.id,
            name=s.name,
            version=s.version,
            tier=s.licence.tier.value,
            licence=s.licence.id,
            legal_basis=s.legal_basis.value,
            provenance=s.provenance.value,
            items=items_consumed.get(s.id, s.items),
            items_source="consumed" if s.id in items_consumed else "declared",
            licence_flags=s.licence.flags.model_dump(),
            attribution=s.citation,
        )
        for s in sorted(sources, key=lambda s: s.id)
    ]

    excluded_records = [
        {"id": d.source_id, "reason": d.reason}
        for d in sorted(excluded or [], key=lambda d: d.source_id)
    ]

    stamp = (now or datetime.now(timezone.utc)).isoformat()

    # Hash covers the substance of the build, not its timestamp, so identical builds
    # at different times produce the same hash.
    payload = json.dumps(
        {
            "profile": profile.id,
            "datasets": [asdict(e) for e in entries],
            "excluded": excluded_records,
            "registry_commit": registry_commit,
        },
        sort_keys=True,
    )
    digest = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return Lineage(
        profile=profile.id,
        built_at=stamp,
        datasets=entries,
        excluded=excluded_records,
        obligations=_obligations(sources, profile),
        legal_opinion_ref=legal_opinion_ref,
        registry_commit=registry_commit,
        content_hash=digest,
    )
