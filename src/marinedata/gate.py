"""The licence gate.

Design rule: **the gate raises, it does not warn.** A warning that scrolls past in a
training log is how tainted data reaches shipped weights. If a caller genuinely wants
the permissive subset, they filter explicitly with :func:`allowed`; they do not get it
by ignoring output.

Nothing here is legal advice. The gate enforces the policy encoded in the registry; it
cannot tell you whether that policy is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import LegalBasis, Tier
from .models import Profile, Source


class LicenceViolation(Exception):
    """Raised when a source is not permitted under the active profile.

    The message names the source, the tier, the profile and the specific reason,
    because the person who sees this is usually not the person who wrote the entry.
    """


@dataclass(frozen=True)
class Decision:
    """Outcome of evaluating one source against one profile."""

    source_id: str
    allowed: bool
    reason: str

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise LicenceViolation(self.reason)


def evaluate(source: Source, profile: Profile, *, legal_opinion_ref: str | None = None) -> Decision:
    """Evaluate one source against one profile. Pure; no I/O, no mutation.

    Checks run cheapest-and-most-absolute first so the reported reason is the most
    fundamental one. A prohibited source should report *why it is prohibited*, not
    that it happens to also fail a flag check.
    """
    lic = source.licence

    # 1. Absolute bars. No profile may override these.
    if lic.tier is Tier.PROHIBITED:
        return Decision(
            source.id,
            False,
            f"{source.id}: tier TX_PROHIBITED — never usable. {source.notes or ''}".strip(),
        )

    if lic.flags.provenance_defective:
        return Decision(
            source.id,
            False,
            f"{source.id}: provenance_defective — the licensor appears to grant rights it "
            f"does not hold. No downstream permission cures this.",
        )

    if source.legal_basis is LegalBasis.UNKNOWN:
        return Decision(
            source.id,
            False,
            f"{source.id}: legal_basis is unknown. Resolve provenance before use.",
        )

    # 2. Disputed verification blocks anything that ships.
    ships = any(t in profile.allow_tiers for t in (Tier.PERMISSIVE, Tier.COPYLEFT)) and (
        Tier.NONCOMMERCIAL not in profile.allow_tiers
    )
    if source.verification.disputed and ships:
        note = " ".join((source.verification.dispute_note or "").split())
        return Decision(
            source.id,
            False,
            f"{source.id}: licence is disputed. {note} "
            f"Resolve before using on shipping profile '{profile.id}'.",
        )

    # 2b. Second-hand verification is fine for research, never for shipping.
    if not source.verification.is_primary and ships:
        return Decision(
            source.id,
            False,
            f"{source.id}: licence was transcribed from a secondary source "
            f"({source.verification.verified_by}). Verify against the primary source "
            f"before using on shipping profile '{profile.id}'.",
        )

    # 3. Tier must be allowed by the profile.
    if lic.tier not in profile.allow_tiers:
        allowed = ", ".join(t.value for t in profile.allow_tiers)
        return Decision(
            source.id,
            False,
            f"{source.id}: tier {lic.tier.value} not permitted by profile "
            f"'{profile.id}' (allows: {allowed}).",
        )

    # 4. Flags denied by the profile.
    for flag in profile.deny_flags:
        if getattr(lic.flags, flag, False):
            return Decision(
                source.id,
                False,
                f"{source.id}: licence flag '{flag}' is denied by profile '{profile.id}'. "
                f"{_flag_explanation(flag)}",
            )

    # 5. TDM requires a counsel opinion on record.
    if source.legal_basis is LegalBasis.TDM:
        if not profile.require_legal_opinion:
            return Decision(
                source.id,
                False,
                f"{source.id}: legal_basis=tdm, but profile '{profile.id}' does not "
                f"declare require_legal_opinion. TDM is an affirmative defence, not a "
                f"permission — it needs an explicit profile.",
            )
        if not legal_opinion_ref:
            return Decision(
                source.id,
                False,
                f"{source.id}: legal_basis=tdm requires legal_opinion_ref to be supplied "
                f"at build time. Pass the counsel opinion identifier.",
            )

    return Decision(source.id, True, f"{source.id}: permitted under '{profile.id}'.")


def _flag_explanation(flag: str) -> str:
    return {
        "no_derivatives": "ND forbids derivative works — masks, crops and augmentations "
        "are all derivatives, so the source cannot be used for training at all.",
        "share_alike": "SA requires derivatives under the same licence; use a profile "
        "that commits to open weights.",
        "contract_gated": "Access required assent to terms, which can override the "
        "statutory TDM exception.",
        "provenance_defective": "The licensor cannot grant what it does not hold.",
        "attribution_required": "Attribution must be carried downstream.",
    }.get(flag, "")


def allowed(
    sources: list[Source], profile: Profile, *, legal_opinion_ref: str | None = None
) -> list[Source]:
    """Return the permitted subset. Explicit filtering — never a silent fallback."""
    return [s for s in sources if evaluate(s, profile, legal_opinion_ref=legal_opinion_ref).allowed]


def enforce(
    sources: list[Source], profile: Profile, *, legal_opinion_ref: str | None = None
) -> list[Source]:
    """Return all sources, or raise on the first violation.

    Use this when the caller has asserted the set should already be clean. The
    exception names every violation, not just the first, so a build can be fixed in
    one pass instead of one error at a time.
    """
    decisions = [evaluate(s, profile, legal_opinion_ref=legal_opinion_ref) for s in sources]
    denied = [d for d in decisions if not d.allowed]
    if denied:
        detail = "\n  - ".join(d.reason for d in denied)
        raise LicenceViolation(
            f"{len(denied)} source(s) not permitted under profile '{profile.id}':\n  - {detail}"
        )
    return sources
