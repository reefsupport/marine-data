"""Gate tests.

Written before any dataset was wired, deliberately. The gate is the entire point of
this package; untested, it gets bypassed under deadline pressure exactly once, and
once is enough to contaminate shipped weights.

Every profile is exercised against every tier, plus each flag and absolute bar.
"""

from __future__ import annotations

from datetime import date

import pytest

from marinedata import LicenceViolation, Registry, evaluate
from marinedata.enums import (
    AccessClass,
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.gate import enforce
from marinedata.models import (
    Access,
    Coverage,
    Licence,
    LicenceFlags,
    Source,
    Verification,
)

_CLASS_OF_TIER = {
    Tier.OWN: AccessClass.OPEN,
    Tier.PERMISSIVE: AccessClass.OPEN,
    Tier.COPYLEFT: AccessClass.OPEN,
    Tier.NONCOMMERCIAL: AccessClass.RESTRICTED_NC,
    Tier.TDM_ONLY: AccessClass.UNKNOWN,
    Tier.PROHIBITED: AccessClass.UNKNOWN,
}


def make_source(
    *,
    source_id: str = "test-source",
    tier: Tier = Tier.PERMISSIVE,
    flags: LicenceFlags | None = None,
    legal_basis: LegalBasis = LegalBasis.LICENCE,
    disputed: bool = False,
    notes: str | None = "test entry",
    access_class: AccessClass | None = None,
) -> Source:
    """Minimal valid source, varied on the axes the gate cares about."""
    return Source(
        id=source_id,
        name="Test Source",
        description="Fixture.",
        licence=Licence(id="TEST", name="Test", tier=tier, flags=flags or LicenceFlags()),
        verification=Verification(
            verified_on=date(2026, 8, 17),
            verified_by="fixture primary source",
            method="licence-file",
            disputed=disputed,
            dispute_note="conflicting sources" if disputed else None,
        ),
        legal_basis=legal_basis,
        access_class=access_class or _CLASS_OF_TIER[tier],
        provenance=Provenance.PUBLIC,
        access=Access(method=AccessMethod.HTTP, uri="https://example.invalid"),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        notes=notes,
    )


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


# ── the core matrix: every profile × every tier ───────────────────────────

EXPECTED = {
    # profile            T0     T1     T2     T3     T4     TX
    "ship-commercial": (True, True, False, False, False, False),
    "ship-open": (True, True, True, False, False, False),
    "ship-noncommercial": (True, True, True, True, False, False),
    "research": (True, True, True, True, False, False),
    "pretrain-eu": (True, True, True, True, True, False),
}

TIERS = (
    Tier.OWN,
    Tier.PERMISSIVE,
    Tier.COPYLEFT,
    Tier.NONCOMMERCIAL,
    Tier.TDM_ONLY,
    Tier.PROHIBITED,
)


@pytest.mark.parametrize("profile_id", sorted(EXPECTED))
def test_tier_matrix(registry: Registry, profile_id: str) -> None:
    """Each profile admits exactly the tiers it declares, and no others."""
    profile = registry.profile(profile_id)
    for tier, should_allow in zip(TIERS, EXPECTED[profile_id], strict=True):
        basis = LegalBasis.TDM if tier is Tier.TDM_ONLY else LegalBasis.LICENCE
        source = make_source(tier=tier, legal_basis=basis)
        decision = evaluate(source, profile, legal_opinion_ref="OPINION-2026-001")
        assert decision.allowed is should_allow, (
            f"profile={profile_id} tier={tier.value}: "
            f"expected allowed={should_allow}, got {decision.allowed} — {decision.reason}"
        )


# ── absolute bars: no profile may override these ──────────────────────────


@pytest.mark.parametrize("profile_id", sorted(EXPECTED))
def test_prohibited_never_allowed(registry: Registry, profile_id: str) -> None:
    source = make_source(tier=Tier.PROHIBITED, notes="provenance defect")
    assert not evaluate(source, registry.profile(profile_id)).allowed


@pytest.mark.parametrize("profile_id", sorted(EXPECTED))
def test_provenance_defective_never_allowed(registry: Registry, profile_id: str) -> None:
    """A defective grant is fatal even when the nominal tier is permissive."""
    source = make_source(tier=Tier.PERMISSIVE, flags=LicenceFlags(provenance_defective=True))
    decision = evaluate(source, registry.profile(profile_id))
    assert not decision.allowed
    assert "provenance_defective" in decision.reason


@pytest.mark.parametrize("profile_id", sorted(EXPECTED))
def test_unknown_legal_basis_never_allowed(registry: Registry, profile_id: str) -> None:
    source = make_source(tier=Tier.TDM_ONLY, legal_basis=LegalBasis.UNKNOWN)
    decision = evaluate(source, registry.profile(profile_id))
    assert not decision.allowed
    assert "unknown" in decision.reason.lower()


@pytest.mark.parametrize("profile_id", sorted(EXPECTED))
def test_no_derivatives_blocked_everywhere(registry: Registry, profile_id: str) -> None:
    """ND blocks training on every profile: masks and crops are derivative acts."""
    source = make_source(tier=Tier.NONCOMMERCIAL, flags=LicenceFlags(no_derivatives=True))
    assert not evaluate(source, registry.profile(profile_id)).allowed


# ── TDM guardrails ────────────────────────────────────────────────────────


def test_tdm_requires_profile_declaring_legal_opinion(registry: Registry) -> None:
    source = make_source(tier=Tier.TDM_ONLY, legal_basis=LegalBasis.TDM)
    decision = evaluate(source, registry.profile("research"), legal_opinion_ref="X")
    assert not decision.allowed


def test_tdm_requires_opinion_ref_at_build_time(registry: Registry) -> None:
    """The profile declaring it is not enough — the reference must actually be supplied."""
    source = make_source(tier=Tier.TDM_ONLY, legal_basis=LegalBasis.TDM)
    decision = evaluate(source, registry.profile("pretrain-eu"), legal_opinion_ref=None)
    assert not decision.allowed
    assert "legal_opinion_ref" in decision.reason


def test_tdm_allowed_with_opinion(registry: Registry) -> None:
    source = make_source(tier=Tier.TDM_ONLY, legal_basis=LegalBasis.TDM)
    decision = evaluate(
        source, registry.profile("pretrain-eu"), legal_opinion_ref="OPINION-2026-001"
    )
    assert decision.allowed


# ── disputed licences ─────────────────────────────────────────────────────


def test_disputed_blocks_shipping_profiles(registry: Registry) -> None:
    source = make_source(tier=Tier.COPYLEFT, disputed=True)
    assert not evaluate(source, registry.profile("ship-open")).allowed


def test_disputed_permitted_for_research(registry: Registry) -> None:
    """Research does not ship, so an unresolved dispute is tolerable there."""
    source = make_source(tier=Tier.COPYLEFT, disputed=True)
    assert evaluate(source, registry.profile("research")).allowed


# ── enforce() reports every violation at once ─────────────────────────────


def test_enforce_raises_and_lists_all_violations(registry: Registry) -> None:
    sources = [
        make_source(source_id="ok", tier=Tier.PERMISSIVE),
        make_source(source_id="bad-nc", tier=Tier.NONCOMMERCIAL),
        make_source(source_id="bad-prohibited", tier=Tier.PROHIBITED, notes="why"),
    ]
    with pytest.raises(LicenceViolation) as exc:
        enforce(sources, registry.profile("ship-commercial"))
    message = str(exc.value)
    assert "bad-nc" in message
    assert "bad-prohibited" in message
    assert "ok" not in message.split("\n")[0]


def test_enforce_passes_clean_set(registry: Registry) -> None:
    sources = [make_source(source_id="ok", tier=Tier.PERMISSIVE)]
    assert enforce(sources, registry.profile("ship-commercial")) == sources
