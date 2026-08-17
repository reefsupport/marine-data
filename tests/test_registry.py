"""Tests over the real registry contents.

These are invariants about the data, not the code. They catch the failure mode that
matters most in a registry: an entry that is syntactically valid but semantically
wrong — a permissive tier on a source nobody verified, or a prohibition with no
explanation.
"""

from __future__ import annotations

import pytest

from marinedata import Registry
from marinedata.enums import LegalBasis, Tier


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def test_registry_loads(registry: Registry) -> None:
    assert len(registry) > 0


def test_every_source_cites_a_primary_source(registry: Registry) -> None:
    """`verified_by` is how we avoid repeating the MarineInst20M mistake."""
    for src in registry:
        assert len(src.verification.verified_by) >= 8, f"{src.id}: weak verification"
        assert src.verification.method, f"{src.id}: no verification method"


def test_prohibited_sources_explain_themselves(registry: Registry) -> None:
    for src in registry:
        if src.licence.tier is Tier.PROHIBITED:
            assert src.notes, f"{src.id}: PROHIBITED without explanation"


def test_disputed_sources_have_notes(registry: Registry) -> None:
    for src in registry:
        if src.verification.disputed:
            assert src.verification.dispute_note, f"{src.id}: disputed without a note"


def test_tdm_sources_declare_tdm_basis(registry: Registry) -> None:
    """A T4 source must say it relies on TDM, or say its basis is unresolved."""
    for src in registry:
        if src.licence.tier is Tier.TDM_ONLY:
            assert src.legal_basis in (LegalBasis.TDM, LegalBasis.UNKNOWN), (
                f"{src.id}: tier T4_TDM_ONLY but legal_basis={src.legal_basis.value}"
            )


def test_no_derivatives_sources_are_flagged_not_just_noted(registry: Registry) -> None:
    """ND must be encoded as a flag, not left in prose where the gate cannot see it."""
    for src in registry:
        text = f"{src.notes or ''} {src.licence.notes or ''}".lower()
        if "nodderivative" in text or "no-derivatives" in text or "noderivatives" in text:
            assert src.licence.flags.no_derivatives, (
                f"{src.id}: mentions no-derivatives in prose but the flag is unset"
            )


def test_auto_provenance_never_supervises(registry: Registry) -> None:
    """Model-generated labels must not be declared as supervision for any axis."""
    for src in registry:
        if src.provenance.value == "auto":
            for ann in src.annotations:
                assert not ann.supervises, (
                    f"{src.id}: pseudo-labels declare supervises={ann.supervises}; "
                    f"machine output must never be treated as ground truth"
                )


def test_partner_data_without_permission_is_not_usable(registry: Registry) -> None:
    """Partner imagery with no written grant must not sit at a usable tier."""
    for src in registry:
        if src.provenance.value == "partner" and src.legal_basis is LegalBasis.UNKNOWN:
            assert src.licence.tier not in (Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT), (
                f"{src.id}: partner data with unknown basis sits at {src.licence.tier.value}"
            )


def test_known_domain_drops_cite_a_source(registry: Registry) -> None:
    for src in registry:
        if src.domain_shift:
            for drop in src.domain_shift.known_drops:
                assert len(drop.source) > 10, (
                    f"{src.id}: known_drop to {drop.to_region.value} has no real citation"
                )


def test_source_ids_are_unique_and_slugged(registry: Registry) -> None:
    ids = [s.id for s in registry]
    assert len(ids) == len(set(ids))
    for sid in ids:
        assert sid == sid.lower()
