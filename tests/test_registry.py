"""Tests over the real registry contents.

These are invariants about the data, not the code. They catch the failure mode that
matters most in a registry: an entry that is syntactically valid but semantically
wrong — a permissive tier on a source nobody verified, or a prohibition with no
explanation.
"""

from __future__ import annotations

import pytest

from marinedata import Registry
from marinedata.enums import AccessMethod, LegalBasis, Redistribution, Tier
from marinedata.registry import _default_root, _read_yaml


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


def test_every_source_declares_redistribution_explicitly(registry: Registry) -> None:
    """The default exists so pydantic never crashes — the raw YAML must not rely on it.

    A silently-defaulted ``unknown`` is indistinguishable from a reviewed ``unknown``,
    which defeats the point of recording a position at all.
    """
    base = _default_root() / "sources"
    for path in sorted(base.rglob("*.yaml")):
        for entry in _read_yaml(path).get("sources", []):
            assert "redistribution" in entry, (
                f"{path}: source '{entry.get('id')}' has no explicit redistribution key"
            )


def test_prohibited_redistribution_matches_prohibited_tier(registry: Registry) -> None:
    """`redistribution: prohibited` and tier `TX_PROHIBITED` must imply each other."""
    for src in registry:
        is_prohibited_tier = src.licence.tier is Tier.PROHIBITED
        is_prohibited_redist = src.redistribution is Redistribution.PROHIBITED
        assert is_prohibited_redist == is_prohibited_tier, (
            f"{src.id}: tier={src.licence.tier.value} "
            f"redistribution={src.redistribution.value} — must match exactly"
        )


def test_tdm_or_unknown_basis_never_redistributes_ok(registry: Registry) -> None:
    """A source with no established legal basis must never claim a redistribution right."""
    for src in registry:
        if src.licence.tier is Tier.TDM_ONLY or src.legal_basis is LegalBasis.UNKNOWN:
            assert src.redistribution is not Redistribution.OK, (
                f"{src.id}: tier={src.licence.tier.value} legal_basis={src.legal_basis.value} "
                f"but redistribution=ok"
            )


def test_every_source_access_method_is_a_valid_enum_member(registry: Registry) -> None:
    """Pinned as an explicit contract, independent of how the field happens to be typed —
    a fetcher dispatch table keyed on a value outside the vocabulary fails silently
    rather than at load."""
    for src in registry:
        assert isinstance(src.access.method, AccessMethod), (
            f"{src.id}: access.method={src.access.method!r} is not a valid AccessMethod member"
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


def test_no_source_claims_own_tier_over_third_party_pixels(registry: Registry) -> None:
    """Regression guard for a real defect found 2026-08-17.

    A single `reef-support-benthic` entry was tiered PROPRIETARY-OWN across 3,957 images
    of which only 1,250 were our pixels; the rest were our masks on XL Catlin Seaview
    photographs. Every gate passed and minted a clean T0_OWN certificate over imagery we
    do not own — the exact false-provenance failure this registry exists to prevent.

    Rights now attach to (source, partition). This asserts the split stayed split.
    """
    own = registry.source("reef-support-benthic-own")
    assert own.licence.tier is Tier.OWN
    assert own.items == 1250, "T0_OWN must cover only pixels we actually own"

    borrowed = registry.source("reef-support-seaview-labels")
    assert borrowed.licence.tier is not Tier.OWN
    assert borrowed.legal_basis is LegalBasis.UNKNOWN

    ids = {s.id for s in registry}
    assert "reef-support-benthic" not in ids, "the conflated entry must not return"


def test_own_tier_sources_are_genuinely_ours(registry: Registry) -> None:
    """T0_OWN asserts we hold the rights outright. Provenance must agree."""
    for src in registry:
        if src.licence.tier is Tier.OWN:
            assert src.provenance.value == "own", (
                f"{src.id}: tier T0_OWN but provenance={src.provenance.value}. "
                f"T0_OWN over third-party pixels mints a false clean certificate."
            )


def test_registry_loads_with_the_network_hard_down(monkeypatch) -> None:
    """Loading, harmonising and gating must never touch the network.

    The taxonomy is anchored on WoRMS/OBIS identifiers, but resolution is CODEGEN — a
    maintainer runs it and commits the result. If a network call ever crept into the
    load path, CI would become flaky, offline work would break, and two builds on
    different days could silently disagree about what a label means.

    The committed YAML is the pin. This test is what keeps that true.
    """
    import socket
    import urllib.request

    def _boom(*args, **kwargs):
        raise AssertionError("network access during registry load")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)

    reg = Registry.load()
    assert len(reg) > 0
    node = reg.label_schema("rs-benthic-v1").node("HC_ORBICELLA")
    assert node.worms_aphia_id == 758259, "frozen AphiaID must resolve offline"
    harmonizer = reg.harmonizer_for("coralscapes")
    assert harmonizer.map_label("massive/meandering bleached").labels
