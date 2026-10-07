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
    # D-B (2026-09-25, delegated): relicensed CC-BY-4.0 for the open dataset — tier is
    # now PERMISSIVE, but `legal_basis` staying OWN is the part this regression guards:
    # we relicense only pixels we actually hold outright, never a source we merely mask.
    assert own.licence.tier is Tier.PERMISSIVE
    assert own.legal_basis is LegalBasis.OWN
    assert own.n_images == 1250, "the relicensed pixels must be only the ones we actually own"

    borrowed = registry.source("reef-support-seaview-labels")
    # Yohan 2026-10-06: the Seaview pixels are CC-BY-3.0 (UQ eSpace UQ_734799), our masks
    # CC-BY-4.0. The pixels' real licence is what the entry carries; it must never be T0_OWN.
    assert borrowed.licence.tier is not Tier.OWN
    assert borrowed.licence.id == "CC-BY-3.0"
    assert borrowed.legal_basis is LegalBasis.LICENCE

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


def test_noaa_esd_coral_bleaching_is_retired_in_favour_of_pifsc(registry: Registry) -> None:
    """S21 (R1): the bucket mirror is a confirmed duplicate of `noaa-pifsc-bleaching`.

    It stays registered (rather than deleted) so the bucket mirror and tar member are
    accounted for, but it must read as retired and never as a source to actively fetch.
    """
    retired = registry.source("noaa-esd-coral-bleaching")
    assert "retired" in retired.tags
    assert "noaa-pifsc-bleaching" in (retired.notes or "")
    assert retired.citation
    live = registry.source("noaa-pifsc-bleaching")
    assert live.legal_basis is LegalBasis.LICENCE

    # The existing metadata-only skip convention (cli.py, verify.py) already excludes
    # it from any active fetch/build enumeration.
    fetchable = [s.id for s in registry if s.loader and s.loader.layout != "metadata-only"]
    assert "noaa-esd-coral-bleaching" not in fetchable


def test_seaview_survey_imagery_licence_resolved(registry: Registry) -> None:
    """S21 (R1 Q2, high confidence): CC BY 3.0 confirmed at researchdata.edu.au."""
    src = registry.source("seaview-survey-imagery")
    assert src.licence.tier is Tier.PERMISSIVE
    assert src.legal_basis is LegalBasis.LICENCE
    assert src.redistribution is Redistribution.OK
    assert src.citation and "10.14264/UQL.2019.930" in src.citation


def test_s21_still_gated_sources_stay_unknown_basis(registry: Registry) -> None:
    """S21 (R1 Q2): entries with no usable licence confirmation remain gated."""
    for source_id in (
        "coralseg-ucsd-mosaics",
        "ibf",
        "coral-health-classification",
        "kaggle-healthy-bleached-corals",
        "coral-bleaching-detection-v2i-multiclass",
    ):
        src = registry.source(source_id)
        assert src.legal_basis is LegalBasis.UNKNOWN, f"{source_id}: expected gated"


def test_roboflow_attribution_added_where_confirmed(registry: Registry) -> None:
    """S21 (R1 Q4 / manager decision 3): confirmed Roboflow attributions carry a
    citation; unconfirmed ones are flagged needs-attribution instead of guessed.

    v13i/v2i (WS-D S37): earlier passes read only the licence line of the shared
    README sidecar and missed the `README.dataset.txt` Universe URL sitting in the
    same file — both are now attributed like the other three, no longer
    needs-attribution."""
    attributed = {
        "roboflow-coral-bleaching-final-v6i": "lockie/coral-bleaching-final",
        "roboflow-coral-bleaching-general-v1-yolov8s": "lockie/coral-bleaching_general",
        "roboflow-coral-reef-classification-v3i": "twork/coral-reef-classification",
        "roboflow-coral-classification-copy-changed-v13i": (
            "maxyn-icaonapo/coral-classification-copy-changed"
        ),
        "roboflow-coral-reef-bleach-detection-v2i": "coralreef/coral-reef-bleach-detection",
    }
    for source_id, url_fragment in attributed.items():
        src = registry.source(source_id)
        assert src.citation and url_fragment in src.citation, source_id
        assert "needs-attribution" not in src.tags, source_id


def test_reef_support_seaview_labels_provenance_is_own(registry: Registry) -> None:
    """S21 (R2 Q8): annotation labour is ours; the mislabeled `partner` provenance is
    corrected to `own`. RESOLVED 2026-10-06 (Yohan): images are CC-BY-3.0 (González-Rivero
    et al., UQ eSpace UQ_734799, attribution), masks CC-BY-4.0 Reef Support; the entry is
    `open` with `legal_basis: licence` (the enum has no `licence+own`; the masks are
    documented in the citation)."""
    src = registry.source("reef-support-seaview-labels")
    assert src.provenance.value == "own"
    assert src.access_class == "open"
    assert src.legal_basis is LegalBasis.LICENCE
    assert src.redistribution is Redistribution.OK
    assert src.licence.id == "CC-BY-3.0" and src.licence.flags.attribution_required
    assert src.licence.tier is not Tier.OWN
    assert src.citation and "10.14264/UQL.2019.930" in src.citation
    assert "CC-BY-4.0" in src.citation and "Reef Support" in src.citation


def test_reefolution_legal_basis_stays_gated(registry: Registry) -> None:
    """S21 (R2 Q6): partnership documented, no written grant found — still unknown."""
    src = registry.source("reefolution")
    assert src.legal_basis is LegalBasis.UNKNOWN
    assert "no written grant" in (src.notes or "").lower()


def test_rs_labelled_masks_split_legal_basis_documented(registry: Registry) -> None:
    """S21 (R2 Q7): the schema has no per-partition legal_basis, so the whole-entry
    value is kept at the stricter (unknown) end rather than blanket-upgraded to own,
    and the split rationale is recorded in notes."""
    src = registry.source("rs-labelled-masks")
    assert src.retired and "D7" in src.retired  # RB-3: duplicate set, no read path
    assert src.legal_basis is LegalBasis.UNKNOWN
    assert src.licence.tier is not Tier.OWN
    notes = (src.verification.verified_by or "") + (src.notes or "")
    assert "reef-support-benthic-own" in notes and "reef-support-seaview-labels" in notes
