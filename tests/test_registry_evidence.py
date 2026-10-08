"""P1.2b: an audited licence class carries its licence id and evidence; the new cross-reference
fields (related, checksums_status, other_versions, n_images_upstream) are coherent."""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.registry import Registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry"
RESTORED = (
    "hacon-aurora-vent-ofos", "marineevt", "mouss", "noaa-pifsc-esa-coral-icra",
    "pangaea-ps101-ofos-arctic", "pangaea-ps118-ofobs-peninsula", "pangaea-ps81-ofos-antarctic",
    "pangaea-ps96-ofos-weddell", "so268-ofos-analysis-ready",
    "noaa-coral-icra", "plasticinwater", "mouss-seg",
)  # fmt: skip
PLACEHOLDER_LICENCES = {"NO-LICENCE-STATED", "unknown", "NOASSERTION"}


@pytest.fixture(scope="module")
def by_id():
    return {s.id: s for s in Registry.load(REGISTRY).sources}


@pytest.mark.parametrize("sid", RESTORED)
def test_restored_class_has_a_licence_id_and_evidence(by_id, sid):
    src = by_id[sid]
    assert src.access_class.value == "open"
    assert src.licence.id not in PLACEHOLDER_LICENCES
    assert src.licence_evidence and ("http" in src.licence_evidence)
    assert src.verification.method in {"dataset-card", "site-terms", "licence-file"}


def test_a_per_file_licence_source_stays_internal_only_without_a_single_licence_id(by_id):
    src = by_id["wikimedia-underwater"]
    assert src.access_class.value == "internal-only" and src.licence_evidence is None


def test_evidence_is_never_attached_to_a_placeholder_licence(by_id):
    for s in by_id.values():
        if s.licence_evidence and not s.licence_per_row:
            assert s.licence.id not in PLACEHOLDER_LICENCES, s.id


def test_checksums_status_missing_excludes_a_checksums_block(by_id):
    for s in by_id.values():
        if s.checksums_status is not None:
            assert s.checksums_status == "missing" and s.checksums is None, s.id


def test_other_versions_never_repeat_the_pin(by_id):
    for s in by_id.values():
        assert s.version not in {o.version for o in s.other_versions}, s.id
        assert len({o.version for o in s.other_versions}) == len(s.other_versions), s.id


def test_n_images_upstream_is_explained(by_id):
    for s in by_id.values():
        if s.n_images_upstream is not None:
            assert s.n_images is not None and s.n_images_note, s.id


def test_related_never_points_at_a_superseded_twin(by_id):
    for s in by_id.values():
        for rel in s.related:
            assert rel in by_id and rel != s.id, (s.id, rel)
            assert by_id[rel].superseded_by != s.id, (s.id, rel)
