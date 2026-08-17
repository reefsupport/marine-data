"""Taxonomic resolution tests.

Split deliberately:

* **Offline** tests exercise parsing, ambiguity detection and drift comparison against
  fixed payloads. They run on every commit.
* **Integration** tests hit WoRMS and OBIS for real and are skipped unless
  ``MARINEDATA_INTEGRATION=1``. They are the drift check.

The offline/online split matters more here than elsewhere: this module is the only one
allowed to touch the network, and `test_registry_loads_with_the_network_hard_down`
guarantees nothing else does.
"""

from __future__ import annotations

import os

import pytest

from marinedata import Registry
from marinedata.taxa import (
    RANK_ORDER,
    AmbiguousName,
    TaxonRecord,
    _from_obis,
    _from_worms,
    check_node,
    resolve_id,
    resolve_name,
)

requires_network = pytest.mark.skipif(
    os.environ.get("MARINEDATA_INTEGRATION") != "1",
    reason="set MARINEDATA_INTEGRATION=1 to run tests that hit WoRMS/OBIS",
)


# ── offline: parsing ──────────────────────────────────────────────────────


def test_parses_a_worms_record() -> None:
    record = _from_worms(
        {
            "AphiaID": 205902,
            "scientificname": "Millepora",
            "rank": "Genus",
            "status": "accepted",
            "kingdom": "Animalia",
            "phylum": "Cnidaria",
            "class": "Hydrozoa",
            "family": "Milleporidae",
        }
    )
    assert record.aphia_id == 205902
    assert record.is_accepted
    assert record.classification["Class"] == "Hydrozoa"
    assert record.obis_url.endswith("/205902")


def test_parses_an_obis_synonym_and_keeps_both_ids() -> None:
    """Montastraea annularis is now Orbicella annularis. Both ids must survive."""
    record = _from_obis(
        {
            "taxonID": 207479,
            "acceptedNameUsageID": 758260,
            "scientificName": "Montastraea annularis",
            "acceptedNameUsage": "Orbicella annularis",
            "taxonRank": "Species",
            "kingdom": "Animalia",
            "phylum": "Cnidaria",
        }
    )
    assert record.aphia_id == 207479
    assert record.accepted_aphia_id == 758260
    fields = record.yaml_fields()
    assert fields["worms_aphia_id"] == 758260
    assert fields["worms_aphia_id_asserted"] == 207479
    assert fields["worms_scientificname"] == "Orbicella annularis"


def test_rank_position_orders_coarse_to_fine() -> None:
    genus = _from_worms(
        {"AphiaID": 1, "scientificname": "X", "rank": "Genus", "status": "accepted"}
    )
    order = _from_worms(
        {"AphiaID": 2, "scientificname": "Y", "rank": "Order", "status": "accepted"}
    )
    assert order.rank_position < genus.rank_position
    assert RANK_ORDER.index("Family") < RANK_ORDER.index("Genus")


def test_informal_rank_is_position_minus_one() -> None:
    giga = _from_worms(
        {
            "AphiaID": 10194,
            "scientificname": "Actinopterygii",
            "rank": "Gigaclass",
            "status": "accepted",
        }
    )
    assert giga.rank_position == -1


# ── offline: drift detection ──────────────────────────────────────────────


class _Node:
    def __init__(self, **kw):
        self.id = kw.get("id", "N")
        self.worms_aphia_id = kw.get("worms_aphia_id")
        self.worms_scientificname = kw.get("worms_scientificname")
        self.worms_rank = kw.get("worms_rank")
        self.worms_status = kw.get("worms_status")


def test_no_drift_when_committed_matches_authority() -> None:
    node = _Node(
        worms_aphia_id=205902,
        worms_scientificname="Millepora",
        worms_rank="Genus",
        worms_status="accepted",
    )
    record = TaxonRecord(205902, "Millepora", "Genus", "accepted")
    assert check_node(node, record) == []


def test_drift_detected_when_the_name_changed() -> None:
    """The exact failure mode that let a red alga sit in a coral schema."""
    node = _Node(
        worms_aphia_id=196197,
        worms_scientificname="Millepora",
        worms_rank="Genus",
        worms_status="accepted",
    )
    record = TaxonRecord(196197, "Cryptonemiaceae", "Family", "unaccepted")
    drifts = check_node(node, record)
    fields = {d.field for d in drifts}
    assert "worms_scientificname" in fields
    assert "worms_rank" in fields
    assert "worms_status" in fields


def test_missing_record_is_reported_as_harder_than_drift() -> None:
    node = _Node(worms_aphia_id=999999999)
    drifts = check_node(node, None)
    assert drifts[0].authority == "NO SUCH RECORD"


def test_reassignment_surfaces_as_id_drift() -> None:
    node = _Node(
        worms_aphia_id=207479,
        worms_scientificname="Orbicella annularis",
        worms_rank="Species",
        worms_status="accepted",
    )
    record = TaxonRecord(
        207479,
        "Montastraea annularis",
        "Species",
        "unaccepted",
        accepted_aphia_id=758260,
        accepted_name="Orbicella annularis",
    )
    assert any(
        d.field == "worms_aphia_id" and d.authority == 758260 for d in check_node(node, record)
    )


# ── integration: the real authority ───────────────────────────────────────


@requires_network
def test_obis_taxon_id_is_the_worms_aphia_id() -> None:
    """The premise of the whole anchoring scheme."""
    from marinedata.taxa import _get

    obis = _get("https://api.obis.org/v3/taxon/Orbicella%20annularis")
    aphia = int(obis["results"][0]["taxonID"])
    worms = resolve_id(aphia)
    assert worms is not None
    assert worms.aphia_id == aphia
    assert worms.scientific_name == "Orbicella annularis"


@requires_network
def test_homonym_raises_rather_than_guessing() -> None:
    """Turbinaria is a coral genus AND a brown alga genus. Guessing put an alga in a
    coral schema on this tool's first run."""
    with pytest.raises(AmbiguousName) as exc:
        resolve_name("Turbinaria")
    assert len(exc.value.candidates) >= 2

    coral = resolve_name("Turbinaria", expect_phylum="Cnidaria")
    assert coral is not None
    assert coral.aphia_id == 206641
    assert coral.classification["Class"] == "Hexacorallia"


@requires_network
def test_millepora_and_acropora_are_in_different_classes() -> None:
    """The fact the whole functional-grouping design rests on."""
    millepora = resolve_name("Millepora", expect_phylum="Cnidaria")
    acropora = resolve_name("Acropora", expect_phylum="Cnidaria")
    assert millepora.classification["Class"] == "Hydrozoa"
    assert acropora.classification["Class"] == "Hexacorallia"


@requires_network
def test_registry_has_no_taxonomic_drift() -> None:
    """THE drift check. Every anchored node must still agree with WoRMS."""
    registry = Registry.load()
    problems = []
    for schema in registry.schemas:
        for node in schema.nodes:
            if node.worms_aphia_id is None:
                continue
            drifts = check_node(node, resolve_id(node.worms_aphia_id))
            problems.extend(d.line() for d in drifts)
    assert not problems, "taxonomic drift:\n" + "\n".join(problems)
