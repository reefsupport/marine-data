"""The FathomNet concept crosswalk — measured coverage, the common-name table, FGVC23 ids.

``fathomnet-concepts`` serves the full FathomNet box vocabulary (1,950 concepts, 150,458 boxes),
FGVC23 (ids -> names through the vendored category key), FGVC25 and the VME subset. The scientific
names resolve through the pinned WoRMS snapshot; the lowercase common names ("sea fan", "bony
fish", ...) are decided by hand in ``vocab/fathomnet-common-names.tsv``, and these tests pin that
table to the crosswalk it generated, so a regenerated crosswalk cannot drift from it silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.schema import Fidelity
from marinedata.task_layers import boxes_table as bt
from marinedata.taxonomy import read_vocab

CROSSWALK_ID = "fathomnet-concepts"
TARGET_SCHEMA = "rs-taxa-v1"

#: Box-weighted mapped coverage measured 2026-10-05 over the 150,454 boxes of ``fathomnet.tsv``:
#: 96.37%. The brief's floor is 95%: a new unmappable label is allowed, a lost common name is not.
COVERAGE_FLOOR = 0.95

_FIDELITY = {
    "exact": Fidelity.EXACT,
    "broader": Fidelity.COARSENED,
    "related": Fidelity.APPROXIMATE,
    "unmapped": Fidelity.UNMAPPABLE,
}


@pytest.fixture(scope="module")
def vocab_dir(registry) -> Path:
    return Path(registry.root) / "taxonomy" / "vocab"


@pytest.fixture(scope="module")
def crosswalk(registry):
    return registry.crosswalk(CROSSWALK_ID)


@pytest.fixture(scope="module")
def by_label(crosswalk):
    return {e.source_label: e for e in crosswalk.edges}


@pytest.fixture(scope="module")
def common(vocab_dir) -> list[dict[str, str]]:
    lines = [x for x in (vocab_dir / "fathomnet-common-names.tsv").read_text().splitlines()
             if x and not x.startswith("#")]  # fmt: skip
    cols = lines[0].split("\t")
    return [dict(zip(cols, x.split("\t"), strict=False)) for x in lines[1:]]


@pytest.fixture(scope="module")
def counts(vocab_dir) -> dict[str, int | None]:
    return read_vocab(vocab_dir / "fathomnet.tsv")[1]


@pytest.fixture(scope="module")
def nodes(registry):
    return {n.id: n for n in registry.label_schema(TARGET_SCHEMA).nodes}


def test_every_measured_label_has_exactly_one_edge(crosswalk, counts):
    edges = [e.source_label for e in crosswalk.edges]
    assert len(edges) == len(set(edges))
    assert not [label for label in counts if label not in set(edges)]


def test_row_weighted_coverage_holds_the_floor(by_label, counts):
    total = sum(counts.values())
    mapped = sum(c for label, c in counts.items() if by_label[label].targets)
    assert total == 150_454
    assert mapped / total >= COVERAGE_FLOOR, f"box-weighted coverage {mapped / total:.4%}"


def test_every_target_exists_on_the_taxon_axis(crosswalk, nodes):
    assert crosswalk.target_schema == TARGET_SCHEMA
    bad = [
        (e.source_label, axis, node_id)
        for e in crosswalk.edges
        for axis, node_id in e.targets.items()
        if node_id not in nodes or nodes[node_id].axis != axis
    ]
    assert not bad, sorted(bad)


def test_unmappable_edges_carry_a_reason(crosswalk):
    silent = [
        e.source_label for e in crosswalk.edges if e.fidelity is Fidelity.UNMAPPABLE and not e.note
    ]
    assert not silent, silent


def test_common_name_table_is_in_box_count_order_and_covers_95_percent(common, counts):
    boxes = [int(r["box_count"]) for r in common]
    assert boxes[:52] == sorted(
        boxes[:52], reverse=True
    )  # the U8a2 block: every label >= 100 boxes
    assert min(boxes[:52]) >= 100
    assert all(counts[r["concept"]] == int(r["box_count"]) for r in common)
    assert sorted(boxes[52:], reverse=True)[:1] and max(boxes[52:]) < 100  # the U8b block


def test_every_common_name_row_is_the_crosswalk_edge(common, by_label, nodes):
    for r in common:
        edge = by_label[r["concept"]]
        assert edge.fidelity is _FIDELITY[r["match_type"]], r["concept"]
        if r.get("target"):  # non-taxon axis (U8b): an NT_* node, no WoRMS id
            assert edge.targets == {"taxon": r["target"]}, r["concept"]
            assert nodes[r["target"]].non_taxon and not r["aphia_id"], r["concept"]
            continue
        if r["match_type"] == "unmapped":
            assert not edge.targets, r["concept"]
            continue
        node = nodes[edge.targets["taxon"]]
        assert str(node.worms_aphia_id) == r["aphia_id"], r["concept"]
        assert node.worms_scientificname == r["scientific_name"], r["concept"]
        assert node.worms_status == "accepted", r["concept"]


@pytest.mark.parametrize(
    ("label", "name", "fidelity"),
    [
        ("brittle star", "Ophiuroidea", Fidelity.EXACT),
        ("sea star", "Asteroidea", Fidelity.EXACT),
        ("stony coral", "Scleractinia", Fidelity.EXACT),
        ("sea fan", "Octocorallia", Fidelity.COARSENED),
        ("bony fish", "Actinopterygii", Fidelity.APPROXIMATE),
        ("marine organism", "Biota", Fidelity.COARSENED),
        # U8b long tail (every unmapped label with >= 10 boxes, WoRMS-confirmed ids)
        ("echinoderm", "Echinodermata", Fidelity.EXACT),
        ("ctenophore", "Ctenophora", Fidelity.EXACT),
        ("chaetognath", "Chaetognatha", Fidelity.EXACT),
        ("hermit crab", "Paguroidea", Fidelity.EXACT),
        ("sea spider", "Pycnogonida", Fidelity.EXACT),
        ("isopod", "Isopoda", Fidelity.EXACT),
        ("sea slug", "Heterobranchia", Fidelity.COARSENED),
        ("Diatom", "Bacillariophyta", Fidelity.EXACT),
    ],
)
def test_common_names_land_on_the_taxon_they_denote(by_label, nodes, label, name, fidelity):
    edge = by_label[label]
    assert nodes[edge.targets["taxon"]].worms_scientificname == name
    assert edge.fidelity is fidelity


def test_bony_fish_is_related_not_broader_and_osteichthyes_is_not_a_node(by_label, nodes):
    edge = by_label["bony fish"]
    assert edge.fidelity is Fidelity.APPROXIMATE  # match_type "related"
    assert "Osteichthyes not a node" in edge.note
    assert not [n for n in nodes.values() if n.worms_scientificname == "Osteichthyes"]  # MAJOR


@pytest.mark.parametrize(
    ("label", "node"),
    [
        ("equipment", "NT_EQUIPMENT"),
        ("Suction Sampler", "NT_EQUIPMENT"),
        ("manipulator", "NT_EQUIPMENT"),
        ("DeepPIV 1.0", "NT_EQUIPMENT"),
        ("trash", "NT_DEBRIS"),
        ("plastic bag", "NT_DEBRIS"),
        ("inconclusive", "NT_UNKNOWN"),
        ("Unknown full image", "NT_UNKNOWN"),
    ],
)
def test_non_taxon_labels_map_exactly_to_the_nt_nodes(by_label, nodes, label, node):
    edge = by_label[label]
    assert edge.targets == {"taxon": node} and edge.fidelity is Fidelity.EXACT
    assert nodes[node].non_taxon


def test_generic_labels_map_to_the_root_only_because_it_exists(by_label, nodes):
    assert nodes["A1"].parent is None and nodes["A1"].worms_scientificname == "Biota"
    assert by_label["marine organism"].targets == {"taxon": "A1"}


def test_non_taxa_and_ambiguous_names_stay_unmapped(common, by_label):
    for label in (
        "__unlabelled",
        "Detritus",
        "Nano plankton",
        "LRJ complex",
        "other",
    ):
        assert by_label[label].fidelity is Fidelity.UNMAPPABLE, label
    assert by_label["LRJ complex"].note.startswith("ambiguous")
    for r in common:
        if r["match_type"] == "unmapped":
            assert r["note"].startswith(("non-taxon", "ambiguous")), r["concept"]


def test_fgvc23_ids_resolve_to_concept_names_with_an_edge(by_label):
    names = bt.load_id_names(bt.FGVC23_KEY)
    assert len(names) == 290
    assert names["1"] == "Actiniaria"
    missing = [n for n in names.values() if n not in by_label]
    assert not missing, missing[:10]
    assert bt.BOX_SOURCES["fathomnet-fgvc23"].id_key == bt.FGVC23_KEY


@pytest.mark.parametrize(
    "source_id",
    ["fathomnet", "fathomnet-fgvc23", "fathomnet-fgvc25", "roboflow-aquarium", "brackishmot"],
)
def test_box_sources_point_at_a_crosswalk(source_id, registry):
    spec = bt.BOX_SOURCES[source_id]
    assert spec.crosswalk_id is not None
    registry.crosswalk(spec.crosswalk_id)


def test_aquarium_crosswalk_maps_its_seven_classes(registry, nodes):
    xw = registry.crosswalk("roboflow-aquarium")
    edges = {e.source_label: e for e in xw.edges}
    assert set(edges) == {"fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"}
    assert all(e.targets for e in edges.values())
    assert edges["fish"].fidelity is Fidelity.COARSENED
    assert nodes[edges["starfish"].targets["taxon"]].worms_scientificname == "Asteroidea"


def test_brackishmot_class_ids_follow_the_paper_class_list(registry, nodes):
    """gt.txt class 1-6 = fish, crab, shrimp, starfish, small fish, jellyfish (arXiv:2302.10645)."""
    xw = registry.crosswalk("brackishmot-class-id")
    edges = {e.source_label: e for e in xw.edges}
    assert set(edges) == {"1", "2", "3", "4", "5", "6"} and all(e.targets for e in edges.values())
    named = {k: nodes[e.targets["taxon"]].worms_scientificname for k, e in edges.items()}
    assert named == {
        "1": "Vertebrata", "2": "Brachyura", "3": "Decapoda", "4": "Asteroidea", "5": "Vertebrata",
        "6": "Medusozoa",
    }  # fmt: skip
    assert edges["2"].fidelity is Fidelity.EXACT and edges["1"].fidelity is Fidelity.COARSENED
    assert edges["6"].fidelity is Fidelity.COARSENED  # jellyfish: Medusozoa, as in roboflow-aquarium


#: Labels with >= 10 boxes that stay unmapped after U8b, each on purpose: natural detritus and
#: structures (no node), ambiguous (homonym, polyphyletic or acronym), absent from WoRMS, or
#: gear that could equally be litter. A new entry here is a decision, not drift.
REMAINING_UNMAPPED_10_PLUS = frozenset({
    "Nano plankton", "Detritus", "LRJ complex", "Pyrosoma detritus", "coral skeleton",
    "detrital aggregate", "other", "none", "fecal cast", "Bathochordaeus sinker", "eggcase",
    "Appendicularia", "worm+", "coiled fecal cast", "Krill molt", "Porifera detritus",
    "Boreomysis californica", "Polychaeta tube", "jelly", "shell fragment", "sea butterfly",
    "Phyllospadix-Zostera detritus", "Teuthoidea", "cable", "Ctenophora", "Sebastomus complex",
    "detritus", "can", "object", "sinker", "polypropylene line", "talus", "amphipod tube mat",
    "Grimalditeuthis bonplandi", "pillow lava",
})  # fmt: skip


def test_long_tail_leaves_only_the_documented_labels_unmapped(by_label, counts):
    left = {
        k
        for k, c in counts.items()
        if (c or 0) >= 10 and by_label[k].fidelity is Fidelity.UNMAPPABLE
    }
    assert left == REMAINING_UNMAPPED_10_PLUS
    mapped = sum(
        c or 0 for k, c in counts.items() if by_label[k].fidelity is not Fidelity.UNMAPPABLE
    )
    assert mapped / sum(c or 0 for c in counts.values()) >= 0.98  # 98.35% measured (was 96.37%)
