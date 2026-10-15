"""Taxonomy 3.0.0 (R3, D9): one edge retargeted, five bridges declared, nothing else moved.

v1.0 is frozen as the published artifact (``test_release_build_frozen.py``,
``test_v1_configs_schema.py``). What a rebuild of a v1.0 config reads from the registry is its
crosswalk edges, so "v1.0 untouched" is pinned here as: every 2.9.0 node and edge is carried into
3.0.0 unchanged, except ``coralscapes-39:unknown hard substrate`` (D9, new tables only).
"""

from pathlib import Path

import pytest
import yaml

from marinedata import taxonomy as tx

D9_EDGE = "coralscapes-39:unknown hard substrate"
BRIDGES = {
    "rs-taxa-to-rs-benthic:Scleractinia": ("A1363", "HC"),
    "rs-taxa-to-rs-benthic:Octocorallia": ("A1341", "SC"),
    "rs-taxa-to-rs-benthic:Actinopterygii": ("A10194", "FISH"),
    "rs-taxa-to-rs-fauna:Actinopterygii": ("A10194", "FISH_ACTINOPT"),
    "rs-taxa-to-rs-fauna:Elasmobranchii": ("A10193", "FISH_ELASMO"),
}


@pytest.fixture(scope="module")
def manifests(registry):
    return tx.load_manifest("2.9.0", registry.root), tx.load_manifest("3.0.0", registry.root)


def test_the_d9_edge_is_the_only_2_9_0_edge_that_changed(manifests):
    before, after = (m["edges"] for m in manifests)
    assert {k for k, v in before.items() if after.get(k) != v} == {D9_EDGE}
    assert before[D9_EDGE] == {"fidelity": "approximate", "targets": {"taxon": "RK"}}
    assert after[D9_EDGE] == {"fidelity": "unmappable", "targets": {}}


def test_no_node_moved_and_only_the_five_bridges_were_added(manifests):
    before, after = manifests
    assert before["nodes"] == after["nodes"]
    assert set(after["edges"]) - set(before["edges"]) == set(BRIDGES)
    assert not set(before["edges"]) - set(after["edges"])


def test_the_working_tree_is_the_3_0_0_release(registry, manifests):
    assert tx.load_meta(registry.root)["version"] == "3.0.0"
    assert tx.manifest(registry, registry.root)["edges"] == manifests[1]["edges"]


@pytest.mark.parametrize("key", sorted(BRIDGES))
def test_a_bridge_is_an_exact_subtree_root_on_one_aphia_id(registry, key):
    cid, label = key.split(":")
    walk = registry.crosswalk(cid)
    source, target = (
        registry.label_schema(walk.source_schema),
        registry.label_schema(walk.target_schema),
    )
    edge = walk.edge(label)
    root_id, target_id = BRIDGES[key]
    root, node = source.node(root_id), target.node(edge.targets[next(iter(edge.targets))])
    assert (edge.source_label_id, node.id) == (root_id, target_id)
    assert (walk.source_schema, edge.fidelity.value) == ("rs-taxa-v1", "exact")
    assert root.name == label and root.worms_aphia_id == node.worms_aphia_id is not None
    assert any(n.parent == root_id for n in source.nodes), "a bridge covers a subtree, not a leaf"


def test_usis_uiis_fish_keeps_its_smallest_accepted_clade_edge(registry):
    edge = registry.crosswalk("usis-uiis").edge(
        "fish"
    )  # WoRMS Pisces is unaccepted: not retargeted
    assert [n for n in edge.targets.values()] == ["A146419"] and edge.fidelity.value == "coarsened"


def test_every_coralscapes_edge_states_its_fidelity_and_d9_is_unmapped(registry):
    path = Path(registry.root) / "crosswalks" / "coralscapes-39.yaml"
    edges = yaml.safe_load(path.read_text())["crosswalks"][0]["edges"]
    assert len(edges) == 39 and all("fidelity" in e for e in edges)
    walk = registry.crosswalk("coralscapes-39")
    assert {f.value: n for f, n in walk.coverage.items() if n} == {
        "exact": 26, "coarsened": 10, "approximate": 2, "unmappable": 1,
    }  # fmt: skip
    taxon, match, aphia, l2 = walk.resolve("unknown hard substrate")
    assert (taxon, match, aphia, l2) == (None, "unmapped", None, None)
