"""Taxonomy 3.0.0 (R3, D9): one edge retargeted, five bridges declared, nothing else moved.

v1.0 is frozen as the published artifact (``test_release_build_frozen.py``,
``test_v1_configs_schema.py``). What a rebuild of a v1.0 config reads from the registry is its
crosswalk edges, so "v1.0 untouched" is pinned here as: every 2.9.0 node and edge is carried into
3.0.0 unchanged, except ``coralscapes-39:unknown hard substrate`` (D9, new tables only).

The same rule holds for rows: ``labels.remap_row`` checks a row's ``class_map`` against the taxonomy
release the row was built under, so the published v1.0 rows (built under 2.9.0) stay valid and a row
stamped 3.0.0 must follow the new rule. The published row is the real v1.0 coralscapes class_map
(``tests/data/label_vocab.json``, dumped from the Hub) over a mask with every class id.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from marinedata import label_schemes as ls
from marinedata import labels
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


def test_the_working_tree_is_3_1_0_and_only_adds_to_the_3_0_0_release(registry, manifests):
    """3.1.0 (v1.2.0, D3) is additive: every 3.0.0 node and edge is still there, unchanged."""
    assert tx.load_meta(registry.root)["version"] == "3.1.0"
    now = tx.manifest(registry, registry.root)
    for key in ("nodes", "edges"):
        old, new = manifests[1][key], now[key]
        assert set(old) <= set(new), key  # nothing removed or renamed
        if isinstance(old, dict):
            assert all(new[k] == v for k, v in old.items()), key  # nothing changed


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


# ── D9 on rows: a frozen row is checked against the release it was built under ───────────────

VOCAB = json.loads((Path(__file__).parent / "data" / "label_vocab.json").read_text())
PUBLISHED = VOCAB["reefsupport/marine-data/coral-masks"]["coralscapes"]["labels"]
V1_0_RELEASE = "2.9.0"  # v1.0 was built after 2.9.0 (2026-10-05) and before 3.0.0
D9_LABEL = "unknown hard substrate"
AS_PUBLISHED = r"coarse 'ABIOTIC' for 'unknown hard substrate' disagrees with the registry"
# a row stamped with a kept release (3.0.0 since 3.1.0 is live) is checked against that release
STAMPED_3_0_0 = (
    r"coarse 'ABIOTIC' for 'unknown hard substrate' disagrees with "
    r"(the registry|taxonomy 3\.0\.0 \(None\))"
)


def _published_row(**extra):
    """The v1.0 coralscapes row: the Hub's class_map, a mask with each of its ids, no stamp."""
    class_map = [{k: v for k, v in e.items() if k != "count"} for e in PUBLISHED]
    mask = np.array([[e["id"] for e in class_map]], dtype=np.uint8)
    return {"source": "coralscapes", "mask": mask, "class_map": class_map, **extra}


def _rebuilt_under_3_0_0(row):
    """The same row with only the D9 entry as a 3.0.0 build writes it: no taxon, no coarse."""
    return {
        **row,
        "class_map": [
            {**e, "taxon_node": None, "coarse": None} if e["label_native"] == D9_LABEL else e
            for e in row["class_map"]
        ],
    }


def test_the_published_row_carries_the_2_9_0_mapping_and_no_stamp():
    d9 = next(e for e in PUBLISHED if e["label_native"] == D9_LABEL)
    assert (d9["taxon_node"], d9["coarse"]) == ("RK", "ABIOTIC")
    assert "taxonomy_version" not in _published_row()


def test_the_published_v1_0_row_passes_unchanged_under_its_release():
    row = _published_row()
    by_argument = labels.remap_row(row, "benthic-coarse", taxonomy_version=V1_0_RELEASE)
    by_stamp = labels.remap_row({**row, "taxonomy_version": V1_0_RELEASE}, "benthic-coarse")
    assert np.array_equal(by_argument["label"], by_stamp["label"])
    # the release only changes the check: the remap reads the current tables, and the scheme sends
    # D9 to 255 either way, so the output equals the row rebuilt under 3.0.0
    rebuilt = labels.remap_row(_rebuilt_under_3_0_0(row), "benthic-coarse")
    assert np.array_equal(by_stamp["label"], rebuilt["label"])
    d9_id = next(e["id"] for e in PUBLISHED if e["label_native"] == D9_LABEL)
    assert by_stamp["label"][row["mask"] == d9_id].tolist() == [labels.IGNORE_INDEX]
    assert row["class_map"] == _published_row()["class_map"] and "label" not in row


def test_a_row_stamped_3_0_0_with_the_old_mapping_raises():
    stamped = _published_row(taxonomy_version="3.0.0")
    with pytest.raises(ValueError, match=STAMPED_3_0_0):
        labels.remap_row(stamped, "benthic-coarse")
    with pytest.raises(ValueError, match=STAMPED_3_0_0):  # the row's stamp wins over the argument
        labels.remap_row(stamped, "benthic-coarse", taxonomy_version=V1_0_RELEASE)
    with pytest.raises(ValueError, match=STAMPED_3_0_0):  # the argument 3.0.0 is no escape either
        labels.remap_row(_published_row(), "benthic-coarse", taxonomy_version="3.0.0")


def test_a_row_stamped_3_0_0_passes_with_d9_unmapped():
    row = _rebuilt_under_3_0_0(_published_row(taxonomy_version="3.0.0"))
    assert labels.remap_row(row, "benthic-coarse")["label"].shape == row["mask"].shape


def test_an_unstamped_row_is_checked_against_the_current_release_as_before():
    with pytest.raises(ValueError, match=AS_PUBLISHED):
        labels.remap_row(_published_row(), "benthic-coarse")
    for blank in (None, ""):  # a nullable stamp column: null is "no stamp"
        with pytest.raises(ValueError, match=AS_PUBLISHED):
            labels.remap_row(_published_row(taxonomy_version=blank), "benthic-coarse")
    assert labels.remap_row(_rebuilt_under_3_0_0(_published_row()), "benthic-coarse")
    assert labels.remap_row(_published_row(), "coral-binary")  # compares no coarse: unaffected


@pytest.mark.parametrize(
    ("stamp", "message"),
    [
        ("9.9.9", "newer than the registry"),
        ("0.1.0", "no release file"),
        ("2.9", "MAJOR.MINOR.PATCH"),
        ("../releases/2.9.0", "MAJOR.MINOR.PATCH"),
    ],
)
def test_a_stamp_that_names_no_usable_release_raises(stamp, message):
    with pytest.raises(ValueError, match=message):
        labels.remap_row(_published_row(taxonomy_version=stamp), "benthic-coarse")


def test_release_for_resolves_the_live_registry_and_the_kept_releases():
    assert ls.current_taxonomy_version() == "3.1.0"
    assert ls.release_for(None) is None and ls.release_for("3.1.0") is None
    kept = ls.release_for("3.0.0")  # a kept release now that 3.1.0 is live
    assert kept is not None and kept.version == "3.0.0"
    release = ls.release_for(V1_0_RELEASE)
    assert release is not None and release.version == V1_0_RELEASE
    assert release.taxon("coralscapes-39", D9_LABEL) == "RK"
    assert release.ancestors("RK") == ("ABIOTIC",)


def _published_coarse():
    scheme = ls.scheme_spec("benthic-coarse")
    for srcs in VOCAB.values():
        for source, entry in srcs.items():
            if source in scheme.sources and entry["kind"] == "class_map":
                for item in entry["labels"]:
                    yield source, item["label_native"], item["coarse"]


def test_the_2_9_0_release_file_rebuilds_every_published_v1_0_coarse_value():
    """The release JSON holds the edges and the node tree the check reads: no tolerance table."""
    release = ls.release_for(V1_0_RELEASE)
    rebuilt = [
        (s, label, coarse)
        for s, label, coarse in _published_coarse()
        if ls.registry_class(s, label, "benthic-coarse", release=release) != coarse
    ]
    assert list(_published_coarse()) and rebuilt == []


def test_d9_is_the_only_published_coarse_value_the_current_registry_disagrees_with():
    differing = {
        (s, label)
        for s, label, coarse in _published_coarse()
        if ls.registry_class(s, label, "benthic-coarse") != coarse
    }
    assert differing == {("coralscapes", D9_LABEL)}
