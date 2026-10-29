"""coralscapes-95 -> rs-benthic-v1 crosswalk: coverage, axis resolution and legacy consistency.

The class table and its Legacy (39-class) column are transcribed from the dataset card of
josauder/coralscapesV2 (revision 8a4567b2fbec14882e09aebff63a7a1ca5e24dca, card dated
2026-09-15). Ids are the card ids, i.e. the
mask pixel values; id 0 is the unlabelled/ignore value and has no edge.

The last block pins the rest of the ``coralscapes-v2`` registry entry: its stem link to V1,
the per-stem split group and the site count.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest
from pydantic import ValidationError

from marinedata.models import SourceLink
from marinedata.registry import Registry
from marinedata.release.stem_links import LinkDecl, declared_links

# (id, name, legacy 39-class name), exactly as in the card table.
CARD_ROWS: tuple[tuple[int, str, str], ...] = (
    (1, "acanthaster planci", "crown of thorn"),
    (2, "acropora alive", "acropora alive"),
    (3, "acropora bleached", "branching bleached"),
    (4, "acropora dead", "branching dead"),
    (5, "algae covered brain coral", "algae covered substrate"),
    (6, "algae covered branching coral", "algae covered substrate"),
    (7, "algae covered lobophylliidae", "algae covered substrate"),
    (8, "algae covered massive coral", "algae covered substrate"),
    (9, "algae covered pocillopora", "algae covered substrate"),
    (10, "algae covered porites", "algae covered substrate"),
    (11, "algae covered substrate", "algae covered substrate"),
    (12, "algae covered table acropora", "algae covered substrate"),
    (13, "anemone", "anemone"),
    (14, "background", "background"),
    (15, "brain coral alive", "meandering alive"),
    (16, "brain coral bleached", "meandering bleached"),
    (17, "brain coral dead", "meandering dead"),
    (18, "branching coral alive", "branching alive"),
    (19, "branching coral bleached", "branching bleached"),
    (20, "branching coral dead", "branching dead"),
    (21, "cirrhipathes", "other coral alive"),
    (22, "dark", "dark"),
    (23, "dead giant clam", "dead clam"),
    (24, "feather worm", "other animal"),
    (25, "fish", "fish"),
    (26, "fungiidae alive", "other coral alive"),
    (27, "fungiidae bleached", "other coral bleached"),
    (28, "fungiidae dead", "other coral dead"),
    (29, "galaxea alive", "other coral alive"),
    (30, "galaxea bleached", "other coral bleached"),
    (31, "galaxea dead", "massive/meandering dead"),
    (32, "giant clam", "clam"),
    (33, "goniopora alive", "other coral alive"),
    (34, "goniopora dead", "other coral dead"),
    (35, "hard coral alive", "other coral alive"),
    (36, "hard coral bleached", "other coral bleached"),
    (37, "hard coral dead", "other coral dead"),
    (38, "hard substrate", "hard substrate"),
    (39, "human", "human"),
    (40, "human-made structure", "hard substrate"),
    (41, "leather coral alive", "other coral alive"),
    (42, "lobophylliidae alive", "meandering alive"),
    (43, "lobophylliidae bleached", "meandering bleached"),
    (44, "lobophylliidae dead", "meandering dead"),
    (45, "massive alive", "massive/meandering alive"),
    (46, "massive bleached", "massive/meandering bleached"),
    (47, "massive dead", "massive/meandering dead"),
    (48, "meandering coral alive", "meandering alive"),
    (49, "meandering coral bleached", "meandering bleached"),
    (50, "meandering coral dead", "meandering dead"),
    (51, "millepora", "millepora"),
    (52, "millepora bleached", "branching bleached"),
    (53, "millepora dead", "branching dead"),
    (54, "nephtheidae alive", "other coral alive"),
    (55, "other animal", "other animal"),
    (56, "other coral alive", "other coral alive"),
    (57, "other coral bleached", "other coral bleached"),
    (58, "other coral dead", "other coral dead"),
    (59, "pavona alive", "other coral alive"),
    (60, "pavona bleached", "other coral bleached"),
    (61, "pavona dead", "other coral dead"),
    (62, "pocillopora alive", "pocillopora alive"),
    (63, "pocillopora bleached", "branching bleached"),
    (64, "pocillopora dead", "branching dead"),
    (65, "porites alive", "massive/meandering alive"),
    (66, "porites bleached", "massive/meandering bleached"),
    (67, "porites dead", "massive/meandering dead"),
    (68, "rubble", "rubble"),
    (69, "sand", "sand"),
    (70, "sea cucumber", "sea cucumber"),
    (71, "sea urchin", "sea urchin"),
    (72, "seagrass", "seagrass"),
    (73, "seriatopora alive", "branching alive"),
    (74, "seriatopora bleached", "branching bleached"),
    (75, "seriatopora dead", "branching dead"),
    (76, "soft coral (malacalcyonacea) alive", "other coral alive"),
    (77, "soft coral (malacalcyonacea) bleached", "other coral bleached"),
    (78, "sponge", "sponge"),
    (79, "stylophora alive", "stylophora alive"),
    (80, "stylophora bleached", "branching bleached"),
    (81, "stylophora dead", "branching dead"),
    (82, "table acropora alive", "table acropora alive"),
    (83, "table acropora bleached", "branching bleached"),
    (84, "table acropora dead", "table acropora dead"),
    (85, "thin plate/encrusting coral alive", "other coral alive"),
    (86, "thin plate/encrusting coral bleached", "other coral bleached"),
    (87, "thin plate/encrusting coral dead", "other coral dead"),
    (88, "transect line", "transect line"),
    (89, "transect tools", "transect tools"),
    (90, "trash", "trash"),
    (91, "turbinaria", "turbinaria"),
    (92, "turbinaria dead", "other coral dead"),
    (93, "water column", "background"),
    (94, "water surface", "background"),
    (95, "xeniidae alive", "other coral alive"),
)
LEGACY_COUNT = 39

# The card spells the "unknown hard substrate" of coralscapes-39 as "hard substrate".
LEGACY_ALIAS = {"hard substrate": "unknown hard substrate"}

# Classes whose taxon target leaves the subtree of the legacy edge's target (rule a/b
# refinements): black
# coral and the soft-coral families (legacy: hard coral), Millepora bleached/dead (MIL is a
# hydrozoan under
# CNIDARIA, legacy: branching hard coral) and human-made structure (legacy: unmappable).
OUTSIDE_LEGACY_SUBTREE = frozenset({40, 41, 76, 77, 52, 53, 21, 54, 95})

EXPECTED_FIDELITY = {"approximate": 9, "coarsened": 25, "exact": 60, "unmappable": 1}

# AphiaIDs re-checked live against WoRMS on 2026-10-29: id -> (AphiaID, scientific name, rank,
# parent).
NEW_NODES = {
    "SC_ALCYONIIDAE": (125269, "Alcyoniidae", "Family", "SC"),
    "SC_NEPHTHEIDAE": (146762, "Nephtheidae", "Family", "SC"),
    "CN_ANTIPATHARIA": (22549, "Antipatharia", "Order", "CNIDARIA"),
}

_LEGACY_TAG = re.compile(r"^\[legacy: ([^\]]+)\]")


def _targets(edge: Any) -> dict[str, str]:
    """Edge targets keyed by plain axis names."""
    return {str(getattr(axis, "value", axis)): node for axis, node in edge.targets.items()}


def _fidelity(edge: Any) -> str:
    return str(getattr(edge.fidelity, "value", edge.fidelity))


def _edge(crosswalk: Any, label: str) -> Any:
    return next((e for e in crosswalk.edges if e.source_label == label), None)


def _legacy_edge(v1: Any, legacy: str) -> Any:
    return _edge(v1, LEGACY_ALIAS.get(legacy, legacy))


def _node(schema: Any, node_id: str) -> Any:
    return next((n for n in schema.nodes if n.id == node_id), None)


def _ancestors(schema: Any, node_id: str) -> list[str]:
    parents = {n.id: n.parent for n in schema.nodes}
    chain: list[str] = []
    current = parents.get(node_id)
    while current is not None:
        chain.append(current)
        current = parents.get(current)
    return chain


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


@pytest.fixture(scope="module")
def v2(registry: Registry) -> Any:
    return registry.crosswalk("coralscapes-95")


@pytest.fixture(scope="module")
def v1(registry: Registry) -> Any:
    return registry.crosswalk("coralscapes-39")


@pytest.fixture(scope="module")
def rsb(registry: Registry) -> Any:
    return registry.label_schema("rs-benthic-v1")


def test_card_table_is_complete() -> None:
    assert [i for i, _, _ in CARD_ROWS] == list(range(1, 96))
    assert len({name for _, name, _ in CARD_ROWS}) == 95
    assert len({legacy for _, _, legacy in CARD_ROWS}) == LEGACY_COUNT == 39


def test_names_equal_the_schema_in_id_order(registry: Registry) -> None:
    schema = registry.label_schema("coralscapes-95")
    assert [n.name for n in schema.nodes] == [name for _, name, _ in CARD_ROWS]


def test_every_id_has_exactly_one_edge(v2: Any) -> None:
    labels = [e.source_label for e in v2.edges]
    assert labels == [name for _, name, _ in CARD_ROWS]
    assert len(set(labels)) == 95
    assert (v2.source_schema, v2.target_schema) == ("coralscapes-95", "rs-benthic-v1")


def test_every_target_resolves_on_its_axis(v2: Any, rsb: Any) -> None:
    for edge in v2.edges:
        for axis, node_id in _targets(edge).items():
            node = _node(rsb, node_id)
            assert node is not None, (edge.source_label, node_id)
            assert str(getattr(node.axis, "value", node.axis)) == axis, (
                edge.source_label,
                axis,
                node_id,
            )


def test_only_hard_substrate_is_unmappable(v2: Any) -> None:
    assert [e.source_label for e in v2.edges if not e.targets] == ["hard substrate"]
    assert all(e.note for e in v2.edges if _fidelity(e) != "exact")


def test_fidelity_counts_are_pinned(v2: Any) -> None:
    assert dict(Counter(_fidelity(e) for e in v2.edges)) == EXPECTED_FIDELITY


def test_every_note_names_its_legacy_class(v2: Any, v1: Any) -> None:
    for (_, name, legacy), edge in zip(CARD_ROWS, v2.edges, strict=True):
        match = _LEGACY_TAG.match(edge.note or "")
        assert match is not None, name
        assert match.group(1) == legacy, name
        assert _legacy_edge(v1, legacy) is not None, legacy


def test_legacy_consistency(v2: Any, v1: Any, rsb: Any) -> None:
    """Taxon equals or descends from the legacy edge's taxon and the condition is equal.

    The only taxon refinements that leave the legacy subtree are the pinned rule a/b ones.
    """
    outside: set[int] = set()
    for (class_id, name, legacy), edge in zip(CARD_ROWS, v2.edges, strict=True):
        new_t, old_t = _targets(edge), _targets(_legacy_edge(v1, legacy))
        assert new_t.get("condition") == old_t.get("condition"), name
        taxon, old_taxon = new_t.get("taxon"), old_t.get("taxon")
        if old_taxon is None:  # legacy unmappable: only human-made structure is refined
            if taxon is not None:
                outside.add(class_id)
            continue
        assert taxon is not None, name
        if taxon != old_taxon and old_taxon not in _ancestors(rsb, taxon):
            outside.add(class_id)
    assert outside == OUTSIDE_LEGACY_SUBTREE


def test_inherited_edges_equal_the_legacy_edge(v2: Any, v1: Any) -> None:
    """A class carrying the same label as its legacy class keeps the coralscapes-39 edge."""
    for (_, name, legacy), edge in zip(CARD_ROWS, v2.edges, strict=True):
        if name == legacy:
            old = _legacy_edge(v1, legacy)
            assert (_targets(edge), _fidelity(edge)) == (_targets(old), _fidelity(old)), name


def test_explicit_rule_a_edges(v2: Any) -> None:
    want = {
        "xeniidae alive": {"taxon": "SC_XENIIDAE", "condition": "HEALTHY"},
        "nephtheidae alive": {"taxon": "SC_NEPHTHEIDAE", "condition": "HEALTHY"},
        "leather coral alive": {"taxon": "SC_ALCYONIIDAE", "condition": "HEALTHY"},
        "soft coral (malacalcyonacea) alive": {"taxon": "SC", "condition": "HEALTHY"},
        "soft coral (malacalcyonacea) bleached": {"taxon": "SC", "condition": "BLEACHED"},
        "cirrhipathes": {"taxon": "CN_ANTIPATHARIA", "condition": "HEALTHY"},
        "acanthaster planci": {"taxon": "COTS"},
        "dark": {"taxon": "DRK"},
        "transect line": {"taxon": "TL"},
        "water column": {"taxon": "WC"},
        "water surface": {"taxon": "WC"},
        "background": {"taxon": "WC"},
        "human-made structure": {"taxon": "HS"},
        "human": {"taxon": "DIV"},
    }
    for label, targets in want.items():
        assert _targets(_edge(v2, label)) == targets, label
    assert _fidelity(_edge(v2, "human-made structure")) == "coarsened"


def test_health_suffix_and_algae_rules(v2: Any) -> None:
    condition_of = {"alive": "HEALTHY", "bleached": "BLEACHED", "dead": "OLD_DEAD"}
    for _, name, _ in CARD_ROWS:
        targets = _targets(_edge(v2, name))
        suffix = name.rsplit(" ", 1)[-1]
        if name.startswith("algae covered "):
            assert targets == {"taxon": "DCA"}, name
        elif suffix in condition_of:
            assert targets.get("condition") == condition_of[suffix], name


def test_a_family_never_maps_to_a_genus(v2: Any, rsb: Any) -> None:
    families = [name for _, name, _ in CARD_ROWS if name.split(" ")[0].endswith("idae")]
    assert {name.split(" ")[0] for name in families} == {
        "fungiidae",
        "lobophylliidae",
        "nephtheidae",
        "xeniidae",
    }
    for name in families:
        node = _node(rsb, _targets(_edge(v2, name))["taxon"])
        assert node.worms_rank != "Genus", (name, node.id)
    assert _targets(_edge(v2, "lobophylliidae alive"))["taxon"] == "HC"
    assert _targets(_edge(v2, "fungiidae alive"))["taxon"] == "HC_FUNGIIDAE"


def test_new_nodes_exist_with_worms_ids_and_are_used(v2: Any, rsb: Any) -> None:
    used = {node for edge in v2.edges for node in edge.targets.values()}
    for node_id, (aphia, name, rank, parent) in NEW_NODES.items():
        node = _node(rsb, node_id)
        assert node is not None, node_id
        assert (node.worms_aphia_id, node.worms_scientificname, node.worms_rank) == (
            aphia,
            name,
            rank,
        )
        assert node.parent == parent
        assert node.worms_status == "accepted"
        assert node_id in used


# -- the coralscapes-v2 source entry: stem link, per-stem split group, site count --------------


@pytest.fixture(scope="module")
def src_v2(registry: Registry) -> Any:
    return registry.source("coralscapes-v2")


def test_source_link_vocabulary_is_closed() -> None:
    ok = {"source_id": "coralscapes", "key": "stem", "relation": "re-annotation"}
    assert SourceLink(**ok).relation == "re-annotation"
    for patch in ({"key": "sha256"}, {"relation": "duplicate"}, {"extra": "x"}):
        with pytest.raises(ValidationError):
            SourceLink(**{**ok, **patch})


def test_v2_declares_its_stem_link_to_v1(registry: Registry, src_v2: Any) -> None:
    link = SourceLink(source_id="coralscapes", key="stem", relation="re-annotation")
    assert src_v2.links == (link,)
    assert registry.source("coralscapes").links == ()
    # the WP4 reader takes the registry objects as they are
    assert declared_links(registry) == (
        LinkDecl("coralscapes-v2", "coralscapes", "stem", "re-annotation"),
    )


def test_v2_split_group_is_one_group_per_stem(registry: Registry, src_v2: Any) -> None:
    stem, path = "site7_000123_004560", "images/site7_000123_004560.png"
    group = src_v2.split_group_for(stem=stem, upstream_path=path, partition="default")
    assert group == f"coralscapes-v2/{stem}"
    other = src_v2.split_group_for(stem="site7_000124_004800", upstream_path=path, partition="x")
    assert other == "coralscapes-v2/site7_000124_004800" != group  # one site is not one group
    v1 = registry.source("coralscapes")  # V1 keeps its per-site group; the stem link joins them
    v1_stem = f"{stem}_leftImg8bit"
    assert v1.split_group_for(stem=v1_stem, upstream_path=path, partition="train") == (
        "coralscapes/site7"
    )


def test_v2_covers_41_dive_sites(src_v2: Any) -> None:
    assert src_v2.coverage.sites == 41
