"""Harmonisation tests.

The behaviour these lock down is the one that matters most in the whole package: an
unmappable source label must leave the axis *unsupervised* rather than being coerced
into the nearest available class. Coercion is the mechanism behind "50% unknown hard
substrate", and it is silent by construction — nothing in a training log reveals it.
"""

from __future__ import annotations

import pytest

from marinedata import Registry
from marinedata.harmonize import HarmonizationError, Harmonizer
from marinedata.schema import (
    Axis,
    Crosswalk,
    CrosswalkEdge,
    Fidelity,
    LabelNode,
    LabelSchema,
)


@pytest.fixture
def tiny_schema() -> LabelSchema:
    return LabelSchema(
        id="tiny",
        name="Tiny",
        canonical=True,
        axes=(Axis.TAXON, Axis.CONDITION),
        nodes=(
            LabelNode(id="BIOTIC", name="Biotic"),
            LabelNode(id="HC", name="Hard coral", parent="BIOTIC"),
            LabelNode(id="SC", name="Soft coral", parent="BIOTIC"),
            LabelNode(id="HEALTHY", name="Healthy", axis=Axis.CONDITION),
        ),
    )


def test_ancestors_walk_up(tiny_schema: LabelSchema) -> None:
    assert tiny_schema.ancestors("HC") == ("BIOTIC",)
    assert tiny_schema.ancestors("BIOTIC") == ()


def test_schema_rejects_dangling_parent() -> None:
    with pytest.raises(ValueError, match="unknown parent"):
        LabelSchema(
            id="bad",
            name="Bad",
            nodes=(LabelNode(id="X", name="X", parent="NOPE"),),
        )


def test_edge_requires_targets_or_unmappable() -> None:
    with pytest.raises(ValueError, match="unmappable"):
        CrosswalkEdge(source_label="orphan")


def test_unmappable_edge_rejects_targets() -> None:
    with pytest.raises(ValueError, match="unmappable but targets"):
        CrosswalkEdge(source_label="x", targets={Axis.TAXON: "HC"}, fidelity=Fidelity.UNMAPPABLE)


def test_harmonizer_rejects_missing_target_node(tiny_schema: LabelSchema) -> None:
    """Fail at construction, not mid-epoch."""
    walk = Crosswalk(
        id="w",
        source_schema="src",
        target_schema="tiny",
        edges=(CrosswalkEdge(source_label="a", targets={Axis.TAXON: "NOT_A_NODE"}),),
    )
    with pytest.raises(HarmonizationError, match="absent from schema"):
        Harmonizer(walk, tiny_schema)


def test_multi_axis_edge(tiny_schema: LabelSchema) -> None:
    walk = Crosswalk(
        id="w",
        source_schema="src",
        target_schema="tiny",
        edges=(
            CrosswalkEdge(
                source_label="live hard coral",
                targets={Axis.TAXON: "HC", Axis.CONDITION: "HEALTHY"},
            ),
        ),
    )
    result = Harmonizer(walk, tiny_schema).map_label("live hard coral")
    assert result.labels[Axis.TAXON].node_id == "HC"
    assert result.labels[Axis.CONDITION].node_id == "HEALTHY"
    assert result.supervised == frozenset({Axis.TAXON, Axis.CONDITION})


def test_unknown_label_drops_rather_than_guessing(tiny_schema: LabelSchema) -> None:
    """⭐ The core invariant. No label, no supervision — never a nearest-neighbour."""
    walk = Crosswalk(
        id="w",
        source_schema="src",
        target_schema="tiny",
        edges=(CrosswalkEdge(source_label="known", targets={Axis.TAXON: "HC"}),),
    )
    result = Harmonizer(walk, tiny_schema).map_label("some gorgonian")
    assert result.dropped is True
    assert result.labels == {}
    assert result.supervised == frozenset()


def test_strict_mode_raises_on_unknown(tiny_schema: LabelSchema) -> None:
    walk = Crosswalk(
        id="w",
        source_schema="src",
        target_schema="tiny",
        edges=(CrosswalkEdge(source_label="known", targets={Axis.TAXON: "HC"}),),
    )
    with pytest.raises(HarmonizationError, match="no edge for"):
        Harmonizer(walk, tiny_schema, strict=True).map_label("unknown")


def test_source_cannot_supervise_an_axis_it_does_not_annotate(tiny_schema: LabelSchema) -> None:
    """A taxon-only point dataset must not gain condition supervision from a label
    that happens to mention bleaching."""
    walk = Crosswalk(
        id="w",
        source_schema="src",
        target_schema="tiny",
        edges=(
            CrosswalkEdge(
                source_label="live hard coral",
                targets={Axis.TAXON: "HC", Axis.CONDITION: "HEALTHY"},
            ),
        ),
    )
    harmonizer = Harmonizer(walk, tiny_schema, supervised_axes=frozenset({Axis.TAXON}))
    result = harmonizer.map_label("live hard coral")
    assert set(result.labels) == {Axis.TAXON}
    assert Axis.CONDITION not in result.supervised


def test_target_schema_mismatch_is_caught(tiny_schema: LabelSchema) -> None:
    walk = Crosswalk(id="w", source_schema="src", target_schema="other")
    with pytest.raises(HarmonizationError, match="but was given schema"):
        Harmonizer(walk, tiny_schema)


# ── the real Coralscapes crosswalk ────────────────────────────────────────


def test_coralscapes_decomposes_into_axes(registry: Registry) -> None:
    """The flat class 'massive/meandering bleached' is really three facts."""
    harmonizer = registry.harmonizer_for("coralscapes")
    assert harmonizer is not None
    result = harmonizer.map_label("massive/meandering bleached")
    assert result.labels[Axis.TAXON].node_id == "HC"
    assert result.labels[Axis.FORM].node_id == "CMM"
    assert result.labels[Axis.CONDITION].node_id == "BLEACHED"


def test_coralscapes_has_no_path_to_soft_coral(registry: Registry) -> None:
    """⭐ The finding this whole project is built around.

    Not one of the 39 Coralscapes classes maps to SC or any of its children. On
    Caribbean reefs, where octocorals are ~24% of our annotations, a model trained on
    this schema cannot express a quarter of what it sees.
    """
    walk = registry.crosswalk("coralscapes-39")
    soft_coral_nodes = {"SC", "SC_SEA_FAN", "SC_SEA_ROD", "SC_SEA_PLUME", "SC_SEA_WHIP"}
    reachable = {node_id for edge in walk.edges for node_id in edge.targets.values()}
    assert not (reachable & soft_coral_nodes), (
        "A soft-coral edge appeared in the Coralscapes crosswalk. If Coralscapes gained "
        "a soft-coral class, this is excellent news and the domain_shift warnings on the "
        "source entry must be updated too."
    )


def test_coralscapes_crosswalk_covers_all_39_classes(registry: Registry) -> None:
    walk = registry.crosswalk("coralscapes-39")
    assert len(walk.edges) == 39, f"expected 39 edges, found {len(walk.edges)}"


def test_coralscapes_records_its_lossiness(registry: Registry) -> None:
    """Coarsened and approximate edges must be labelled as such, not silently exact."""
    walk = registry.crosswalk("coralscapes-39")
    coverage = walk.coverage
    assert coverage[Fidelity.COARSENED] > 0
    assert coverage[Fidelity.APPROXIMATE] > 0
    # 'unknown hard substrate' is the dumping ground — it must never claim exactness.
    edge = walk.edge("unknown hard substrate")
    assert edge is not None
    assert edge.fidelity is Fidelity.APPROXIMATE


def test_every_crosswalk_target_resolves(registry: Registry) -> None:
    for walk in registry.crosswalks:
        target = registry.label_schema(walk.target_schema)
        Harmonizer(walk, target)  # raises if any edge points at a missing node


# ── schema validation: four holes found and closed 2026-08-17 ─────────────


def test_self_parent_cycle_rejected_at_construction() -> None:
    """Previously loaded clean; leaves() silently returned () and the cycle surfaced
    only if someone happened to call ancestors()."""
    with pytest.raises(ValueError, match="cycle in parent chain"):
        LabelSchema(id="x", name="x", nodes=(LabelNode(id="A", name="A", parent="A"),))


def test_multi_node_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="cycle in parent chain"):
        LabelSchema(
            id="x",
            name="x",
            nodes=(
                LabelNode(id="X", name="X", parent="Y"),
                LabelNode(id="Y", name="Y", parent="X"),
            ),
        )


def test_cross_axis_parent_rejected() -> None:
    """A taxon descending from a condition makes 'ancestor' meaningless and would
    silently corrupt any rollup."""
    with pytest.raises(ValueError, match="cross-axis parent"):
        LabelSchema(
            id="x",
            name="x",
            axes=(Axis.TAXON, Axis.CONDITION),
            nodes=(
                LabelNode(id="BLEACHED", name="Bleached", axis=Axis.CONDITION),
                LabelNode(id="HC", name="Hard coral", axis=Axis.TAXON, parent="BLEACHED"),
            ),
        )


def test_duplicate_node_ids_rejected() -> None:
    """The second node was silently unreachable via node()."""
    with pytest.raises(ValueError, match="duplicate node id"):
        LabelSchema(
            id="x",
            name="x",
            nodes=(LabelNode(id="D", name="first"), LabelNode(id="D", name="second")),
        )


def test_node_on_undeclared_axis_rejected() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        LabelSchema(
            id="x",
            name="x",
            axes=(Axis.TAXON,),
            nodes=(LabelNode(id="Z", name="Z", axis=Axis.CONDITION),),
        )


def test_valid_deep_chain_still_works() -> None:
    schema = LabelSchema(
        id="ok",
        name="ok",
        nodes=(
            LabelNode(id="ROOT", name="r"),
            LabelNode(id="MID", name="m", parent="ROOT"),
            LabelNode(id="LEAF", name="l", parent="MID"),
        ),
    )
    assert schema.ancestors("LEAF") == ("MID", "ROOT")
    assert [n.id for n in schema.leaves()] == ["LEAF"]
