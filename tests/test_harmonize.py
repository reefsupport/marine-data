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
from marinedata.task import TaskProjector


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


def test_roboflow_unhealthy_never_resolves_to_bleached(registry: Registry) -> None:
    """WS-D S24 D1: the condition axis is flat (BLEACHED is a sibling of DISEASED/
    RECENTLY_DEAD/OLD_DEAD, not their ancestor), so a Roboflow contributor's binary
    "Unhealthy" call must never be asserted as the specific diagnosis BLEACHED. It
    resolves to UNHEALTHY, their shared condition-axis parent, and the real
    ``bleaching-condition`` task (targeting the six leaves) abstains on it rather than
    guessing."""
    walk = registry.crosswalk("roboflow-bleaching-condition-hu")
    target = registry.label_schema(walk.target_schema)
    harmonizer = Harmonizer(walk, target)

    result = harmonizer.map_label("Unhealthy")
    node_id = result.labels[Axis.CONDITION].node_id
    assert node_id == "UNHEALTHY"
    assert node_id != "BLEACHED"

    task = registry.task("bleaching-condition")
    assert set(task.classes) == {
        "HEALTHY",
        "PALE",
        "BLEACHED",
        "DISEASED",
        "RECENTLY_DEAD",
        "OLD_DEAD",
    }, "bleaching-condition classes must stay the six leaves; UNHEALTHY is an ancestor."
    projector = TaskProjector(task, target)
    projection = projector.project(node_id)
    assert projection.target_class is None
    assert "coarser" in projection.reason


def test_every_crosswalk_target_resolves(registry: Registry) -> None:
    for walk in registry.crosswalks:
        target = registry.label_schema(walk.target_schema)
        Harmonizer(walk, target)  # raises if any edge points at a missing node


def test_suim_scene_classes_are_all_coarsened_or_exact_by_design(registry: Registry) -> None:
    """SUIM's 8 scene classes never claim precision the source doesn't have.

    HD (divers) and BW (background) map exactly onto their own dedicated non-benthic
    nodes. Every other class is necessarily coarser: RI ("reefs/invertebrates")
    flattens hard coral, soft coral and other fauna into one bucket the source cannot
    itself disaggregate, so it must land on BIOTIC rather than guessing HC.
    """
    harmonizer = registry.harmonizer_for("suim")
    assert harmonizer is not None
    assert harmonizer.map_label("HD").labels[Axis.TAXON].node_id == "DIV"
    assert harmonizer.map_label("BW").labels[Axis.TAXON].node_id == "WC"
    result = harmonizer.map_label("RI")
    assert result.labels[Axis.TAXON].node_id == "BIOTIC"
    assert result.labels[Axis.TAXON].fidelity is Fidelity.COARSENED


def test_seaclear_separates_debris_biota_and_hardware(registry: Registry) -> None:
    """⭐ The scheme mixes three unrelated things — litter, incidental biota, and ROV
    hardware — and only the abstention machinery keeps them from colliding on one axis.
    """
    harmonizer = registry.harmonizer_for("seaclear-marine-debris")
    assert harmonizer is not None
    assert harmonizer.map_label("can_metal").labels[Axis.TAXON].node_id == "TRASH"
    assert harmonizer.map_label("animal_sponge").labels[Axis.TAXON].node_id == "PORIFERA"
    assert harmonizer.map_label("animal_fish").labels[Axis.TAXON].node_id == "FISH"
    assert harmonizer.map_label("rov_bluerov").labels[Axis.TAXON].node_id == "NON_BENTHIC"
    unknown = harmonizer.map_label("unknown_instance")
    assert Axis.TAXON not in unknown.supervised, (
        "unknown_instance is the source's own 'cannot tell' bucket — mapping it to "
        "anything would fabricate a determination the annotators declined to make."
    )


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


# ── taxonomic anchoring: OBIS/WoRMS ───────────────────────────────────────


def test_every_aphia_id_carries_its_verified_facts(registry: Registry) -> None:
    """A bare AphiaID cannot be reviewed, so it must not stand alone.

    Three of the first eight AphiaIDs entered here by hand were wrong — one pointed at
    Cryptonemiaceae (a red alga), one at Aspidostoma fallax (a bryozoan), one at a
    species rather than the intended genus. Nothing in the system could tell, because
    an integer looks the same whether it is right or wrong. Freezing the scientific
    name, rank and status alongside makes a wrong id visible in review and detectable
    by a drift check.
    """
    for schema in registry.schemas:
        for node in schema.nodes:
            if node.worms_aphia_id is None:
                continue
            assert node.worms_scientificname, f"{node.id}: AphiaID without a frozen name"
            assert node.worms_rank, f"{node.id}: AphiaID without a frozen rank"
            assert node.worms_status, f"{node.id}: AphiaID without a frozen status"
            assert node.worms_checked_on, f"{node.id}: AphiaID never checked"


def test_no_unaccepted_taxa_anchor_the_canonical_schema(registry: Registry) -> None:
    """An unaccepted WoRMS record means the name has been superseded."""
    for schema in registry.schemas:
        if not schema.canonical:
            continue
        for node in schema.nodes:
            if node.worms_status:
                assert node.worms_status == "accepted", (
                    f"{node.id}: anchored on a '{node.worms_status}' WoRMS record "
                    f"({node.worms_scientificname}). Re-resolve to the accepted id."
                )


def test_millepora_is_a_hydrozoan_not_an_alga(registry: Registry) -> None:
    """Regression guard for the specific corrected error, and for the design premise.

    Millepora sits in Hydrozoa while Scleractinia sits in Hexacorallia, yet every reef
    protocol scores both as hard coral. That is why functional grouping cannot be a
    single-parent tree walk.
    """
    schema = registry.label_schema("rs-benthic-v1")
    mil = schema.node("MIL")
    assert mil.worms_aphia_id == 205902
    assert mil.worms_scientificname == "Millepora"
    assert mil.worms_rank == "Genus"


def test_frozen_worms_fields_require_an_id() -> None:
    with pytest.raises(ValueError, match="no worms_aphia_id"):
        LabelNode(id="X", name="X", worms_scientificname="Something")


# ── the populated label dictionaries ──────────────────────────────────────


def test_our_own_data_can_be_harmonised(registry: Registry) -> None:
    """Until this crosswalk existed, our own Caribbean corpus — the only dense
    octocoral labelling anywhere — could not enter a harmonised training set at all."""
    harmonizer = registry.harmonizer_for("reef-support-benthic-own")
    assert harmonizer is not None
    assert harmonizer.map_label("Hard Coral").labels[Axis.TAXON].node_id == "HC"
    assert harmonizer.map_label("Soft Coral").labels[Axis.TAXON].node_id == "SC"
    assert harmonizer.map_label("Milleporid").labels[Axis.TAXON].node_id == "MIL"
    assert harmonizer.map_label("SCALE").labels[Axis.TAXON].node_id == "SCL"


def test_coralscapes_schema_matches_its_crosswalk(registry: Registry) -> None:
    """The dataset's own dictionary and its crosswalk must enumerate the same labels."""
    schema = registry.label_schema("coralscapes-39")
    walk = registry.crosswalk("coralscapes-39")
    assert len(schema.nodes) == 39
    schema_labels = {n.name for n in schema.nodes}
    walk_labels = {e.source_label for e in walk.edges}
    assert schema_labels == walk_labels


def test_catami_maps_multiple_axes_at_once(registry: Registry) -> None:
    """CATAMI is morphology-first, so its coral children populate the FORM axis —
    the cleanest multi-axis mapping in the registry, and the reason form is separate."""
    harmonizer = registry.harmonizer_for("benthicnet-1m")
    assert harmonizer is not None
    branching = harmonizer.map_label("Corals/Branching")
    assert branching.labels[Axis.TAXON].node_id == "HC"
    assert branching.labels[Axis.FORM].node_id == "CB"

    bleached = harmonizer.map_label("Corals/Bleached")
    assert bleached.labels[Axis.CONDITION].node_id == "BLEACHED"


def test_catami_hydrocorals_reach_millepora(registry: Registry) -> None:
    """CATAMI files Hydrocorals OUTSIDE Corals — taxonomically right, operationally
    awkward. The crosswalk is where that gets reconciled, which is precisely why
    functional grouping is a lookup rather than a tree walk."""
    harmonizer = registry.harmonizer_for("benthicnet-1m")
    assert harmonizer.map_label("Hydrocorals").labels[Axis.TAXON].node_id == "MIL"


def test_catami_reaches_soft_coral(registry: Registry) -> None:
    """The class Coralscapes cannot express at all."""
    harmonizer = registry.harmonizer_for("benthicnet-1m")
    for label in ("Black & Octocorals", "Black & Octocorals/Sea fans"):
        result = harmonizer.map_label(label)
        assert result.labels[Axis.TAXON].node_id.startswith("SC")


def test_every_crosswalk_declares_its_lossiness_honestly(registry: Registry) -> None:
    """A crosswalk claiming 100% exact is almost certainly not being honest — these
    vocabularies were designed independently and do not align perfectly."""
    for walk in registry.crosswalks:
        coverage = walk.coverage
        if len(walk.edges) >= 20 and not walk.exact_by_construction:
            assert coverage[Fidelity.EXACT] < len(walk.edges), (
                f"{walk.id}: every edge claims exact fidelity across {len(walk.edges)} "
                f"independently-designed labels, which is not credible"
            )


# ── coverage of the populated schemas and crosswalks ─────────────────────


def test_every_declared_schema_has_a_vocabulary(registry: Registry) -> None:
    """A schema registered as a bare id with no nodes is a placeholder, not a dictionary.

    Five of seven schemas were in that state before this pass: coralscapes-39,
    catami-1.4, coralnet-labelset, worms-genus and agrra-benthic were all registered
    with zero label nodes, so no dataset actually had a label dictionary.

    Open vocabularies are the deliberate exception — worms-species, sonotype,
    dataset-native and mermaid-attributes (D3a2: a 259-name vocabulary with no
    honest crosswalk yet) cannot be enumerated and say so in their descriptions.
    """
    OPEN = {"worms-species", "sonotype", "dataset-native", "mermaid-attributes"}
    # Closed (not open by nature) vocabularies whose label list is registered but not
    # yet transcribed from upstream — distinct from OPEN, which can never be enumerated.
    # Remove an id here once its `nodes` land.
    PENDING_ENUMERATION = {"deolho-21"}
    exempt = OPEN | PENDING_ENUMERATION
    empty = [s.id for s in registry.schemas if not s.nodes and s.id not in exempt]
    assert not empty, f"schemas registered with no vocabulary: {empty}"


def test_open_vocabularies_declare_why_they_are_open(registry: Registry) -> None:
    """An empty schema must justify itself, or it is indistinguishable from an oversight."""
    for schema_id in ("worms-species", "sonotype", "dataset-native"):
        schema = registry.label_schema(schema_id)
        assert not schema.nodes
        assert len(schema.description) > 100, (
            f"{schema_id}: an open vocabulary needs a description explaining why it "
            f"cannot be enumerated"
        )


def test_every_source_with_labels_declares_a_schema(registry: Registry) -> None:
    """54 of 59 sources had no schema at all before this pass."""
    from marinedata.enums import AnnotationKind

    gaps = [
        s.id
        for s in registry
        if s.loader
        and s.loader.layout != "metadata-only"
        and not s.loader.schema_id
        and any(a.supervises for a in s.annotations)
    ]
    assert not gaps, f"sources carrying labels but no declared vocabulary: {gaps}"
    assert AnnotationKind  # imported for the reader's benefit


def test_crosswalk_targets_resolve_in_the_canonical_schema(registry: Registry) -> None:
    """Every crosswalk must build — a target node that does not exist fails here, not
    at epoch 1."""
    assert registry.crosswalks, "no crosswalks loaded"
    for walk in registry.crosswalks:
        target = registry.label_schema(walk.target_schema)
        for edge in walk.edges:
            for axis, node_id in edge.targets.items():
                node = target.node(node_id)
                assert node is not None, (
                    f"{walk.id}: '{edge.source_label}' -> {axis.value}:{node_id} "
                    f"is absent from {target.id}"
                )
                assert node.axis is axis, (
                    f"{walk.id}: '{edge.source_label}' targets {node_id} on axis "
                    f"{axis.value} but that node is on {node.axis.value}"
                )


def test_agrra_distinguishes_pale_from_bleached(registry: Registry) -> None:
    """Evidence for splitting STRESSED into PALE/BLEACHED.

    AGRRA — the Caribbean regional standard — scores them separately, so a merged
    condition node would discard signal that practitioners in our own region record.
    """
    walk = registry.crosswalk("agrra-benthic")
    pale = walk.edge("C_PALE")
    bleached = walk.edge("C_BLEACHED")
    assert pale and bleached
    assert pale.targets[Axis.CONDITION] == "PALE"
    assert bleached.targets[Axis.CONDITION] == "BLEACHED"


def test_millepora_is_not_folded_into_hard_coral(registry: Registry) -> None:
    """Fire coral is a hydrozoan, and the schemes disagree about it.

    AGRRA surveys it as its own shape group; CoralNet holds both answers at once. Our
    canonical schema keeps MIL distinct from HC so the distinction survives, rather than
    being resolved by whichever crosswalk was written last.
    """
    rs = registry.label_schema("rs-benthic-v1")
    mil, hc = rs.node("MIL"), rs.node("HC")
    assert mil is not None and hc is not None
    assert mil.parent != "HC", "MIL must not descend from HC — it is a different class"
    assert mil.worms_aphia_id == 205902, "Millepora, verified against WoRMS"
    assert hc.worms_aphia_id == 1363, "Scleractinia, verified against WoRMS"

    agrra = registry.crosswalk("agrra-benthic")
    assert agrra.edge("FIRE").targets[Axis.TAXON] == "MIL"
    assert agrra.edge("CORAL").targets[Axis.TAXON] == "HC"


# ── label auditing: declared crosswalk edges vs labels in real data ───────


def test_observed_labels_reads_both_meta_shapes() -> None:
    """Loaders record native labels under two keys depending on cardinality."""
    from marinedata.labelcheck import observed_labels
    from marinedata.sample import Sample

    single = Sample(source_id="s", key="k", meta={"native_label": "Hard Coral"})
    many = Sample(source_id="s", key="k", meta={"native_labels": ["sand", "sand", "HC"]})
    assert observed_labels(single) == ["Hard Coral"]
    assert observed_labels(many) == ["sand", "sand", "HC"]
    assert observed_labels(Sample(source_id="s", key="k")) == []


def test_audit_flags_silent_drops_and_weights_them_by_instance() -> None:
    """A label with no edge is a SILENT DROP — supervision discarded with no error.

    Coverage is instance-weighted on purpose: one unmapped label covering 40% of
    annotations matters far more than ten appearing once each, and a type count hides
    exactly that.
    """
    from collections import Counter

    from marinedata.labelcheck import LabelAudit

    audit = LabelAudit(
        source_id="x",
        crosswalk_id="cw",
        samples=10,
        observed=Counter({"mapped": 60, "orphan": 40}),
        unmapped=("orphan",),
    )
    assert audit.instances == 100
    assert audit.coverage == pytest.approx(0.6)
    assert "SILENT DROP" in audit.report()


def test_audit_reports_unseen_edges_without_calling_them_errors() -> None:
    """A bounded sample sees a bounded vocabulary.

    Auditing 40 frames from one Colombian site reports SCALE and Milleporid as unseen,
    because that site genuinely has neither. The crosswalk is right; the sample is
    partial. The report must not present that as a defect.
    """
    from collections import Counter

    from marinedata.labelcheck import LabelAudit

    audit = LabelAudit(
        source_id="reef-support-benthic-own",
        crosswalk_id="reef-support-labelbox",
        samples=40,
        observed=Counter({"Hard Coral": 90, "Soft Coral": 10}),
        dead_edges=("Milleporid", "SCALE"),
    )
    assert audit.coverage == 1.0
    report = audit.report()
    assert "unseen edge" in report
    assert "prompt to look, not proof" in report
    assert "SILENT DROP" not in report


def test_audit_with_no_crosswalk_treats_every_label_as_unmapped() -> None:
    """A source with no crosswalk cannot map anything — coverage must read 0, not 100%."""
    from collections import Counter

    from marinedata.labelcheck import LabelAudit

    audit = LabelAudit(
        source_id="y",
        crosswalk_id=None,
        samples=5,
        observed=Counter({"a": 3, "b": 2}),
        unmapped=("a", "b"),
        detail="no crosswalk declared",
    )
    assert audit.coverage == 0.0
