"""The MERMAID attribute crosswalk — completeness, validity and measured coverage.

The point of these tests is that MERMAID's vocabulary is *live*: attribute names appear
as partners annotate. A label with no edge is not a harmless gap, it is a point that
leaves the corpus silently (``Harmonizer.map_label`` returns ``dropped=True`` in
non-strict mode), so the invariant worth defending is total coverage of the measured
vocabulary — every label mapped or explicitly declared unmappable, never absent.

The vocabulary is vendored in ``tests/data/mermaid_labels.tsv`` with its row counts so
the coverage floor is a real weighted number rather than a label count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.schema import Fidelity

VOCAB = Path(__file__).parent / "data" / "mermaid_labels.tsv"
CROSSWALK_ID = "mermaid-attributes"
TARGET_SCHEMA = "rs-benthic-v1"

#: Row-weighted mapped coverage measured 2026-09-18 over the full 464,025-row corpus:
#: 462,578 rows (99.69%). Only ``Other`` and ``Unknown`` are unmappable. The floor sits
#: half a point below the measurement so a new unmappable label is allowed, but a
#: regression toward the old ``coralnet-labelset`` mapping (24.9%) fails loudly.
COVERAGE_FLOOR = 0.9919


def _vocab() -> list[tuple[str, int]]:
    out = []
    for line in VOCAB.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        label, rows = line.split("\t")
        out.append((label, int(rows)))
    return out


@pytest.fixture(scope="module")
def crosswalk(registry):
    return registry.crosswalk(CROSSWALK_ID)


def test_vocabulary_fixture_is_the_measured_one():
    vocab = _vocab()
    assert len(vocab) == 259
    assert sum(rows for _, rows in vocab) == 464_025


def test_every_measured_label_has_exactly_one_edge(crosswalk):
    edges = [e.source_label for e in crosswalk.edges]
    missing = [label for label, _ in _vocab() if edges.count(label) != 1]
    assert not missing, f"labels without exactly one edge: {missing}"


def test_no_duplicate_source_labels(crosswalk):
    labels = [e.source_label for e in crosswalk.edges]
    assert len(labels) == len(set(labels))


def test_no_edges_beyond_the_measured_vocabulary(crosswalk):
    """An edge for a label the corpus never contained is a guess, not a mapping."""
    vocab = {label for label, _ in _vocab()}
    assert {e.source_label for e in crosswalk.edges} == vocab


def test_every_target_node_exists_in_the_canonical_schema(registry, crosswalk):
    schema = registry.label_schema(TARGET_SCHEMA)
    known = {node.id for node in schema.nodes}
    assert crosswalk.target_schema == TARGET_SCHEMA
    unknown = {
        (e.source_label, axis, node_id)
        for e in crosswalk.edges
        for axis, node_id in e.targets.items()
        if node_id not in known
    }
    assert not unknown, f"targets absent from {TARGET_SCHEMA}: {sorted(unknown)}"


def test_every_target_sits_on_the_axis_it_claims(registry, crosswalk):
    schema = registry.label_schema(TARGET_SCHEMA)
    axis_of = {node.id: node.axis for node in schema.nodes}
    wrong = {
        (e.source_label, axis, node_id)
        for e in crosswalk.edges
        for axis, node_id in e.targets.items()
        if axis_of[node_id] != axis
    }
    assert not wrong, f"axis mismatch: {sorted(wrong)}"


def test_unmappable_edges_carry_a_reason(crosswalk):
    """An unmappable edge without a note is indistinguishable from an oversight."""
    silent = [
        e.source_label for e in crosswalk.edges if e.fidelity is Fidelity.UNMAPPABLE and not e.note
    ]
    assert not silent, f"unmappable without a reason: {silent}"


def test_row_weighted_coverage_holds_the_measured_floor(crosswalk):
    by_label = {e.source_label: e for e in crosswalk.edges}
    total = sum(rows for _, rows in _vocab())
    mapped = sum(
        rows
        for label, rows in _vocab()
        if (edge := by_label.get(label)) is not None and edge.targets
    )
    assert mapped / total >= COVERAGE_FLOOR, (
        f"row-weighted coverage {mapped / total:.4%} fell below {COVERAGE_FLOOR:.2%}"
    )


def test_the_source_declares_this_crosswalk(registry):
    """Authoring the file is only half of it — the source has to point at it."""
    spec = registry.source("mermaid-aws").loader
    assert spec is not None and spec.crosswalk_id == CROSSWALK_ID


def test_bare_substrate_retargeted_to_rk(crosswalk):
    """S21 (R3 Q12a): retargeted from ABIOTIC(coarsened) to RK(exact)."""
    edge = next(e for e in crosswalk.edges if e.source_label == "Bare substrate")
    assert edge.targets["taxon"] == "RK"
    assert edge.fidelity is Fidelity.EXACT


def test_dead_coral_retargeted_to_transition(crosswalk):
    """S21 (R3 Q12b): retargeted from DC(approximate) to TRANSITION(coarsened)."""
    edge = next(e for e in crosswalk.edges if e.source_label == "Dead coral")
    assert edge.targets["taxon"] == "TRANSITION"
    assert edge.fidelity is Fidelity.COARSENED


def test_harmonizer_maps_the_homonym_traps(registry):
    """MERMAID disambiguates two genera that share the name Turbinaria; we must too."""
    harmonizer = registry.harmonizer_for("mermaid-aws")
    assert harmonizer is not None
    assert harmonizer.map_label("Turbinaria-coral").labels["taxon"].node_id == "HC_TURBINARIA"
    assert harmonizer.map_label("Turbinaria-algae").labels["taxon"].node_id == "SW"
    assert harmonizer.map_label("Turbinaria turbinata").labels["taxon"].node_id == "SW"
    assert harmonizer.map_label("Other").dropped is True
