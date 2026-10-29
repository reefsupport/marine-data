"""Label matching across all sources (registry only, no image data).

Runs the verifier of ``scripts/verify_label_matching.py`` on the live registry: every source that
declares a label schema or a crosswalk must resolve into ``rs-benthic-v1`` with nothing unresolved,
ambiguous or missing. Synthetic registries then show that each hard check can fail, so a green run
is not vacuous. Method and results: ``docs/label-matching.md``.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from marinedata.gates import verify_matching as vm
from marinedata.gates import verify_matching_cross as cross
from marinedata.gates.verify_matching_render import blocks, splice
from marinedata.registry import Registry
from marinedata.schema import Axis, Fidelity

ROOT = Path(__file__).resolve().parents[1]
REPORT = cross.run()
WITH_CROSSWALK = [r for r in REPORT["sources"] if r["crosswalk"]]
REQUIRED_COUNTS = {
    "labels", "edges", "exact", "coarsened", "ignored", "unresolved", "ambiguous", "missing",
}  # fmt: skip


def _declares(node: Any) -> bool:
    if isinstance(node, dict):
        has = any(node.get(k) for k in ("schema_id", "crosswalk_id"))
        return has or any(_declares(v) for v in node.values())
    if isinstance(node, list):
        return any(_declares(v) for v in node)
    return False


def _replace(obj: Any, **changes: Any) -> Any:
    if hasattr(obj, "model_copy"):
        return obj.model_copy(update=changes)
    return dataclasses.replace(obj, **changes)


def _run_all(edit: Callable[[Any], Any]) -> dict[str, Any]:
    """Run the verifier on a registry whose crosswalks have all been passed through ``edit``."""
    registry = copy.copy(Registry.load())
    registry._crosswalks = {cw.id: edit(cw) for cw in registry.crosswalks}
    return cross.run(registry=registry)


def test_no_failures() -> None:
    assert REPORT["failures"] == []


@pytest.mark.parametrize("rec", WITH_CROSSWALK, ids=lambda r: r["id"])
def test_source_resolves(rec: dict[str, Any]) -> None:
    counts = rec["counts"]
    assert set(counts) >= REQUIRED_COUNTS
    assert (counts["unresolved"], counts["ambiguous"], counts["missing"]) == (0, 0, 0)
    assert counts["duplicate"] == 0
    assert rec["failures"] == []
    assert rec["status"] in {"pass", "canonical-native"}


def test_every_declaring_source_is_verified() -> None:
    declared: set[str] = set()
    for path in sorted((ROOT / "registry" / "sources").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for entry in doc.get("sources", []) if isinstance(doc, dict) else []:
            if isinstance(entry, dict) and entry.get("id") and _declares(entry):
                declared.add(str(entry["id"]))
    assert declared, "no registry source declares a schema or a crosswalk"
    assert declared <= {r["id"] for r in REPORT["sources"]}, sorted(
        declared - {r["id"] for r in REPORT["sources"]}
    )


def test_sources_without_a_crosswalk_say_why() -> None:
    gaps = [r for r in REPORT["sources"] if r["status"] == "no-crosswalk"]
    assert all(r["reason"] for r in gaps) or not gaps


def test_node_matrix_covers_every_target_node() -> None:
    nodes = {n.id for n in vm.make_context().schemas[vm.TARGET].nodes}
    assert set(REPORT["agreement"]["nodes"]) == nodes


def test_new_nodes_resolve() -> None:
    new = REPORT["new_nodes"]
    assert new["failures"] == []
    assert set(new["nodes"]) == set(cross.NEW_NODES)
    assert all(n["present"] and n["parent_present"] for n in new["nodes"].values())


def test_unmappable_edges_are_listed_with_notes() -> None:
    items = REPORT["unmappable"]
    assert items
    assert all(i["note"] for i in items)
    assert any(i["crosswalk"] == "coralscapes-95" and i["label"] == "hard substrate" for i in items)
    assert REPORT["summary"]["edges_by_fidelity"]["ignored"] == len(items)


def test_coralscapes_v2_pairs_with_its_legacy_classes() -> None:
    legacy = REPORT["legacy"]["coralscapes-95"]
    assert legacy["paired"] == legacy["edges"]
    assert legacy["untagged"] == [] and legacy["unpaired"] == []
    assert sum(legacy["kinds"].values()) == legacy["paired"]


def test_ignore_index_is_255_in_every_scheme() -> None:
    ignore = REPORT["ignore"]
    assert ignore["failures"] == []
    assert ignore["checks"] > 0


def test_generated_files_are_current() -> None:
    hint = "run: uv run python scripts/verify_label_matching.py --write"
    result = (ROOT / "results" / "label-matching-v1.2.0.json").read_text()
    assert result == cross.dumps(REPORT), hint
    doc = (ROOT / "docs" / "label-matching.md").read_text()
    assert splice(doc, blocks(REPORT)) == doc, hint


# ── the checks can fail ──────────────────────────────────────────────────────────


def test_a_target_that_is_not_a_node_fails() -> None:
    def edit(cw: Any) -> Any:
        for i, edge in enumerate(cw.edges):
            if edge.fidelity is Fidelity.EXACT and Axis.TAXON in edge.targets:
                bad = _replace(edge, targets={Axis.TAXON: "NOT_A_NODE"})
                return _replace(cw, edges=[*cw.edges[:i], bad, *cw.edges[i + 1 :]])
        return cw

    report = _run_all(edit)
    assert report["summary"]["unresolved"] >= 1
    assert report["failures"]


def test_a_label_without_an_edge_fails() -> None:
    report = _run_all(lambda cw: _replace(cw, edges=cw.edges[1:]))
    assert report["summary"]["missing"] >= 1
    assert report["failures"]


def test_an_unmappable_edge_without_a_note_fails() -> None:
    def edit(cw: Any) -> Any:
        edges = [
            _replace(e, note=None) if e.fidelity is Fidelity.UNMAPPABLE else e for e in cw.edges
        ]
        return _replace(cw, edges=edges)

    assert any("without a note" in f for f in _run_all(edit)["failures"])


def test_small_helpers() -> None:
    assert {"hard", "coral"} <= cross.tokens("Hard Corals")
    assert cross.code_like("1012") and cross.code_like("CCA") and not cross.code_like("Sand")
    assert cross.health_words("Bleached Acropora") == {"bleached"}
    with pytest.raises(ValueError):
        splice("no markers here", {"sources": "x"})
