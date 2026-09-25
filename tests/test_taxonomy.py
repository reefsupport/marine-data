"""WP-7: the unified taxonomy is anchored, crosswalked and gated — offline only."""

from __future__ import annotations

import shutil
import socket
import urllib.request

import pytest

from marinedata import taxonomy as tx
from marinedata.registry import Registry
from marinedata.schema import LabelNode


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test here reads ONLY the pinned snapshot; any socket is a failure."""

    def refuse(*_a, **_k):
        raise AssertionError("network access from a snapshot-only test")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


def test_network_is_really_blocked():
    with pytest.raises(AssertionError, match="network"):
        urllib.request.urlopen("https://www.marinespecies.org/rest/AphiaRecordByAphiaID/1")


def test_every_taxon_node_has_aphia_or_non_taxon(reg):
    snapshot = tx.load_snapshot(reg.root)
    assert tx.check_nodes(reg, snapshot) == []
    counts = tx.node_counts(reg)
    assert counts["nodes"] == counts["aphia"] + counts["non_taxon"]
    assert counts["aphia"] > 800


def test_every_vocab_has_no_silent_drop_and_meets_floor(reg):
    exceptions = tx.load_meta(reg.root)["coverage_exceptions"]
    audits = tx.audit_all(reg, reg.root)
    assert len(audits) >= 15
    for a in audits:
        assert a.silent_drops == (), a.line()
        assert a.dead_targets == (), a.line()
        assert all(a.unmappable.values()), f"{a.source_id}: unmappable without a reason"
        assert a.coverage >= tx.MIN_MAPPED or a.source_id in exceptions, a.line()


def test_gate_passes_on_a_release(reg):
    """The per-release gate (the one that actually blocks a build) is scoped to that
    release's sources, so it passes even though the full registry (below) does not."""
    stamp = tx.assert_release_gate(reg, reg.root, ["ruod", "noaa-benthic-t1"])
    assert stamp["taxonomy_version"] == tx.load_meta(reg.root)["version"]


def test_gate_on_the_full_registry_fails_only_on_no_crosswalk_yet(reg):
    """WP-7b (2026-09-25): `no_crosswalk_yet` stopped being a gate exemption. 4 of the
    23 sources it listed got real crosswalks (reefolution, coralscop-masks-rs,
    labeled-fishes-in-the-wild, mouss-detection); the other 19 are not staged in
    rs-storage-open (or are access-blocked / have no closed label vocabulary at all —
    coralvqa, marineinst20m) and cannot be honestly crosswalked without inventing label
    names. Every remaining failure must be exactly one of those 19, never a silent
    drop, a dead target or an unmappable-with-no-reason."""
    meta = tx.load_meta(reg.root)
    no_crosswalk = set(meta.get("no_crosswalk_yet") or {})
    fails = tx.gate(reg, reg.root)
    assert fails, "expected the 19 unresolved no_crosswalk_yet sources to fail"
    for line in fails:
        sid = line.split(":", 1)[0]
        assert sid in no_crosswalk, f"unexpected gate failure outside no_crosswalk_yet: {line}"
    assert {line.split(":", 1)[0] for line in fails} == no_crosswalk


def _copy_root(reg, tmp_path):
    root = tmp_path / "registry"
    shutil.copytree(reg.root / "taxonomy", root / "taxonomy")
    return root


def test_gate_fails_on_silent_drop(reg, tmp_path):
    root = _copy_root(reg, tmp_path)
    vocab = root / "taxonomy" / "vocab" / "ruod.tsv"
    vocab.write_text(vocab.read_text() + "kraken\t5\t\n")
    fails = tx.gate(reg, root, source_ids=["ruod"])
    assert any("SILENT DROP 'kraken'" in f for f in fails)


def test_gate_fails_under_floor_without_exception(reg, tmp_path):
    root = _copy_root(reg, tmp_path)
    meta = root / "taxonomy" / "taxonomy.yaml"
    text = meta.read_text().replace("  plc-beijbom2015: >-", "  plc-beijbom2015-x: >-")
    meta.write_text(text)
    fails = tx.gate(reg, root, source_ids=["plc-beijbom2015"])
    assert any("plc-beijbom2015" in f and "< 95%" in f for f in fails)


def test_gate_fails_on_uncovered_labelled_source(reg, tmp_path):
    root = _copy_root(reg, tmp_path)
    meta = root / "taxonomy" / "taxonomy.yaml"
    meta.write_text(meta.read_text().replace("  ozfish:", "  ozfish-x:"))
    fails = tx.gate(reg, root, source_ids=["ozfish"])
    assert fails == ["ozfish: labelled source with no crosswalk and no documented exception"]


def test_frozen_release_matches_working_tree(reg):
    version = tx.load_meta(reg.root)["version"]
    d = tx.diff(tx.load_manifest(version, reg.root), tx.manifest(reg, reg.root))
    for key in ("nodes", "edges"):
        assert d[key] == {"added": [], "removed": [], "changed": {}}, key


def test_diff_reports_added_removed_changed(reg):
    a = tx.manifest(reg, reg.root)
    b = {**a, "taxonomy_version": "1.1.0", "nodes": dict(a["nodes"]), "edges": dict(a["edges"])}
    gone = next(iter(b["nodes"]))
    del b["nodes"][gone]
    b["nodes"]["rs-taxa-v1:A999999999"] = {"parent": None}
    edge = next(iter(b["edges"]))
    b["edges"][edge] = {**b["edges"][edge], "fidelity": "approximate"}
    d = tx.diff(a, b)
    assert d["to"] == "1.1.0"
    assert d["nodes"]["removed"] == [gone]
    assert d["nodes"]["added"] == ["rs-taxa-v1:A999999999"]
    assert list(d["edges"]["changed"]) == [edge] or a["edges"][edge]["fidelity"] == "approximate"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"non_taxon": True},
        {"non_taxon_reason": "substrate: sand"},
        {"non_taxon": True, "non_taxon_reason": "vibes: nope"},
        {"non_taxon": True, "non_taxon_reason": "substrate: sand", "worms_aphia_id": 1},
    ],
)
def test_non_taxon_validator_rejects(kwargs):
    with pytest.raises(ValueError):
        LabelNode(id="X", name="x", **kwargs)


def test_non_taxon_validator_accepts():
    node = LabelNode(id="SD", name="sand", non_taxon=True, non_taxon_reason="substrate: sand")
    assert node.non_taxon and not node.is_taxon
