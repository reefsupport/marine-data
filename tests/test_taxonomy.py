"""WP-7: the unified taxonomy is anchored, crosswalked and gated — offline only."""

from __future__ import annotations

import shutil
import socket
import urllib.request

import pytest

from marinedata import coralnet_labels as cn
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
    names. WP-7c moved coralvqa and marineinst20m to `crosswalk: open_vocabulary`, so 17
    remain. Every remaining failure must be one of them, never a silent drop, a dead
    target or an unmappable-with-no-reason. (The unscoped gate; CI runs scoped_gate.)"""
    meta = tx.load_meta(reg.root)
    no_crosswalk = set(meta.get("no_crosswalk_yet") or {})
    assert len(no_crosswalk) == 17
    fails = tx.gate(reg, reg.root)
    assert fails, "expected the 17 unresolved no_crosswalk_yet sources to fail"
    staged = set(tx.staged_sources(reg))
    for line in fails:
        sid = line.split(":", 1)[0]
        # WP-7d (D-S1): a not-staged source with a crosswalk but no vocab TSV also shows here
        unmeasured = "but no vocab TSV" in line and sid not in staged
        assert sid in no_crosswalk or unmeasured, f"unexpected gate failure: {line}"
    assert no_crosswalk <= {line.split(":", 1)[0] for line in fails}


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


def _copy_registry(tmp_path, old: str, new: str):
    root = tmp_path / "registry"
    shutil.copytree(Registry.load().root, root)
    path = root / "sources" / "fish.yaml"
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new))
    return Registry.load(root), root


def test_scoped_gate_passes_the_registry_and_lists_the_rest(reg):
    """D-Q: nothing staged or released fails today; the 17 not-staged no_crosswalk_yet
    sources are listed (for --strict), not failed."""
    fails, listed = tx.scoped_gate(reg, reg.root)
    assert fails == []
    no_crosswalk = set(tx.load_meta(reg.root)["no_crosswalk_yet"])
    unmeasured = {line.split(":", 1)[0] for line in listed if "but no vocab TSV" in line}
    assert {line.split(":", 1)[0] for line in listed} == no_crosswalk | unmeasured
    assert not unmeasured & set(tx.staged_sources(reg))
    assert {"reefolution", "coralscop-masks-rs"} <= set(tx.staged_sources(reg))
    assert "ozfish" not in tx.staged_sources(reg)


def _ozfish_uri_line() -> str:
    """Read ozfish's real `uri:` line straight from the registry, so this test
    doesn't drift out of sync with a literal that no longer matches the file."""
    text = (Registry.load().root / "sources" / "fish.yaml").read_text()
    start = text.index("  - id: ozfish")
    end = text.index("\n  - id: ", start + 1)
    for line in text[start:end].splitlines(keepends=True):
        if line.strip().startswith("uri:"):
            return line
    raise AssertionError("ozfish uri not found in registry/sources/fish.yaml")


def test_scoped_gate_fails_a_staged_source_missing_a_crosswalk(tmp_path):
    staged, root = _copy_registry(
        tmp_path,
        _ozfish_uri_line(),
        "      uri: s3://rs-storage-open/sources/ozfish/2026-09-25-test\n",
    )
    assert "ozfish" in tx.staged_sources(staged)
    fails, listed = tx.scoped_gate(staged, root)
    assert [f.split(":", 1)[0] for f in fails] == ["ozfish"]
    assert "no_crosswalk_yet is not a release-gate exemption" in fails[0]
    assert not any(line.startswith("ozfish:") for line in listed)


def test_scoped_gate_fails_a_source_named_in_a_release_manifest(reg, tmp_path):
    release = tmp_path / "RELEASE.json"
    release.write_text('{"sources": [{"id": "ozfish"}, {"id": "ruod"}]}')
    fails, _listed = tx.scoped_gate(reg, reg.root, releases=[release])
    assert [f.split(":", 1)[0] for f in fails] == ["ozfish"]


def test_open_vocabulary_passes_and_needs_the_declaration(reg, tmp_path):
    release = tmp_path / "RELEASE.json"
    release.write_text('{"sources": [{"id": "coralvqa"}, {"id": "marineinst20m"}]}')
    fails, listed = tx.scoped_gate(reg, reg.root, releases=[release])
    assert fails == []
    assert not any(line.split(":", 1)[0] in {"coralvqa", "marineinst20m"} for line in listed)
    root = _copy_root(reg, tmp_path)
    meta = root / "taxonomy" / "taxonomy.yaml"
    meta.write_text(
        meta.read_text().replace("    crosswalk: open_vocabulary", "    crosswalk: guess", 1)
    )
    fails, _listed = tx.scoped_gate(reg, root, releases=[release])
    assert fails == ["coralvqa: declared crosswalk 'guess' needs 'open_vocabulary' + a reason"]


def test_reefolution_meets_the_floor_without_an_exception(reg):
    """WP-7c: every observed Reefolution code resolves via its CoralNet label id."""
    audit = next(a for a in tx.audit_all(reg, reg.root) if a.source_id == "reefolution")
    assert audit.weighted and sum(audit.counts.values()) == 43500
    assert audit.coverage >= tx.MIN_MAPPED
    assert "reefolution" not in (tx.load_meta(reg.root).get("coverage_exceptions") or {})


FROZEN_TAXONOMY_VERSION = "2.2.1"
"""Charter D-AD: the v2 release ships taxonomy 2.1.0. Bumping it is a deliberate act —
freeze a new ``registry/taxonomy/releases/<v>.json`` and update this pin together."""


def test_frozen_release_matches_working_tree(reg):
    version = tx.load_meta(reg.root)["version"]
    assert version == FROZEN_TAXONOMY_VERSION
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


# ── WP-7d: missing-vocab gate (D-S1) and the CoralNet label-id resolver (D-S2) ──────


def test_every_staged_labelled_source_has_a_measured_vocab(reg):
    """D-S1: an unmeasured staged source is not a pass. WP-7d measured the last 9."""
    vocab = {a.source_id for a in tx.audit_all(reg, reg.root)}
    declared = tx.load_meta(reg.root).get("source_crosswalks") or {}
    for sid in set(tx.labelled_sources(reg)) & set(tx.staged_sources(reg)):
        assert sid in vocab or sid in declared, f"{sid} is staged with no vocab TSV"


def test_scoped_gate_fails_a_staged_source_with_no_vocab(reg, tmp_path):
    root = _copy_root(reg, tmp_path)
    (root / "taxonomy" / "vocab" / "coralscop-masks-rs.tsv").unlink()
    fails, _listed = tx.scoped_gate(reg, root)
    assert fails == [
        "coralscop-masks-rs: crosswalk 'coralscop-masks-rs' but no vocab TSV "
        "(coverage unmeasured; add registry/taxonomy/vocab/<source>.tsv)"
    ]


def test_coralnet_label_id_rederives_reefolution(reg):
    """D-S2: Reefolution's codes -> CoralNet ids (committed in its vocab description column,
    from labelset.csv) -> the coralnet-label-id resolver gives 100% and WP-7c's targets."""
    _head, counts = tx.read_vocab(reg.root / "taxonomy" / "vocab" / "reefolution.tsv")
    ids = {}
    for line in (reg.root / "taxonomy" / "vocab" / "reefolution.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].startswith("CoralNet "):
            ids[parts[0]] = parts[2].split()[1]
    assert set(ids) == set(counts) and len(ids) == 64
    resolver = tx.crosswalk_for(reg, reg.root, cn.CROSSWALK_ID)
    wp7c = reg.crosswalk("reefolution")
    for code, lid in ids.items():
        edge = resolver.edge(lid)
        assert edge is not None and edge.targets == wp7c.edge(code).targets, code
    hit = sum(counts[c] for c, lid in ids.items() if resolver.edge(lid).targets)
    assert hit == sum(counts.values()) == 43500


def test_coralnet_resolver_falls_back_to_the_functional_group(reg):
    resolver = cn.resolver(reg, reg.root)
    assert len(resolver.table) > 12000
    row = resolver.table[8384]  # "0 - no bleaching", Hard coral, used by none of our sources
    assert row["functional_group"] == "Hard coral" and resolver.curated.edge("8384") is None
    edge = resolver.edge("8384")
    assert edge.targets == {"taxon": "HC"} and edge.note.startswith("auto: CoralNet 8384")
    assert resolver.edge("999999999") is None


def test_coralnet_page_parsers():
    row = cn.parse_list(
        '<tr data-label-id="7"><td class="name"><a href="/label/7/">Acro &amp; co</a></td>'
        '<td>Hard coral</td><td><div class="meter" title="91%"></div></td>'
        '<td class="status-cell"><img src="/x/label-icon-duplicate__1.png" '
        'title="Duplicate of Acropora" /></td><td>Acr</td></tr>'
    )
    assert row == [
        {"label_id": 7, "name": "Acro & co", "functional_group": "Hard coral", "popularity": 91,
         "verified": False, "duplicate": True, "duplicate_of": "Acropora",
         "calcification_rates": False, "short_code": "Acr"}
    ]  # fmt: skip
    page = (
        '<div class="line">Name: XENIIDAE</div><div class="line">Verified:\n  No\n</div>'
        "Used in 104 sources\n and for 95920 annotations"
        "<dt>Description:</dt>\n<dd><p>Family Xeniidae</p></dd>"
    )
    d = cn.parse_detail(page)
    assert (d["name"], d["verified"], d["used_in_sources"], d["description"]) == (
        "XENIIDAE", False, 104, "Family Xeniidae")  # fmt: skip


def test_binary_not_bleached_labels_share_one_rule(reg):
    """WP-7d (manager decision 2): a binary bleached / not-bleached split assessed no other
    condition, so its "healthy" class only asserts "not bleached". NOAA PIFSC's CORAL,
    Roboflow's Healthy and Reef Support's non_bleached map by one rule: HEALTHY, never exact."""
    from marinedata.schema import Axis, Fidelity

    pairs = [
        ("noaa-pifsc-bleaching-condition", "CORAL"),
        ("roboflow-bleaching-condition-hb", "Healthy"),
        ("roboflow-bleaching-condition-hu", "Healthy"),
        ("reef-support-bleaching-condition", "non_bleached"),
    ]
    edges = {pair: reg.crosswalk(pair[0]).edge(pair[1]) for pair in pairs}
    assert all(e is not None for e in edges.values()), edges
    kinds = {(tuple(sorted(e.targets.items())), e.fidelity) for e in edges.values()}
    assert kinds == {(((Axis.CONDITION, "HEALTHY"),), Fidelity.COARSENED)}, edges


def test_coralscop_coral_targets_the_coarsest_node_holding_hard_and_soft_coral(reg):
    """WP-7d (manager decision 1, vocabulary 2.0.0): CoralSCOP masks are class-agnostic
    coral, so the edge must not assert Scleractinia (HC); it targets the lowest common
    ancestor of HC and SC."""
    from marinedata.schema import Axis

    nodes = {n.id: n for n in reg.label_schema("rs-benthic-v1").nodes}
    target = reg.crosswalk("coralscop-masks-rs").edge("coral").targets[Axis.TAXON]
    assert target == nodes["HC"].parent == nodes["SC"].parent == "CNIDARIA"
