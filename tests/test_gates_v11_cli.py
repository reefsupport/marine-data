"""v1.1.0 gate CLI/orchestrator (G1-G9 into GATES.json), G8 images-scoped privacy and G9 labelmap.

The gate bodies of G1-G7 have their own fixtures in test_gates_v11.py. Here the CLI runs join,
leak and splits_v11 for real on that synthetic build; frozen, bytes_once, masks, decon, privacy and
labelmap are stubbed in the CLI tests (G8 and G9 have their own real tests below). No model weights
are downloaded: the detector is a fake, and the YuNet loader is mocked."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from test_gates_v11 import SHAS, build_release, write_report, write_shard, write_v1

from marinedata.cli import main
from marinedata.gates import (
    FAIL_CLOSED,
    REQUIRED,
    V11_REQUIRED,
    finish,
    merge_gates_json,
    privacy_v11,
    publish_verdict,
    skip,
)
from marinedata.gates.labelmap import load_release, run_labelmap_gate
from marinedata.gates.leak import id_tags
from marinedata.gates.privacy_detectors import Detection
from marinedata.gates.v11_runner import RUNNERS, V11_ALL, V11Options, run_v11_gate
from marinedata.gates.yunet_model import YuNetModelError
from marinedata.hf_export_v11 import label_map_rows
from marinedata.label_schemes import current_taxonomy_version
from marinedata.registry import Registry
from marinedata.task_layers.boxes_table import BOX_SOURCES
from marinedata.task_layers.image_labels_table import axes_resolver_for

STUBBED = ("frozen", "bytes_once", "masks", "decon", "privacy", "labelmap")


LIVE_TAXONOMY = current_taxonomy_version()  # the labelmap gate wants the live release's stamp


def stub(name, how="pass"):
    def run(build, options):
        if how == "fail":
            return finish(name, {"stub": 1}, [f"{name}: stub failure"])
        if how == "skip":
            return skip(name, "stub skipped")
        return finish(name, {"stub": 1}, [])

    return run


def stub_gates(monkeypatch, names=STUBBED, **how):
    for name in names:
        monkeypatch.setitem(RUNNERS, name, stub(name, how.get(name, "pass")))


def seed_v10(build):
    """What `release gate --all` adds: the v1.0 gates publish_verdict still requires."""
    names = [n for n in (*REQUIRED, *FAIL_CLOSED) if n not in V11_ALL]
    merge_gates_json(build / "GATES.json", [finish(n, {}, []) for n in names])


def entries(build):
    return json.loads((build / "GATES.json").read_text())


# CLI / orchestrator


def test_labelmap_is_a_v11_required_gate_and_all_nine_run():
    assert "labelmap" in V11_REQUIRED and len(V11_ALL) == 9
    assert {*V11_REQUIRED, "masks", "privacy"} == set(V11_ALL)


def test_cli_autodetects_a_v11_build_and_all_pass_allows_publish(tmp_path, monkeypatch, capsys):
    build = build_release(tmp_path)
    stub_gates(monkeypatch, V11_ALL)
    assert main(["release", "gate", str(build)]) == 0
    assert set(V11_ALL) <= set(entries(build))
    assert "publish verdict: BLOCKED" in capsys.readouterr().out  # v1.0 gates not yet in the file
    seed_v10(build)
    required, problems = publish_verdict(build)
    assert problems == [] and set(V11_ALL) <= set(required)


def test_e2e_real_join_leak_splits_pass_then_a_join_failure_blocks(tmp_path, monkeypatch):
    build, v1 = build_release(tmp_path), write_v1(tmp_path)
    stub_gates(monkeypatch)
    assert main(["release", "gate", str(build), "--v1-dir", str(v1)]) == 0
    gates = entries(build)
    assert all(gates[n]["passed"] for n in V11_ALL), gates
    seed_v10(build)
    assert publish_verdict(build)[1] == []
    orphan = [{"image_sha256": "f" * 64, "source_id": "srcA"}]
    write_shard(build, "points", "train", orphan, shard=1)  # G1: a row with no image
    assert main(["release", "gate", str(build), "--v1-dir", str(v1)]) == 1
    assert entries(build)["join"]["passed"] is False
    assert any(p.startswith("gate join: failed") for p in publish_verdict(build)[1])


@pytest.mark.parametrize("how", ["fail", "skip"])
def test_a_required_gate_that_fails_or_only_skips_exits_nonzero_and_blocks(
    tmp_path, monkeypatch, how
):
    build = build_release(tmp_path)
    stub_gates(monkeypatch, V11_ALL, leak=how)
    assert main(["release", "gate", str(build), "--v11"]) == 1
    seed_v10(build)
    problems = publish_verdict(build)[1]
    assert len(problems) == 1 and problems[0].startswith(f"gate leak: {how}ed"[:9])


def test_rehearsal_with_allow_unsplit_skips_never_passes_and_still_blocks(tmp_path):
    build, v1 = build_release(tmp_path), write_v1(tmp_path)
    write_report(build, unsplit_images=2, allow_unsplit=True)
    gates = ["--only", "splits_v11", "--only", "decon", "--v1-dir", str(v1)]
    assert main(["release", "gate", str(build), *gates]) == 1  # a real run refuses unsplit
    assert main(["release", "gate", str(build), *gates, "--rehearsal"]) == 0
    got = entries(build)
    assert all("skipped" in got[n] and got[n]["passed"] is False for n in ("splits_v11", "decon"))
    seed_v10(build)
    blocked = publish_verdict(build)[1]
    assert any(p.startswith("gate splits_v11: skipped") for p in blocked)
    assert any(p.startswith("gate decon: skipped") for p in blocked)


def test_a_gate_that_raises_is_a_failed_gate_not_a_crash(tmp_path, monkeypatch):
    def boom(build, options):
        raise RuntimeError("no look")

    monkeypatch.setitem(RUNNERS, "frozen", boom)
    result = run_v11_gate("frozen", tmp_path, V11Options())
    assert not result.passed and result.counts["failed_gate_error"] == 1


# G9 labelmap


def an_exact_edge():
    for key, edge in load_release("3.0.0")["edges"].items():
        cid, native = key.split(":", 1)
        if edge["fidelity"] == "exact" and set(edge["targets"]) == {"taxon"}:
            return cid, native, edge["targets"]["taxon"]
    raise AssertionError("no exact single-taxon edge in taxonomy 3.0.0")


def label_build(tmp_path, table=None, field=None, value=None):
    cid, native, taxon = an_exact_edge()
    common = {"source_id": "srcA", "taxon_node_id": taxon, "form_node_id": None}
    common |= {"condition_node_id": None, "fidelity": "exact", "taxonomy_version": LIVE_TAXONOMY}
    rows = {
        "label-map": {**common, "crosswalk_id": cid, "native_label": native, "mapped": True},
        "points": {**common, "label_native": native},
    }
    if table:
        rows[table] = {**rows[table], field: value}
    build = tmp_path / "labels"
    for config, row in rows.items():
        write_shard(build, config, "train", [row])
    return build


def test_labelmap_passes_when_every_class_id_resolves(tmp_path):
    result = run_labelmap_gate(label_build(tmp_path))
    assert result.passed, result.failures
    assert result.counts["label_map_rows"] == 1 and result.counts["points_rows"] == 1


@pytest.mark.parametrize(
    ("table", "field", "value", "kind"),
    [
        ("label-map", "taxon_node_id", "no-such-node", "unknown_node"),
        ("points", "fidelity", "fuzzy", "bad_fidelity"),
        ("label-map", "taxonomy_version", "2.0.0", "stale_taxonomy"),
        ("label-map", "native_label", "no such label", "unresolved_edge"),
        ("label-map", "fidelity", "coarsened", "edge_mismatch"),
        ("points", "label_native", "never mapped", "unresolved_label"),
    ],
)
def test_labelmap_fails_an_unresolved_id_or_a_bad_fidelity(tmp_path, table, field, value, kind):
    result = run_labelmap_gate(label_build(tmp_path, table, field, value))
    assert not result.passed and result.counts[f"failed_{kind}"] >= 1, result.counts


def test_labelmap_fails_without_a_label_map_config(tmp_path):
    write_shard(tmp_path / "b", "points", "train", [{"source_id": "srcA", "label_native": "x"}])
    assert run_labelmap_gate(tmp_path / "b").counts["failed_no_label_map"] == 1


def an_edge_into(namespace):
    """``(crosswalk id, native label, column, bare id)`` of an exact one-target edge of taxonomy
    3.0.0 whose target is a node of ``namespace`` (node keys are ``<namespace>:<id>``)."""
    release = load_release("3.0.0")
    for key, edge in release["edges"].items():
        targets = edge.get("targets") or {}
        if edge["fidelity"] == "exact" and len(targets) == 1:
            ((axis, ident),) = targets.items()
            if f"{namespace}:{ident}" in release["nodes"] and ident not in release["nodes"]:
                cid, native = key.split(":", 1)
                return cid, native, f"{axis}_node_id", ident
    raise AssertionError(f"no exact one-target edge into {namespace} in taxonomy 3.0.0")


def bare_id_build(tmp_path, cid, native, column, ident, table="boxes"):
    row = {"source_id": "srcA", "taxon_node_id": None, "form_node_id": None}
    row |= {"condition_node_id": None, column: ident, "fidelity": "exact"}
    row |= {"taxonomy_version": LIVE_TAXONOMY}
    build = tmp_path / "bare"
    mapped = {**row, "crosswalk_id": cid, "native_label": native, "mapped": True}
    write_shard(build, "label-map", "train", [mapped])
    write_shard(build, table, "train", [{**row, "label_native": native}])
    return build


@pytest.mark.parametrize("namespace", ["rs-taxa-v1", "rs-fauna-v1"])
def test_labelmap_resolves_a_bare_id_in_every_taxonomy_namespace(tmp_path, namespace):
    cid, native, column, ident = an_edge_into(namespace)
    result = run_labelmap_gate(bare_id_build(tmp_path, cid, native, column, ident))
    assert result.passed, result.failures
    assert result.counts["boxes_rows"] == 1 and result.counts["ambiguous_bare_ids"] == 0


def test_labelmap_still_fails_a_bare_id_no_namespace_has(tmp_path):
    result = run_labelmap_gate(label_build(tmp_path, "points", "taxon_node_id", "A0"))
    assert not result.passed and result.counts["failed_unknown_node"] == 1, result.counts


def test_labelmap_counts_an_ambiguous_bare_id_without_failing(tmp_path, monkeypatch):
    release = {
        "nodes": {"ns-a:X1": {}, "ns-b:X1": {}, "ns-a:X2": {}},
        "edges": {"cw:lbl": {"fidelity": "exact", "targets": {"taxon": "X1"}}},
    }
    monkeypatch.setattr("marinedata.gates.labelmap.load_release", lambda *_: release)
    build = bare_id_build(tmp_path, "cw", "lbl", "taxon_node_id", "X1")
    result = run_labelmap_gate(build, version=LIVE_TAXONOMY)
    assert result.passed, result.failures
    assert result.counts["ambiguous_bare_ids"] == 1


def vocab_labels(source_id):
    vocab = Path(__file__).resolve().parents[1] / "registry" / "taxonomy" / "vocab"
    lines = [x for x in (vocab / f"{source_id}.tsv").read_text().splitlines() if x[:1] not in "#"]
    return [r["label"] for r in csv.DictReader(lines, delimiter="\t")]


def test_labelmap_passes_the_four_box_sources_run_4_failed_on(tmp_path):
    """Run 4: 330 findings, all from roboflow-aquarium, uiis, uiis10k and usis10k (bare rs-taxa /
    NT_* ids, and no label-map edge because their loaders declare no crosswalk)."""
    registry = Registry.load()
    sources = ("roboflow-aquarium", "uiis", "uiis10k", "usis10k")
    seen = [{"source_id": s, "label_native": n} for s in sources for n in vocab_labels(s)]
    assert len(seen) == 31
    map_rows = label_map_rows(registry, sources, LIVE_TAXONOMY, seen)
    mapped = {(r["source_id"], r["native_label"]): r["fidelity"] for r in map_rows if r["mapped"]}
    assert {(r["source_id"], r["label_native"]) for r in seen} <= set(mapped)
    boxes = []
    for r in seen:
        resolve = axes_resolver_for(registry, BOX_SOURCES[r["source_id"]].crosswalk_id)
        got = resolve(r["label_native"])
        fid = mapped[(r["source_id"], r["label_native"])]
        boxes.append(
            {**r, "taxon_node_id": got.taxon, "form_node_id": got.form, "fidelity": fid}
            | {"condition_node_id": got.condition, "taxonomy_version": LIVE_TAXONOMY}
        )
    write_shard(tmp_path / "b", "label-map", "train", map_rows)
    write_shard(tmp_path / "b", "boxes", "train", boxes)
    result = run_labelmap_gate(tmp_path / "b")
    assert result.passed, result.failures
    assert result.counts["boxes_rows"] == 31
    assert {"A137039", "A146419", "NT_DIVER"} <= {r["taxon_node_id"] for r in boxes}


# G8 privacy, images config only


class FakeDetector:
    name, version = "fake", "1"

    def __init__(self, flagged=()):
        self.flagged, self.seen = set(flagged), []

    def detect(self, image):
        self.seen.append(image)
        return [Detection("face", 0.97)] if image in self.flagged else []


def jpeg(colour):
    out = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(out, "JPEG")
    return out.getvalue()


def image_row(data):
    sha = hashlib.sha256(data).hexdigest()
    return {"image_sha256": sha, "source_id": "srcA", "image": {"bytes": data, "path": None}}


def test_privacy_scans_the_images_config_only_and_lists_the_detected_sha256(tmp_path):
    a, b, frozen = jpeg((10, 10, 10)), jpeg((200, 30, 30)), jpeg((30, 30, 200))
    build = tmp_path / "priv"
    write_shard(build, "images", "train", [image_row(a), image_row(b)])
    write_shard(build, "coralseg", "train", [image_row(frozen)])  # frozen v1.0 config
    detector = FakeDetector(flagged={b, frozen})
    result = privacy_v11.run_privacy_v11_gate(build, detector=detector)
    sha = hashlib.sha256(b).hexdigest()
    assert not result.passed and frozen not in detector.seen
    assert result.counts["scope"] == "images" and result.counts["images_scanned"] == 2
    assert result.counts["detected_image_sha256"] == [sha]
    assert any(sha in line and "face_detected" in line for line in result.failures)


def test_privacy_passes_a_clean_images_config(tmp_path):
    build = tmp_path / "priv"
    write_shard(build, "images", "train", [image_row(jpeg((1, 2, 3)))])
    result = privacy_v11.run_privacy_v11_gate(build, detector=FakeDetector())
    assert result.passed, result.failures
    assert result.counts["detector"] == "fake 1"


def test_privacy_loads_yunet_only_through_the_sha256_verified_path(monkeypatch):
    asked = []

    class FakeYuNet:
        def __init__(self, path):
            self.path = path

    def verified(path=None):
        asked.append(path)
        return Path("pinned.onnx")

    monkeypatch.setattr(privacy_v11, "verified_model_path", verified)
    monkeypatch.setattr(privacy_v11, "YuNetDetector", FakeYuNet)
    detector = privacy_v11.load_yunet_detector("given.onnx")
    assert asked == ["given.onnx"] and detector.path == Path("pinned.onnx")


def test_privacy_fails_closed_when_the_model_cannot_be_verified(tmp_path, monkeypatch):
    def refuse(path=None):
        raise YuNetModelError("sha256 mismatch")

    monkeypatch.setattr(privacy_v11, "verified_model_path", refuse)
    build = tmp_path / "priv"
    write_shard(build, "images", "train", [image_row(jpeg((1, 2, 3)))])
    result = privacy_v11.run_privacy_v11_gate(build)
    assert not result.passed and result.counts["detector"] == "unavailable"


# Run 6: GATES.json is uploaded, so it names no source that stays out of the open release


def plant(note):
    """A passing decon whose entry carries ``note`` (what a gate that echoes an id would write)."""

    def run(build, options):
        return finish("decon", {"note": note}, [])

    return run


def test_the_merged_gates_json_is_rescanned_and_a_planted_id_fails_leak(
    tmp_path, monkeypatch, capsys
):
    build, v1 = build_release(tmp_path), write_v1(tmp_path)
    stub_gates(monkeypatch)
    monkeypatch.setitem(RUNNERS, "decon", plant("ncsrc"))
    call = ["release", "gate", str(build), "--v1-dir", str(v1)]
    assert main(call) == 0  # ncsrc is not a non-open source here: there is nothing to find
    assert entries(build)["leak"]["passed"] is True
    capsys.readouterr()
    (build / "GATES.json").unlink()  # G3 must not see the planted id in a file of an earlier call
    assert main([*call, "--nc-id", "ncsrc"]) == 1
    leak, shown = entries(build)["leak"], capsys.readouterr()
    tag = id_tags("ncsrc", {"ncsrc"})[0]
    assert leak["passed"] is False and leak["counts"]["gates_json_leaks"] == 1
    assert leak["failures"][0] == f"GATES.json leak: {tag} in the open upload set"
    assert "ncsrc" not in json.dumps(leak) + shown.out + shown.err
    seed_v10(build)
    assert any(p.startswith("gate leak: failed") for p in publish_verdict(build)[1])


def test_open_decon_publishes_labels_for_non_open_benchmarks_and_the_nc_flavour_keeps_ids(
    tmp_path, capsys
):
    build = build_release(tmp_path)
    hidden, bench = tmp_path / "hidden.json", tmp_path / "ncbench.txt"
    hidden.write_text(json.dumps({"sha256": [hashlib.sha256(b"other").hexdigest()]}))
    bench.write_text(hashlib.sha256(b"x").hexdigest() + "\n" + SHAS[2] + "\n")
    call = ["release", "gate", str(build), "--only", "decon", "--hidden-test", str(hidden)]
    call += ["--benchmark", str(bench), "--nc-id", "ncbench"]
    assert main(call) == 1  # a benchmark overlap with no registry entry fails closed
    decon, shown = entries(build)["decon"], capsys.readouterr()
    tag = id_tags("ncbench", {"ncbench"})[0]
    assert decon["passed"] is False and tag in json.dumps(decon)
    assert "ncbench" not in json.dumps(decon) + shown.out + shown.err
    assert "leak" not in entries(build)  # the rescan found nothing to report
    (build / "GATES.json").unlink()
    assert main([*call, "--flavour", "nc"]) == 1
    assert "ncbench" in json.dumps(entries(build)["decon"])  # the nc GATES.json stays private
