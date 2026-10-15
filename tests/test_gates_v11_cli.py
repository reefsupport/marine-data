"""v1.1.0 gate CLI/orchestrator (G1-G9 into GATES.json), G8 images-scoped privacy and G9 labelmap.

The gate bodies of G1-G7 have their own fixtures in test_gates_v11.py. Here the CLI runs join,
leak and splits_v11 for real on that synthetic build; frozen, bytes_once, masks, decon, privacy and
labelmap are stubbed in the CLI tests (G8 and G9 have their own real tests below). No model weights
are downloaded: the detector is a fake, and the YuNet loader is mocked."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from test_gates_v11 import build_release, write_report, write_shard, write_v1

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
from marinedata.gates.privacy_detectors import Detection
from marinedata.gates.v11_runner import RUNNERS, V11_ALL, V11Options, run_v11_gate
from marinedata.gates.yunet_model import YuNetModelError

STUBBED = ("frozen", "bytes_once", "masks", "decon", "privacy", "labelmap")


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
    common |= {"condition_node_id": None, "fidelity": "exact", "taxonomy_version": "3.0.0"}
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
