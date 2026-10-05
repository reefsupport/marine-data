"""WP-U15 registry patch: the applier (text-level, idempotent) and the applied registry."""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from marinedata.ingest_source import IngestSpec
from marinedata.ingest_subset import check_spec
from marinedata.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "apply_label_kind_patch", ROOT / "scripts/apply_label_kind_patch.py"
)
patcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patcher)


def _tiny_root(tmp_path: Path) -> Path:
    (tmp_path / "registry/ingest-specs").mkdir(parents=True)
    (tmp_path / "registry/sources").mkdir()
    specs = tmp_path / "registry/ingest-specs"
    (specs / "a.yaml").write_text("# keep me\nid: a\nlicense: x\n")
    (specs / "large-scale-fish.yaml").write_text(
        "id: large-scale-fish\nmeasured:\n  items: 3\n  label_type: old\n  method: m\nnotes: n\n"
    )
    (tmp_path / "registry/sources/s.yaml").write_text(
        "sources:\n- id: a\n  tags:\n  - one\n- id: other\n  tags: [x]\n"
    )
    (tmp_path / "registry/sources/fish.yaml").write_text(
        "sources:\n  - id: large-scale-fish\n    tags: [out-of-water]\n    annotations:\n      - kind: none\n        supervises: []\n  - id: z\n    tags: [q]\n"
    )
    patch = {"patches": [
        {"id": "a", "spec_file": "registry/ingest-specs/a.yaml", "spec_label_type": "image-level labels, 2 classes",
         "source_file": "registry/sources/s.yaml", "source_annotations": [{"kind": "image-label", "supervises": []}]},
        {"id": "large-scale-fish", "spec_file": "registry/ingest-specs/large-scale-fish.yaml", "spec_label_type": "classes plus masks",
         "source_file": "registry/sources/fish.yaml", "source_annotations": [{"kind": "dense-mask", "supervises": []}]},
    ]}  # fmt: skip
    (tmp_path / "patch.yaml").write_text(yaml.safe_dump(patch))
    return tmp_path


def test_applier_edits_text_in_place_and_is_idempotent(tmp_path):
    root = _tiny_root(tmp_path)
    assert patcher.apply(root / "patch.yaml", root) == {"specs": 2, "sources": 2}
    snapshot = {p: p.read_text() for p in root.rglob("*.yaml")}
    patcher.apply(root / "patch.yaml", root)
    assert snapshot == {p: p.read_text() for p in root.rglob("*.yaml")}
    a = (root / "registry/ingest-specs/a.yaml").read_text()
    assert yaml.safe_load(a)["measured"]["method"] and yaml.safe_load(a)["measured"]["date"]
    assert (
        a.startswith("# keep me\n")
        and yaml.safe_load(a)["measured"]["label_type"] == "image-level labels, 2 classes"
    )
    fish = yaml.safe_load((root / "registry/ingest-specs/large-scale-fish.yaml").read_text())
    assert fish["measured"]["items"] == 3 and fish["measured"]["label_type"] == "classes plus masks"
    assert fish["measured"]["quarantined"] is True and fish["measured"]["release_excluded"] is True
    srcs = {
        s["id"]: s
        for s in yaml.safe_load((root / "registry/sources/s.yaml").read_text())["sources"]
    }
    assert srcs["a"]["annotations"] == [{"kind": "image-label", "supervises": []}] and srcs[
        "other"
    ]["tags"] == ["x"]
    fish_src = {
        s["id"]: s
        for s in yaml.safe_load((root / "registry/sources/fish.yaml").read_text())["sources"]
    }
    assert fish_src["large-scale-fish"]["tags"] == [
        "out-of-water",
        "release-excluded",
        "quarantined",
    ]
    assert fish_src["large-scale-fish"]["annotations"] == [{"kind": "dense-mask", "supervises": []}]
    assert fish_src["z"]["tags"] == ["q"]


def test_applied_registry_loads_and_carries_the_triage():
    reg = Registry.load()
    specs = ROOT / "registry/ingest-specs"
    for sid in ("floating-marine-debris", "noaa-dsc-csv", "large-scale-fish"):
        measured = IngestSpec.load(specs / f"{sid}.yaml").measured
        assert measured["release_excluded"] is True and measured["release_exclusion_reason"], sid
    assert IngestSpec.load(specs / "large-scale-fish.yaml").measured["quarantined"] is True
    assert {"release-excluded", "quarantined"} <= set(reg.source("large-scale-fish").tags)
    assert IngestSpec.load(specs / "aqualoc.yaml").measured["label_type"].startswith("none")
    assert [a.kind.value for a in reg.source("aqualoc").annotations] == ["camera-pose"]
    assert "depth-map" in [a.kind.value for a in reg.source("usod10k").annotations]
    assert IngestSpec.load(specs / "euvp.yaml").measured["label_type"].startswith("paired")
    for sid in ("euvp", "lsui", "aqualoc", "noaa-dsc-csv", "large-scale-fish"):
        assert check_spec(yaml.safe_load((specs / f"{sid}.yaml").read_text())) == [], sid
