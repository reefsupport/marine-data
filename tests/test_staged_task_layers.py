"""``--v11`` builds boxes and instances from the staged trees (part B of the overlay)."""

from __future__ import annotations

import contextlib
import hashlib
import json
from itertools import takewhile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_boxes_instance import _doc
from test_boxes_sonar import _meta, _pq
from test_release import _two_source_fixture, _write_map

from marinedata.registry import Registry
from marinedata.release import build_release
from marinedata.task_layers import boxes_table as bt
from marinedata.task_layers import staged_annotations as SA
from marinedata.task_layers.configs import build_all_configs, resolve_tasklabels_root
from marinedata.task_layers.sources.boxes_yolo import yolo_names
from marinedata.task_layers.sources.masks_instance import INSTANCE_SOURCES
from marinedata.task_layers.v11_tables import write_v11_report

ROOT = Path(__file__).resolve().parents[1]
DATA, LOCKS = ROOT / "data", ROOT / "releases" / "marine-data" / "v1.1.0"
AQ, UIIS, DEBRIS = "roboflow-aquarium", "uiis", "synthetic-seabed-debris"
FINAL = "synthetic_seabed_dataset_v1_0_zip_synthetic_seabed_dataset_final_img_0000"
BACKGROUND = FINAL.replace("final", "backgrounds")
AQUARIUM_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _lay(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    """A staged tree as the VM caches it (no CHECKSUMS.sha256); returns ``{rel: sha256}``."""
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}


def _aquarium(root: Path, *, labels: bool = True) -> dict[str, str]:
    """yolo: two labelled images (3 boxes), one with an empty label file; no docs/data.yaml."""
    text = {
        "aq_train_a": "0 0.5 0.5 0.2 0.2\n2 0.25 0.25 0.1 0.1\n",
        "aq_val_b": "4 0.5 0.5 0.4 0.4\n",
    }
    files, meta = {"INGEST.json": b"{}", "LICENSE": b"CC BY 4.0\n"}, []
    for stem, split in (("aq_train_a", "train"), ("aq_val_b", "val"), ("aq_test_c", "test")):
        files[f"images/{stem}.jpg"] = f"jpeg {stem}".encode()
        files[f"labels/files/{stem}.txt"] = text.get(stem, "").encode() if labels else b""
        refs = f"['labels/files/{stem}.txt']"
        meta.append({"stem": stem, "width": "100", "height": "50", "label_refs": refs,
                     "split_hint": split, "upstream_split": split})  # fmt: skip
    files["metadata.parquet"] = _pq(meta)
    return _lay(root, files)


def _uiis(root: Path) -> dict[str, str]:
    """coco-instances: the boxes and the instances come from the same two documents (3 each)."""
    train = _doc(["a.jpg", "b.jpg"], [(1, 1, [10, 5, 20, 10]), (1, 1, [0, 0, 5, 5])])
    val = _doc(["c.jpg"], [(1, 1, [0, 0, 50, 25])])
    return _lay(root, {
        "labels/files/annotations_train.json": train, "labels/files/annotations_val.json": val,
        "images/train_a.jpg": b"jpeg train_a", "images/val_c.jpg": b"jpeg val_c",
        "INGEST.json": b"{}", "LICENSE": b"Apache-2.0\n",
    })  # fmt: skip


def _debris(root: Path) -> dict[str, str]:
    """synthetic-json: one document for the staged final composite, one for none (orphan)."""
    doc = {"image": "img_0000.png", "source": "gen_full_a", "fauna": [{"box": [0, 0, 77, 77]}],
           "placed": [{"class": "Tire", "box": [0, 0, 384, 384]}]}  # fmt: skip
    return _lay(root, {
        f"images/{FINAL}.png": b"png final", f"images/{BACKGROUND}.png": b"png background",
        "labels/files/unresolved-0.json": json.dumps(doc).encode(),
        "labels/files/unresolved-1.json": json.dumps({**doc, "image": "img_0001.png"}).encode(),
        "metadata.parquet": _meta([FINAL, BACKGROUND], 768, 768),
        "INGEST.json": b"{}", "LICENSE": b"CC BY-NC 4.0\n",
    })  # fmt: skip


@pytest.fixture
def trees(tmp_path):
    roots = {sid: tmp_path / "cache" / sid for sid in (AQ, UIIS, DEBRIS)}
    shas = {AQ: _aquarium(roots[AQ]), UIIS: _uiis(roots[UIIS]), DEBRIS: _debris(roots[DEBRIS])}
    return SimpleNamespace(roots=roots, shas=shas)


def test_aquarium_names_are_vendored_with_their_recorded_sha256(registry):
    spec = bt.BOX_SOURCES[AQ]
    assert spec.names_rel is None and spec.names_key == bt.AQUARIUM_NAMES_KEY
    raw = (Path(registry.root) / spec.names_key).read_bytes()
    head = list(takewhile(lambda line: line.startswith(b"#"), raw.split(b"\n")))
    body = raw[sum(len(line) + 1 for line in head) :]
    recorded = next(line.split()[2] for line in head if line.startswith(b"# sha256:"))
    assert hashlib.sha256(body).hexdigest() == recorded.decode() and len(body) == 133
    assert yolo_names(body.decode()) == dict(enumerate(AQUARIUM_NAMES))


def test_producers_are_exactly_the_locked_box_and_instance_sources(registry):
    locked: dict[str, str] = {}
    for flavour in ("open", "nc"):
        lock = yaml.safe_load((LOCKS / flavour / "release.lock.yaml").read_text())
        locked |= {s["id"]: s["version"] for s in lock["sources"]}
    producers = SA.default_producers()
    by_table = {t: [p for p in producers if p.table == t] for t in ("boxes", "masks", "points")}
    assert {p.source_id for p in by_table["boxes"]} == set(bt.BOX_SOURCES) & set(locked)
    assert {p.source_id for p in by_table["masks"]} == set(INSTANCE_SOURCES) & set(locked)
    assert {p.source_id for p in by_table["points"]} == {SA.SEAVIEW}
    for p in producers:  # the producer's pin is the locked version
        assert p.version(registry) == locked.get(p.source_id, p.version(registry))
    assert {p.source_id: p.version(registry) for p in by_table["boxes"]} == {
        s: locked[s] for s in (AQ, UIIS, "uiis10k", "usis10k", DEBRIS)
    }


def test_overlay_adds_boxes_and_instances_keyed_by_image_sha256_and_touches_nothing_else(
    registry, trees
):
    base = resolve_tasklabels_root(registry)
    plain = build_all_configs(registry, base)
    with SA.staged_overlay(registry, base, trees.roots) as overlay:
        staged = build_all_configs(registry, overlay.root)
        layers = {(layer.table, layer.source_id): layer for layer in overlay.layers}
        written = sorted(p.relative_to(overlay.root).as_posix() for p in
                         (overlay.root / SA.ANNOTATIONS_DIR).rglob("*.parquet") if p.is_file()
                         and p.parts[-3] in ("boxes", "masks") and not p.is_symlink())  # fmt: skip
    assert not overlay.root.exists()
    kept = lambda cfgs: {c: r.rows for c, r in cfgs.items() if c not in ("boxes", "instances")}  # noqa: E731
    assert kept(staged) == kept(plain)  # bleaching, semseg, points, ... as without the overlay
    assert {k: (v.rows, v.images, v.reused) for k, v in layers.items()} == {
        ("boxes", AQ): (3, 2, False), ("boxes", UIIS): (3, 2, False),
        ("boxes", DEBRIS): (2, 1, False), ("masks", UIIS): (3, 2, False),
    }  # fmt: skip
    assert written == [
        f"_annotations/boxes/{AQ}/zip-5d30fed3dd5a.parquet",
        f"_annotations/boxes/{DEBRIS}/1.0.0.parquet",
        f"_annotations/boxes/{UIIS}/rev-44ca5db46599.parquet",
        f"_annotations/masks/{UIIS}/rev-44ca5db46599.parquet",
    ]
    boxes = {s: [r for r in staged["boxes"].rows if r["source_id"] == s] for s in trees.roots}
    assert tuple(r for r in staged["boxes"].rows if r["source_id"] not in trees.roots) == (
        plain["boxes"].rows
    )
    sha = lambda sid, *stems: {trees.shas[sid][f"images/{s}.jpg"] for s in stems}  # noqa: E731
    assert {r["image_sha256"] for r in boxes[AQ]} == sha(AQ, "aq_train_a", "aq_val_b")
    assert sorted(r["label_native"] for r in boxes[AQ]) == ["fish", "penguin", "shark"]
    assert {r["image_sha256"] for r in boxes[UIIS]} == sha(UIIS, "train_a", "val_c")
    assert {r["image_sha256"] for r in boxes[DEBRIS]} == {trees.shas[DEBRIS][f"images/{FINAL}.png"]}
    assert {r["licence_class"] for r in boxes[DEBRIS]} == {"restricted-nc"}
    assert (
        layers[("boxes", DEBRIS)].detail["counts"]["orphan"] == 1
    )  # unresolved-1 has no composite
    instances = [r for r in staged["instances"].rows if r["source_id"] == UIIS]
    assert sorted(r["instance_id"] for r in instances) == [1, 1, 2]
    assert {r["image_sha256"] for r in instances} == sha(UIIS, "train_a", "val_c")
    assert {r["mask_ref"] for r in instances} == {None} and layers[("masks", UIIS)].detail[
        "config"
    ] == "instances"


def test_a_tree_that_yields_no_box_fails_closed_and_says_why(registry, tmp_path):
    root = tmp_path / AQ
    _aquarium(root, labels=False)
    with pytest.raises(SA.StagedLayerError, match=rf"boxes/{AQ} .*no box was read"):  # noqa: SIM117
        with SA.staged_overlay(registry, DATA, {AQ: root}):
            pytest.fail("the overlay must not be handed out empty")


def test_a_missing_document_names_its_key(registry, tmp_path):
    root = tmp_path / UIIS
    _uiis(root)
    (root / "labels/files/annotations_val.json").unlink()
    with pytest.raises(SA.StagedLayerError, match=r"annotations_val\.json"):  # noqa: SIM117
        with SA.staged_overlay(registry, DATA, {UIIS: root}):
            pytest.fail("the overlay must not be handed out without the val document")


def test_layer_tallies_land_in_v11_tables_json(registry, trees, tmp_path):
    with SA.staged_overlay(registry, DATA, trees.roots) as overlay:
        records = [SA.layer_record(layer) for layer in overlay.layers]
        lines = [SA.layer_line(layer) for layer in overlay.layers]
    report = json.loads(write_v11_report({}, tmp_path, staged_layers=records).read_text())
    assert report["staged_layers"] == records and len(records) == 4
    tally = {(r["table"], r["source_id"]): (r["rows"], r["images"], r["reused"]) for r in records}
    assert tally == {
        ("boxes", AQ): (3, 2, False), ("boxes", UIIS): (3, 2, False),
        ("boxes", DEBRIS): (2, 1, False), ("masks", UIIS): (3, 2, False),
    }  # fmt: skip
    assert f"boxes/{AQ} zip-5d30fed3dd5a: 3 rows, 2 images (built from the staged tree)" in lines
    assert "staged_layers" not in json.loads(write_v11_report({}, tmp_path).read_text())


def test_build_release_writes_the_layer_tallies_of_a_v11_build(monkeypatch, tmp_path):
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(split_map, {"shared/dup": "val", "src-a/only": "train", "src-b/only": "test"})
    layer = SA.StagedLayer("boxes", AQ, "zip-5d30fed3dd5a", 3, 2, False, {"counts": {"read": 3}})

    @contextlib.contextmanager
    def fake_overlay(reg, base_dir, admitted, *, digest):
        yield SimpleNamespace(root=tmp_path / "the-overlay", layers=(layer,))

    monkeypatch.setattr(SA, "staged_overlay", fake_overlay)
    monkeypatch.setattr("marinedata.task_layers.configs.build_all_configs", lambda *a, **k: {})
    args = dict(release="r7k", split_map=split_map, roots=roots, tasks="v2", tasklabels_root=DATA)
    plain = build_release(registry, out_dir=tmp_path / "plain", **args)
    assert plain.staged_layers == ()
    built = build_release(registry, out_dir=tmp_path / "v11", v11=True, allow_unsplit=True, **args)
    report = json.loads((built.out_dir / "task_layers" / "V11_TABLES.json").read_text())
    assert built.staged_layers == (layer,) and report["staged_layers"] == [SA.layer_record(layer)]
