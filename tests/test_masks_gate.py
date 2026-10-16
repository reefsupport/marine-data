"""Masks gate (Phase A, P1.5): one fixture per failure mode plus a passing set (images <= 64x64)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from marinedata.cli import main
from marinedata.gates import GATES_FILE, MAX_FAILURES
from marinedata.gates.masks import run_masks_gate
from marinedata.hf_export import load_mask_geo
from marinedata.hf_parquet import ConfigSpec, EmbeddedBlob, ExportRow, write_shard
from marinedata.task_layers import mask_configs as mc
from marinedata.task_layers import mask_encode as me

SEM = ConfigSpec("coral-masks", mc.SEMANTIC_COLUMNS, "fixture")
INST = ConfigSpec("instance-masks", mc.INSTANCE_COLUMNS, "fixture")


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _index_png(arr) -> bytes:  # a palette PNG, the way staged semantic masks arrive
    buf = io.BytesIO()
    Image.frombytes("P", (arr.shape[1], arr.shape[0]), arr.tobytes()).save(buf, format="PNG")
    return buf.getvalue()


def _photo(w=8, h=6, orientation=None) -> bytes:
    buf = io.BytesIO()
    im = Image.fromarray(np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3))
    if orientation is None:
        im.save(buf, format="PNG")
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _sem_mask(w=8, h=6, fill=None):
    if fill is not None:
        return np.full((h, w), fill, np.uint8)
    m = np.full((h, w), 1, np.uint8)
    m[:, w // 2 :] = 2
    m[0, 0] = 255
    return m


def _inst_mask(w=8, h=6, ids=(1, 2)):
    m = np.zeros((h, w), np.uint16)
    for k, i in enumerate(ids):
        m[k : k + 2, 2 * k : 2 * k + 2] = i
    return m


def _row(i, image, mask, *, kind="semantic", source="suim", ids=(1, 2), exif=False) -> ExportRow:
    sha = f"{i:064x}"
    w, h = Image.open(io.BytesIO(image)).size
    values = {
        "image_sha256": sha, "source": source, "licence": "CC-BY-4.0", "annotation_licence": None,
        "attribution": None, "split": "train", "split_group": None, "annotator_type": "human",
        "width": w, "height": h, "lat": None, "lon": None, "exif_transposed": exif,
    }  # fmt: skip
    entry = {"label_native": None, "taxon_node": None, "coarse": None}
    if kind == "semantic":
        values["class_map"] = [{"id": c, **entry} for c in ids]
    else:
        values["instances"] = [{"id": c, **entry, "bbox_xyxy_norm": None} for c in ids]
    blobs = (
        EmbeddedBlob("image", f"{sha}.png", len(image), lambda b=image: b),
        EmbeddedBlob("mask", f"{source}/{sha}.png", len(mask), lambda b=mask: b),
    )
    return ExportRow(values=values, blobs=blobs)


def _build(root, semantic=(), instance=()):
    if semantic:
        write_shard(root / "coral-masks" / "train-00000-of-00001.parquet", SEM, list(semantic))
    if instance:
        write_shard(root / "instance-masks" / "train-00000-of-00001.parquet", INST, list(instance))
    return root


def _good_semantic(n=3):
    return [_row(i, _photo(), _png(_sem_mask())) for i in range(n)]


def _good_instance(n=2):
    return [
        _row(100 + i, _photo(), _png(_inst_mask()), kind="instance", source="uiis10k")
        for i in range(n)
    ]


def _failed(result) -> set[str]:
    return {k.removeprefix("failed_") for k in result.counts if k.startswith("failed_")}


def test_passing_set(tmp_path):
    result = run_masks_gate(_build(tmp_path, _good_semantic(), _good_instance()))
    assert result.passed and result.failures == []
    assert (result.counts["rows"], result.counts["semantic_rows"]) == (5, 3)
    assert result.counts["empty_by_source"]["suim"] == {"rows": 3, "empty": 0, "rate": 0.0}


def test_size_mismatch(tmp_path):
    bad = _row(1, _photo(8, 8), _png(_sem_mask(8, 6)))
    result = run_masks_gate(_build(tmp_path, [bad, *_good_semantic()]))
    assert not result.passed and _failed(result) == {"size_mismatch"}
    assert "size_mismatch: mask 8x6 != image 8x8" in result.failures[0]


def test_exif_orientation_with_mask_on_the_transposed_pixels(tmp_path):
    # stored 8x6, orientation 6 -> displayed 6x8; the mask was drawn on the displayed pixels
    bad = _row(1, _photo(8, 6, orientation=6), _png(_sem_mask(6, 8)))
    result = run_masks_gate(_build(tmp_path, [bad]))
    assert {"exif_orientation", "size_mismatch"} <= _failed(result)
    assert "drawn on the transposed pixels" in " ".join(result.failures)


def test_mask_id_outside_class_map(tmp_path):
    mask = _sem_mask()
    mask[2, 2] = 9
    result = run_masks_gate(_build(tmp_path, [_row(1, _photo(), _png(mask))]))
    assert _failed(result) == {"unknown_class_id"} and "[9]" in result.failures[0]


def test_ignore_index_listed_as_a_class(tmp_path):
    result = run_masks_gate(
        _build(tmp_path, [_row(1, _photo(), _png(_sem_mask()), ids=(1, 2, 255))])
    )
    assert _failed(result) == {"ignore_as_class"}


def test_semantic_mask_must_be_uint8(tmp_path):
    row = _row(1, _photo(), _png(_sem_mask().astype(np.uint16)))
    assert _failed(run_masks_gate(_build(tmp_path, [row]))) == {"mask_dtype"}


def test_instance_ids_must_be_one_to_n(tmp_path):
    row = _row(1, _photo(), _png(_inst_mask(ids=(1, 3))), kind="instance", ids=(1, 3))
    assert _failed(run_masks_gate(_build(tmp_path, instance=[row]))) == {"instance_ids"}


def test_instance_mask_paints_an_id_without_an_entry(tmp_path):
    row = _row(1, _photo(), _png(_inst_mask(ids=(1, 2))), kind="instance", ids=(1,))
    result = run_masks_gate(_build(tmp_path, instance=[row]))
    assert _failed(result) == {"instance_ids"} and "no instances[] entry" in result.failures[0]


def test_instance_mask_must_be_uint16(tmp_path):
    row = _row(1, _photo(), _png(_inst_mask().astype(np.uint8)), kind="instance")
    assert _failed(run_masks_gate(_build(tmp_path, instance=[row]))) == {"mask_dtype"}


def test_empty_mask_rate_per_source(tmp_path):
    empty = _png(_sem_mask(fill=255))
    rows = [_row(i, _photo(), empty if i < 2 else _png(_sem_mask())) for i in range(10)]
    build = _build(tmp_path, rows)
    result = run_masks_gate(build)
    assert not result.passed and _failed(result) == {"empty_rate"}
    assert result.counts["empty_by_source"]["suim"] == {"rows": 10, "empty": 2, "rate": 0.2}
    assert run_masks_gate(build, empty_threshold=0.25).passed


def test_a_build_without_mask_rows_fails(tmp_path):
    result = run_masks_gate(tmp_path)
    assert not result.passed and "no mask rows found" in result.failures[0]


def test_failures_are_capped_but_counted(tmp_path):
    rows = [_row(i, _photo(8, 8), _png(_sem_mask(8, 6))) for i in range(MAX_FAILURES + 10)]
    result = run_masks_gate(_build(tmp_path, rows))
    assert len(result.failures) == MAX_FAILURES
    assert (
        result.counts["failed_size_mismatch"]
        == result.counts["failures_total"]
        == MAX_FAILURES + 10
    )


def test_cli_gate_merges_into_gates_json(tmp_path, capsys):
    build = _build(tmp_path / "ok", _good_semantic())
    (build / GATES_FILE).write_text(json.dumps({"privacy": {"passed": True}}))
    assert main(["release", "gate", "--masks", str(build)]) == 0
    gates = json.loads((build / GATES_FILE).read_text())
    assert gates["privacy"] == {"passed": True} and gates["masks"]["passed"] is True
    bad = _build(tmp_path / "bad", [_row(1, _photo(8, 8), _png(_sem_mask(8, 6)))])
    assert main(["release", "gate", "--masks", str(bad)]) == 1
    assert json.loads((bad / GATES_FILE).read_text())["masks"]["passed"] is False
    assert "size_mismatch" in capsys.readouterr().err


def test_fit_never_resizes_unless_allowed():
    small = _index_png(np.full((4, 4), 1, np.uint8))
    kwargs = {"encoding": "png-indexed", "ignore_value": None, "size": (8, 8)}
    with pytest.raises(me.MaskEncodingError, match="refusing to resize"):
        me.semantic_png(small, **kwargs)
    with me.allow_resize():
        out = me.semantic_png(small, **kwargs)
    assert Image.open(io.BytesIO(out)).size == (8, 8)
    with pytest.raises(me.MaskEncodingError):  # the escape does not outlive the block
        me.semantic_png(small, **kwargs)


def _record(sha):
    classes = [{"id": c, "label_native": f"c{c}"} for c in (1, 2)]
    return {"image_sha256": sha, "source_id": "coralscapes", "mask_key": "m",
            "hf_class_map": json.dumps(classes)}  # fmt: skip


def _semantic_row(path, mask: bytes, sha: str):
    return mc.semantic_row(_record(sha), mc.ImageInfo(path, "train"), lambda key: mask, {}, None)


def test_exporter_transposes_by_exif_and_records_it(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(_photo(8, 6, orientation=6))
    row = _semantic_row(path, _index_png(_sem_mask(6, 8)), "a" * 64)  # mask on the displayed pixels
    assert row.values["exif_transposed"] is True
    assert (row.values["width"], row.values["height"]) == (6, 8)
    image = Image.open(io.BytesIO({b.column: b.read() for b in row.blobs}["image"]))
    assert image.size == (6, 8) and image.getexif().get(0x0112, 1) == 1
    write_shard(tmp_path / "out" / "coral-masks" / "train-00000-of-00001.parquet", SEM, [row])
    assert run_masks_gate(tmp_path / "out").passed


def test_exporter_keeps_upright_images_byte_identical(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(_photo(8, 6))
    row = _semantic_row(path, _index_png(_sem_mask()), "b" * 64)
    assert row.values["exif_transposed"] is False
    assert {b.column: b.read() for b in row.blobs}["image"] == path.read_bytes()


def test_exporter_refuses_a_mask_on_the_stored_pixels_of_a_rotated_image(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(_photo(8, 6, orientation=6))
    row = _semantic_row(path, _index_png(_sem_mask(8, 6)), "c" * 64)
    with pytest.raises(me.MaskEncodingError, match="refusing to resize"):
        [b.read() for b in row.blobs]


def test_load_mask_geo_reads_a_csv(tmp_path):
    path = tmp_path / "geo.csv"
    path.write_text("image_sha256,lat,lon\nabc,1.5,2.5\n")
    assert load_mask_geo(path)["abc"] == (1.5, 2.5)


def test_v1_dir_masks_count_when_the_build_has_none(tmp_path):
    build, v1 = tmp_path / "build", tmp_path / "v1"
    build.mkdir()
    _build(v1, _good_semantic(), _good_instance())
    rehearsal = run_masks_gate(build)
    assert not rehearsal.passed and "no mask rows found" in rehearsal.failures[0]
    result = run_masks_gate(build, v1_dir=v1)
    assert result.passed and result.counts["rows"] == 5 and result.counts["v1_dir_scanned"] == 1
    neither = run_masks_gate(build, v1_dir=tmp_path / "empty")
    assert not neither.passed and "no mask rows found" in neither.failures[0]


def test_v1_dir_is_not_scanned_when_the_build_has_mask_rows(tmp_path):
    build, v1 = tmp_path / "build", tmp_path / "v1"
    _build(build, _good_semantic())
    _build(v1, _good_semantic(), _good_instance())
    result = run_masks_gate(build, v1_dir=v1)
    assert result.passed and result.counts["rows"] == 3 and "v1_dir_scanned" not in result.counts


def test_runner_passes_v1_dir_to_the_masks_gate(tmp_path):
    from marinedata.gates.runner import GateOptions, run_gate

    build = tmp_path / "build"
    build.mkdir()
    v1 = _build(tmp_path / "v1", _good_semantic())
    assert run_gate("masks", build, GateOptions(v1_dir=v1)).passed
