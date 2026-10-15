"""PR-B: the v1.1.0 task tables (metadata, points, bleaching, boxes, semseg, instances)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.registry import Registry
from marinedata.release.build import build_release
from marinedata.task_layers.boxes_sonar import SONAR
from marinedata.task_layers.configs.assemble import build_all_configs, write_configs
from marinedata.task_layers.configs.common import ConfigResult
from marinedata.task_layers.mask_classmap import palette_json, palette_rgb
from marinedata.task_layers.mask_encode import palette_png
from marinedata.task_layers.v11_tables import (
    V11_CONFIG_IDS,
    UnsplitImagesError,
    V11Inputs,
    _arrow_type,
    build_v11_tables,
    core_columns,
    fidelity_of,
    mark_near_dup_conflicts,
    v11_schemas,
    write_v11_report,
)

SEAVIEW, SONAR_SRC, CORAL, OPT = (
    "seaview-survey-imagery", "sss-mine-detection", "coralscapes", "reefolution",
)  # fmt: skip
A, B, C, D = (f"{i:064x}" for i in (1, 2, 3, 4))
STAMP = {"image_sha256", "source_id", "split", "taxonomy_version", "bytes_config"}


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _stage(tmp_path: Path, staged: dict[str, list[tuple[str, str]]]) -> dict[str, Path]:
    """One staged root per source with a ``metadata.parquet`` of (sha, split_group) images."""
    roots = {}
    for source_id, items in staged.items():
        root = tmp_path / "roots" / source_id
        root.mkdir(parents=True)
        rows = [
            {"image_sha256": s, "split_group": g, "license": "CC BY 4.0", "lat": 1.5, "lon": 2.5}
            for s, g in items
        ]
        pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
        roots[source_id] = root
    return roots


def _inputs(roots: dict[str, Path], **kw) -> V11Inputs:
    return V11Inputs(taxonomy_version="3.0.0", roots=roots, **kw)


def _box(sha: str, source_id: str, **kw) -> dict:
    return {
        "image_sha256": sha, "source_id": source_id, "ann_id": sha[-1], "label_native": "mine",
        "match_type": "exact", "label_status": "ok", **kw,
    }  # fmt: skip


def test_fidelity_of_match_type() -> None:
    kinds = ("exact", "broader", "narrower", "related", "unmapped", None)
    assert [fidelity_of(m) for m in kinds] == [
        "exact", "coarsened", "approximate", "approximate", "unmappable", None,
    ]  # fmt: skip


def test_metadata_row_per_image_names_the_config_holding_the_bytes(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {CORAL: [(A, "g1"), (B, "g2")], OPT: [(B, "g2"), (C, "g3")]})
    inputs = _inputs(roots, split_of={A: "train", B: "val", C: "test"})
    rows = {r["image_sha256"]: r for r in build_v11_tables(registry, {}, inputs)["metadata"].rows}
    assert set(rows) == {A, B, C}  # one row per image_sha256, B once
    assert rows[A]["bytes_config"] == "coral-masks"  # v1.0 images reference the frozen config
    assert rows[B]["bytes_config"] == "coral-masks"  # the v1.0 config that embeds it wins
    assert rows[C]["bytes_config"] == "images"
    for row in rows.values():
        assert row["taxonomy_version"] == "3.0.0" and row["split"] and row["source_id"]
        assert row["bytes_sha256"] is None and not ({"image", "bytes"} & set(row))


def test_seaview_points_one_row_per_point_with_label_id_and_fidelity(registry, tmp_path) -> None:
    base = tmp_path / "labels"
    (base / "_annotations" / "points" / SEAVIEW).mkdir(parents=True)
    pts = [
        ("a", "coral.hard", "exact", A),
        ("b", "algae.turf", "broader", A),
        ("c", None, "unmapped", B),
    ]
    rows = [
        {
            "image_sha256": sha, "source_id": SEAVIEW, "source_version": "v1",
            "ann_id": f"{SEAVIEW}:{i}", "label_native": native, "annotator_type": "human",
            "x": 0.1 * (i + 1), "y": 0.5, "taxon_node_id": node, "match_type": mt,
        }
        for i, (native, node, mt, sha) in enumerate(pts)
    ]  # fmt: skip
    pq.write_table(
        pa.Table.from_pylist(rows), base / "_annotations" / "points" / SEAVIEW / "v1.parquet"
    )
    roots = _stage(tmp_path, {SEAVIEW: [(A, "g1"), (B, "g2")]})
    out = build_all_configs(
        registry, base, None, v11=_inputs(roots, split_of={A: "train", B: "test"})
    )
    points = out["points"].rows
    assert len(points) == 3  # one row per point, the unmapped one carried
    assert [(r["taxon_node_id"], r["fidelity"]) for r in points] == [
        ("coral.hard", "exact"), ("algae.turf", "coarsened"), (None, "unmappable"),
    ]  # fmt: skip
    assert all(set(r) >= STAMP and r["bytes_config"] == "images" for r in points)
    assert [r["split"] for r in points] == ["train", "train", "test"]
    paths = write_configs(out, tmp_path / "out", v11_schemas(out))
    assert pq.read_table(paths["points"]).num_rows == 3
    assert pq.read_table(paths["metadata"]).num_rows == 2


def test_bleaching_near_dup_images_with_different_labels_are_conflict() -> None:
    def row(sha: str, cond: str, source: str = "s1") -> dict:
        return {
            "sha256": sha,
            "source_id": source,
            "canonical_condition": cond,
            "label_status": "ok",
        }

    rows = [row(A, "bleached"), row(B, "healthy", "s2"), row(C, "healthy"), row(C, "bleached")]
    out = mark_near_dup_conflicts(rows, {A: "n1", B: "n1"})  # A and B are near-dups; C stands alone
    assert [r["label_status"] for r in out] == ["conflict", "conflict", "ok", "ok"]
    assert [r["near_dup_images"] for r in out] == [2, 2, 1, 1]  # all rows kept, none dropped
    agree = mark_near_dup_conflicts(
        [row(A, "healthy"), row(B, "healthy", "s2")], {A: "n1", B: "n1"}
    )
    assert {r["label_status"] for r in agree} == {"ok"}


def test_bleaching_table_marks_conflict_and_stamps_rows(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {"noaa-pifsc-bleaching": [(A, "g1"), (B, "g2")]})
    rows = tuple(
        {"sha256": s, "source_id": "noaa-pifsc-bleaching", "canonical_condition": c,
         "match_type": "exact", "label_status": "ok"}
        for s, c in ((A, "bleached"), (B, "healthy"))
    )  # fmt: skip
    inputs = _inputs(roots, split_of={A: "train", B: "train"}, near_dup_group_of={A: "n", B: "n"})
    out = build_v11_tables(registry, {"bleaching": ConfigResult("bleaching", rows, {})}, inputs)
    assert [r["label_status"] for r in out["bleaching"].rows] == ["conflict", "conflict"]
    assert all(set(r) >= STAMP for r in out["bleaching"].rows)


def test_semseg_palette_png_is_mode_p_with_the_declared_palette() -> None:
    arr = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)
    img = Image.open(io.BytesIO(palette_png(arr)))
    assert img.mode == "P"
    assert np.array_equal(np.array(img), arr)  # class ids are untouched
    palette = img.getpalette()
    ids = (0, 1, 2, 3, 4, 255)
    assert [tuple(palette[3 * i : 3 * i + 3]) for i in ids] == [palette_rgb(i) for i in ids]
    assert len({palette_rgb(i) for i in range(5)}) == 5 and palette_rgb(255) == (0, 0, 0)


def test_semseg_rows_reference_masks_and_never_carry_bytes(registry, tmp_path) -> None:
    class_map = json.dumps([{"id": i, "label_native": f"c{i}"} for i in (0, 1)])
    row = {
        "sha256": A, "source_id": CORAL, "label_origin": "human", "mask_key": "masks/a.png",
        "class_counts": "{}", "canonical_class_counts": "{}", "label_status": "ok",
        "hf_class_map": class_map,
    }  # fmt: skip
    roots = _stage(tmp_path, {CORAL: [(A, "g1")]})
    out = build_v11_tables(
        registry,
        {"semseg": ConfigResult("semseg", (row,), {})},
        _inputs(roots, split_of={A: "train"}),
    )
    (got,) = out["semseg"].rows
    assert set(got) >= STAMP and got["split"] == "train" and got["taxonomy_version"] == "3.0.0"
    assert (got["mask_config"], got["bytes_config"], got["mask_ref"]) == (
        "coral-masks", "coral-masks", "masks/a.png",
    )  # fmt: skip
    assert json.loads(got["palette"]) == json.loads(palette_json(class_map))
    assert json.loads(got["palette"])[0] == {"id": 0, "rgb": list(palette_rgb(0))}
    assert not ({"image", "mask", "bytes"} & set(got))


def test_sonar_boxes_carry_modality_sonar(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {SONAR_SRC: [(A, "g1")]})
    row = _box(A, SONAR_SRC, attrs=json.dumps(dict(SONAR)))
    out = build_v11_tables(
        registry,
        {"boxes": ConfigResult("boxes", (row,), {})},
        _inputs(roots, split_of={A: "train"}),
    )
    (got,) = out["boxes"].rows
    assert got["modality"] == "sonar" and got["fidelity"] == "exact" and set(got) >= STAMP


def test_inherited_split_map_withheld_groups_and_admitted_roots(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {SONAR_SRC: [(A, "g1"), (B, "g2"), (C, "g3")]})
    boxes = (*(_box(s, SONAR_SRC) for s in (A, B, C)), _box(D, "synthetic-seabed-debris"))
    inputs = _inputs(
        roots, assignments={"g1": "val", "g3": "test_ood"}, withheld_groups=frozenset({"g2"})
    )
    out = build_v11_tables(registry, {"boxes": ConfigResult("boxes", boxes, {})}, inputs)
    got = {r["image_sha256"]: r["split"] for r in out["boxes"].rows}
    assert got == {A: "val", C: "test_ood"}  # g2 withheld; the nc source is outside the roots
    assert {r["image_sha256"] for r in out["metadata"].rows} == {A, C}
    assert out["metadata"].notes["report"]["withheld_images"] == 1


def test_every_table_has_the_declared_schema(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {SONAR_SRC: [(A, "g1")]})
    out = build_v11_tables(registry, {}, _inputs(roots, split_of={A: "train"}))
    paths = write_configs(out, tmp_path / "out", v11_schemas(out))
    for cid in V11_CONFIG_IDS:
        schema = pq.read_schema(paths[cid])
        assert set(schema.names) >= STAMP, cid
        for name, kind in core_columns(cid):
            want = _arrow_type(kind)
            assert name in schema.names and (want is None or schema.field(name).type == want)
    report = json.loads(write_v11_report(out, tmp_path / "out").read_text())
    assert report["taxonomy_version"] == "3.0.0" and report["tables"]["metadata"]["rows"] == 1
    assert report["tables"]["metadata"]["unsplit"] == 0


def test_v11_needs_the_v2_task_layers(registry, tmp_path) -> None:
    with pytest.raises(ValueError, match="tasks='v2'"):
        build_release(
            registry, release="v1.1.0", split_map=tmp_path / "SPLIT_MAP.json", roots={},
            out_dir=tmp_path, v11=True,
        )  # fmt: skip


def test_unsplit_images_fail_the_build_naming_the_count_and_the_first_five(registry, tmp_path):
    shas = sorted(f"{i:064x}" for i in range(10, 18))
    staged = [(s, f"g{i}") for i, s in enumerate(shas, start=10)] + [(A, "g1")]
    roots = _stage(tmp_path, {SONAR_SRC: staged})
    with pytest.raises(UnsplitImagesError) as err:
        build_v11_tables(registry, {}, _inputs(roots, assignments={"g1": "val"}))
    msg = str(err.value)
    assert msg.startswith("8 admitted image(s) have no split in the split map")
    assert all(s in msg for s in shas[:5]) and not any(s in msg for s in shas[5:])


def test_allow_unsplit_keeps_null_splits_and_the_report_records_them(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {SONAR_SRC: [(A, "g1"), (B, "g2")]})
    boxes = ConfigResult("boxes", (_box(A, SONAR_SRC), _box(B, SONAR_SRC)), {})
    inputs = _inputs(roots, assignments={"g1": "val"}, allow_unsplit=True)
    out = build_v11_tables(registry, {"boxes": boxes}, inputs)
    for cid in ("metadata", "boxes"):
        assert {r["image_sha256"]: r["split"] for r in out[cid].rows} == {A: "val", B: None}
    (tmp_path / "out").mkdir()
    report = json.loads(write_v11_report(out, tmp_path / "out").read_text())
    assert report["allow_unsplit"] is True and report["unsplit_images"] == 1
    assert report["tables"]["metadata"]["unsplit"] == 1
    assert report["tables"]["boxes"]["unsplit"] == 1
