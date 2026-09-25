"""WP-8c: the 3 today-available producers, the 5 v2 configs, and the human/model
eval-split invariant. Builds synthetic staged trees (never the real caches — CI-portable)
against the real registry, so crosswalk/rollup behaviour is exercised for real."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.registry import Registry
from marinedata.task_layers.configs import (
    assert_no_mixed_origin_in_eval,
    build_all_configs,
    write_configs,
)
from marinedata.task_layers.producers.coralscapes_semseg import produce_semseg
from marinedata.task_layers.producers.coralvqa_vqa import produce_vqa
from marinedata.task_layers.producers.reefolution_points import produce_points


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _write_reefolution_tree(tmp_path):
    root = tmp_path / "reefolution"
    images = root / "images" / "default"
    images.mkdir(parents=True)
    (images / "img1.jpg").write_bytes(b"fake-jpeg-bytes-1")
    (images / "img2.jpg").write_bytes(b"fake-jpeg-bytes-2")
    pq.write_table(
        pa.table(
            {
                "stem": ["img1", "img2"],
                "partition": ["default", "default"],
                "width": [100, 100],
                "height": [100, 100],
            }
        ),
        root / "metadata.parquet",
    )
    (root / "labels").mkdir()

    # img1: 6 "Acr" (-> HC, rolls up) + 4 "Sand" (-> SD -> ABIOTIC) => dominant HC (60%).
    # img2: 1 unmapped label only => 100% unmapped, excluded from D-Y (< 10 known).
    def _pt(stem, row, col, label):
        return {"stem": stem, "partition": "default", "row": row, "col": col, "label": label}

    rows = [_pt("img1", i, i, "Acr") for i in range(6)]
    rows += [_pt("img1", i, i, "Sand") for i in range(4)]
    rows += [_pt("img2", 0, 0, "Zzz_unmapped")]
    pq.write_table(pa.Table.from_pylist(rows), root / "labels" / "points.parquet")
    return root


def _write_coralscapes_tree(tmp_path):
    root = tmp_path / "coralscapes"
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    (images / "m1.png").write_bytes(b"fake-png-bytes")
    # index 5 = "sand" (-> SD -> ABIOTIC); index 0 = ignore (dropped); index 2 = "trash"
    # (-> TRASH -> ABIOTIC too, per the real schema) — both known classes roll up to the
    # same coarse bucket, so the coarse rollup below is 100% ABIOTIC, 0% unknown.
    mask = Image.new("P", (10, 10), color=0)
    mask.putpalette([0] * 768)  # an unset palette lets PIL re-quantize indices on save
    pixels = mask.load()
    for x in range(10):
        for y in range(10):
            pixels[x, y] = 5 if x < 8 else 2
    mask.save(masks / "m1.png")
    return root


def _write_coralvqa(tmp_path):
    images_dir = tmp_path / "coralvqa_images"
    images_dir.mkdir()
    (images_dir / "v1.jpg").write_bytes(b"fake-vqa-jpeg")
    jsonl = tmp_path / "CoralVQA_train.jsonl"
    record = {
        "id": "CoralVQA",
        "image": "v1.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n[vqa] How many genera?"},
            {"from": "gpt", "value": "2"},
        ],
    }
    jsonl.write_text(json.dumps(record) + "\n")
    return jsonl, images_dir


def _stage_tasklabels(
    tmp_path, registry, reefolution_root, coralscapes_root, vqa_jsonl, vqa_images
):
    base = tmp_path / "release"
    points_out = base / "_tasklabels" / "reefolution" / "points.parquet"
    semseg_out = base / "_tasklabels" / "coralscapes" / "semseg.parquet"
    vqa_out = base / "_tasklabels" / "coralvqa" / "vqa.parquet"
    n_points = produce_points(reefolution_root, points_out)
    n_semseg = produce_semseg(coralscapes_root, semseg_out)
    n_vqa = produce_vqa(vqa_jsonl, vqa_images, vqa_out)
    return base, n_points, n_semseg, n_vqa


def test_producers_write_dz2_columns(tmp_path, registry):
    reef = _write_reefolution_tree(tmp_path)
    coral = _write_coralscapes_tree(tmp_path)
    jsonl, images_dir = _write_coralvqa(tmp_path)
    base, n_points, n_semseg, n_vqa = _stage_tasklabels(
        tmp_path, registry, reef, coral, jsonl, images_dir
    )

    assert n_points == 11  # 6 + 4 + 1
    assert n_semseg == 1
    assert n_vqa == 1

    points = pq.read_table(base / "_tasklabels" / "reefolution" / "points.parquet").to_pylist()
    assert {"sha256", "source_id", "label_origin", "native_label", "x", "y"} <= points[0].keys()
    assert all(0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0 for p in points)

    semseg = pq.read_table(base / "_tasklabels" / "coralscapes" / "semseg.parquet").to_pylist()
    assert {"sha256", "source_id", "label_origin", "mask_key", "class_counts"} <= semseg[0].keys()
    counts = json.loads(semseg[0]["class_counts"])
    assert counts == {"sand": 80, "trash": 20}  # index 0 (ignore) dropped

    vqa = pq.read_table(base / "_tasklabels" / "coralvqa" / "vqa.parquet").to_pylist()
    expected_vqa_cols = {
        "sha256",
        "source_id",
        "label_origin",
        "question",
        "answer",
        "qtype",
        "split_hint",
    }
    assert expected_vqa_cols <= vqa[0].keys()
    assert vqa[0]["qtype"] == "vqa"
    assert vqa[0]["question"] == "How many genera?"


def test_build_all_configs_and_rollup(tmp_path, registry):
    reef = _write_reefolution_tree(tmp_path)
    coral = _write_coralscapes_tree(tmp_path)
    jsonl, images_dir = _write_coralvqa(tmp_path)
    base, *_ = _stage_tasklabels(tmp_path, registry, reef, coral, jsonl, images_dir)

    results = build_all_configs(registry, base)
    assert set(results) == {"points", "vqa", "semseg", "benthic-coarse", "benthic-cover"}

    points = results["points"]
    assert points.n_images == 2
    assert points.unmapped_by_source["reefolution"] == pytest.approx(1 / 11)

    coarse_by_sha = {row["sha256"]: row for row in results["benthic-coarse"].rows}
    img1_sha = next(r["sha256"] for r in results["points"].rows if r["native_label"] == "Acr")
    assert coarse_by_sha[img1_sha]["benthic_dominant"] == "HC"

    img2_sha = next(
        r["sha256"] for r in results["points"].rows if r["native_label"] == "Zzz_unmapped"
    )
    assert coarse_by_sha[img2_sha]["benthic_dominant"] is None  # excluded: 100% unknown

    cover_by_sha = {row["sha256"]: row for row in results["benthic-cover"].rows}
    assert json.loads(cover_by_sha[img1_sha]["benthic_cover"])["HC"] == pytest.approx(0.6)

    mask_sha = results["semseg"].rows[0]["sha256"]
    coarse_mask_row = coarse_by_sha[mask_sha]
    # sand -> SD -> ABIOTIC (80/80 known px, trash excluded as unknown) => dominant ABIOTIC
    assert coarse_mask_row["benthic_dominant"] == "ABIOTIC"

    written = write_configs(results, tmp_path / "out")
    assert set(written) == set(results)
    for path in written.values():
        assert path.is_file()


def test_no_mixed_human_model_in_eval_split():
    rows = [
        {"sha256": "a", "label_origin": "human"},
        {"sha256": "b", "label_origin": "human"},
        {"sha256": "c", "label_origin": "model"},
    ]
    ok_splits = {"a": "test", "b": "test", "c": "train"}
    assert_no_mixed_origin_in_eval(rows, ok_splits)  # model row is train-only: fine

    bad_splits = {"a": "test", "b": "test", "c": "test"}
    with pytest.raises(ValueError, match="mixed"):
        assert_no_mixed_origin_in_eval(rows, bad_splits)
