"""WS-D S55 — Hub export layout: shard plan, embedded-image Parquet, split rules, commits."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from marinedata.hf_card import _supervises, render_card, render_licence, unsupervised_configs
from marinedata.hf_export import (
    IMAGES,
    MASKS,
    HFExportError,
    SampleRow,
    build_layout,
    export,
    hf_split,
    read_manifest,
)
from marinedata.hf_parquet import (
    ConfigSpec,
    ExportRow,
    HFParquetError,
    files_per_folder,
    greedy_chunks,
    plan_config,
    row_groups,
    shard_name,
    write_shard,
)
from marinedata.hf_upload import main as upload_main
from marinedata.hf_upload import plan_commits

pq = pytest.importorskip("pyarrow.parquet")
Image = pytest.importorskip("PIL.Image")


def _png(path: Path, shade: int) -> Path:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buf, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


def _row(tmp_path: Path, i: int, **kw) -> SampleRow:
    image = _png(tmp_path / "src" / f"img{i}.png", i * 20)
    base = dict(
        task_id="t",
        raw_split="train",
        image_sha256=f"{i:064x}",
        source_id="s",
        sample_key=f"images/default/img{i}.png",
        image=image,
    )
    base.update(kw)
    return SampleRow(**base)


def test_shard_names_are_deterministic_and_counted():
    assert shard_name("images", "train", 0, 24) == "data/images/train-00000-of-00024.parquet"
    with pytest.raises(HFParquetError):
        shard_name("images", "train", 24, 24)


def test_greedy_chunks_respect_the_byte_target():
    rows = [ExportRow(values={}, file_bytes=400) for _ in range(5)]
    chunks = greedy_chunks(rows, 1000)  # 400 + 256 overhead per row -> one row per chunk
    assert [len(c) for c in chunks] == [1, 1, 1, 1, 1]
    assert [len(c) for c in greedy_chunks(rows, 1400)] == [2, 2, 1]
    assert greedy_chunks([], 1000) == []


def test_plan_config_skips_empty_splits_and_names_final_counts():
    spec = ConfigSpec("t", (("a", "string"),), shard_target_bytes=600)
    rows = [ExportRow(values={"a": str(i)}) for i in range(3)]
    plans = plan_config(spec, {"train": rows, "validation": rows[:1]})
    assert [p.name for p in plans] == [
        "data/t/train-00000-of-00002.parquet",
        "data/t/train-00001-of-00002.parquet",
        "data/t/validation-00000-of-00001.parquet",
    ]


def test_write_shard_embeds_image_bytes_with_hf_features(tmp_path):
    files = [_png(tmp_path / f"{i}.png", i * 40) for i in range(5)]
    spec = ConfigSpec(
        "images",
        (("image_sha256", "string"), ("image", "image")),
        row_group_target_bytes=2 * (max(f.stat().st_size for f in files) + 256),
    )
    rows = [
        ExportRow(
            values={"image_sha256": str(i)},
            file=f,
            file_path=f"s/{f.name}",
            file_bytes=f.stat().st_size,
        )
        for i, f in enumerate(files)
    ]
    out = tmp_path / "data/images/train-00000-of-00001.parquet"
    write_shard(out, spec, rows)
    meta = pq.ParquetFile(out)
    assert meta.metadata.num_row_groups == 3  # 2 + 2 + 1
    features = json.loads(meta.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    assert features["image"] == {"_type": "Image"}
    table = pq.read_table(out)
    assert table.column("image").to_pylist()[3]["bytes"] == files[3].read_bytes()
    assert not meta.metadata.row_group(0).column(1).is_stats_set  # no blob stats in footer


def test_split_names_map_probe_and_val_to_validation():
    assert hf_split("val") == hf_split("probe") == "validation"
    with pytest.raises(HFExportError):
        hf_split("holdout")


def test_read_manifest_counts_duplicate_rows(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("image_sha256\tsplit\naa\ttrain\naa\ttrain\nbb\ttest\n")
    assert read_manifest(path) == {("aa", "train"): 2, ("bb", "test"): 1}


def test_layout_embeds_each_image_once_and_tasks_are_label_only(tmp_path):
    a = _row(tmp_path, 1, label="HEALTHY")
    a_twin = _row(tmp_path, 1, source_id="s2", label="HEALTHY")
    mask = _png(tmp_path / "m" / "img2.png", 1)
    b = _row(
        tmp_path,
        2,
        raw_split="probe",
        mask=mask,
        mask_values="0=unlabelled,1=HC",
        mask_class_map='{"0": null, "1": "HC"}',
    )
    plain = [replace(r, label=None, mask_class_map=None) for r in (a, b)]
    layout = build_layout({"task": [a, a_twin, b], "pre": plain})
    _, splits = layout[IMAGES]
    assert {s: len(r) for s, r in splits.items()} == {"train": 1, "validation": 1}
    assert splits["train"][0].values["source_ids"] == "s,s2"
    assert layout["task"][0].image_column is None
    assert len(layout["task"][1]["train"]) == 2  # one row per (image, source)
    assert layout["pre"][0].columns[-1][0] == "sample_key"  # unlabelled -> membership only
    assert len(layout[MASKS][1]["validation"]) == 1


def test_one_image_in_two_splits_raises(tmp_path):
    a = _row(tmp_path, 1)
    with pytest.raises(HFExportError, match="lands in splits"):
        build_layout({"x": [a], "y": [_row(tmp_path, 1, raw_split="test")]})


def test_export_summary_and_card_list_every_config(tmp_path):
    layout = build_layout(
        {"task": [_row(tmp_path, 1, label="HC"), _row(tmp_path, 2, raw_split="test")]}
    )
    summary = export(layout, tmp_path / "hf")
    assert summary["files_per_folder"]["data"] == 2  # images/ + task/
    assert (tmp_path / "hf/data/images/train-00000-of-00001.parquet").is_file()
    release = {
        "release": "v1",
        "split_map_sha256": "x",
        "never_eval_near_dup_excluded": {"count": 1},
        "near_dup": {
            "algorithm": "dhash-64",
            "pil_version": "12.3.0",
            "union_max_hamming": 4,
            "never_eval_exclude_max_hamming": 8,
            "chain_guard_fraction": 0.01,
        },
    }
    sources = [
        {
            "id": "s",
            "version": "1",
            "licence": "CC-BY-4.0",
            "licence_hf": "cc-by-4.0",
            "tier": "T1",
            "citation": "A|B",
        }
    ]
    card = render_card(summary, release, sources, pretty_name="RS v1")
    assert "path: data/task/test-*.parquet" in card and "default: true" in card
    assert "flat sand" in card and "Hamming 8" in card
    assert "CC-BY-4.0" in render_licence(sources)
    # row 2 (test) has no label -> "task" still supervised by row 1; a null-only config is flagged
    assert unsupervised_configs(tmp_path / "hf", summary) == ()
    note = render_card(
        summary, release, sources, pretty_name="RS v1", repo_id="o/r", unsupervised=("task",)
    )
    assert "`task` has **no supervision" in note and 'load_dataset("o/r"' in note
    assert _supervises("mask_class_map", '{"0": null}') is False


def test_commit_plan_bounds_files_and_puts_readme_last(tmp_path):
    for i in range(205):
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / f"f{i:03d}.parquet").write_bytes(b"x")
    (tmp_path / "README.md").write_text("card")
    commits = plan_commits(tmp_path)
    assert [len(c.files) for c in commits] == [99, 99, 8]
    assert commits[-1].files[-1] == "README.md"
    assert len(plan_commits(tmp_path, max_bytes=2)) == 104  # 103 data pairs/single + README
    with pytest.raises(ValueError):
        plan_commits(tmp_path, max_files=100)


def test_upload_is_dry_run_unless_both_flags(tmp_path, capsys):
    (tmp_path / "README.md").write_text("card")
    assert upload_main([str(tmp_path), "--repo-id", "org/x"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert upload_main([str(tmp_path), "--repo-id", "org/x", "--execute"]) == 2
    assert (
        upload_main([str(tmp_path), "--repo-id", "org/x", "--execute", "--confirm-yohan-go"]) == 2
    )  # visibility must be explicit


def test_files_per_folder_counts_direct_entries():
    counts = files_per_folder(
        ["README.md", "data/a/x.parquet", "data/a/y.parquet", "data/b/z.parquet"]
    )
    assert counts == {"": 2, "data": 2, "data/a": 2, "data/b": 1}


def test_image_row_groups_are_capped_at_100_rows():
    rows = [ExportRow(values={}, file_bytes=10) for _ in range(250)]
    image_spec = ConfigSpec("images", (("image", "image"),))
    assert [len(g) for g in row_groups(image_spec, rows)] == [100, 100, 50]
    assert [len(g) for g in row_groups(ConfigSpec("t", (("a", "string"),)), rows)] == [250]
