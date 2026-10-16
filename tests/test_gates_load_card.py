from __future__ import annotations

import json

import pyarrow as pa
import pytest
from _gates_build import (
    IMAGE_COLS,
    card_text,
    edit_release,
    images_info,
    make_build,
    smooth_png,
    write_config,
)

from marinedata.gates.card import run_card_gate, size_category
from marinedata.gates.croissant import run_croissant_gate
from marinedata.gates.load import run_load_gate

CONFIGS = {
    "images": {"train": 16, "validation": 2, "test": 2},
    "metadata": {"train": 16, "validation": 2, "test": 2},
    "fixture-masks": {"train": 4},
}


def _has(result, kind: str) -> bool:
    return any(f"{kind}:" in line for line in result.failures)


def test_load_passes_and_reads_the_sample(tmp_path):
    result = run_load_gate(make_build(tmp_path))
    assert result.passed, result.failures
    assert result.counts["rows_read"] == 44 and result.counts["configs"] == 3


def test_load_reads_only_sample_rows_per_split(tmp_path):
    result = run_load_gate(make_build(tmp_path), sample_rows=5)
    assert result.passed and result.counts["rows_read"] == 22


@pytest.mark.parametrize(
    ("kind", "build_kw"),
    [
        ("no_release", {"release": False}),
        ("release_without_configs", {"release_configs": False}),
        ("load_error", {"card": False}),
        ("split_mismatch", {"card_configs": {**CONFIGS, "images": {"train": 16, "validation": 2}}}),
        ("schema_mismatch", {"info_lines": images_info(sha_dtype="int64")}),
    ],
)
def test_load_failure_modes(tmp_path, kind, build_kw):
    result = run_load_gate(make_build(tmp_path, **build_kw))
    assert not result.passed and _has(result, kind), result.failures


def test_load_release_and_shard_disagreements(tmp_path):
    build = make_build(tmp_path)
    edit_release(build, configs={**CONFIGS, "ghost": {"train": 1}})
    assert _has(run_load_gate(build), "missing_config")
    edit_release(build, configs={k: v for k, v in CONFIGS.items() if k != "metadata"})
    assert _has(run_load_gate(build), "unlisted_config")
    edit_release(build, configs={**CONFIGS, "images": {"train": 99, "validation": 2, "test": 2}})
    result = run_load_gate(build)
    assert _has(result, "count_mismatch") and "images/train" in result.failures[0]


def test_load_schema_differs_between_shards(tmp_path):
    build = make_build(tmp_path)
    cols = {**IMAGE_COLS, "source_id": (pa.int64(), {"dtype": "int64", "_type": "Value"})}
    row = {"image_sha256": "a" * 64, "image": {"bytes": smooth_png(90), "path": None}}
    rows = [
        {**row, "source_id": 1, "split_group": "x1"},
        {**row, "source_id": 2, "split_group": "x2"},
    ]
    write_config(build, "images", {"validation": rows}, cols)
    assert _has(run_load_gate(build), "schema_mismatch")


def test_card_passes_with_and_without_dataset_info(tmp_path):
    assert run_card_gate(make_build(tmp_path / "a")).passed
    build = make_build(tmp_path / "b")
    size = sum(p.stat().st_size for p in (build / "data" / "images").glob("*.parquet"))
    info = images_info(rows=CONFIGS["images"], download_size=size)
    (build / "README.md").write_text(card_text(CONFIGS, info_lines=info))
    result = run_card_gate(build)
    assert result.passed, result.failures


@pytest.mark.parametrize(
    ("kind", "build_kw"),
    [
        ("unparseable_card", {"card": False}),
        ("missing_config_section", {"card_configs": {"images": CONFIGS["images"]}}),
        ("extra_config_section", {"card_configs": {**CONFIGS, "ghost": {"train": 1}}}),
        ("split_mismatch", {"card_configs": {**CONFIGS, "images": {"train": 16}}}),
        ("count_mismatch", {"info_lines": images_info(rows={"train": 99})}),
        ("size_mismatch", {"info_lines": images_info(download_size=1)}),
        ("size_category", {"categories": ["1M<n<10M"]}),
        ("size_category", {"categories": []}),
    ],
)
def test_card_failure_modes(tmp_path, kind, build_kw):
    result = run_card_gate(make_build(tmp_path, **build_kw))
    assert not result.passed and _has(result, kind), result.failures


def test_card_that_is_not_yaml_is_unparseable(tmp_path):
    build = make_build(tmp_path)
    (build / "README.md").write_text("---\nconfigs: [unclosed\n---\n")
    assert _has(run_card_gate(build), "unparseable_card")


def test_size_category_bands():
    assert [size_category(n) for n in (0, 999, 1000, 99_999, 10**6, 10**13)] == [
        "n<1K", "n<1K", "1K<n<10K", "10K<n<100K", "1M<n<10M", "n>1T",
    ]  # fmt: skip


def test_croissant_skipped_when_absent_or_validator_missing(tmp_path, monkeypatch):
    assert run_croissant_gate(tmp_path).skipped
    doc = {"@context": {}, "@type": "sc:Dataset", "name": "fx"}
    (tmp_path / "croissant.json").write_text(json.dumps(doc))
    monkeypatch.setitem(__import__("sys").modules, "mlcroissant", None)
    result = run_croissant_gate(tmp_path)
    assert result.skipped and "mlcroissant" in result.skipped and not result.passed


def test_croissant_malformed_files_fail_without_the_validator(tmp_path):
    (tmp_path / "croissant.json").write_text("{nope")
    assert _has(run_croissant_gate(tmp_path), "unparseable_json")
    (tmp_path / "croissant.json").write_text('{"@context": {}, "@type": "sc:Dataset"}')
    assert _has(run_croissant_gate(tmp_path), "missing_key")


def test_croissant_validator_runs_when_installed(tmp_path):
    pytest.importorskip("mlcroissant")
    doc = {"@context": {}, "@type": "sc:Dataset", "name": "fx"}
    (tmp_path / "croissant.json").write_text(json.dumps(doc))
    assert run_croissant_gate(tmp_path).skipped is None


FROZEN = {"coral-masks": {"train": 3}}


def _v1_tree(root):
    """A v1.0 tree holding the frozen config's shards (the shape of the Hub repo's data/ folder)."""
    row = {
        "image_sha256": "b" * 64,
        "image": {"bytes": smooth_png(90), "path": None},
        "split_group": "g1",
    }
    root.mkdir(parents=True, exist_ok=True)
    write_config(root, "coral-masks", {"train": [row]}, IMAGE_COLS)
    return root


def test_load_skips_frozen_configs_the_build_lacks(tmp_path):
    pytest.importorskip("datasets")
    build = make_build(tmp_path, card_configs={**FROZEN, **CONFIGS})  # the frozen config is first
    result = run_load_gate(build)
    assert result.passed, result.failures
    assert result.counts["skipped_frozen"] == 1 and result.counts["configs"] == 3
    assert result.counts["rows_read"] == 44


def test_load_reads_frozen_configs_from_v1_dir(tmp_path):
    pytest.importorskip("datasets")
    build = make_build(tmp_path / "b", card_configs={**FROZEN, **CONFIGS})
    result = run_load_gate(build, v1_dir=_v1_tree(tmp_path / "v1"))
    assert result.passed, result.failures
    assert result.counts["configs"] == 4 and "skipped_frozen" not in result.counts


@pytest.mark.parametrize(
    ("card_configs", "v1_dir_given"),
    [({**FROZEN, **CONFIGS}, True), ({"ghost": {"train": 1}, **CONFIGS}, False)],
)
def test_load_fails_when_a_pattern_matches_in_neither_tree(tmp_path, card_configs, v1_dir_given):
    pytest.importorskip("datasets")
    build = make_build(tmp_path / "b", card_configs=card_configs)
    result = run_load_gate(build, v1_dir=tmp_path / "v1" if v1_dir_given else None)
    assert _has(result, "config_files_missing") and _has(result, "first_config_unresolved")


def test_card_counts_configs_under_v1_dir_as_present(tmp_path):
    pytest.importorskip("datasets")
    cards = {**CONFIGS, **FROZEN}
    build = make_build(tmp_path / "b", card_configs=cards)
    assert _has(run_card_gate(build), "extra_config_section")  # a rehearsal has no v1.0 tree
    v1 = _v1_tree(tmp_path / "v1")
    assert run_card_gate(build, v1_dir=v1).passed
    ghost = make_build(tmp_path / "c", card_configs={**cards, "ghost": {"train": 1}})
    result = run_card_gate(ghost, v1_dir=v1)
    assert _has(result, "extra_config_section") and [f.split()[0] for f in result.failures] == [
        "ghost"
    ]
