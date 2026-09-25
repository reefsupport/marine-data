"""WP-3 AC: ``datasets.load_dataset(<local _hf/v1 dir>, <config>, streaming=True)`` for every
config in the card. A tiny fixture proves the mechanism always (no real build needed); the
real-build test parametrizes over every ``config_name`` in ``_hf/v1/README.md`` and skips
cleanly when that directory is absent (large, built outside the repo, WP-2/S59's output)."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

datasets = pytest.importorskip("datasets")

HF_DIR = Path(os.environ.get("MARINEDATA_HF_DIR", Path.home() / "dev/reefsupport/data/_hf/v1"))


def _card_configs(hf_dir: Path) -> list[str]:
    text = (hf_dir / "README.md").read_text(encoding="utf-8")
    _, front_matter, _ = text.split("---", 2)
    meta = yaml.safe_load(front_matter)
    return [c["config_name"] for c in meta["configs"]]


def _first_split(hf_dir: Path, config: str) -> str:
    text = (hf_dir / "README.md").read_text(encoding="utf-8")
    _, front_matter, _ = text.split("---", 2)
    meta = yaml.safe_load(front_matter)
    entry = next(c for c in meta["configs"] if c["config_name"] == config)
    return entry["data_files"][0]["split"]


def test_tiny_fixture_streams_and_has_expected_columns(tmp_path):
    """Always runs — proves streaming works even when no real _hf/v1 build exists."""
    (tmp_path / "data" / "tiny").mkdir(parents=True)
    pq.write_table(
        pa.table({"image_sha256": ["a", "b", "c"], "label": ["x", "y", "z"]}),
        tmp_path / "data/tiny/train-00000-of-00001.parquet",
    )
    (tmp_path / "README.md").write_text(
        "---\n"
        "configs:\n"
        "- config_name: tiny\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/tiny/train-*.parquet\n"
        "---\n# tiny\n"
    )
    ds = datasets.load_dataset(str(tmp_path), name="tiny", streaming=True, split="train")
    rows = list(ds.take(3))
    assert len(rows) == 3
    assert set(rows[0]) == {"image_sha256", "label"}


@pytest.mark.slow
class TestRealBuildStreaming:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self):
        if not (HF_DIR / "README.md").is_file():
            pytest.skip(f"{HF_DIR} absent — real _hf/v1 build not present on this machine")

    _configs = _card_configs(HF_DIR) if (HF_DIR / "README.md").is_file() else []

    @pytest.mark.parametrize("config", _configs)
    def test_config_streams_first_rows(self, config):
        split = _first_split(HF_DIR, config)
        ds = datasets.load_dataset(str(HF_DIR), name=config, streaming=True, split=split)
        rows = list(ds.take(3))
        assert len(rows) == 3
        columns = set(rows[0])
        assert columns, f"{config}: no columns in a streamed row"
        for row in rows:
            assert set(row) == columns, f"{config}: inconsistent columns across rows"
