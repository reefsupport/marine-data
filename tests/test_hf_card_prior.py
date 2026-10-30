"""``hf_card --v1-dir``: the card declares every config folder of the prior tree (PR-4a)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from test_hf_card_v11 import CFF, REPO, V1_OPEN, _export
from test_hf_export_v11 import build as _build
from test_hf_export_v11 import registry as _registry

from marinedata import hf_card_prior as prior
from marinedata.hf_card_v11 import V1_CONFIGS, v1_entries, write_card_files
from marinedata.hf_export_v11 import V11ExportError

build, registry = _build, _registry
SOURCE = "reef-support-bleaching"
ROWS = {"train": 4, "validation": 2, "test": 1}


def _shard(root: Path, config: str, name: str, rows: int) -> None:
    path = root / "data" / config / name
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"{i:064x}" for i in range(rows)]
    pq.write_table(pa.table({"image_sha256": ids, "source_id": [SOURCE] * rows}), path)


def snapshot(root: Path) -> Path:
    """The Hub repo as it is before the publish: the four v1.0 folders and condition-masks."""
    for config in V1_OPEN:
        for split, rows in ROWS.items():
            _shard(root, config, f"{split}-00000-of-00001.parquet", rows)
    _shard(root, "condition-masks", "train-00000-of-00001.parquet", 3)
    return root


def _configs(readme: str) -> list[dict]:
    return yaml.safe_load(readme.split("---\n", 2)[1])["configs"]


def _card(registry, out: Path, **kwargs) -> str:
    write_card_files(
        out,
        registry,
        flavour="open",
        repo_id=REPO["open"],
        cff=CFF,
        released="2026-10-16",
        **kwargs,
    )
    return (out / "README.md").read_text()


def test_prior_configs_are_the_snapshot_folders_the_build_does_not_ship(tmp_path):
    found = prior.prior_configs(snapshot(tmp_path), {"metadata", "images"}, V1_CONFIGS["open"])
    assert [p.name for p in found] == [*V1_OPEN, "condition-masks"]
    coral, masks = found[0], found[-1]
    assert (coral.release, coral.rows) == ("v1.0", 7)
    assert [split for split, _, _ in coral.splits] == ["train", "validation", "test"]
    assert (masks.release, masks.rows, masks.sources) == ("v1.1.1", 3, (SOURCE,))
    assert masks.data_files == [{"split": "train", "path": "data/condition-masks/train-*.parquet"}]


def test_a_config_the_build_ships_is_not_a_prior_config(tmp_path):
    found = prior.prior_configs(snapshot(tmp_path), {"coral-masks"}, V1_CONFIGS["open"])
    assert "coral-masks" not in [p.name for p in found]


def test_v1_entries_stay_byte_for_byte_and_others_list_their_splits(tmp_path):
    found = prior.prior_configs(snapshot(tmp_path), set(), V1_CONFIGS["open"])
    out = prior.entries(found, {e["config_name"]: e for e in v1_entries("open")})
    assert out[:4] == v1_entries("open")
    assert out[4] == {"config_name": "condition-masks", "data_files": found[-1].data_files}


def test_a_snapshot_without_parquet_is_refused(tmp_path):
    with pytest.raises(V11ExportError, match="--v1-dir"):
        prior.prior_configs(tmp_path, set(), ())


def test_the_card_declares_every_hub_folder_when_given_the_prior_tree(registry, build, tmp_path):
    exported = _export(registry, build, tmp_path)
    out = exported[0] if isinstance(exported, tuple) else exported
    before = _card(registry, out)
    after = _card(registry, out, v1_dir=snapshot(tmp_path / "hub"))
    names0 = [c["config_name"] for c in _configs(before)]
    names = [c["config_name"] for c in _configs(after)]
    assert names0[:4] == V1_OPEN and "condition-masks" not in names0  # no --v1-dir: unchanged
    assert "Frozen prior-release config" not in before
    assert names == [*V1_OPEN, "condition-masks", *names0[4:]]
    assert len(names) == len(set(names))
    assert _configs(after)[:4] == v1_entries("open")
    row = (
        "| `condition-masks` | Frozen prior-release "
        "config (v1.1.1) | 3 | `reef-support-bleaching` |"
    )
    assert row in after
    assert "| `coral-masks` | Frozen prior-release config (v1.0) | 7 |" in after
    assert "carried unchanged and not rebuilt" in after
