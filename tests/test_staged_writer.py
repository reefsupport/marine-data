"""D-D writer: objects below the threshold, WebDataset shards + index.parquet above it."""

from __future__ import annotations

import datetime as dt
import hashlib
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

import pyarrow.parquet as pq
from _wp6_fixtures import png

from marinedata.adapters import Decoded, RemoteItem
from marinedata.staged_writer import LayoutError, StagedWriter, WriterConfig

ITEM = RemoteItem("x.tar", "http://example/x.tar")


def _cfg(layout: str, threshold: int = 100, shard_bytes: int = 40_000) -> WriterConfig:
    return WriterConfig(
        "syn",
        "v1",
        "CC-BY-4.0",
        "Test",
        dt.date(2026, 9, 25),
        layout=layout,
        threshold=threshold,
        shard_bytes=shard_bytes,
        defaults={"platform": "diver"},
    )


def _fill(root: Path, cfg: WriterConfig, n: int) -> StagedWriter:
    w = StagedWriter(root, cfg)
    for i in range(n):
        w.add(
            ITEM,
            Decoded(
                f"x.tar#d/{i:04d}.png",
                png(i, 24),
                ".png",
                labels={"cls": str(i % 3)},
                fields={"depth_m": 3.0, "depth_source": "metadata"},
            ),
        )
    w.finish_item("f" * 64)
    w.finalize()
    return w


def test_synthetic_large_source_round_trips_shard_and_index(tmp_path):
    n = 250  # > threshold 100 -> shards (the 200k rule with a test-sized threshold)
    w = _fill(tmp_path / "a", _cfg("shards"), n)
    shards = sorted((tmp_path / "a/images").glob("shard-*.tar"))
    assert len(shards) > 1 and not list((tmp_path / "a/images").glob("*.png"))
    index = pq.read_table(tmp_path / "a/images/index.parquet").to_pylist()
    assert len(index) == n
    by_member = {}
    for shard in shards:
        raw = shard.read_bytes()
        with tarfile.open(shard) as tar:
            for m in tar:
                data = tar.extractfile(m).read()
                by_member[m.name] = data
                assert m.mtime == 0 and m.uid == 0 and m.uname == ""
        for rec in (r for r in index if r["shard"] == f"images/{shard.name}"):
            chunk = raw[rec["offset"] : rec["offset"] + rec["size"]]
            assert hashlib.sha256(chunk).hexdigest() == rec["sha256"], "offset is a range read"
    assert len(by_member) == n
    for row in w.rows:
        data = by_member[row.image_member]
        assert hashlib.sha256(data).hexdigest() == row.image_sha256
        assert row.image_path.startswith("images/shard-") and row.depth_zone == "shallow"
        assert row.upstream_digest == "f" * 64 and row.platform == "diver"
    assert (
        w.files[f"images/{shards[0].name}"][0] == hashlib.sha256(shards[0].read_bytes()).hexdigest()
    )

    w2 = _fill(tmp_path / "b", _cfg("shards"), n)
    assert {k: v for k, v in w.files.items()} == {k: v for k, v in w2.files.items()}, (
        "deterministic: a re-run produces byte-identical shards"
    )


def test_below_threshold_stays_per_object(tmp_path):
    w = _fill(tmp_path, _cfg("objects"), 20)
    assert len(list((tmp_path / "images").glob("*.png"))) == 20
    assert not (tmp_path / "images/index.parquet").exists()
    assert all(r.image_member is None for r in w.rows)
    labels = pq.read_table(tmp_path / "labels/image_labels.parquet")
    assert labels.num_rows == 20


def test_objects_layout_refuses_to_cross_threshold(tmp_path):
    with pytest.raises(LayoutError, match="shards"):
        _fill(tmp_path, _cfg("objects", threshold=10), 11)


def test_label_files_pair_by_basename(tmp_path):
    cfg = _cfg("objects")
    cfg.label_stem_suffix = "_mask"
    w = StagedWriter(tmp_path, cfg)
    w.add(
        ITEM,
        Decoded("z.zip#masks/a_mask.png", b"", ".png", label_files={"masks/a_mask.png": png(1)}),
    )
    w.add(ITEM, Decoded("z.zip#imgs/a.png", png(2), ".png"))
    w.finish_item(None)
    w.finalize()
    assert w.rows[0].label_refs == ("labels/files/masks_a_mask.png",)
