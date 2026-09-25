"""WP-3 AC: WebDataset export from the ``images`` config + label joins, one shard at a
time. Verified with :mod:`tarfile` always, and with :mod:`webdataset` when installed."""

from __future__ import annotations

import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.wds_export import build_label_index, export_wds


def _make_hf_dir(tmp_path):
    hf_dir = tmp_path / "hf"
    (hf_dir / "data" / "images").mkdir(parents=True)
    (hf_dir / "data" / "benthic-coarse").mkdir(parents=True)

    images = pa.table(
        {
            "image_sha256": ["aaa", "bbb"],
            "image": [
                {"bytes": b"\xff\xd8fake-jpg", "path": "aaa.jpg"},
                {"bytes": b"\x89PNGfake", "path": "bbb.png"},
            ],
            "source_id": ["src1", "src1"],
            "source_ids": ["src1", "src1"],
            "split_group": ["g1", "g2"],
        },
        schema=pa.schema(
            [
                ("image_sha256", pa.string()),
                ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ("source_id", pa.string()),
                ("source_ids", pa.string()),
                ("split_group", pa.string()),
            ]
        ),
    )
    pq.write_table(images, hf_dir / "data/images/train-00000-of-00001.parquet")

    labels = pa.table(
        {
            "image_sha256": ["aaa"],
            "source_id": ["src1"],
            "sample_key": ["k1"],
            "label": ["coral"],
            "native_label": ["Coral"],
            "label_reason": ["expert"],
            "mask_class_map": ["{}"],
        }
    )
    pq.write_table(labels, hf_dir / "data/benthic-coarse/train-00000-of-00001.parquet")
    return hf_dir


def test_build_label_index_joins_on_sha(tmp_path):
    hf_dir = _make_hf_dir(tmp_path)
    index = build_label_index(hf_dir)
    assert set(index) == {"aaa"}
    assert index["aaa"]["benthic-coarse"]["label"] == "coral"


def test_export_wds_writes_one_tar_with_image_and_json_members(tmp_path):
    hf_dir = _make_hf_dir(tmp_path)
    out_dir = tmp_path / "out"
    results = export_wds(hf_dir, out_dir, limit_shards=1)
    assert len(results) == 1
    result = results[0]
    assert result.n_images == 2
    assert result.tar_path.name == "train-00000-of-00001.tar"

    with tarfile.open(result.tar_path) as tar:
        names = sorted(tar.getnames())
        assert names == ["aaa.jpg", "aaa.json", "bbb.json", "bbb.png"]
        meta = json.loads(tar.extractfile("aaa.json").read())
        assert meta["image_sha256"] == "aaa"
        assert meta["labels"]["benthic-coarse"]["label"] == "coral"
        assert tar.extractfile("aaa.jpg").read() == b"\xff\xd8fake-jpg"
        meta_b = json.loads(tar.extractfile("bbb.json").read())
        assert meta_b["labels"] == {}


def test_limit_shards_caps_input_shards_processed(tmp_path):
    hf_dir = _make_hf_dir(tmp_path)
    # a second, later train shard that limit_shards=1 must not touch
    images2 = pa.table(
        {
            "image_sha256": ["ccc"],
            "image": [{"bytes": b"zzz", "path": "ccc.jpg"}],
            "source_id": ["src1"],
            "source_ids": ["src1"],
            "split_group": ["g3"],
        },
        schema=pa.schema(
            [
                ("image_sha256", pa.string()),
                ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ("source_id", pa.string()),
                ("source_ids", pa.string()),
                ("split_group", pa.string()),
            ]
        ),
    )
    pq.write_table(images2, hf_dir / "data/images/train-00001-of-00002.parquet")
    # rename the first shard to match a real 2-part split naming
    (hf_dir / "data/images/train-00000-of-00001.parquet").rename(
        hf_dir / "data/images/train-00000-of-00002.parquet"
    )
    out_dir = tmp_path / "out"
    results = export_wds(hf_dir, out_dir, limit_shards=1)
    assert len(results) == 1
    assert results[0].tar_path.name == "train-00000-of-00002.tar"


def test_webdataset_can_read_the_shard_if_installed(tmp_path):
    wds = pytest.importorskip("webdataset")
    hf_dir = _make_hf_dir(tmp_path)
    out_dir = tmp_path / "out"
    results = export_wds(hf_dir, out_dir, limit_shards=1)
    ds = wds.WebDataset(str(results[0].tar_path), shardshuffle=False)
    samples = list(ds)
    assert len(samples) == 2
    keys = sorted(s["__key__"] for s in samples)
    assert keys == ["aaa", "bbb"]
