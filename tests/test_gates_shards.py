"""``gates.shards.parse_shard``: the shard file names a release export writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.gates.shards import RELEASE_ID, ShardName, parse_shard, shard_split


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # v1.0: one stem per split
        ("train-00000-of-00004.parquet", ShardName("train", None, None, 0, 4)),
        ("test_ood-00003-of-00010.parquet", ShardName("test_ood", None, None, 3, 10)),
        ("val-00000-of-00001.parquet", ShardName("val", None, None, 0, 1)),
        # v1.1: <split>-<source>
        (
            "test-noaa-pifsc-bleaching-00000-of-00001.parquet",
            ShardName("test", "noaa-pifsc-bleaching", None, 0, 1),
        ),
        # an additive export: <split>-<source>-<release-id>
        (
            "train-noaa-pifsc-bleaching-v1.2.0-00000-of-00001.parquet",
            ShardName("train", "noaa-pifsc-bleaching", "v1.2.0", 0, 1),
        ),
        # a source that ends in a version-like token is still the source
        (
            "train-seaview-v1-v1.2.0-00002-of-00010.parquet",
            ShardName("train", "seaview-v1", "v1.2.0", 2, 10),
        ),
        # the release alone is not a source
        ("train-v1.2.0-00000-of-00001.parquet", ShardName("train", None, "v1.2.0", 0, 1)),
    ],
)
def test_parse_shard_takes_both_name_shapes_apart(name, expected):
    assert parse_shard(name) == expected
    assert parse_shard(Path("/x/data/images") / name) == expected  # only the file name counts


@pytest.mark.parametrize(
    "name",
    [
        "train.parquet",
        "train-0000-of-00004.parquet",  # the index has five digits
        "train-00000-of-00004.csv",
        "00000-of-00004.parquet",
        "README.md",
    ],
)
def test_parse_shard_refuses_a_name_that_is_not_a_shard(name):
    assert parse_shard(name) is None


def test_the_infix_changes_the_release_and_nothing_else():
    plain = parse_shard("train-noaa-pifsc-bleaching-00000-of-00001.parquet")
    infixed = parse_shard("train-noaa-pifsc-bleaching-v1.2.0-00000-of-00001.parquet")
    assert plain and infixed
    assert (plain.split, plain.source, plain.index, plain.total) == (
        infixed.split,
        infixed.source,
        infixed.index,
        infixed.total,
    )
    assert (plain.release, infixed.release) == (None, "v1.2.0")


@pytest.mark.parametrize(
    ("name", "split"),
    [
        ("train-00000-of-00004.parquet", "train"),
        ("val-00000-of-00001.parquet", "validation"),
        ("valid-00000-of-00001.parquet", "validation"),
        ("probe-noaa-v1.2.0-00000-of-00001.parquet", "validation"),
        ("train-noaa-pifsc-bleaching-v1.2.0-00000-of-00001.parquet", "train"),
        ("test_ood-a-00000-of-00001.parquet", "test_ood"),
        ("val-notes.parquet", "validation"),  # not a shard name: its first token, aliased
    ],
)
def test_shard_split_keeps_the_aliases_and_ignores_the_infix(name, split):
    assert shard_split(Path(name)) == split


@pytest.mark.parametrize("release", ["v1.2.0", "v10.20.30"])
def test_release_id_takes_three_numbers(release):
    assert RELEASE_ID.fullmatch(release)


@pytest.mark.parametrize(
    "release", ["v1.2", "1.2.0", "V1.2.0", "v1.2.0-rc1", "v1.2.0.1", " v1.2.0", "v1.2.0\n", ""]
)
def test_release_id_refuses_anything_else(release):
    assert not RELEASE_ID.fullmatch(release)
