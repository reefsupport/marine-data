"""Eval-split resolution + the rev-layout staged reader (WP-R8)."""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_bench_manifest import BLUE, RED, FakeS3Client

from marinedata.bench_manifest import iter_bucket_images, resolve_eval_split
from marinedata.benchmarks import BenchmarkEntry, Obtain, UpstreamSplit


def _split(eval_split: str, heldout_val: str | None = None, image_subdir: str | None = None):
    return UpstreamSplit(
        rule="dirs",
        eval_split=eval_split,
        heldout_val=heldout_val,
        counts={},
        definition_url="https://example.org",
        image_subdir=image_subdir,
    )


def test_labelled_row_matches_case_insensitively_and_returns_registry_name() -> None:
    assert resolve_eval_split("val", "a/b.png", _split("TE", "VAL")) == "VAL"
    assert resolve_eval_split("train", "a/b.png", _split("TE", "VAL")) is None


def test_unlabelled_row_falls_back_to_directory_never_to_a_guess() -> None:
    s = _split("TE", "VAL", image_subdir="RGB")
    assert resolve_eval_split(None, "U/U/TE/RGB/1.png", s) == "TE"
    assert resolve_eval_split(None, "U/U/TR/RGB/1.png", s) is None
    assert resolve_eval_split(None, "no-split-evidence.png", s) is None
    # 3+ letter split names also match a longer directory name (Validate -> val)
    assert resolve_eval_split(None, "Validate/Ferny/1.jpg", _split("test", "val")) == "val"
    assert resolve_eval_split(None, "Training/Ferny/1.jpg", _split("test", "val")) is None


def test_image_subdir_drops_masks_beside_the_images() -> None:
    s = _split("TE", image_subdir="RGB")
    assert resolve_eval_split(None, "U/TE/GT/1.png", s) is None
    assert resolve_eval_split("TE", "U/TE/depth/1.png", s) is None


def test_eval_split_all_keeps_everything() -> None:
    assert resolve_eval_split(None, "x.png", _split("all")) == "all"


def _staged_entry(split: UpstreamSplit) -> BenchmarkEntry:
    return BenchmarkEntry(
        id="fakebench",
        name="F",
        task="cls",
        catalog_id="fakebench",
        registry_id=None,
        upstream_split=split,
        split_verified=False,
        verified_by="unit test",
        obtain=Obtain(status="staged", via="t"),
        policy="exclude",
        policy_reason="t",
    )


def test_staged_rev_layout_fetches_only_eval_rows_in_order() -> None:
    root = "sources/fakebench/rev-1/"
    rows = [
        {
            "stem": f"s{i}",
            "image_path": f"images/s{i}.png",
            "upstream_split": None,
            "upstream_path": f"D/{d}/{k}/s{i}.png",
        }
        for i, (d, k) in enumerate([("TE", "RGB"), ("TE", "GT"), ("TR", "RGB"), ("TE", "RGB")])
    ]
    meta = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), meta)
    objects = {root + "metadata.parquet": meta.getvalue()}
    objects.update({root + f"images/s{i}.png": (RED if i % 2 == 0 else BLUE) for i in range(4)})
    entry = _staged_entry(_split("TE", image_subdir="RGB"))
    images = list(iter_bucket_images(FakeS3Client(objects), "b", entry, layout="staged"))
    assert [i.stem for i in images] == ["s0", "s3"]
    assert {i.upstream_split for i in images} == {"TE"}
    assert images[0].upstream_path == "D/TE/RGB/s0.png"


@pytest.mark.parametrize("layout", ["auto", "staged"])
def test_missing_metadata_is_a_named_error(layout: str) -> None:
    from marinedata.bench_manifest import ManifestBuildError

    with pytest.raises(ManifestBuildError):
        list(iter_bucket_images(FakeS3Client({}), "b", _staged_entry(_split("TE")), layout=layout))
