"""Unit tests for the generic benchmark eval-image manifest builder (P1)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.bench_manifest import (
    ManifestBuildError,
    RawImage,
    build_manifest,
    iter_bucket_images,
    iter_upstream_images,
    resolve_source,
    spec_resolves,
    write_manifest,
)
from marinedata.benchmarks import BenchmarkEntry, Obtain, UpstreamSplit


def _entry(benchmark_id: str = "fakebench", eval_split: str = "test") -> BenchmarkEntry:
    return BenchmarkEntry(
        id=benchmark_id,
        name="Fake Bench",
        task="cls",
        catalog_id=benchmark_id,
        registry_id=None,
        upstream_split=UpstreamSplit(
            rule="test only",
            eval_split=eval_split,
            counts={eval_split: 2},
            definition_url="https://example.org",
        ),
        split_verified=True,
        verified_by="unit test fixture",
        obtain=Obtain(status="staged", via="unit test"),
        policy="exclude",
        policy_reason="unit test fixture",
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


RED = _png_bytes((255, 0, 0))
BLUE = _png_bytes((0, 0, 255))


class FakeS3Client:
    """Minimal ``list_objects_v2``/``get_object`` double — no network, no pagination."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(self, Bucket: str, Prefix: str, ContinuationToken: str | None = None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[Key])}


def _stream_part_bytes(rows: list[dict]) -> bytes:
    table = pa.Table.from_pylist(rows)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def test_iter_bucket_images_stream_parts_filters_eval_split():
    part = _stream_part_bytes(
        [
            {"stem": "a", "upstream_split": "test", "image": RED},
            {"stem": "b", "upstream_split": "train", "image": BLUE},
        ]
    )
    client = FakeS3Client({"sources/fakebench/_stream/rev-1/part-0.parquet": part})
    entry = _entry(eval_split="test")
    images = list(iter_bucket_images(client, "rs-storage-open", entry))
    assert [i.stem for i in images] == ["a"]
    assert images[0].data == RED


def test_iter_bucket_images_hf_style_bytes_struct_column():
    rows = [{"stem": "a", "upstream_split": "test", "image": {"bytes": RED, "path": "a.png"}}]
    table = pa.Table.from_pylist(rows)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    client = FakeS3Client({"sources/fakebench/_stream/rev-1/part-0.parquet": sink.getvalue()})
    images = list(iter_bucket_images(client, "rs-storage-open", _entry(eval_split="test")))
    assert images[0].data == RED


def test_iter_bucket_images_sample_schema_metadata_only_part():
    # Real marineeval shape (verified 2026-09-30): the _stream/ part carries only
    # metadata (stem, image_path, split_hint) — no embedded bytes column — and the
    # actual object lives in the sibling rev-dir (without the _stream/ segment).
    part = _stream_part_bytes(
        [
            {"stem": "a", "image_path": "images/a.jpg", "split_hint": "test"},
            {"stem": "b", "image_path": "images/b.jpg", "split_hint": "train"},
        ]
    )
    client = FakeS3Client(
        {
            "sources/fakebench/_stream/rev-1/part-0.parquet": part,
            "sources/fakebench/rev-1/images/a.jpg": RED,
            "sources/fakebench/rev-1/images/b.jpg": BLUE,
        }
    )
    images = list(iter_bucket_images(client, "rs-storage-open", _entry(eval_split="test")))
    assert [i.stem for i in images] == ["a"]
    assert images[0].data == RED
    assert images[0].upstream_path == "images/a.jpg"


def test_bucket_has_no_stream_prefix_raises_no_embedded_bytes():
    # staged-layout fallback with no metadata.parquet present -> KeyError surfaces
    client = FakeS3Client({})
    with pytest.raises(KeyError):
        list(iter_bucket_images(client, "rs-storage-open", _entry()))


class FakeDecoded:
    def __init__(self, upstream_id: str, data: bytes, split_hint: str | None) -> None:
        self.upstream_id = upstream_id
        self.data = data
        self.split_hint = split_hint


class FakeAdapter:
    def __init__(self, decoded: list[FakeDecoded]) -> None:
        self._decoded = decoded

    def samples(self, tmp_dir: Path):
        for d in self._decoded:
            yield None, None, d


def test_iter_upstream_images_filters_eval_split(tmp_path: Path):
    adapter = FakeAdapter(
        [
            FakeDecoded("test/a.png", RED, "test"),
            FakeDecoded("train/b.png", BLUE, "train"),
        ]
    )
    entry = _entry(eval_split="test")
    images = list(iter_upstream_images(entry, tmp_path, tmp_path, adapter=adapter))
    assert [i.stem for i in images] == ["a"]


def test_iter_upstream_images_eval_split_all_keeps_everything(tmp_path: Path):
    adapter = FakeAdapter([FakeDecoded("1.png", RED, None), FakeDecoded("2.png", BLUE, None)])
    entry = _entry(eval_split="all")
    images = list(iter_upstream_images(entry, tmp_path, tmp_path, adapter=adapter))
    assert len(images) == 2


def test_iter_upstream_images_max_bytes_stops_stream(tmp_path: Path):
    # RED/BLUE fixtures are small PNGs; three of them exceed a 2x-one-image cap.
    adapter = FakeAdapter(
        [
            FakeDecoded("1.png", RED, "all"),
            FakeDecoded("2.png", BLUE, "all"),
            FakeDecoded("3.png", RED, "all"),
        ]
    )
    entry = _entry(eval_split="all")
    images = list(
        iter_upstream_images(
            entry, tmp_path, tmp_path, adapter=adapter, max_bytes=len(RED) - 1
        )
    )
    assert [i.stem for i in images] == ["1"]


def test_spec_resolves(tmp_path: Path):
    (tmp_path / "u45.yaml").write_text("id: u45\n")
    assert spec_resolves(_entry("u45"), tmp_path) is True
    assert spec_resolves(_entry("nope"), tmp_path) is False


def test_build_manifest_computes_hash_columns():
    entry = _entry()
    images = iter([RawImage("a.png", "a", "test", RED)])
    table, n_before = build_manifest(entry, images, Path("/nonexistent/does-not-exist.parquet"))
    assert n_before == 0
    row = table.to_pylist()[0]
    assert row["sha256"] == hashlib.sha256(RED).hexdigest()
    assert row["embedding_ref"] == row["sha256"]
    assert row["embedding_model"] == "sscd_disc_mixup"
    assert set(row) == {
        "benchmark_id", "upstream_path", "stem", "upstream_split", "sha256", "pixel_sha256",
        "width", "height", "dhash", "phash", "phash64", "margin", "embedding_ref",
        "embedding_model",
    }


def test_build_manifest_is_resumable(tmp_path: Path):
    entry = _entry()
    existing = tmp_path / "fakebench.parquet"
    table1, _ = build_manifest(entry, iter([RawImage("a.png", "a", "test", RED)]), existing)
    write_manifest(table1, existing)
    # second run re-offers "a" (already present, must be skipped) plus a new "b"
    table2, n_before = build_manifest(
        entry,
        iter([RawImage("a.png", "a", "test", RED), RawImage("b.png", "b", "test", BLUE)]),
        existing,
    )
    assert n_before == 1
    assert sorted(r["stem"] for r in table2.to_pylist()) == ["a", "b"]


def test_build_manifest_no_rows_raises():
    entry = _entry()
    with pytest.raises(ManifestBuildError):
        build_manifest(entry, iter([]), Path("/nonexistent/does-not-exist.parquet"))


def test_resolve_source_auto_picks_bucket_when_checksums_present():
    client = FakeS3Client({"sources/fakebench/_stream/rev-1/CHECKSUMS.sha256": b""})
    assert resolve_source("auto", _entry(), client, "rs-storage-open") == "bucket"


def test_resolve_source_auto_picks_upstream_when_no_checksums():
    client = FakeS3Client({})
    assert resolve_source("auto", _entry(), client, "rs-storage-open") == "upstream"


def test_resolve_source_explicit_bypasses_bucket_check():
    assert resolve_source("upstream", _entry(), None, "rs-storage-open") == "upstream"
