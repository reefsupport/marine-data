"""Unit tests for the generic benchmark eval-image manifest builder (P1)."""

from __future__ import annotations

import hashlib
import io
import re
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


def _entry(
    benchmark_id: str = "fakebench",
    eval_split: str = "test",
    heldout_val: str | None = None,
) -> BenchmarkEntry:
    counts = {eval_split: 2}
    if heldout_val is not None:
        counts[heldout_val] = 2
    return BenchmarkEntry(
        id=benchmark_id,
        name="Fake Bench",
        task="cls",
        catalog_id=benchmark_id,
        registry_id=None,
        upstream_split=UpstreamSplit(
            rule="test only",
            eval_split=eval_split,
            heldout_val=heldout_val,
            counts=counts,
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


def test_iter_bucket_images_multi_split_includes_heldout_val():
    """usis10k-style entries pin ``eval_split: test`` + ``heldout_val: val`` —
    both must be pulled into the manifest, ``train`` must not (BENCH-evalsplit)."""
    part = _stream_part_bytes(
        [
            {"stem": "a", "upstream_split": "test", "image": RED},
            {"stem": "b", "upstream_split": "val", "image": BLUE},
            {"stem": "c", "upstream_split": "train", "image": RED},
        ]
    )
    client = FakeS3Client({"sources/fakebench/_stream/rev-1/part-0.parquet": part})
    entry = _entry(eval_split="test", heldout_val="val")
    images = list(iter_bucket_images(client, "rs-storage-open", entry))
    assert sorted(i.stem for i in images) == ["a", "b"]


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


def test_iter_upstream_images_multi_split_includes_heldout_val(tmp_path: Path):
    """fathomnet-vme-style entries pin ``eval_split: test`` + ``heldout_val: val`` —
    decontamination must cover both held-out sets, not just ``eval_split`` (BENCH-evalsplits)."""
    adapter = FakeAdapter(
        [
            FakeDecoded("test/a.png", RED, "test"),
            FakeDecoded("val/b.png", BLUE, "val"),
            FakeDecoded("train/c.png", RED, "train"),
        ]
    )
    entry = _entry(eval_split="test", heldout_val="val")
    images = list(iter_upstream_images(entry, tmp_path, tmp_path, adapter=adapter))
    assert sorted(i.stem for i in images) == ["a", "b"]


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


def test_build_manifest_raises_loud_on_empty_bytes_not_a_silent_skip():
    """BENCH-fix3 B guardrail: zero-length image bytes is a decoder/adapter bug (the
    empty-bytes sha256 ``e3b0c442...``), never an ordinary per-item skip — it must
    raise immediately rather than let `build_manifest` log-and-continue through every
    row and surface only as a misleading "0 rows found"."""
    entry = _entry()
    images = iter([RawImage("a.png", "a", "test", b"")])
    with pytest.raises(ManifestBuildError, match="zero-length image bytes"):
        build_manifest(entry, images, Path("/nonexistent/does-not-exist.parquet"))


def test_iter_upstream_images_skips_label_only_empty_data(tmp_path: Path):
    """A label-only Decoded (annotation JSON paired by label_files, or an unresolved
    caption-json ref) always carries ``data == b""`` and must never be treated as an
    eval image — this is the fathomnet-vme/uiis root cause."""
    adapter = FakeAdapter(
        [
            FakeDecoded("annotations/train.json", b"", "train"),
            FakeDecoded("train/b.png", BLUE, "train"),
        ]
    )
    entry = _entry(eval_split="train")
    images = list(iter_upstream_images(entry, tmp_path, tmp_path, adapter=adapter))
    assert [i.stem for i in images] == ["b"]


def test_iter_upstream_images_stem_unique_across_hash_fragment(tmp_path: Path):
    """Two synthetic ``key#i`` ids sharing the same base file (e.g. one caption-json
    record per image) must not collapse to the same stem via naive ``Path(...).stem``
    truncation — that silently drops every record but the first in build_manifest's
    dedup-by-stem."""
    adapter = FakeAdapter(
        [
            FakeDecoded("coco_test.json#0", RED, "test"),
            FakeDecoded("coco_test.json#1", BLUE, "test"),
        ]
    )
    entry = _entry(eval_split="test")
    images = list(iter_upstream_images(entry, tmp_path, tmp_path, adapter=adapter))
    assert len({i.stem for i in images}) == 2


def test_resolve_source_auto_picks_bucket_when_checksums_present():
    client = FakeS3Client({"sources/fakebench/_stream/rev-1/CHECKSUMS.sha256": b""})
    assert resolve_source("auto", _entry(), client, "rs-storage-open") == "bucket"


def test_resolve_source_auto_picks_upstream_when_no_checksums():
    client = FakeS3Client({})
    assert resolve_source("auto", _entry(), client, "rs-storage-open") == "upstream"


def test_resolve_source_explicit_bypasses_bucket_check():
    assert resolve_source("upstream", _entry(), None, "rs-storage-open") == "upstream"


# --------------------------------------------------------------------- BENCH-checkpoint


def test_build_manifest_checkpoints_at_row_threshold(tmp_path: Path):
    entry = _entry()
    ckpt = tmp_path / "fakebench.parquet"
    mid_run: dict[str, object] = {}

    def images():
        yield RawImage("a.png", "a", "test", RED)
        yield RawImage("b.png", "b", "test", BLUE)
        # a checkpoint must already exist once the tiny 2-row threshold is crossed,
        # well before the generator (and build_manifest) finishes
        mid_run["exists"] = ckpt.is_file()
        if ckpt.is_file():
            mid_run["rows"] = len(pq.read_table(ckpt).to_pylist())
        yield RawImage("c.png", "c", "test", RED)

    build_manifest(
        entry, images(), ckpt, checkpoint_every_rows=2, checkpoint_every_s=10_000
    )
    assert mid_run["exists"] is True
    assert mid_run["rows"] == 2


def test_build_manifest_checkpoints_on_interrupt_and_resumes(tmp_path: Path):
    entry = _entry()
    ckpt = tmp_path / "fakebench.parquet"

    def images_then_interrupt():
        yield RawImage("a.png", "a", "test", RED)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        build_manifest(entry, images_then_interrupt(), ckpt, checkpoint_every_rows=1)

    assert ckpt.is_file()
    assert [r["stem"] for r in pq.read_table(ckpt).to_pylist()] == ["a"]

    # simulated rerun: resumes from the checkpoint, re-offers "a" (skipped) plus "b"
    table2, n_before = build_manifest(
        entry,
        iter([RawImage("a.png", "a", "test", RED), RawImage("b.png", "b", "test", BLUE)]),
        ckpt,
    )
    assert n_before == 1
    assert sorted(r["stem"] for r in table2.to_pylist()) == ["a", "b"]


def test_build_manifest_progress_line_format(capsys: pytest.CaptureFixture[str]):
    entry = _entry()
    images = iter([RawImage("a.png", "a", "test", RED)])
    build_manifest(
        entry, images, Path("/nonexistent/does-not-exist.parquet"),
        progress_every=1, progress_every_s=10_000,
    )
    err = capsys.readouterr().err
    assert re.search(
        r"^fakebench rows=1 bytes=[\d.]+MB rate=[\d.]+img/min skipped=0$", err, re.MULTILINE
    )
