"""Generic per-benchmark eval-image manifest builder (P1, design doc §1-2, §5).

``marinedata bench manifest <id> [--source bucket|upstream|auto]`` builds
``registry/benchmarks/manifests/<id>.parquet`` — the same 14 columns as the two
hand-built manifests from ``d376e95`` (coralscapes, suim): ``benchmark_id``,
``upstream_path``, ``stem``, ``upstream_split``, ``sha256``, ``pixel_sha256``,
``width``, ``height``, ``dhash``, ``phash``, ``phash64``, ``margin``,
``embedding_ref``, ``embedding_model``. ``embedding_ref`` is the image's own
``sha256`` (a pointer into a later SSCD embedding cache keyed by hash, not a
vector), so building never needs the embedder or its weights.

Two ways to get eval-split image bytes:

* ``bucket``: streams ``sources/<id>/_stream/rev-*/part-*.parquet`` (embedded
  bytes, like :func:`marinedata.dedup.corpus.iter_hf_images`) from
  ``rs-storage-open`` via ``boto3.get_object``, in memory — no disk staging.
  Falls back to the staged ``images/`` layout (``metadata.parquet`` + loose
  objects, like :func:`marinedata.dedup.corpus.iter_staged`) when no
  ``_stream`` prefix exists.
* ``upstream``: reuses the ingest adapter framework
  (:mod:`marinedata.adapters`, :class:`marinedata.ingest_source.IngestSpec`) —
  the same spec resolution ``ingest-batch`` uses — to stream the benchmark's
  eval-split members straight through, hash, and discard.

``auto`` picks ``bucket`` when a ``CHECKSUMS.sha256`` marker exists under
``sources/<id>/`` there, else ``upstream``.

Resumable: an existing manifest's ``stem``s are kept and only new ones are
hashed and appended; the whole table (old + new) is rewritten, since this is
a decontamination-gate input, never a live-appended log.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from .benchmarks import BenchmarkEntry, UpstreamSplit
from .dedup.features import FeatureError, features_from_bytes

SourceKind = Literal["bucket", "upstream", "auto"]

MANIFEST_COLUMNS = (
    "benchmark_id",
    "upstream_path",
    "stem",
    "upstream_split",
    "sha256",
    "pixel_sha256",
    "width",
    "height",
    "dhash",
    "phash",
    "phash64",
    "margin",
    "embedding_ref",
    "embedding_model",
)
EMBEDDING_MODEL = "sscd_disc_mixup"  # the model d376e95's hand-built manifests name
CHECKSUM_SUFFIX = "CHECKSUMS.sha256"


class ManifestBuildError(RuntimeError):
    """No streamable eval-split image source found for a benchmark (a hole in
    coverage; the design doc requires this to fail loudly, never silently)."""


@dataclass(frozen=True)
class RawImage:
    """One eval-split image, resolved down to bytes, before hashing."""

    upstream_path: str
    stem: str
    upstream_split: str
    data: bytes


def build_row(benchmark_id: str, img: RawImage) -> dict[str, Any]:
    feat = features_from_bytes(img.data)
    return {
        "benchmark_id": benchmark_id,
        "upstream_path": img.upstream_path,
        "stem": img.stem,
        "upstream_split": img.upstream_split,
        "sha256": feat.sha256,
        "pixel_sha256": feat.pixel_sha256,
        "width": feat.width,
        "height": feat.height,
        "dhash": f"{feat.dhash:016x}",
        "phash": feat.phash.hex(),
        "phash64": f"{feat.phash64:016x}",
        "margin": feat.margin,
        "embedding_ref": feat.sha256,
        "embedding_model": EMBEDDING_MODEL,
    }


# --------------------------------------------------------------------------- bucket


def _list_all(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return sorted(keys)
        token = resp["NextContinuationToken"]


def bucket_has_checksums(client: Any, bucket: str, benchmark_id: str) -> bool:
    keys = _list_all(client, bucket, f"sources/{benchmark_id}/")
    return any(k.endswith(CHECKSUM_SUFFIX) for k in keys)


def _image_bytes_field(schema: pa.Schema) -> tuple[str, str | None]:
    """``(column, subfield)`` — subfield is ``"bytes"`` for an HF-style
    ``{bytes, path}`` struct column, ``None`` for a plain binary column.
    Raises if the part carries no embedded image bytes (a keys-only layout,
    not yet wired here)."""
    for f in schema:
        if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type):
            return f.name, None
        if pa.types.is_struct(f.type) and any(c.name == "bytes" for c in f.type):
            return f.name, "bytes"
    raise ManifestBuildError(f"no embedded-bytes column in schema {schema.names}")


def _rev_prefix(part_key: str) -> str:
    """``sources/<id>/_stream/rev-x/part-N.parquet`` -> ``sources/<id>/rev-x/`` —
    the sibling revision directory that carries loose image objects for the
    "sample schema" stream layout, where the part is metadata-only (real
    ``marineeval`` shape, verified 2026-09-30: ``image_path`` column, no
    embedded bytes)."""
    without_stream = part_key.replace("_stream/", "", 1)
    return without_stream.rsplit("/", 1)[0] + "/"


def _get_object_bytes(client: Any, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def iter_bucket_images(client: Any, bucket: str, entry: BenchmarkEntry) -> Iterator[RawImage]:
    parts = [
        k
        for k in _list_all(client, bucket, f"sources/{entry.id}/_stream/")
        if k.endswith(".parquet")
    ]
    split = entry.upstream_split
    if parts:
        yield from _iter_stream_parts(client, bucket, parts, split)
        return
    yield from _iter_staged_bucket(client, bucket, entry.id, split)


def _iter_stream_parts(
    client: Any, bucket: str, parts: list[str], split: UpstreamSplit
) -> Iterator[RawImage]:
    eval_split = split.eval_split
    eval_splits = split.eval_splits
    for key in parts:
        table = pq.read_table(pa.BufferReader(_get_object_bytes(client, bucket, key)))
        names = table.schema.names
        split_col = next(
            (c for c in ("upstream_split", "split", "split_hint") if c in names), None
        )
        try:
            img_col, sub = _image_bytes_field(table.schema)
            external = False
        except ManifestBuildError:
            if "image_path" not in names:
                raise
            img_col, sub, external = "image_path", None, True
        path_col = next(
            (c for c in ("upstream_path", "key", "path", "image_path") if c in names), None
        )
        rev_prefix = _rev_prefix(key) if external else None
        for row in table.to_pylist():
            row_split = (row.get(split_col) if split_col else None) or eval_split
            if "all" not in eval_splits and row_split not in eval_splits:
                continue
            raw = row[img_col]
            if external:
                data = _get_object_bytes(client, bucket, rev_prefix + str(raw))
            else:
                data = raw[sub] if sub else raw
            upath = row.get(path_col) or row.get("stem") or ""
            stem = row.get("stem") or Path(str(upath)).stem
            yield RawImage(str(upath), stem, row_split, data)


def _iter_staged_bucket(
    client: Any, bucket: str, benchmark_id: str, split: UpstreamSplit
) -> Iterator[RawImage]:
    eval_split = split.eval_split
    eval_splits = split.eval_splits
    meta_key = f"sources/{benchmark_id}/metadata.parquet"
    meta_bytes = _get_object_bytes(client, bucket, meta_key)
    meta = pq.read_table(pa.BufferReader(meta_bytes)).to_pylist()
    root = f"sources/{benchmark_id}/images/"
    by_stem = {Path(k).stem: k for k in _list_all(client, bucket, root)}
    for row in meta:
        row_split = row.get("upstream_split") or eval_split
        if "all" not in eval_splits and row_split not in eval_splits:
            continue
        stem = row["stem"]
        key = by_stem.get(stem)
        if key is None:
            continue
        data = _get_object_bytes(client, bucket, key)
        yield RawImage(key[len(f"sources/{benchmark_id}/") :], stem, row_split, data)


# ------------------------------------------------------------------------- upstream


def _spec_path(entry: BenchmarkEntry, specs_dir: Path) -> Path:
    return specs_dir / f"{entry.registry_id or entry.id}.yaml"


def iter_upstream_images(
    entry: BenchmarkEntry,
    specs_dir: Path,
    tmp_dir: Path,
    adapter: Any | None = None,
    max_bytes: int | None = None,
) -> Iterator[RawImage]:
    """``adapter`` is injectable for tests; production always loads the same spec
    ``ingest-batch`` would (:meth:`marinedata.ingest_source.IngestSpec.load` +
    :func:`marinedata.adapters.make_adapter`).

    ``max_bytes``, when set, stops the stream once the cumulative size of
    fetched sample bytes (eval-split or not — the adapter has already
    downloaded them) reaches the cap. The run is resumable: a later call
    with the same ``tmp_dir``-backed manifest picks up from the stems
    already recorded in the existing parquet."""
    if adapter is None:
        from .adapters import make_adapter
        from .ingest_source import IngestSpec

        spec = IngestSpec.load(_spec_path(entry, specs_dir))
        adapter = make_adapter(spec.adapter, spec.params)
    eval_split = entry.upstream_split.eval_split
    eval_splits = entry.upstream_split.eval_splits
    total_bytes = 0
    for _item, _fetched, decoded in adapter.samples(tmp_dir):
        total_bytes += len(decoded.data)
        split = decoded.split_hint or eval_split
        if "all" in eval_splits or split in eval_splits:
            stem = Path(decoded.upstream_id).stem
            yield RawImage(decoded.upstream_id, stem, split, decoded.data)
        if max_bytes is not None and total_bytes >= max_bytes:
            break


def spec_resolves(entry: BenchmarkEntry, specs_dir: Path) -> bool:
    """Cheap, no-download check: does an ingest spec exist for this benchmark id?"""
    return _spec_path(entry, specs_dir).is_file()


# --------------------------------------------------------------------------- build


def resolve_source(
    requested: SourceKind, entry: BenchmarkEntry, client: Any, bucket: str
) -> Literal["bucket", "upstream"]:
    if requested != "auto":
        return requested
    return "bucket" if bucket_has_checksums(client, bucket, entry.id) else "upstream"


def build_manifest(
    entry: BenchmarkEntry, images: Iterator[RawImage], existing: Path, log: Any = print
) -> tuple[pa.Table, int]:
    """Returns ``(table, n_rows_before)`` — old rows kept verbatim, new ones hashed."""
    rows: list[dict[str, Any]] = []
    if existing.is_file():
        rows = pq.read_table(existing).to_pylist()
    seen = {r["stem"] for r in rows}
    n_before = len(rows)
    for img in images:
        if img.stem in seen:
            continue
        try:
            rows.append(build_row(entry.id, img))
        except FeatureError as exc:
            log(f"{entry.id}: skip {img.stem}: {exc}")
            continue
        seen.add(img.stem)
    if not rows:
        raise ManifestBuildError(f"{entry.id}: no eval-split images found (0 rows)")
    return pa.Table.from_pylist(rows, schema=None), n_before


def write_manifest(table: pa.Table, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
