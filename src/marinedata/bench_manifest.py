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

BENCH-checkpoint: a long ``upstream`` build (e.g. fathomnet-vme fetching every
image by its own ``source_url``, one MBARI request each) used to run silent
and write the manifest only at the very end — a caller-side wall-clock alarm
killing it lost every row. :func:`build_manifest` now prints progress and
checkpoints the parquet mid-run (atomically), and :func:`iter_upstream_images`
bounds each upstream fetch so one hung request can no longer stall the whole
stream indefinitely.
"""

from __future__ import annotations

import itertools
import os
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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

# BENCH-checkpoint defaults (design doc addendum, 2026-10-01).
PROGRESS_EVERY_IMAGES = 100
PROGRESS_EVERY_S = 60.0
CHECKPOINT_EVERY_ROWS = 500
CHECKPOINT_EVERY_S = 300.0
REQUEST_TIMEOUT_S = 60.0
REQUEST_MAX_RETRIES = 3

_TERMINATING_SIGNALS = tuple(
    sig for sig in (getattr(signal, "SIGALRM", None), getattr(signal, "SIGTERM", None)) if sig
)


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


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def _get_with_retry(client: Any, bucket: str, key: str, tries: int = 3) -> bytes:
    for attempt in range(tries):
        try:
            return _get_object_bytes(client, bucket, key)
        except Exception:  # transient network/5xx; the last attempt re-raises
            if attempt == tries - 1:
                raise
            time.sleep(1.0 + attempt)
    raise AssertionError("unreachable")


_DONE = object()
STAGED_GET_WORKERS = 6


def _ordered_map(
    fn: Callable[[Any], Any], items: Iterable[Any], workers: int = STAGED_GET_WORKERS
) -> Iterator[tuple[Any, Any]]:
    """``(item, fn(item))`` in input order with ``workers`` calls in flight — bounded, so a
    long stream never holds more than ``4 * workers`` results in memory."""
    it = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        window: deque[tuple[Any, Any]] = deque(
            (item, pool.submit(fn, item)) for item in itertools.islice(it, workers * 4)
        )
        while window:
            item, fut = window.popleft()
            result = fut.result()
            nxt = next(it, _DONE)
            if nxt is not _DONE:
                window.append((nxt, pool.submit(fn, nxt)))
            yield item, result


def _norm(name: Any) -> str:
    return str(name).strip().lower()


def _dir_parts(path: str) -> list[str]:
    """Lowercased directory names of an upstream path (``a.zip#d/x.png`` -> ``[a.zip, d]``)."""
    parts = [p for p in str(path).replace("#", "/").replace("\\", "/").split("/") if p]
    return [_norm(p) for p in parts[:-1]]


def resolve_eval_split(row_split: Any, upstream_path: str, split: UpstreamSplit) -> str | None:
    """The canonical eval-split name a row belongs to, or ``None`` when it is not eval.

    A labelled row is matched case-insensitively against ``eval_splits`` (the adapter says
    ``val``, the registry ``VAL``). An *unlabelled* row falls back to its directory names
    (``.../TE/RGB/x.png`` -> ``TE``; a directory also matches when it starts with a split
    name of 3+ letters, ``Validate`` -> ``val``) — never to a guess (BENCH-trashsplit):
    no label and no matching directory means excluded, unless every row is eval (``all``).
    ``image_subdir`` additionally drops masks/depth maps shipped beside the images.
    """
    eval_splits = sorted(split.eval_splits)
    dirs = _dir_parts(upstream_path)
    if split.image_subdir and _norm(split.image_subdir) not in dirs:
        return None
    if "all" in eval_splits:
        return str(row_split) if row_split is not None else split.eval_split
    if row_split is not None:
        return next((s for s in eval_splits if _norm(s) == _norm(row_split)), None)
    for name in eval_splits:
        n = _norm(name)
        if any(d == n or (len(n) >= 3 and d.startswith(n)) for d in dirs):
            return name
    return None


def iter_bucket_images(
    client: Any, bucket: str, entry: BenchmarkEntry, layout: str = "auto"
) -> Iterator[RawImage]:
    parts = (
        []
        if layout == "staged"
        else [
            k
            for k in _list_all(client, bucket, f"sources/{entry.id}/_stream/")
            if k.endswith(".parquet")
        ]
    )
    split = entry.upstream_split
    if parts:
        yield from _iter_stream_parts(client, bucket, parts, split)
        return
    yield from _iter_staged_bucket(client, bucket, entry.id, split)


def _iter_stream_parts(
    client: Any, bucket: str, parts: list[str], split: UpstreamSplit
) -> Iterator[RawImage]:
    for key in parts:
        table = pq.read_table(pa.BufferReader(_get_object_bytes(client, bucket, key)))
        names = table.schema.names
        split_col = next((c for c in ("upstream_split", "split", "split_hint") if c in names), None)
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
            upath = row.get(path_col) or row.get("stem") or ""
            row_split = resolve_eval_split(row.get(split_col) if split_col else None, upath, split)
            if row_split is None:
                continue
            raw = row[img_col]
            if external:
                data = _get_object_bytes(client, bucket, rev_prefix + str(raw))
            else:
                data = raw[sub] if sub else raw
            stem = row.get("stem") or Path(str(upath)).stem
            yield RawImage(str(upath), stem, row_split, data)


def _staged_root(keys: list[str], benchmark_id: str) -> str:
    """The prefix holding ``metadata.parquet`` + ``images/``: the flat
    ``sources/<id>/`` layout, else the newest ``sources/<id>/<rev>/`` revision."""
    flat = f"sources/{benchmark_id}/metadata.parquet"
    if flat in keys:
        return f"sources/{benchmark_id}/"
    revs = sorted(k for k in keys if k.endswith("/metadata.parquet") and "/_stream/" not in k)
    if not revs:
        raise ManifestBuildError(
            f"{benchmark_id}: no metadata.parquet under sources/{benchmark_id}/"
        )
    return revs[-1].rsplit("/", 1)[0] + "/"


def _iter_staged_bucket(
    client: Any, bucket: str, benchmark_id: str, split: UpstreamSplit
) -> Iterator[RawImage]:
    keys = _list_all(client, bucket, f"sources/{benchmark_id}/")
    root = _staged_root(keys, benchmark_id)
    meta = pq.read_table(
        pa.BufferReader(_get_object_bytes(client, bucket, root + "metadata.parquet"))
    ).to_pylist()
    present = set(keys)
    by_stem = {Path(k).stem: k for k in keys if k.startswith(root + "images/")}
    jobs: list[tuple[str, str, str, str]] = []
    for row in meta:
        upath = row.get("upstream_path") or row.get("upstream_id") or row.get("stem") or ""
        row_split = resolve_eval_split(row.get("upstream_split"), upath, split)
        if row_split is None:
            continue
        stem = row["stem"]
        image_path = row.get("image_path")
        key = root + str(image_path) if image_path and root + str(image_path) in present else None
        key = key or by_stem.get(stem)
        if key is None:
            continue
        jobs.append(
            (key, str(upath) if row.get("upstream_path") else key[len(root) :], stem, row_split)
        )
    for (_, shown, stem, row_split), data in _ordered_map(
        lambda job: _get_with_retry(client, bucket, job[0]), jobs
    ):
        yield RawImage(shown, stem, row_split, data)


# ------------------------------------------------------------------------- upstream


def _spec_path(entry: BenchmarkEntry, specs_dir: Path) -> Path:
    return specs_dir / f"{entry.registry_id or entry.id}.yaml"


def iter_upstream_images(
    entry: BenchmarkEntry,
    specs_dir: Path,
    tmp_dir: Path,
    adapter: Any | None = None,
    max_bytes: int | None = None,
    request_timeout_s: float = REQUEST_TIMEOUT_S,
    max_retries: int = REQUEST_MAX_RETRIES,
    log: Any = _log_stderr,
) -> Iterator[RawImage]:
    """``adapter`` is injectable for tests; production always loads the same spec
    ``ingest-batch`` would (:meth:`marinedata.ingest_source.IngestSpec.load` +
    :func:`marinedata.adapters.make_adapter`).

    ``max_bytes``, when set, stops the stream once the cumulative size of
    fetched sample bytes (eval-split or not — the adapter has already
    downloaded them) reaches the cap. The run is resumable: a later call
    with the same ``tmp_dir``-backed manifest picks up from the stems
    already recorded in the existing parquet.

    BENCH-checkpoint: some adapters (e.g. the caption/COCO decoder resolving
    each record's own ``source_url``) make one network request per image with
    no bound of their own. Each ``next()`` step is run on a single worker
    thread and bounded to ``request_timeout_s``; a stall counts as a retry
    and, past ``max_retries`` consecutive stalls, that fetch slot is logged
    and skipped (never silently) so one hung request can't stall the whole
    stream forever — the caller's wall-clock alarm is still the hard bound.
    """
    if adapter is None:
        from .adapters import make_adapter
        from .ingest_source import IngestSpec

        spec = IngestSpec.load(_spec_path(entry, specs_dir))
        adapter = make_adapter(
            spec.adapter,
            {**spec.params, "eval_splits": sorted(entry.upstream_split.eval_splits)},
        )
    eval_split = entry.upstream_split.eval_split
    eval_splits = entry.upstream_split.eval_splits
    total_bytes = 0
    it = iter(adapter.samples(tmp_dir))
    stalls = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            future = pool.submit(next, it)
            try:
                _item, _fetched, decoded = future.result(timeout=request_timeout_s)
            except StopIteration:
                return
            except FutureTimeoutError:
                stalls += 1
                log(
                    f"{entry.id}: fetch stalled past {request_timeout_s:.0f}s "
                    f"(attempt {stalls}/{max_retries})"
                )
                if stalls >= max_retries:
                    log(f"{entry.id}: skip — giving up after {max_retries} stalled fetches")
                    stalls = 0
                continue
            stalls = 0
            total_bytes += len(decoded.data)
            # WP-BENCH-fix3 B: label-only / unresolved Decoded records (annotation JSON
            # paired by ``label_files``, or a caption-json ref that never resolved) always
            # carry ``data == b""``. They are never eval images; treating them as one was
            # exactly why fathomnet-vme/uiis manifests were all `cannot decode image
            # e3b0c44...` (sha256 of empty bytes) — the hash of nothing, not real corruption.
            if not decoded.data:
                continue
            # BENCH-trashsplit: an image with no split evidence at all (no annotation-file
            # split, no train/val/test path segment — e.g. TrashCan's loose
            # ``original_data/*``) must never be guessed into the eval split. Only
            # ``eval_split: all`` entries (eval-only datasets with no split structure to
            # read, e.g. u45) still label such images with ``eval_split`` as a fallback.
            split = decoded.split_hint
            if split is None:
                if "all" not in eval_splits:
                    continue
                split = eval_split
            if "all" in eval_splits or split in eval_splits:
                stem = _stem(decoded.upstream_id)
                yield RawImage(decoded.upstream_id, stem, split, decoded.data)
            if max_bytes is not None and total_bytes >= max_bytes:
                break


def _stem(upstream_id: str) -> str:
    """``Path(upstream_id).stem`` alone collapses every synthetic ``key#i`` id (e.g. one
    caption-json record per image) onto the SAME stem (``coco_test``) since
    ``Path.stem`` only strips the outer ``.json#i`` suffix — a silent dedup-by-stem
    collision in :func:`build_manifest` that would keep just one row per source file.
    Strip the real extension off the base path only; keep the ``#frag`` discriminator
    verbatim so each record stays a distinct stem."""
    base, sep, frag = upstream_id.partition("#")
    return Path(base).stem + sep + frag


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


def _atomic_write_manifest(table: pa.Table, out_path: Path) -> None:
    """Same bytes as :func:`write_manifest`, but crash-safe: write to a sibling
    tmp file and ``os.replace`` it over the target, so a checkpoint mid-write can
    never leave a half-written (or truncated) parquet behind."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp-{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)


def build_manifest(
    entry: BenchmarkEntry,
    images: Iterator[RawImage],
    existing: Path,
    log: Any = print,
    *,
    out_path: Path | None = None,
    checkpoint_every_rows: int = CHECKPOINT_EVERY_ROWS,
    checkpoint_every_s: float = CHECKPOINT_EVERY_S,
    progress_every: int = PROGRESS_EVERY_IMAGES,
    progress_every_s: float = PROGRESS_EVERY_S,
) -> tuple[pa.Table, int]:
    """Returns ``(table, n_rows_before)`` — old rows kept verbatim, new ones hashed.

    BENCH-checkpoint: prints one progress line to stderr every
    ``progress_every`` images or ``progress_every_s`` seconds, and atomically
    checkpoints the in-progress manifest (to ``out_path``, default ``existing``
    — the same stem-keyed file a rerun resumes from) every
    ``checkpoint_every_rows`` new rows or ``checkpoint_every_s`` seconds, and
    once more on ``KeyboardInterrupt``/``SIGALRM``/``SIGTERM`` before
    re-raising/exiting. The final write (by the caller, via
    :func:`write_manifest`) is unchanged.
    """
    ckpt_path = out_path or existing
    rows: list[dict[str, Any]] = []
    if existing.is_file():
        rows = pq.read_table(existing).to_pylist()
    seen = {r["stem"] for r in rows}
    n_before = len(rows)

    def _checkpoint_now() -> None:
        if rows:
            _atomic_write_manifest(pa.Table.from_pylist(rows, schema=None), ckpt_path)

    def _signal_checkpoint(signum: int, _frame: Any) -> None:
        _checkpoint_now()
        log(f"{entry.id}: checkpoint on signal {signum}, {len(rows)} rows -> {ckpt_path}")
        raise SystemExit(f"{entry.id}: interrupted by signal {signum}")

    previous_handlers = {
        sig: signal.signal(sig, _signal_checkpoint) for sig in _TERMINATING_SIGNALS
    }
    start = time.monotonic()
    last_progress_t = start
    last_checkpoint_t = start
    rows_since_checkpoint = 0
    processed = 0
    skipped = 0
    bytes_total = 0
    try:
        for img in images:
            processed += 1
            bytes_total += len(img.data)

            now = time.monotonic()
            if processed % progress_every == 0 or (now - last_progress_t) >= progress_every_s:
                elapsed_min = max((now - start) / 60.0, 1e-9)
                _log_stderr(
                    f"{entry.id} rows={len(rows)} bytes={bytes_total / 1e6:.1f}MB "
                    f"rate={processed / elapsed_min:.1f}img/min skipped={skipped}"
                )
                last_progress_t = now

            if img.stem in seen:
                continue
            if not img.data:
                # WP-BENCH-fix3 B: an empty-bytes RawImage is never per-item corruption
                # (that is FeatureError, below) — it means an upstream code path handed
                # the decoder a zero-length payload. Silently `log`-and-skip let every
                # row in a benchmark fail the same way and masked it as an ordinary "0
                # rows" result; raise loud.
                raise ManifestBuildError(
                    f"{entry.id}: {img.stem} ({img.upstream_path}): zero-length image bytes "
                    "from upstream — decoder/adapter bug, not a per-item skip"
                )
            try:
                rows.append(build_row(entry.id, img))
            except FeatureError as exc:
                log(f"{entry.id}: skip {img.stem}: {exc}")
                skipped += 1
                continue
            seen.add(img.stem)
            rows_since_checkpoint += 1

            due_rows = rows_since_checkpoint >= checkpoint_every_rows
            due_time = (now - last_checkpoint_t) >= checkpoint_every_s
            if due_rows or due_time:
                _checkpoint_now()
                rows_since_checkpoint = 0
                last_checkpoint_t = now
    except KeyboardInterrupt:
        _checkpoint_now()
        log(f"{entry.id}: checkpoint on KeyboardInterrupt, {len(rows)} rows -> {ckpt_path}")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if not rows:
        raise ManifestBuildError(f"{entry.id}: no eval-split images found (0 rows)")
    return pa.Table.from_pylist(rows, schema=None), n_before


def write_manifest(table: pa.Table, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
