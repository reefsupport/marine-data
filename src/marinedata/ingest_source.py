"""``marinedata ingest-source``: any open source -> ``sources/<id>/<version>/`` on S3 (WP-6).

source (adapter) -> staged tree in a bounded temp dir -> streamed, resumable upload ->
verify every key -> registry stub. See ``docs/design/ingestion.md`` for the state
machine; ``docs/ingest-howto.md`` for the recipe.

Resume model: staging is deterministic (pinned upstream revision, sorted enumeration,
byte-identical shard tars), so a killed run is resumed by re-running the same command.
Every file already on S3 with equal size + ETag/sha256 is skipped, and a partially sent
multipart upload continues from its checkpoint. ``CHECKSUMS.sha256`` is uploaded LAST:
its presence on S3 is the "version complete" marker.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import checksums, sample_schema
from .adapters import SPOOLED, make_adapter, suffix_of
from .adapters._http import HashingReader
from .adapters.decode import IMAGE_SUFFIXES
from .concurrency import HostLimiter, retry_with_backoff
from .s3_upload import DEFAULT_PART, DiskGuard, GiB, local_digest, upload_file
from .staged_writer import DEFAULT_SHARD_BYTES, DEFAULT_THRESHOLD, StagedWriter, WriterConfig
from .subset_filter import SubsetFilter

_BIG_PREFIXES = ("images/", "labels/")
_CONTAINERS = (".tar", ".tar.gz", ".tgz", *SPOOLED)

DEFAULT_JOBS = 8
MAX_JOBS = 32
DEFAULT_MAX_PER_HOST = 4
DEFAULT_PART_JOBS = 4
# A killed-then-resumed spooled fetch can wait this long for another in-flight fetch to
# drain (guard.reserve_blocking) before it gives up and raises TempCapError (D-L).
_RESERVE_WAIT_S = 300.0


def _materialize(fetched: Any) -> Any:
    """Fully drain a *simple* (non-container) stream now, in the fetch worker thread —
    this is the network-bound part worth overlapping across ``--jobs``. ``decode()``
    then replays the bytes through a fresh :class:`HashingReader`, so its hash/size
    accounting (and upstream-digest verification) is unchanged; only when the bytes
    were transferred moves earlier. Tar/zip/parquet containers are untouched: they are
    read member-by-member to keep memory bounded (see staged_writer/decode module docs).
    """
    if fetched.stream is not None:
        data = fetched.stream.raw.read()
        fetched.stream.raw.close()
        fetched.stream = HashingReader(io.BytesIO(data))
    return fetched


def _fetch_one(
    adapter: Any,
    item: Any,
    tmp: Path,
    guard: DiskGuard,
    limiter: HostLimiter,
    *,
    reserve_timeout: float = 0.0,
) -> Any:
    """One item's fetch, with (D-L) disk-budget admission for spooled downloads and
    (D-G) per-host politeness + retry-with-backoff-and-jitter for the network call.
    Used on both the sequential (``--jobs 1``) and pooled paths, so behaviour is
    identical either way — only *when* the bytes move differs. ``reserve_timeout=0``
    (the sequential path) fails fast exactly like the pre-WP-6b ``guard.reserve()``;
    the pooled path waits (``_RESERVE_WAIT_S``) for a sibling fetch to drain instead."""
    if suffix_of(item.key) in SPOOLED:
        guard.reserve_blocking(item.size or 0, timeout=reserve_timeout)
    with limiter.acquire(item.url):
        fetched = retry_with_backoff(lambda: adapter.fetch(item, tmp))
    if suffix_of(item.key) not in _CONTAINERS:
        fetched = _materialize(fetched)
    return fetched


class _Prefetcher:
    """Bounded (``--jobs``) sliding window of fetch futures, consumed strictly in
    enumeration order — so staging (shard offsets, sample order, CHECKSUMS) is exactly
    as deterministic as the ``--jobs 1`` sequential path regardless of N (fetch
    *completion* order may differ; *consumption* order never does)."""

    def __init__(
        self,
        adapter: Any,
        items: list,
        tmp: Path,
        guard: DiskGuard,
        jobs: int,
        limiter: HostLimiter,
    ) -> None:
        self._items = deque(items)
        self._adapter, self._tmp, self._guard, self._limiter = adapter, tmp, guard, limiter
        self.pool = ThreadPoolExecutor(max_workers=jobs)
        self._futures: deque[Future] = deque()
        for _ in range(jobs):
            self._submit_next()

    def _submit_next(self) -> None:
        if not self._items:
            return
        item = self._items.popleft()
        self._futures.append(
            self.pool.submit(
                _fetch_one,
                self._adapter,
                item,
                self._tmp,
                self._guard,
                self._limiter,
                reserve_timeout=_RESERVE_WAIT_S,
            )
        )

    def __bool__(self) -> bool:
        return bool(self._futures)

    def next(self) -> Any:
        fut = self._futures.popleft()
        self._submit_next()
        return fut.result()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


@dataclass
class IngestSpec:
    id: str
    adapter: str
    params: dict[str, Any]
    license: str
    attribution: str
    citation: str = ""
    homepage: str = ""
    notes: str = ""
    version: str | None = None
    layout: str = "auto"
    expected_images: int | None = None
    shard_threshold: int = DEFAULT_THRESHOLD
    shard_bytes: int = DEFAULT_SHARD_BYTES
    defaults: dict[str, Any] = field(default_factory=dict)
    naive_datetime_is_utc: bool = False
    label_stem_suffix: str = ""
    lineage_root_digest: str | None = None
    max_images: int | None = None
    bucket: str = "rs-storage-open"
    prefix: str = "sources"
    remote: str = "rs-hel1"
    temp_cap_gb: float = 6.0
    disk_floor_gib: float = 40.0
    # Documentation-only blocks (SPEC-w3): ignored by the runner, validated by
    # marinedata.ingest_subset.check_spec. `measured` = totals read from upstream
    # metadata; `subset` = the stratified target for sources > 1M items or > 1 TB.
    measured: dict[str, Any] = field(default_factory=dict)
    subset: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, adapter: str | None = None) -> IngestSpec:
        raw = yaml.safe_load(path.read_text()) or {}
        if adapter:
            if raw.get("adapter") not in (None, adapter):
                raise ValueError(f"spec says adapter {raw['adapter']!r}, CLI says {adapter!r}")
            raw["adapter"] = adapter
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"{path}: unknown spec keys {unknown}")
        spec = cls(**raw)
        if not spec.license.strip() or not spec.attribution.strip():
            raise ValueError(f"{path}: license and attribution are required (D-C)")
        return spec


@dataclass
class IngestReport:
    source_id: str
    version: str
    prefix: str
    layout: str
    images: int = 0
    files: int = 0
    bytes: int = 0
    uploaded: int = 0
    skipped: int = 0
    verified: int = 0
    root_digest: str = ""
    peak_temp_bytes: int = 0
    dry_run: bool = False
    plan: dict[str, Any] = field(default_factory=dict)


def _choose_layout(spec: IngestSpec, items: list) -> str:
    if spec.layout in {"objects", "shards"}:
        return spec.layout
    n = spec.expected_images
    if n is None and items and not any(suffix_of(i.key) in _CONTAINERS for i in items):
        n = sum(suffix_of(i.key) in IMAGE_SUFFIXES for i in items)  # loose images (+labels)
    if n is None and spec.max_images is not None:
        n = spec.max_images
    return "shards" if n is not None and n > spec.shard_threshold else "objects"


class _Uploader:
    """Upload closed files, then delete the local copy (D-F: only after verify)."""

    def __init__(
        self,
        client: Any,
        spec: IngestSpec,
        key_prefix: str,
        root: Path,
        work: Path,
        part_size: int,
        report: IngestReport,
        part_jobs: int = 1,
    ) -> None:
        self.client, self.spec, self.key_prefix, self.root = client, spec, key_prefix, root
        self.ckpt = work / "checkpoints"
        self.ledger_path = work / "uploaded.json"
        self.ledger: dict[str, dict[str, Any]] = (
            json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        )
        self.part_size, self.report, self.part_jobs = part_size, report, part_jobs
        self._lock = threading.Lock()  # WP-6b: push() may run from several PUT threads

    def push(self, rel: str) -> None:
        path = self.root / rel
        d = local_digest(path, self.part_size)
        res = upload_file(
            self.client,
            self.spec.bucket,
            f"{self.key_prefix}/{rel}",
            path,
            checkpoint_dir=self.ckpt,
            part_size=self.part_size,
            digest=d,
            part_jobs=self.part_jobs,
        )
        with self._lock:
            self.ledger[rel] = {"sha256": d.sha256, "size": d.size, "etag": res.etag}
            tmp = self.ledger_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.ledger, sort_keys=True))
            tmp.replace(self.ledger_path)
            self.report.skipped += int(res.skipped)
            self.report.uploaded += int(not res.skipped)
        if rel.startswith(_BIG_PREFIXES):
            path.unlink()

    def verify(self, rels: list[str]) -> int:
        ok = 0
        for rel in rels:
            want = self.ledger[rel]
            head = self.client.head_object(Bucket=self.spec.bucket, Key=f"{self.key_prefix}/{rel}")
            etag = str(head["ETag"]).strip('"')
            if int(head["ContentLength"]) != want["size"] or (
                etag != want["etag"]
                and (head.get("Metadata") or {}).get("sha256") != want["sha256"]
            ):
                raise RuntimeError(f"verify failed: {rel}")
            ok += 1
        return ok


def _stub(spec: IngestSpec, report: IngestReport) -> str:
    stub = {
        "id": spec.id,
        "name": spec.id,
        "version": report.version,
        "licence": spec.license,
        "attribution": spec.attribution,
        "citation": spec.citation or None,
        "homepage": spec.homepage or None,
        "access": {"method": spec.adapter, "params": spec.params},
        "items": report.images,
        "staged": {
            "bucket": spec.bucket,
            "prefix": report.prefix,
            "root_digest": report.root_digest,
            "layout": report.layout,
            "files": report.files,
            "bytes": report.bytes,
        },
    }
    return (
        "# registry stub written by `marinedata ingest-source` — complete description,\n"
        "# capabilities, annotations and coverage before merging into registry/sources.\n"
        + yaml.safe_dump(stub, sort_keys=False, allow_unicode=True)
    )


def fetch_throughput(
    spec: IngestSpec,
    work: Path,
    *,
    jobs: int = 1,
    max_per_host: int = DEFAULT_MAX_PER_HOST,
    limit: int | None = None,
) -> dict[str, Any]:
    """WP-6b ``--fetch-only`` benchmark: drain up to ``limit`` items through the exact
    fetch path ``run_ingest`` uses (retry+jitter, per-host limiting, `--jobs` prefetch) —
    no staging, no S3 — to measure files/s scaling of ``--jobs`` alone."""
    jobs = max(1, min(int(jobs), MAX_JOBS))
    adapter = make_adapter(spec.adapter, spec.params)
    items = list(adapter.enumerate())
    if limit is not None:
        items = items[:limit]
    tmp = work / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(work, int(spec.temp_cap_gb * GiB), int(spec.disk_floor_gib * GiB))
    limiter = HostLimiter(max_per_host)
    started = time.monotonic()
    if jobs > 1:
        prefetcher = _Prefetcher(adapter, items, tmp, guard, jobs, limiter)
        try:
            for _ in items:
                prefetcher.next()
        finally:
            prefetcher.close()
    else:
        for item in items:
            _fetch_one(adapter, item, tmp, guard, limiter)
    elapsed = time.monotonic() - started
    return {
        "items": len(items),
        "jobs": jobs,
        "seconds": elapsed,
        "files_per_s": (len(items) / elapsed) if elapsed > 0 else float(len(items)),
    }


def run_ingest(
    spec: IngestSpec,
    work: Path,
    *,
    client: Any = None,
    dry_run: bool = False,
    part_size: int = DEFAULT_PART,
    guard: DiskGuard | None = None,
    fetch_date: dt.date | None = None,
    jobs: int = DEFAULT_JOBS,
    max_per_host: int = DEFAULT_MAX_PER_HOST,
    part_jobs: int = DEFAULT_PART_JOBS,
) -> IngestReport:
    jobs = max(1, min(int(jobs), MAX_JOBS))
    adapter = make_adapter(spec.adapter, spec.params)
    version = spec.version or adapter.resolve_version()
    if spec.version:
        adapter.resolve_version()  # still pin + gate-check upstream
    items = list(adapter.enumerate())
    layout = _choose_layout(spec, items)
    key_prefix = f"{spec.prefix}/{spec.id}/{version}"
    report = IngestReport(spec.id, version, key_prefix, layout, dry_run=dry_run)
    report.plan = {
        "items": len(items),
        "declared_bytes": sum(i.size or 0 for i in items),
        "first_keys": [i.key for i in items[:5]],
        "target": f"s3://{spec.bucket}/{key_prefix}/",
    }
    # WP-6e-B: `subset.enforce: true` -> per-sample predicates + per-group cap (D-AB).
    subset = SubsetFilter.from_spec(spec.subset, Path(__file__).resolve().parents[2])
    if subset is not None:
        report.plan["subset"] = "enforced"
    if dry_run:
        return report
    guard = guard or DiskGuard(work, int(spec.temp_cap_gb * GiB), int(spec.disk_floor_gib * GiB))
    guard.check()
    if client is None:
        from .s3_upload import client_from_rclone

        client = client_from_rclone(spec.remote)
    root = work / "stage" / spec.id / version
    tmp = work / "tmp"
    writer = StagedWriter(
        root,
        WriterConfig(
            spec.id,
            version,
            spec.license,
            spec.attribution,
            fetch_date or dt.date.today(),
            layout=layout,
            threshold=spec.shard_threshold,
            shard_bytes=spec.shard_bytes,
            defaults=spec.defaults,
            naive_datetime_is_utc=spec.naive_datetime_is_utc,
            label_stem_suffix=spec.label_stem_suffix,
            lineage_root_digest=spec.lineage_root_digest,
        ),
    )
    up = _Uploader(client, spec, key_prefix, root, work, part_size, report, part_jobs=part_jobs)
    flush_at = guard.temp_cap_bytes // 2
    live = 0
    upstream: list[dict[str, Any]] = []
    limiter = HostLimiter(max_per_host)
    # WP-6b: fetch (network) is prefetched jobs-deep, consumed strictly in item order —
    # so decode/write/CHECKSUMS stay exactly as deterministic as `--jobs 1`. PUT (also
    # network) is likewise pooled: object identity, not order, decides the final state,
    # and CHECKSUMS.sha256 is only pushed once every earlier push has completed (below).
    prefetcher = _Prefetcher(adapter, items, tmp, guard, jobs, limiter) if jobs > 1 else None
    put_pool = ThreadPoolExecutor(max_workers=jobs) if jobs > 1 else None

    def flush() -> None:
        nonlocal live
        guard.sample()  # a real walk at the local maximum -> honest peak
        rels = writer.drain_closed()
        if put_pool is not None and len(rels) > 1:
            list(put_pool.map(up.push, rels))
        else:
            for rel in rels:
                up.push(rel)
        live = guard.sample()

    try:
        for item in items:
            fetched = (
                prefetcher.next()
                if prefetcher is not None
                else _fetch_one(adapter, item, tmp, guard, limiter)
            )
            live += fetched.size
            truncated = False
            try:
                for decoded in adapter.decode(fetched):
                    if subset is not None and not subset.admit(decoded):
                        continue
                    if writer.add(item, decoded) is not None:
                        live += len(decoded.data)
                    guard.observe(live)
                    if live >= flush_at:
                        flush()
                    if spec.max_images and len(writer.rows) >= spec.max_images:
                        truncated = True
                        break
                if truncated and fetched.stream is not None:
                    fetched.stream.raw.close()
                    fetched.stream = None
                else:
                    fetched.close()
            finally:
                if fetched.path is not None:
                    fetched.path.unlink(missing_ok=True)
                    live -= fetched.size
            writer.finish_item(fetched.sha256)
            upstream.append(
                {"key": item.key, "url": item.url, "sha256": fetched.sha256, "truncated": truncated}
            )
            if truncated:
                break
    except BaseException:
        if prefetcher is not None:
            prefetcher.close()
        if put_pool is not None:
            put_pool.shutdown(wait=False)
        raise
    if prefetcher is not None:
        prefetcher.close()
    writer.finalize()
    report.images = len(writer.rows)
    if subset is not None:
        report.plan["subset"] = subset.stats()
    sample_schema.write_samples(root / "metadata.parquet", writer.rows)
    ingest_json = {
        "ingest": "marinedata.ingest_source",
        "schema_version": sample_schema.SCHEMA_VERSION,
        "source_id": spec.id,
        "version": version,
        "adapter": spec.adapter,
        "params": spec.params,
        "layout": layout,
        "images": report.images,
        "fetch_date": (fetch_date or dt.date.today()).isoformat(),
        "upstream": upstream,
        "license": spec.license,
        "attribution": spec.attribution,
        "truncated_by_max_images": bool(upstream and upstream[-1]["truncated"]),
    }
    (root / "INGEST.json").write_text(json.dumps(ingest_json, indent=2, sort_keys=True) + "\n")
    (root / "LICENSE").write_text(
        f"License: {spec.license}\nAttribution: {spec.attribution}\n"
        + (f"Citation: {spec.citation}\n" if spec.citation else "")
    )
    for name in ("metadata.parquet", "INGEST.json", "LICENSE"):
        writer.register(root / name)
    flush()
    digests = {rel: sha for rel, (sha, _) in writer.files.items()}
    manifest = "".join(checksums.iter_lines(digests)).encode()
    (root / checksums.CHECKSUM_FILE).write_bytes(manifest)
    report.root_digest = hashlib.sha256(manifest).hexdigest()
    report.files = len(digests) + 1
    report.bytes = sum(size for _, size in writer.files.values()) + len(manifest)
    up.push(checksums.CHECKSUM_FILE)  # LAST: the version-complete marker
    if put_pool is not None:
        put_pool.shutdown(wait=True)
    report.verified = up.verify(sorted([*digests, checksums.CHECKSUM_FILE]))
    report.peak_temp_bytes = guard.peak_bytes
    (work / f"registry-stub-{spec.id}.yaml").write_text(_stub(spec, report))
    return report


def report_json(report: IngestReport) -> str:
    return json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)


def summary(report: IngestReport) -> Mapping[str, Any]:
    return {
        k: getattr(report, k)
        for k in (
            "source_id",
            "version",
            "images",
            "files",
            "uploaded",
            "skipped",
            "verified",
            "root_digest",
        )
    }
