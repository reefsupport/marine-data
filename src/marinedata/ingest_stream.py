"""WP-6h stream (diskless) staging: item bytes live in memory and go straight to S3.

Disk mode (:func:`marinedata.ingest_source.run_ingest`) writes every staged file under
``<work>/stage/<id>/<version>/`` and uploads it at flush points (up to ``temp_cap/2`` of
images, plus each whole 1 GiB shard tar, on local disk). Stream mode keeps the SAME
:class:`~marinedata.staged_writer.StagedWriter` (same stems, shard bytes, index, labels)
but swaps its sink: a closed file is PUT from memory, and a shard tar is sent as a
multipart upload while it is written (at most one part buffered). The end-of-run files
come from the shared :func:`~marinedata.ingest_source.final_artifacts`, so the staged
tree on S3 is byte-identical to disk mode's.

Local disk in stream mode: spooled archives only (``.zip``/``.parquet``/rar/video/rosbag,
admitted by :class:`StreamGuard` only if ``free - size >= stream_disk_floor_gib``), plus
the listing cache, the MISSING journal and the registry stub (a few KB).

Memory: :class:`MemoryBudget` bounds prefetched item bytes + queued PUT bodies + the
shard part buffer at ``memory_cap`` (512 MiB), over by at most one object when nothing
in flight could free bytes (the alternative is deadlocking the in-order consumer).

Resume: every checkpoint (``checkpoint_items`` items in objects layout; each shard close
that falls on an item boundary in shards layout) waits for every PUT, then writes

    sources/<id>/_stream/<version>/part-NNNNN.parquet   rows staged since the last part
    sources/<id>/_stream/<version>/part-NNNNN.json      state delta — written LAST (commit)

outside the version prefix, so the staged tree itself is unchanged. A re-run lists those
parts, restores the writer, skips the items they cover (no re-fetch), and skips any PUT
whose key already exists on S3 with equal size + ETag.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import threading
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import checksums
from .adapters import suffix_of
from .concurrency import HostLimiter, retry_with_backoff, tag_s3_key
from .ingest_missing import MissingLedger, decode_status, fetch_status
from .ingest_source import (
    _CONTAINERS,
    _RESERVE_WAIT_S,
    IngestReport,
    IngestSpec,
    _fetch_one,
    _stub,
    final_artifacts,
    writer_config,
)
from .s3_upload import DEFAULT_PART, DiskFloorError, DiskGuard, GiB, MiB
from .sample_schema import SampleRow, samples_bytes
from .staged_writer import StagedWriter

DEFAULT_MEMORY_CAP = 512 * MiB
DEFAULT_CHECKPOINT_ITEMS = 1000
UNKNOWN_ITEM_BYTES = 8 * MiB  # reservation for an item with no declared size
STREAM_DIR = "_stream"
_SUBSET_COUNTERS = ("seen", "rejected_filter", "rejected_hash", "rejected_cap")


class ResumeMismatch(RuntimeError):
    """The upstream listing no longer matches the stream checkpoints on S3."""


class StreamGuard(DiskGuard):
    """Stream-mode floor: only a spooled archive touches disk, so it is admitted only if
    it fits above the (low) floor — ``free - size >= floor`` — not merely ``free >= floor``."""

    def reserve(self, nbytes: int) -> None:
        free = self.free_bytes()
        if free - nbytes < self.floor_bytes:
            raise DiskFloorError(
                f"spooled archive {nbytes / GiB:.1f} GiB does not fit: free {free / GiB:.1f}"
                f" GiB - archive < floor {self.floor_bytes / GiB:.1f} GiB — pause"
            )
        super().reserve(nbytes)


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


class MemoryBudget:
    """In-flight bytes: prefetched items + queued PUT bodies + shard part buffers."""

    def __init__(self, cap: int = DEFAULT_MEMORY_CAP) -> None:
        self.cap, self.used, self.peak, self._sending = int(cap), 0, 0, 0
        self._cv = threading.Condition()

    def acquire(self, n: int) -> None:
        """Block while ``n`` more would pass the cap AND an in-flight PUT will free bytes;
        with nothing in flight, proceed (over by one object) rather than deadlock."""
        with self._cv:
            while self.used + n > self.cap and self._sending:
                self._cv.wait(timeout=1.0)
            self.used += n
            self.peak = max(self.peak, self.used)

    def try_acquire(self, n: int) -> bool:
        with self._cv:
            if self.used + n > self.cap:
                return False
            self.used += n
            self.peak = max(self.peak, self.used)
            return True

    def adjust(self, delta: int) -> None:
        with self._cv:
            self.used += delta
            self.peak = max(self.peak, self.used)
            self._cv.notify_all()

    def release(self, n: int) -> None:
        self.adjust(-n)

    def sending(self, delta: int) -> None:
        with self._cv:
            self._sending += delta
            self._cv.notify_all()


def _list_existing(client: Any, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    """``rel -> (size, etag)`` for every object under ``prefix/``."""
    out: dict[str, tuple[int, str]] = {}
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/")
    for page in pages:
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1 :]
            out[rel] = (int(obj["Size"]), str(obj["ETag"]).strip('"'))
    return out


class _Putter:
    """PUTs from memory on a pool; records sha256/size/ETag per rel for verify."""

    def __init__(
        self,
        client: Any,
        bucket: str,
        key_prefix: str,
        budget: MemoryBudget,
        jobs: int,
        existing: Mapping[str, tuple[int, str]],
    ) -> None:
        self.client, self.bucket, self.key_prefix = client, bucket, key_prefix
        self.budget, self.existing = budget, existing
        self.pool = ThreadPoolExecutor(max_workers=max(1, jobs))
        self.pending: list[Future] = []
        self.ledger: dict[str, dict[str, Any]] = {}
        self.uploaded = self.skipped = 0
        self._lock = threading.Lock()

    def key(self, rel: str) -> str:
        return f"{self.key_prefix}/{rel}"

    def record(self, rel: str, sha: str, size: int, etag: str, *, sent: bool = False) -> None:
        with self._lock:
            self.ledger[rel] = {"sha256": sha, "size": size, "etag": etag}
            self.uploaded += int(sent)

    def put(self, rel: str, data: bytes) -> None:
        sha, etag = hashlib.sha256(data).hexdigest(), _md5(data)
        self.record(rel, sha, len(data), etag)
        if self.existing.get(rel) == (len(data), etag):
            with self._lock:
                self.skipped += 1
            return
        self.budget.acquire(len(data))
        self.budget.sending(+1)
        self.pending.append(self.pool.submit(self._send, rel, data, sha))

    def _send(self, rel: str, data: bytes, sha: str) -> None:
        try:
            tag_s3_key(
                self.key(rel),
                lambda: retry_with_backoff(
                    lambda: self.client.put_object(
                        Bucket=self.bucket, Key=self.key(rel), Body=data, Metadata={"sha256": sha}
                    )
                ),
            )
            with self._lock:
                self.uploaded += 1
        finally:
            self.budget.release(len(data))
            self.budget.sending(-1)

    def drain(self) -> None:
        pending, self.pending = self.pending, []
        for fut in pending:
            fut.result()

    def verify(self, files: Mapping[str, tuple[str, int]]) -> int:
        ok = 0
        for rel in sorted(files):
            sha, size = files[rel]
            head = self.client.head_object(Bucket=self.bucket, Key=self.key(rel))
            etag = str(head["ETag"]).strip('"')
            if int(head["ContentLength"]) != size or (
                etag != self.ledger.get(rel, {}).get("etag")
                and (head.get("Metadata") or {}).get("sha256") != sha
            ):
                raise RuntimeError(f"verify failed: {rel}")
            ok += 1
        return ok

    def close(self, *, wait: bool) -> None:
        self.pool.shutdown(wait=wait, cancel_futures=not wait)


class _S3Stream(io.RawIOBase):
    """A shard tar written straight into a multipart upload (≤ one part buffered). Part
    boundaries and ETag match :func:`marinedata.s3_upload.local_digest` exactly; a tar
    that never exceeds one part goes as a single PUT, as disk mode's ``upload_file`` does."""

    def __init__(self, putter: _Putter, rel: str, part_size: int) -> None:
        super().__init__()
        self.putter, self.rel, self.part_size = putter, rel, part_size
        self.sha = hashlib.sha256()
        self.pos = 0
        self._buf = bytearray()
        self._md5s: list[str] = []
        self._upload_id: str | None = None
        putter.budget.adjust(part_size)

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def write(self, b) -> int:  # type: ignore[override]
        data = bytes(b)
        self.sha.update(data)
        self.pos += len(data)
        self._buf += data
        while len(self._buf) > self.part_size:  # the LAST part may be exactly part_size
            self._send_part(bytes(self._buf[: self.part_size]))
            del self._buf[: self.part_size]
        return len(data)

    def _send_part(self, chunk: bytes) -> None:
        client, bucket, key = self.putter.client, self.putter.bucket, self.putter.key(self.rel)
        if self._upload_id is None:
            resp = tag_s3_key(
                key,
                lambda: retry_with_backoff(
                    lambda: client.create_multipart_upload(Bucket=bucket, Key=key)
                ),
            )
            self._upload_id = str(resp["UploadId"])
        n, md5 = len(self._md5s) + 1, _md5(chunk)
        resp = tag_s3_key(
            key,
            lambda: retry_with_backoff(
                lambda: client.upload_part(
                    Bucket=bucket, Key=key, UploadId=self._upload_id, PartNumber=n, Body=chunk
                )
            ),
        )
        if str(resp["ETag"]).strip('"') != md5:
            raise RuntimeError(f"{key} part {n}: ETag {resp['ETag']} != local md5 {md5}")
        self._md5s.append(md5)

    def close(self) -> None:
        if self.closed:
            return
        try:
            data, self._buf = bytes(self._buf), bytearray()
            if self._upload_id is None:
                self.putter.put(self.rel, data)
            else:
                self._send_part(data)
                parts = [{"PartNumber": i + 1, "ETag": f'"{m}"'} for i, m in enumerate(self._md5s)]
                tag_s3_key(
                    self.putter.key(self.rel),
                    lambda: retry_with_backoff(
                        lambda: self.putter.client.complete_multipart_upload(
                            Bucket=self.putter.bucket,
                            Key=self.putter.key(self.rel),
                            UploadId=self._upload_id,
                            MultipartUpload={"Parts": parts},
                        )
                    ),
                )
                joined = b"".join(bytes.fromhex(m) for m in self._md5s)
                etag = f"{_md5(joined)}-{len(self._md5s)}"
                self.putter.record(self.rel, self.sha.hexdigest(), self.pos, etag, sent=True)
        finally:
            self.putter.budget.release(self.part_size)
            super().close()


class _S3Sink:
    """StagedWriter sink for stream mode: every file goes to S3, none to disk."""

    def __init__(self, putter: _Putter, part_size: int) -> None:
        self.putter, self.part_size = putter, part_size

    def write(self, rel: str, data: bytes) -> None:
        self.putter.put(rel, data)

    def open_stream(self, rel: str) -> _S3Stream:
        return _S3Stream(self.putter, rel, self.part_size)


class _StreamPrefetcher:
    """``_Prefetcher`` with memory admission: each item's bytes are reserved (declared
    size, else ``min(UNKNOWN_ITEM_BYTES, cap / jobs)``) before its fetch is queued, and
    corrected to the real size once fetched; the window shrinks to what the cap admits."""

    def __init__(
        self,
        adapter: Any,
        items: list,
        tmp: Path,
        guard: DiskGuard,
        jobs: int,
        limiter: HostLimiter,
        budget: MemoryBudget,
    ) -> None:
        self._items = deque(items)
        self._adapter, self._tmp, self._guard, self._limiter = adapter, tmp, guard, limiter
        self._budget, self._jobs = budget, max(1, jobs)
        self.pool = ThreadPoolExecutor(max_workers=self._jobs)
        self._futures: deque[tuple[Future, int]] = deque()
        self._top_up()

    def _top_up(self) -> None:
        """Queue fetches while the window has room AND the budget admits the item. Only
        the consumer frees these bytes, so it never blocks here: an item that does not
        fit waits for a later top-up — unless the window is empty (progress over cap)."""
        while self._items and len(self._futures) < self._jobs:
            item = self._items[0]
            guess = min(UNKNOWN_ITEM_BYTES, self._budget.cap // self._jobs)
            held = 0 if suffix_of(item.key) in _CONTAINERS else int(item.size or guess)
            if self._futures and not self._budget.try_acquire(held):
                return
            if not self._futures:
                self._budget.adjust(held)
            self._items.popleft()
            fut = self.pool.submit(
                _fetch_one,
                self._adapter,
                item,
                self._tmp,
                self._guard,
                self._limiter,
                reserve_timeout=_RESERVE_WAIT_S,
            )
            self._futures.append((fut, held))

    def next(self) -> tuple[Any, int]:
        if not self._futures:
            self._top_up()
        fut, held = self._futures.popleft()
        self._top_up()
        try:
            return fut.result(), held
        except BaseException:
            self._budget.release(held)
            raise

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _rows_from_parquet(data: bytes) -> list[SampleRow]:
    import pyarrow.parquet as pq

    rows = []
    for rec in pq.read_table(io.BytesIO(data)).to_pylist():
        rec["label_refs"] = tuple(rec.get("label_refs") or ())
        rows.append(SampleRow(**rec))
    return rows


class _Parts:
    """Checkpoint parts under ``sources/<id>/_stream/<version>/``."""

    def __init__(self, client: Any, bucket: str, prefix: str) -> None:
        self.client, self.bucket, self.prefix = client, bucket, prefix

    def _get(self, name: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{name}")
        return obj["Body"].read()

    def _put(self, name: str, data: bytes) -> None:
        key = f"{self.prefix}/{name}"
        tag_s3_key(
            key,
            lambda: retry_with_backoff(
                lambda: self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
            ),
        )

    def load(self) -> list[tuple[dict[str, Any], list[SampleRow]]]:
        names = set(_list_existing(self.client, self.bucket, self.prefix))
        out: list[tuple[dict[str, Any], list[SampleRow]]] = []
        while f"part-{len(out):05d}.json" in names:  # contiguous committed parts only
            n = len(out)
            state = json.loads(self._get(f"part-{n:05d}.json"))
            rows = _rows_from_parquet(self._get(f"part-{n:05d}.parquet")) if state["rows"] else []
            out.append((state, rows))
        return out

    def save(self, n: int, state: dict[str, Any], rows: list[SampleRow]) -> None:
        if rows:
            self._put(f"part-{n:05d}.parquet", samples_bytes(rows))
        self._put(f"part-{n:05d}.json", json.dumps(state, sort_keys=True).encode())  # commit


def _subset_state(subset: Any) -> dict[str, Any] | None:
    if subset is None:
        return None
    return {"kept": dict(subset.kept), **{k: getattr(subset, k) for k in _SUBSET_COUNTERS}}


def _restore_subset(subset: Any, state: Mapping[str, Any] | None) -> None:
    if subset is None or not state:
        return
    subset.kept.clear()
    subset.kept.update(state["kept"])
    for k in _SUBSET_COUNTERS:
        setattr(subset, k, int(state[k]))


def run_stream(
    spec: IngestSpec,
    work: Path,
    *,
    adapter: Any,
    items: list,
    report: IngestReport,
    subset: Any,
    client: Any,
    guard: DiskGuard | None,
    part_size: int = DEFAULT_PART,
    fetch_date: dt.date | None = None,
    jobs: int = 8,
    max_per_host: int = 4,
    memory_cap: int | None = None,
    checkpoint_items: int | None = None,
) -> IngestReport:
    """Stage ``items`` (already listed by ``run_ingest``) from memory straight to S3."""
    if client is None:
        from .s3_upload import client_from_rclone

        client = client_from_rclone(spec.remote)
    floor = int(spec.stream_disk_floor_gib * GiB)
    guard = guard or StreamGuard(work, int(spec.temp_cap_gb * GiB), floor)
    guard.check()
    version, layout, key_prefix = report.version, report.layout, report.prefix
    budget = MemoryBudget(memory_cap or DEFAULT_MEMORY_CAP)
    every = max(1, int(checkpoint_items or DEFAULT_CHECKPOINT_ITEMS))
    existing = _list_existing(client, spec.bucket, key_prefix)
    putter = _Putter(client, spec.bucket, key_prefix, budget, jobs, existing)
    parts = _Parts(client, spec.bucket, f"{spec.prefix}/{spec.id}/{STREAM_DIR}/{version}")
    loaded = parts.load()
    if loaded:  # one fetch_date for the whole version, however many resumes it took
        fetch_date = dt.date.fromisoformat(loaded[0][0]["fetch_date"])
    fetch_date = fetch_date or dt.date.today()
    writer = StagedWriter(
        work / "stage" / spec.id / version,
        writer_config(spec, version, layout, fetch_date),
        sink=_S3Sink(putter, part_size),
    )
    missing = MissingLedger(work / "missing" / f"{spec.id}-{version}.tsv")  # D-AF
    upstream: list[dict[str, Any]] = []
    done = 0
    for state, rows in loaded:
        writer.restore(rows, state)
        missing.restore(state["missing"], state["missing_attempted"], state["missing_consecutive"])
        upstream.extend(state["upstream"])
        for rel, etag in state["etags"].items():
            putter.ledger[rel] = {"etag": etag}
        _restore_subset(subset, state["subset"])
        done = int(state["items_done"])
    if loaded:
        last = loaded[-1][0]
        if last["layout"] != layout or done > len(items) or items[done - 1].key != last["last_key"]:
            raise ResumeMismatch(
                f"{spec.id}/{version}: listing no longer matches {STREAM_DIR} checkpoint "
                f"part-{len(loaded) - 1:05d} (item {done}: {last['last_key']!r})"
            )
        report.plan["resumed_items"] = done
    part_no, mark = len(loaded), writer.mark()
    up_n, miss_n, since, last_shard = len(upstream), len(missing.rows), 0, writer.shard_no

    def checkpoint(items_done: int, last_key: str) -> None:
        nonlocal part_no, mark, up_n, miss_n, since, last_shard
        writer.drain_closed()
        putter.drain()  # every object of every item up to here is durable on S3
        delta = writer.delta(mark)
        rows = delta.pop("rows")
        state = {
            "part": part_no,
            "items_done": items_done,
            "last_key": last_key,
            "fetch_date": fetch_date.isoformat(),
            "layout": layout,
            "rows": len(rows),
            **delta,
            "etags": {r: putter.ledger[r]["etag"] for r in delta["files"] if r in putter.ledger},
            "upstream": upstream[up_n:],
            "missing": [list(r) for r in missing.rows[miss_n:]],
            "missing_attempted": missing.attempted,
            "missing_consecutive": missing.consecutive,
            "subset": _subset_state(subset),
        }
        parts.save(part_no, state, rows)
        part_no, mark, since, last_shard = part_no + 1, writer.mark(), 0, writer.shard_no
        up_n, miss_n = len(upstream), len(missing.rows)

    limiter = HostLimiter(max_per_host)
    prefetcher = _StreamPrefetcher(
        adapter, items[done:], work / "tmp", guard, jobs, limiter, budget
    )
    try:
        for idx, item in enumerate(items[done:], start=done):
            try:
                fetched, held = prefetcher.next()
            except Exception as exc:
                status = fetch_status(exc)
                if status is None:
                    raise
                missing.skip(item.key, item.url, status)  # D-AF; raises past a threshold
                continue
            actual = 0 if suffix_of(item.key) in _CONTAINERS else int(fetched.size or 0)
            budget.adjust(actual - held)
            truncated = wrote = failed = False
            try:
                try:
                    for decoded in adapter.decode(fetched):
                        if subset is not None and not subset.admit(decoded):
                            continue
                        wrote = True
                        writer.add(item, decoded)
                        if spec.max_images and len(writer.rows) >= spec.max_images:
                            truncated = True
                            break
                    if truncated and fetched.stream is not None:
                        fetched.stream.raw.close()
                        fetched.stream = None
                    else:
                        fetched.close()
                except Exception as exc:
                    status = decode_status(exc, wrote=wrote)
                    if status is None:
                        raise
                    if fetched.stream is not None:
                        fetched.stream.raw.close()
                        fetched.stream = None
                    missing.skip(item.key, item.url, status)
                    failed = True
            finally:
                if fetched.path is not None:
                    fetched.path.unlink(missing_ok=True)
                budget.release(actual)
            if failed:
                continue
            missing.ok()
            writer.finish_item(fetched.sha256)
            writer.drain_closed()  # the sink already sent them
            upstream.append(
                {
                    "key": item.key,
                    "url": item.url,
                    "sha256": fetched.sha256,
                    "upstream_sha256_match": fetched.upstream_match,
                    "truncated": truncated,
                }
            )
            if truncated:
                break
            since += 1
            if not writer.shard_open and (
                (layout == "objects" and since >= every)
                or (layout == "shards" and writer.shard_no > last_shard)
            ):
                checkpoint(idx + 1, item.key)
    except BaseException:
        prefetcher.close()
        putter.close(wait=False)
        raise
    prefetcher.close()
    missing.finish()  # D-AF: every attempted item failed -> abort, never an empty version
    report.plan["peak_memory_items_bytes"] = budget.peak  # before the whole-file artifacts
    writer.finalize()
    report.images = len(writer.rows)
    report.missing = len(missing.rows)
    if subset is not None:
        report.plan["subset"] = subset.stats()
    for rel, data in final_artifacts(spec, report, writer.rows, upstream, missing, fetch_date):
        writer.add_bytes(rel, data)
    writer.drain_closed()
    putter.drain()
    digests = {rel: sha for rel, (sha, _) in writer.files.items()}
    manifest = "".join(checksums.iter_lines(digests)).encode()
    report.root_digest = hashlib.sha256(manifest).hexdigest()
    report.files = len(digests) + 1
    report.bytes = sum(size for _, size in writer.files.values()) + len(manifest)
    putter.put(checksums.CHECKSUM_FILE, manifest)  # LAST: the version-complete marker
    putter.drain()
    putter.close(wait=True)
    files = {**writer.files, checksums.CHECKSUM_FILE: (report.root_digest, len(manifest))}
    report.verified = putter.verify(files)
    report.uploaded, report.skipped = putter.uploaded, putter.skipped
    report.peak_temp_bytes = guard.peak_bytes
    report.plan["mode"] = "stream"
    report.plan["peak_memory_bytes"] = budget.peak
    work.mkdir(parents=True, exist_ok=True)
    (work / f"registry-stub-{spec.id}.yaml").write_text(_stub(spec, report))
    return report
