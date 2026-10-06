"""``registry verify``: every s3 pointer in the registry must name a stored, checksummed tree.

Offline (default): each source's ``access.params.bucket`` + ``prefix`` is looked up in a *listing
snapshot* (a parquet/TSV/CSV of bucket keys, e.g. the reorg inventory) and must hold a
``CHECKSUMS.sha256``. ``--live`` instead HEADs that one key per source with credentials read via
:mod:`configparser` (never printed). Sources that are retired or not an s3 pointer are skipped and
counted, never failed.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Source

MANIFEST = "CHECKSUMS.sha256"
OK, NO_MANIFEST, MISSING, SKIPPED = "ok", "no-manifest", "missing", "skipped"


@dataclass(frozen=True)
class Check:
    source_id: str
    status: str
    uri: str
    detail: str = ""

    def line(self) -> str:
        return f"{self.status:<12} {self.source_id:<36} {self.uri}  {self.detail}".rstrip()


def _bucket_of_name(name: str) -> str | None:
    for bucket in ("rs-storage-open", "rs-storage-private"):
        if bucket in name:
            return bucket
    return None


def load_listing(path: Path, bucket: str | None = None) -> dict[str, set[str]]:
    """``{bucket: {key, ...}}`` from a parquet (``key`` [+ ``bucket``] columns) or a TSV/CSV whose
    first column is the key. ``bucket`` defaults to the one named in the file name."""
    path = Path(path)
    bucket = bucket or _bucket_of_name(path.name)
    out: dict[str, set[str]] = {}
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        keys = table.column("key").to_pylist()
        buckets = table.column("bucket").to_pylist() if "bucket" in table.column_names else None
        for i, key in enumerate(keys):
            b = buckets[i] if buckets else bucket
            if b is None:
                raise ValueError(f"{path}: no bucket column and none inferable from the name")
            out.setdefault(b, set()).add(key)
        return out
    if bucket is None:
        raise ValueError(f"{path}: pass BUCKET=PATH (bucket not inferable from the name)")
    delim = "," if path.suffix == ".csv" else "\t"
    with path.open(newline="") as fh:
        for row in csv.reader(fh, delimiter=delim):
            if row and row[0] and row[0] != "key":
                out.setdefault(bucket, set()).add(row[0])
    return out


def _pointer(source: Source) -> tuple[str, str] | None:
    """``(bucket, prefix-with-trailing-slash)`` of an s3 source, else ``None``."""
    if source.access.method.value != "s3":
        return None
    params = source.access.params or {}
    bucket, prefix = params.get("bucket"), params.get("prefix")
    if not bucket or not prefix:  # fall back to ``s3://bucket/prefix`` in the uri
        rest = (source.access.uri or "").removeprefix("s3://")
        bucket, _, prefix = rest.partition("/")
    if not bucket or not prefix:
        return None
    return str(bucket), str(prefix).rstrip("/") + "/"


def verify_sources(
    sources: Iterable[Source],
    has_key: Callable[[str, str], bool],
    *,
    has_any: Callable[[str, str], bool],
) -> list[Check]:
    """One :class:`Check` per source. ``has_key(bucket, key)`` and ``has_any(bucket, prefix)`` are
    supplied by the offline listing or by a live client."""
    checks: list[Check] = []
    for s in sorted(sources, key=lambda s: s.id):
        if s.retired:
            checks.append(Check(s.id, SKIPPED, s.access.uri or "-", "retired"))
            continue
        ptr = _pointer(s)
        if ptr is None:
            checks.append(
                Check(s.id, SKIPPED, s.access.uri or "-", f"method {s.access.method.value}")
            )
            continue
        bucket, prefix = ptr
        uri = f"s3://{bucket}/{prefix}"
        if has_key(bucket, prefix + MANIFEST):
            checks.append(Check(s.id, OK, uri))
        elif has_any(bucket, prefix):
            checks.append(Check(s.id, NO_MANIFEST, uri, "objects but no CHECKSUMS.sha256"))
        else:
            checks.append(Check(s.id, MISSING, uri, "nothing under the prefix"))
    return checks


def verify_offline(sources: Iterable[Source], listing: Mapping[str, set[str]]) -> list[Check]:
    def has_key(bucket: str, key: str) -> bool:
        return key in listing.get(bucket, ())

    def has_any(bucket: str, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in listing.get(bucket, ()))

    return verify_sources(sources, has_key, has_any=has_any)


def verify_live(sources: Iterable[Source], client: Any) -> list[Check]:
    """HEAD ``<prefix>CHECKSUMS.sha256`` per source; a 404 falls back to a 1-key LIST."""
    from botocore.exceptions import ClientError

    def has_key(bucket: str, key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def has_any(bucket: str, prefix: str) -> bool:
        try:
            resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        except ClientError:  # e.g. NoSuchBucket for a third-party bucket (imos-data)
            return False
        return bool(resp.get("KeyCount"))

    return verify_sources(sources, has_key, has_any=has_any)


def summarise(checks: list[Check]) -> str:
    counts = {s: sum(c.status == s for c in checks) for s in (OK, NO_MANIFEST, MISSING, SKIPPED)}
    return "  ".join(f"{k}={v}" for k, v in counts.items())


def failed(checks: Iterable[Check]) -> list[Check]:
    return [c for c in checks if c.status in (NO_MANIFEST, MISSING)]
