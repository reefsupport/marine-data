"""Anonymous single-key reads of a staged tree in the object store (P1.2).

Shared by ``scripts/registry_sync_bucket.py`` (writes the registry's counts and digests) and
``tests/test_registry_counts.py`` (the nightly drift check). Only ``GET`` of named keys, never a
listing: ``CHECKSUMS.sha256``, ``INGEST.json`` and the footer of ``metadata.parquet`` (a suffix
range read). ``root_digest`` is the sha256 of the CHECKSUMS.sha256 file itself, as
``checksums.py`` and ``manifest_identity.py`` define it.
"""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.error
import urllib.request

ENDPOINT = "hel1.your-objectstorage.com"
CHECKSUMS_NAME = "CHECKSUMS.sha256"
OPEN_BUCKET = "rs-storage-open"
_TAIL = 262_144


def key_url(bucket: str, prefix: str, name: str) -> str:
    return f"https://{bucket}.{ENDPOINT}/{prefix.strip('/')}/{name}"


def get(url: str, byte_range: str | None = None, timeout: float = 90.0) -> bytes | None:
    """One anonymous GET; ``None`` when the key is absent or not publicly readable (403/404)."""
    headers = {"User-Agent": "marinedata-registry-probe/1"}
    if byte_range:
        headers["Range"] = byte_range
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def read_checksums(bucket: str, prefix: str) -> tuple[str, int, int] | None:
    """``(root_digest, n_files, n_bytes_read)`` of a tree's CHECKSUMS.sha256."""
    body = get(key_url(bucket, prefix, CHECKSUMS_NAME))
    if body is None:
        return None
    return (
        hashlib.sha256(body).hexdigest(),
        sum(1 for ln in body.splitlines() if ln.strip()),
        len(body),
    )


def read_ingest_images(bucket: str, prefix: str) -> int | None:
    """``images`` from INGEST.json, or ``None`` when the key or the field is missing."""
    body = get(key_url(bucket, prefix, "INGEST.json"))
    if body is None:
        return None
    try:
        n = json.loads(body).get("images")
    except (ValueError, AttributeError):
        return None
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def read_parquet_rows(bucket: str, prefix: str) -> tuple[int, int] | None:
    """``(rows, gets)`` from the footer of metadata.parquet, via suffix range reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    url = key_url(bucket, prefix, "metadata.parquet")
    tail = get(url, f"bytes=-{_TAIL}")
    if tail is None or len(tail) < 12 or tail[-4:] != b"PAR1":
        return None
    gets = 1
    need = struct.unpack("<I", tail[-8:-4])[0] + 8
    if need > len(tail):
        tail = get(url, f"bytes=-{need}")
        gets += 1
        if tail is None:
            return None
    return pq.read_metadata(pa.BufferReader(b"PAR1" + tail)).num_rows, gets


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def probe_tree(row: dict) -> dict:
    """Read what the listing row says exists (nothing is probed for): at most 3 GETs per tree."""
    bucket, sid, version = row["bucket"], row["source_id"], row["version"]
    prefix = f"sources/{sid}/{version}"
    out: dict = {
        "bucket": bucket,
        "id": sid,
        "version": version,
        "gets": 0,
        "bytes_read": 0,
        "size_bytes": int(row.get("bytes") or 0),
        "n_objects": int(row.get("n_objects") or 0),
    }
    if bucket != OPEN_BUCKET:
        return {**out, "skipped": "private bucket: no anonymous read"}
    if _truthy(row.get("has_checksums")):
        got = read_checksums(bucket, prefix)
        out["gets"] += 1
        if got is None:
            out["error"] = "CHECKSUMS.sha256 listed but not readable"
        else:
            out["root_digest"], out["files"], out["bytes_read"] = got
    images, method = None, None
    if _truthy(row.get("has_ingest_json")):
        images = read_ingest_images(bucket, prefix)
        out["gets"] += 1
        method = "ingest-json" if images is not None else None
    if images is None and _truthy(row.get("has_metadata_parquet")) and out["gets"] <= 1:
        got = read_parquet_rows(bucket, prefix)
        out["gets"] += got[1] if got else 1
        if got is not None:
            images, method = got[0], "metadata-parquet"
    listed = str(row.get("images_in_listing") or "").strip()
    if images is None and listed.isdigit():
        images, method = int(listed), "listing"
    out["images"], out["method"] = images, method
    return out
