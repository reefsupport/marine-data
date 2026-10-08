"""Read-only access to a staged tree for the resolvers: cached anonymous GETs, ranged GETs, rows.

Every read is a single-key anonymous HTTPS GET on ``rs-storage-open`` (never a listing, never a
write). Tables are cached on local disk when a cache directory is given and verified against the
tree's ``CHECKSUMS.sha256``; ranged GETs (EXIF) are metered by :class:`RangeLedger`.
"""

from __future__ import annotations

import hashlib
import io
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...task_layers.s3_keyed import PUBLIC_HOST, fetch_small

Fetch = Callable[..., bytes]
RangeFetch = Callable[[str, int], bytes]
MAX_RANGE_REQUESTS = 2000
MAX_RANGE_BYTES = 150 * (1 << 20)
EXIF_WINDOW = 64 * 1024


class ChecksumMismatch(ValueError):
    """A fetched table does not match the tree's CHECKSUMS.sha256 entry."""


class RangeBudgetExceeded(RuntimeError):
    pass


@dataclass
class RangeLedger:
    """Counts ranged image reads; raises once either cap would be crossed."""

    max_requests: int = MAX_RANGE_REQUESTS
    max_bytes: int = MAX_RANGE_BYTES
    requests: int = 0
    bytes: int = 0

    def charge(self, n: int) -> None:
        if self.requests + 1 > self.max_requests or self.bytes + n > self.max_bytes:
            raise RangeBudgetExceeded(f"ranged-read budget hit at {self.requests} requests")
        self.requests += 1
        self.bytes += n


@dataclass(frozen=True)
class StagedImage:
    partition: str
    stem: str
    rel_path: str
    sha256: str
    row: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolveContext:
    """What a resolver may read. ``get`` takes a tree-relative key; ``get_range`` a tree-relative
    image key and a byte count (the first ``n`` bytes)."""

    source_id: str
    version: str
    get: Callable[[str], bytes]
    get_range: RangeFetch | None = None
    default_platform: str | None = None
    checksums: Mapping[str, str] = field(default_factory=dict)
    ledger: RangeLedger = field(default_factory=RangeLedger)


def make_getter(
    tree: str,
    checksums: Mapping[str, str],
    cache_dir: str | Path | None = None,
    fetch: Fetch = fetch_small,
) -> Callable[[str], bytes]:
    """``get(rel)`` -> bytes of ``tree + rel``; cached on disk and sha256-checked when ``rel`` is
    listed in ``checksums``."""
    root = Path(cache_dir) if cache_dir else None

    def get(rel: str) -> bytes:
        path = root / (tree + rel).replace("/", "__") if root else None
        data = path.read_bytes() if path is not None and path.is_file() else None
        if data is None:
            data = fetch(tree + rel)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        want = checksums.get(rel)
        if want and hashlib.sha256(data).hexdigest() != want:
            if path is not None and path.is_file():
                path.unlink()
            raise ChecksumMismatch(f"{tree}{rel}: sha256 differs from CHECKSUMS.sha256")
        return data

    return get


def http_range(tree: str, ledger: RangeLedger) -> RangeFetch:
    """Anonymous ranged GET of the first ``n`` bytes of ``tree + rel`` (EXIF), metered."""

    def get_range(rel: str, n: int) -> bytes:
        ledger.charge(n)
        url = f"{PUBLIC_HOST}/{urllib.parse.quote(tree + rel)}"
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read(n)

    return get_range


def staged_images(meta_rows: list[dict], checksums: Mapping[str, str]) -> list[StagedImage]:
    """Join ``metadata.parquet`` rows to image paths + sha256. A row's own ``image_path`` /
    ``image_sha256`` win; slimmer tables (NOAA, Coralscapes: no sha column) fall back to the
    ``images/<partition>/<stem>.<ext>`` entry of CHECKSUMS."""
    index: dict[tuple[str, str], str] = {}
    for rel in checksums:
        parts = rel.split("/")
        if parts[0] == "images" and len(parts) >= 3:
            index.setdefault((parts[1], Path(parts[-1]).stem), rel)
    out: list[StagedImage] = []
    for r in meta_rows:
        partition = str(r.get("partition") or "default")
        stem = str(r["stem"])
        rel = r.get("image_path") or index.get((partition, stem)) or ""
        sha = r.get("image_sha256") or checksums.get(rel, "")
        out.append(StagedImage(partition, stem, rel, sha, r))
    return out


def read_rows(data: bytes) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()
