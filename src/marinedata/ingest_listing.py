"""Listing cache for ``ingest-batch`` (INT-ingest5c).

A source's enumeration (version + items) is written once as parquet under the queue's
work dir, keyed by a hash of the fields that decide the listing (adapter, params,
version), so a resumed pass reuses it instead of re-crawling upstream (GBIF: ~1,200 API
requests; Commons: a full category walk). Adapters opt in with ``listing_cacheable =
True`` and, when fetch/decode depend on per-item state gathered during enumeration,
``listing_state(item)`` / ``restore_listing(items, states)``. State that does not survive
a JSON round trip unchanged is never cached (the source just re-lists, as before).

On a cache hit ``resolve_version()`` is not called again: the version is the one pinned
when the listing was cached, which keeps a resumed pass on the same version it started.
Delete ``<work>/_listings/`` to force a fresh listing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .adapters import RemoteItem

LISTING_DIR = "_listings"
_FORMAT = "1"


def spec_hash(spec: Any) -> str:
    basis = {"adapter": spec.adapter, "params": spec.params, "version": spec.version}
    blob = json.dumps(basis, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_path(cache_dir: Path, spec: Any) -> Path:
    return cache_dir / f"{spec.id}-{spec_hash(spec)}.parquet"


def _roundtrips(value: Any) -> bool:
    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


def save(path: Path, version: str, items: list[RemoteItem], states: list[Any]) -> bool:
    if not all(_roundtrips(s) for s in states):
        return False
    table = pa.table(
        {
            "key": pa.array([i.key for i in items], pa.string()),
            "url": pa.array([i.url for i in items], pa.string()),
            "size": pa.array([i.size for i in items], pa.int64()),
            "md5": pa.array([i.md5 for i in items], pa.string()),
            "sha256": pa.array([i.sha256 for i in items], pa.string()),
            "parts": pa.array([list(i.parts) for i in items], pa.list_(pa.string())),
            "state": pa.array(
                [None if s is None else json.dumps(s, sort_keys=True) for s in states],
                pa.string(),
            ),
        }
    ).replace_schema_metadata({"version": version, "format": _FORMAT})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return True


def load(path: Path) -> tuple[str, list[RemoteItem], list[Any]] | None:
    if not path.exists():
        return None
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if meta.get(b"format") != _FORMAT.encode() or b"version" not in meta:
        return None
    rows = table.to_pylist()
    items = [
        RemoteItem(
            key=r["key"],
            url=r["url"],
            size=r["size"],
            md5=r["md5"],
            sha256=r["sha256"],
            parts=tuple(r["parts"] or ()),
        )
        for r in rows
    ]
    states = [None if r["state"] is None else json.loads(r["state"]) for r in rows]
    return meta[b"version"].decode(), items, states


def list_source(spec: Any, adapter: Any, cache_dir: Path | None = None) -> tuple[str, list]:
    """``(version, items)`` for ``spec``: from the cache when the adapter opts in and a
    listing exists, else from upstream (then cached when the adapter opts in)."""
    cacheable = cache_dir is not None and getattr(adapter, "listing_cacheable", False)
    path = cache_path(cache_dir, spec) if cacheable and cache_dir is not None else None
    if path is not None:
        hit = load(path)
        if hit is not None:
            version, items, states = hit
            restore = getattr(adapter, "restore_listing", None)
            if restore is not None:
                restore(items, states)
            return version, items
    version = spec.version or adapter.resolve_version()
    if spec.version:
        adapter.resolve_version()  # still pin + gate-check upstream
    items = list(adapter.enumerate())
    if path is not None:
        state_of = getattr(adapter, "listing_state", lambda _i: None)
        save(path, version, items, [state_of(i) for i in items])
    return version, items
