"""Read-only inputs for the normalisers: staged files on ``rs-storage-open`` + the registry.

Nothing here writes to the bucket. Small files come through the public anonymous GET
(:func:`marinedata.task_layers.s3_keyed.fetch_small`); listings and ranged parquet reads use a
boto3 client whose credentials are read with ``configparser`` and never printed.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..checksums import parse_checksums
from ..licence_class import per_row_source, source_class
from ..registry import _default_root
from ..task_layers.s3_keyed import FetchFailed, fetch_small
from .base import NormContext
from .defaults import registry_entry, registry_fetch_date, validated_defaults

BUCKET = "rs-storage-open"
Fetch = Callable[..., bytes]


def s3_client():
    import configparser
    import os

    import boto3

    sec = _cfg(configparser, os)
    ep = sec["endpoint"]
    return boto3.client(
        "s3",
        aws_access_key_id=sec["access_key_id"],
        aws_secret_access_key=sec["secret_access_key"],
        endpoint_url=ep if ep.startswith("http") else f"https://{ep}",
    )


def _cfg(configparser, os):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.config/rclone/rclone.conf"))
    return cfg["rs-hel1"]


def spec_fields(source_id: str, root: str | Path | None = None) -> dict[str, str]:
    """``license``/``attribution``/``citation``/``homepage``/``version`` from the ingest spec,
    falling back to the registry source (``licence.name``)."""
    import yaml

    base = Path(root) if root else _default_root()
    spec = base / "ingest-specs" / f"{source_id}.yaml"
    data: dict[str, Any] = yaml.safe_load(spec.read_text()) if spec.is_file() else {}
    version = data.get("version") or (data.get("params") or {}).get("version") or ""
    out = {k: str(data.get(k) or "") for k in ("license", "attribution", "citation", "homepage")}
    out["version"] = str(version)
    if not out["license"]:
        from ..registry import Registry

        try:
            src = Registry.load(base).source(source_id)
        except KeyError:
            return out
        out |= {
            "license": src.licence.name,
            "attribution": out["attribution"] or src.name,
            "citation": src.citation or "",
            "homepage": src.homepage or "",
            "version": out["version"] or src.version,
        }
    return out


def _get(fetch: Fetch, key: str) -> bytes | None:
    try:
        return fetch(key, tries=2)
    except (FetchFailed, OSError, ValueError):
        return None


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of one bucket object through ranged GETs."""

    def __init__(self, client, key: str) -> None:
        self.c, self.key, self.pos = client, key, 0
        self.size = client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        d = self.c.get_object(Bucket=BUCKET, Key=self.key, Range=rng)["Body"].read()
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def _parquet_head(client, key: str, limit: int) -> list[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(io.BufferedReader(_RangeFile(client, key), 1 << 20))
    for batch in pf.iter_batches(batch_size=limit):
        return batch.to_pylist()
    return []


def _list(client, prefix: str, limit: int) -> list[dict]:
    r = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=min(limit, 1000))
    return list(r.get("Contents", []))


def load_inputs(
    source_id: str,
    version: str | None = None,
    limit: int = 1000,
    fetch: Fetch = fetch_small,
    client=None,
    registry_root: str | Path | None = None,
    events_path: str | Path | None = None,
    meow_path: str | Path | None = None,
) -> tuple[str, list[dict], NormContext]:
    """``(version, staged rows <= limit, context)`` for one source version. ``events_path`` =
    cached MERMAID sample-event JSON, ``meow_path`` = a local MEOW polygon file (no download)."""
    spec = spec_fields(source_id, registry_root)
    version = version or spec["version"]
    tree = f"sources/{source_id}/{version}/"
    sums = _get(fetch, tree + "CHECKSUMS.sha256")
    ingest = _get(fetch, tree + "INGEST.json")
    ctx = NormContext(
        registry_licence=spec["license"],
        attribution=spec["attribution"],
        citation=spec["citation"],
        homepage=spec["homepage"],
        source_class=source_class(source_id, root=registry_root),
        per_row=per_row_source(source_id, root=registry_root),
        checksums=parse_checksums(sums.decode()) if sums else {},
        ingest=json.loads(ingest) if ingest else {},
        version=version,
    )
    entry = registry_entry(source_id, registry_root)
    defaults = validated_defaults(entry)
    fetched, fetched_from = (None, "")
    if not ctx.ingest:  # INGEST.json missing: registry date, else the earliest LastModified
        fetched, fetched_from = registry_fetch_date(entry)
        if fetched is None:
            client = client or s3_client()
            fetched, fetched_from = earliest_modified(client, tree)
    ctx = replace(
        ctx,
        fetch_date_fallback=fetched,
        fetch_date_origin=fetched_from,
        default_platform=defaults.get("platform", ""),
        default_habitat=defaults.get("habitat", ""),
        split_rule=split_rule(source_id, registry_root),
        events=json.loads(Path(events_path).read_text()) if events_path else {},
        meow=_meow(meow_path),
    )
    rows: list[dict] = []
    meta = _get(fetch, tree + "metadata.parquet")
    if meta:
        import pyarrow.parquet as pq

        rows = pq.read_table(io.BytesIO(meta)).slice(0, limit).to_pylist()
    elif source_id == "fathomnet":
        client = client or s3_client()
        keys = [o["Key"] for o in _list(client, tree + "labels/files/", limit)]
        with ThreadPoolExecutor(16) as pool:
            blobs = list(pool.map(lambda k: _get(fetch, k), keys))
        labels = {json.loads(b)["uuid"]: json.loads(b) for b in blobs if b}
        ctx = _with_labels(ctx, labels)
        rows = [{"stem": u, "partition": "default"} for u in sorted(labels)]
    elif source_id == "inat-marine":
        client = client or s3_client()
        key = "sources/inat-marine/_manifest/inat-marine-manifest.parquet"
        rows = _parquet_head(client, key, limit)
    elif not ctx.checksums:
        client = client or s3_client()
        for o in _list(client, tree + "images/", limit):
            rel = o["Key"][len(tree) :]
            stem = Path(rel).stem
            rows.append(
                {"stem": stem, "partition": "default", "image_path": rel, "image_bytes": o["Size"]}
            )
    return version, rows, ctx


def _with_labels(ctx: NormContext, labels: dict[str, Any]) -> NormContext:
    return replace(ctx, labels=labels)


def earliest_modified(client, prefix: str) -> tuple[Any, str]:
    """``(date, origin)`` of the earliest ``LastModified`` on the first listing page of
    ``prefix`` (one request, <= 1000 objects); ``(None, "")`` when the prefix is empty."""
    stamps = [o["LastModified"] for o in _list(client, prefix, 1000) if o.get("LastModified")]
    if not stamps:
        return None, ""
    return min(stamps).date(), "s3:earliest LastModified (first page)"


def split_rule(source_id: str, root: str | Path | None = None):
    """The source's registry ``SplitGroupRule`` (``None`` when the source is not registered)."""
    from ..registry import Registry

    try:
        return Registry.load(Path(root) if root else _default_root()).source(source_id).split_group
    except KeyError:
        return None


def _meow(path: str | Path | None):
    if not path:
        return ()
    from ..geo_meow import load_meow_polygons

    return load_meow_polygons(Path(path))
