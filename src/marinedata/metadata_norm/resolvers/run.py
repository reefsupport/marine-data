"""``marinedata metadata resolve``: read a staged source anonymously, resolve, write local files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...checksums import parse_checksums
from ...task_layers.s3_keyed import fetch_small
from ..defaults import registry_entry, validated_defaults
from ..stage import spec_fields
from .registry import resolver_for
from .schema import write_outputs
from .staged import (
    Fetch,
    RangeLedger,
    ResolveContext,
    http_range,
    make_getter,
    read_rows,
    staged_images,
)


def default_version(source_id: str, registry_root: str | Path | None = None) -> str:
    """The ingest spec's version, else the registry entry's."""
    from ...registry import Registry, _default_root

    spec = spec_fields(source_id, registry_root)["version"]
    return (
        spec
        or Registry.load(Path(registry_root) if registry_root else _default_root())
        .source(source_id)
        .version
    )


def resolve_source(
    source_id: str,
    out_dir: str | Path,
    version: str | None = None,
    cache_dir: str | Path | None = None,
    fetch: Fetch = fetch_small,
    registry_root: str | Path | None = None,
    range_fetch: Any = None,
) -> Path:
    """Resolve every staged image of ``source_id`` and write the three output files; returns the
    ``<out_dir>/<source>/<version>`` directory. Reads only: single-key GETs and
    ranged image heads."""
    resolver = resolver_for(source_id)
    version = version or default_version(source_id, registry_root)
    tree = f"sources/{source_id}/{version}/"
    sums = fetch(tree + "CHECKSUMS.sha256")
    checksums = parse_checksums(sums.decode())
    get = make_getter(tree, checksums, cache_dir, fetch)
    images = staged_images(read_rows(get("metadata.parquet")), checksums)
    ledger = RangeLedger()
    ctx = ResolveContext(
        source_id=source_id,
        version=version,
        get=get,
        get_range=range_fetch or http_range(tree, ledger),
        default_platform=validated_defaults(registry_entry(source_id, registry_root)).get(
            "platform"
        ),
        checksums=checksums,
        ledger=ledger,
    )
    result = resolver(images, ctx)
    return write_outputs(
        out_dir, source_id, version, result.rows, result.unresolved, {"notes": dict(result.notes)}
    )


def cmd_resolve(args: argparse.Namespace) -> int:
    target = resolve_source(args.source, args.out, args.version, args.cache)
    print(f"wrote {target}/samples.parquet, coverage.json, unresolved.tsv")
    return 0
