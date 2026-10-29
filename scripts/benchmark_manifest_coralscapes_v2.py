#!/usr/bin/env python3
"""Build ``registry/benchmarks/manifests/coralscapes-v2.parquet`` from the staged V2 tree, offline.

``marinedata bench manifest`` fetches eval frames from the bucket or upstream (network). The V2
manifest is built where the staged tree already is: on VM 900 (rs-server-2), after the staging step
wrote ``sources/coralscapes-v2/2.0/`` and before ``release build``. It reads the tree's
``metadata.parquet`` and the image files it names; nothing leaves the machine.

VM command (from the marine-data checkout, no network, no bucket)::

    uv run python scripts/benchmark_manifest_coralscapes_v2.py \\
        --staged <staging root>/sources/coralscapes-v2/2.0

Rows: every frame whose ``upstream_split`` is the registry entry's ``eval_split`` (test) or
``heldout_val`` (validation): 473 + 170 for ``coralscapes-v2`` in ``registry/benchmarks.yaml``.
Columns are those of every manifest (``marinedata.bench_manifest.MANIFEST_COLUMNS``); ``stem``
is the V2 ``upstream_id`` when the tree carries one (``site<N>_<frame>_<ts>``, the key the V1
twins link by), else its ``stem``. A split count that differs from the registry exits 2 before
anything is hashed unless ``--allow-count-mismatch``; ``--dry-run`` runs that check and builds
nothing. An existing manifest is resumed by stem, as ``bench manifest`` does. Until this has
run, ``coralscapes-v2`` has no manifest and the G7 coverage rule fails closed on it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "coralscapes-v2"
DEFAULT_REGISTRY = REPO / "registry" / "benchmarks.yaml"
DEFAULT_OUT = REPO / "registry" / "benchmarks" / "manifests" / f"{BENCHMARK_ID}.parquet"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SPLIT_NAMES = {"val": "validation", "valid": "validation"}


def split_name(record: dict[str, Any]) -> str:
    name = str(record.get("upstream_split") or "").strip().lower()
    return SPLIT_NAMES.get(name, name)


def frame_stem(record: dict[str, Any]) -> str:
    return str(record.get("upstream_id") or record["stem"])


def expected_counts(entry: Any) -> dict[str, int]:
    """``{split: n}`` the registry entry declares for its eval and held-out splits."""
    split = entry.upstream_split
    return {
        name: int(split.counts[name])
        for name in (split.eval_split, split.heldout_val)
        if name and split.counts.get(name) is not None
    }


def eval_frames(staged: Path, wanted: frozenset[str]) -> list[dict[str, Any]]:
    """The ``metadata.parquet`` records of ``staged`` in one of the ``wanted`` splits, by stem."""
    import pyarrow.parquet as pq

    meta = staged / "metadata.parquet"
    if not meta.is_file():
        raise SystemExit(
            f"no metadata.parquet under {staged}: --staged is sources/{BENCHMARK_ID}/<version>"
        )
    records = [r for r in pq.read_table(meta).to_pylist() if split_name(r) in wanted]
    return sorted(records, key=frame_stem)


def image_file(staged: Path, record: dict[str, Any]) -> Path:
    """The record's image: its own ``image_path``, else ``images/**/<stem>.<ext>``."""
    rel = record.get("image_path")
    if rel and (staged / str(rel)).is_file():
        return staged / str(rel)
    found = sorted(
        p
        for p in (staged / "images").rglob(f"{record['stem']}.*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise SystemExit(f"{frame_stem(record)}: no image file under {staged / 'images'}")
    return found[0]


def raw_images(staged: Path, records: list[dict[str, Any]]) -> Iterator[Any]:
    from marinedata.bench_manifest import RawImage

    for record in records:
        path = image_file(staged, record)
        yield RawImage(
            upstream_path=str(record.get("upstream_path") or record.get("image_path") or path.name),
            stem=frame_stem(record),
            upstream_split=split_name(record),
            data=path.read_bytes(),
        )


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--staged", type=Path, required=True, help="staged tree with metadata.parquet"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="manifest parquet to write")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="benchmarks.yaml")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="count the eval frames and stop")
    args = parser.parse_args(argv)

    from marinedata.bench_manifest import build_manifest, write_manifest
    from marinedata.benchmarks import BenchmarkRegistry

    entry = BenchmarkRegistry.load(args.registry).by_id(BENCHMARK_ID)
    want = expected_counts(entry)
    records = eval_frames(args.staged, frozenset(want))
    found = dict(sorted(Counter(split_name(r) for r in records).items()))
    print(f"{BENCHMARK_ID}: eval frames {found}; registry {dict(sorted(want.items()))}")
    if found != dict(sorted(want.items())) and not args.allow_count_mismatch:
        print(
            "split counts differ from the registry (--allow-count-mismatch to override)",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return 0
    table, before = build_manifest(
        entry, raw_images(args.staged, records), args.out, _log, out_path=args.out
    )
    write_manifest(table, args.out)
    print(f"wrote {table.num_rows} rows ({before} resumed) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
