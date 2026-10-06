"""``marinedata metadata normalise`` — local-only metadata normalisation (WP-U14a).

Reads one staged source version (read-only), runs its normaliser and writes
``<out>/<source>/<version>/metadata.normalised.parquet``. Nothing is written to the bucket.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_normalise(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from .metadata_norm import normalise
    from .metadata_norm.fill import fill_rates
    from .metadata_norm.stage import load_inputs

    if args.limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        return 2
    try:
        version, staged, ctx = load_inputs(
            args.source,
            args.version,
            args.limit,
            events_path=args.events,
            meow_path=args.meow,
        )
    except Exception as exc:  # network / registry failure: report, never traceback
        print(f"error: {args.source}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    table = normalise(args.source, version, staged, ctx)
    out = Path(args.out) / args.source / version
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metadata.normalised.parquet"
    pq.write_table(table, path, compression="zstd")
    filled = sum(1 for v in fill_rates(table).values() if v > 0)
    print(
        f"{args.source} {version}: {table.num_rows} rows, "
        f"{filled}/{table.num_columns} columns filled -> {path}"
    )
    return 0


def add_metadata_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``metadata`` sub-command onto ``sub`` (called from ``cli.build_parser``)."""
    p = sub.add_parser("metadata", help="Normalise staged metadata (local output only)")
    msub = p.add_subparsers(dest="metadata_command", required=True)
    n = msub.add_parser("normalise", help="Write a SampleRow-shaped normalised parquet locally")
    n.add_argument("--source", required=True, help="Source id")
    n.add_argument("--version", help="Staged version (default: the ingest spec's)")
    n.add_argument("--out", required=True, help="Local output directory")
    n.add_argument("--limit", type=int, default=1000, help="Max rows (default 1000)")
    n.add_argument("--events", help="Cached MERMAID sample-event JSON (see metadata_norm.geo)")
    n.add_argument("--meow", help="Local MEOW polygon file (lat/lon -> ecoregion; no download)")
    n.set_defaults(func=_cmd_normalise)
