"""``marinedata tasks build-points``: a task-layer table from a staged tree (P3.1).

Reads one staged source by anonymous single-key GETs (no listing, no image) and writes
``<out>/points.parquet``, ``summary.json`` and ``unmatched.tsv`` locally. Nothing goes to a bucket.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_build_points(args: argparse.Namespace) -> int:
    from .task_layers.seaview_points import SOURCE_ID, run_build_points

    if args.source != SOURCE_ID:
        print(
            f"error: build-points supports {SOURCE_ID} only "
            "(reefolution and mermaid-aws have their own producers)",
            file=sys.stderr,
        )
        return 2
    try:
        summary = run_build_points(
            args.out,
            version=args.version,
            cache_dir=args.cache,
            master_sample_bytes=args.master_sample_bytes,
            worms_cache_path=args.worms_cache,
            worms=not args.no_worms,
            annotations_root=args.annotations_root,
            metadata_dir=args.metadata_dir,
        )
    except Exception as exc:  # network / registry failure: report, never a traceback
        print(f"error: {args.source}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    t = summary["totals"]
    print(
        f"{args.source} {summary['source_version']}: {t['points']} points on "
        f"{t['images_with_points']} images, crosswalked {summary['crosswalked_pct']}%, "
        f"WoRMS {summary['worms_aphia_pct']}%, {t['annotation_ids_without_image']} ids and "
        f"{t['images_without_annotation_rows']} images unmatched -> {args.out}"
    )
    if j := summary.get("metadata_join"):
        print(f"metadata join: lat/lon {j['lat_lon_pct']}%, date {j['date_pct']}% of {j['points']}")
    return 0


def add_tasks_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``tasks`` sub-command onto ``sub`` (called from ``cli.build_parser``)."""
    p = sub.add_parser("tasks", help="Build task-layer tables from staged trees (local output)")
    tsub = p.add_subparsers(dest="tasks_command", required=True)
    b = tsub.add_parser("build-points", help="Benthic points table for a staged source (Seaview)")
    b.add_argument("--source", required=True, help="Source id (seaview-survey-imagery)")
    b.add_argument(
        "--out", required=True, help="Output directory: points.parquet, summary.json, unmatched.tsv"
    )
    b.add_argument("--version", help="Staged version (default: the ingest spec's)")
    b.add_argument("--cache", help="Directory to cache the small tables (CSVs, metadata.parquet)")
    b.add_argument(
        "--annotations-root",
        help="Also write <root>/_annotations/points/<source>/<version>.parquet "
        "(the layout build_points_config reads)",
    )
    b.add_argument(
        "--metadata-dir",
        help="Resolver output (metadata resolve --out): also write points_joined.parquet, "
        "points + lat/lon/date/site/licence columns by image_sha256, and metadata_join.json",
    )
    b.add_argument(
        "--master-sample-bytes",
        type=int,
        default=8 * (1 << 20),
        help="Head sample of the master CSV to cross-check (0 = skip)",
    )
    b.add_argument("--worms-cache", help="WoRMS name cache (default: taxonomy/worms-cache.json)")
    b.add_argument("--no-worms", action="store_true", help="Never call the WoRMS REST API")
    b.set_defaults(func=_cmd_build_points)
