"""``marinedata ingest-source <adapter> <spec.yaml> [--dry-run]`` (WP-6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_ingest_source(args: argparse.Namespace) -> int:
    import json

    from .adapters import AccessRefused, NoStageableItems
    from .ingest_source import IngestSpec, fetch_throughput, report_json, run_ingest
    from .s3_upload import MiB

    spec = IngestSpec.load(Path(args.spec), args.adapter)
    if args.max_images is not None:
        spec.max_images = args.max_images
    if not args.dry_run and not args.work and not args.fetch_only:
        print(
            "error: --work is required (a directory under your scratch $SP/<n>/)", file=sys.stderr
        )
        return 2
    try:
        if args.fetch_only:
            # WP-6b: network-light throughput probe — never uploads, never stages.
            result = fetch_throughput(
                spec,
                Path(args.work or "."),
                jobs=args.jobs,
                max_per_host=args.max_per_host,
                limit=args.limit,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        report = run_ingest(
            spec,
            Path(args.work or "."),
            dry_run=args.dry_run,
            part_size=args.part_size_mib * MiB,
            jobs=args.jobs,
            max_per_host=args.max_per_host,
            part_jobs=args.part_jobs,
        )
    except AccessRefused as exc:
        print(f"NEEDS-YOHAN\t{exc.url}\t{exc.needs}", file=sys.stderr)
        return 3
    except NoStageableItems as exc:
        # D-R4 root cause fix: 0 stageable items is never a false "ok" (see
        # marinedata.adapters.NoStageableItems) — a real, non-zero exit + reason.
        print(f"NEEDS-ADAPTER\t{exc.reason}\t{exc}", file=sys.stderr)
        return 4
    print(report_json(report))
    return 0


def add_ingest_source_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ingest-source", help="Adapter-driven ingest of an open source to sources/<id>/<version>/"
    )
    p.add_argument("adapter", choices=["hf", "http", "zenodo", "bucket", "github"])
    p.add_argument("spec", help="Ingest spec yaml (see docs/ingest-howto.md)")
    p.add_argument("--dry-run", action="store_true", help="Resolve + enumerate only")
    p.add_argument("--work", help="Work/temp dir (bounded by the spec's temp_cap_gb)")
    p.add_argument("--max-images", type=int, default=None, help="Smoke cap")
    p.add_argument("--part-size-mib", type=int, default=64)
    p.add_argument(
        "--jobs", type=int, default=8, help="Bounded fetch/PUT concurrency, 1-32 (WP-6b)"
    )
    p.add_argument("--max-per-host", type=int, default=4, help="Per-host in-flight cap (WP-6b)")
    p.add_argument("--part-jobs", type=int, default=4, help="Parallel multipart parts (WP-6b)")
    p.add_argument(
        "--fetch-only",
        action="store_true",
        help="Network-light throughput probe: fetch, don't stage or upload (WP-6b)",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap items fetched with --fetch-only")
    p.set_defaults(func=_cmd_ingest_source)
