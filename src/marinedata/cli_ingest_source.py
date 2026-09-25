"""``marinedata ingest-source <adapter> <spec.yaml> [--dry-run]`` (WP-6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_ingest_source(args: argparse.Namespace) -> int:
    from .adapters import AccessRefused
    from .ingest_source import IngestSpec, report_json, run_ingest
    from .s3_upload import MiB

    spec = IngestSpec.load(Path(args.spec), args.adapter)
    if args.max_images is not None:
        spec.max_images = args.max_images
    if not args.dry_run and not args.work:
        print(
            "error: --work is required (a directory under your scratch $SP/<n>/)", file=sys.stderr
        )
        return 2
    try:
        report = run_ingest(
            spec, Path(args.work or "."), dry_run=args.dry_run, part_size=args.part_size_mib * MiB
        )
    except AccessRefused as exc:
        print(f"NEEDS-YOHAN\t{exc.url}\t{exc.needs}", file=sys.stderr)
        return 3
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
    p.set_defaults(func=_cmd_ingest_source)
