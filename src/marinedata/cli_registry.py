"""``marinedata registry verify`` wiring (RB-3). Kept out of :mod:`marinedata.cli`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .registry import Registry
from .registry_verify import failed, load_listing, summarise, verify_live, verify_offline


def _cmd_registry_verify(args: argparse.Namespace) -> int:
    reg = Registry.load()
    sources = [reg.source(i) for i in args.source_id] if args.source_id else list(reg.sources)
    if args.live:
        from .s3_upload import client_from_rclone  # creds via configparser, never printed

        checks = verify_live(sources, client_from_rclone())
    else:
        if not args.listing:
            raise ValueError("offline verify needs --listing [BUCKET=]PATH (or use --live)")
        listing: dict[str, set[str]] = {}
        for spec in args.listing:
            bucket, _, path = spec.rpartition("=")
            for b, keys in load_listing(Path(path), bucket or None).items():
                listing.setdefault(b, set()).update(keys)
        checks = verify_offline(sources, listing)
    for c in checks:
        if args.all or c.status in ("missing", "no-manifest"):
            print(c.line())
    print(f"\n{summarise(checks)}")
    return 1 if failed(checks) and not args.report_only else 0


def add_registry_subparsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("registry", help="Registry pointer checks against the bucket")
    rsub = p.add_subparsers(dest="registry_cmd", required=True)
    v = rsub.add_parser(
        "verify", help="Every s3 pointer must hold a CHECKSUMS.sha256 (offline snapshot or --live)"
    )
    v.add_argument("source_id", nargs="*")
    v.add_argument(
        "--listing",
        action="append",
        default=[],
        metavar="[BUCKET=]PATH",
        help="listing snapshot (parquet with a key column, or TSV/CSV); repeatable",
    )
    v.add_argument("--live", action="store_true", help="HEAD each CHECKSUMS.sha256 (needs creds)")
    v.add_argument("--all", action="store_true", help="also print the ok rows")
    v.add_argument("--report-only", action="store_true", help="always exit 0")
    v.set_defaults(func=_cmd_registry_verify)
