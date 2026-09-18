"""``marinedata ingest`` sub-command — parser wiring plus ``_cmd_ingest`` (D3f).

Split out of :mod:`marinedata.cli` to keep that module under the 400-line cap. Pure
move: no behaviour change. ``cli.py`` imports both names from here and re-exports
``_cmd_ingest``, so ``from .cli import _cmd_ingest`` still resolves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .registry import Registry


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Stage one source's declared version into ``<out>/sources/<source_id>/<version>/``."""
    from .fetch import cache_root
    from .gate import LicenceViolation
    from .ingest import IngestError, stage_source

    reg = Registry.load()
    source = reg.source(args.source_id)
    profile = reg.profile(args.profile)
    out, cap = Path(args.out), args.slice_cap_bytes
    try:
        result = stage_source(
            source,
            cache_root=cache_root(),
            out_root=out,
            profile=profile,
            slice_cap_bytes=cap,
            workers=args.workers,
        )
    except (IngestError, LicenceViolation) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    m = result.manifest
    print(
        f"{result.source_id}@{result.version}  {result.images} images  → {result.root}  "
        f"root_digest={m.root_digest}  files={m.files}  size_bytes={m.size_bytes}"
    )
    return 0


def add_ingest_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``ingest`` sub-command onto ``sub`` (called from ``cli.build_parser``)."""
    p_ingest = sub.add_parser("ingest", help="Stage a source version into sources/<id>/<version>/")
    p_ingest.add_argument("source_id")
    p_ingest.add_argument("--out", required=True, help="Root directory to stage into")
    p_ingest.add_argument("--profile", required=True, help="Release profile (see profiles.yaml)")
    p_ingest.add_argument(
        "--slice-cap-bytes", type=int, default=None, help="S3 sources only: cap staged bytes"
    )
    p_ingest.add_argument(
        "--workers",
        type=int,
        default=None,
        help="S3 sources only: concurrent image downloads (default 8)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)
