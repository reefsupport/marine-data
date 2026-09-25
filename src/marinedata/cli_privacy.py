"""``marinedata privacy-scan`` sub-command — face/person detection for WP-5b.

Split out of :mod:`marinedata.cli` for the same reason as :mod:`marinedata.cli_ingest`.
Requires the ``privacy`` extra (``uv sync --extra privacy``): torch, torchvision,
opencv-python-headless.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .privacy import scan


def _cmd_privacy_scan(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    parquet_paths = sorted(input_dir.glob("*.parquet"))
    if not parquet_paths:
        print(f"no parquet shards found under {input_dir}", file=sys.stderr)
        return 2

    def log(msg: str) -> None:
        print(f"[privacy-scan] {msg}", file=sys.stderr)

    start = time.time()
    stats = scan(
        parquet_paths=parquet_paths,
        output_path=Path(args.output),
        face_weights=Path(args.face_weights),
        person_weights=Path(args.person_weights) if args.person_weights else None,
        batch_size=args.batch_size,
        limit=args.limit,
        flush_every=args.flush_every,
        log=log,
    )
    elapsed = time.time() - start
    print(
        f"scanned={stats.scanned} skipped_cached={stats.skipped_cached} "
        f"decode_errors={stats.decode_errors} faces_found={stats.faces_found} "
        f"people_found={stats.people_found} elapsed_s={elapsed:.1f}"
    )
    return 0


def add_privacy_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "privacy-scan",
        help="Scan HF-build image shards for identifiable faces / people (WP-5b)",
    )
    p.add_argument("--input", required=True, help="Directory of images/*.parquet shards")
    p.add_argument("--output", required=True, help="Output privacy.parquet path (resumable)")
    p.add_argument("--face-weights", required=True, help="Path to YuNet ONNX weights")
    p.add_argument(
        "--person-weights",
        default=None,
        help="TORCH_HOME cache dir pre-populated with the SSDLite COCO checkpoint (optional)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Stop after N new images (testing)")
    p.add_argument("--flush-every", type=int, default=500)
    p.set_defaults(func=_cmd_privacy_scan)
