"""CLI wiring for WP-3's format-hardening commands: ``manifest``, ``verify-release`` and
``export-wds``. Split out of :mod:`marinedata.cli` the same way ``cli_ingest``/``cli_release``/
``cli_splitmap`` are, so ``cli.py`` only imports and registers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import build_manifest, read_manifest
from .verify_release import verify
from .wds_export import export_wds


def _cmd_manifest(args: argparse.Namespace) -> int:
    dest = build_manifest(args.dir, args.out)
    n = len(read_manifest(dest))
    print(f"manifest: {n} files -> {dest}")
    return 0


def _cmd_verify_release(args: argparse.Namespace) -> int:
    deep = args.deep
    if deep is not None and deep != "all":
        deep = int(deep)
    report = verify(
        args.target,
        manifest_path=args.manifest,
        deep=deep,
        strict=args.strict,
        scratch=args.scratch,
    )
    print(report.summary(strict=args.strict))
    return 0 if report.ok(strict=args.strict) else 1


def _cmd_export_wds(args: argparse.Namespace) -> int:
    results = export_wds(args.hf_dir, args.out, limit_shards=args.limit_shards)
    for r in results:
        print(f"{r.tar_path}: {r.n_images} images")
    return 0


def add_verify_subparsers(sub: argparse._SubParsersAction) -> None:
    p_manifest = sub.add_parser(
        "manifest",
        help="Write MANIFEST.tsv (path, size, sha256, s3_etag, parquet rows) for a build",
    )
    p_manifest.add_argument("dir", type=Path)
    p_manifest.add_argument("--out", type=Path, default=None, help="default: <dir>/MANIFEST.tsv")
    p_manifest.set_defaults(func=_cmd_manifest)

    p_verify_release = sub.add_parser(
        "verify-release",
        help="Check a build (local dir or s3://bucket/prefix) against its MANIFEST.tsv",
    )
    p_verify_release.add_argument("target", help="local directory or s3://bucket/prefix")
    p_verify_release.add_argument("--manifest", type=Path, default=None)
    p_verify_release.add_argument(
        "--deep", default=None, help="N random objects, or 'all' (s3 only)"
    )
    p_verify_release.add_argument("--strict", action="store_true", help="unlisted files also fail")
    p_verify_release.add_argument("--scratch", type=Path, default=None)
    p_verify_release.set_defaults(func=_cmd_verify_release)

    p_wds = sub.add_parser(
        "export-wds", help="WebDataset tars from the images config + label joins"
    )
    p_wds.add_argument("hf_dir", type=Path)
    p_wds.add_argument("out", type=Path)
    p_wds.add_argument("--limit-shards", type=int, default=None)
    p_wds.set_defaults(func=_cmd_export_wds)
