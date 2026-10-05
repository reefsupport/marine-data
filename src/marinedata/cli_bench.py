"""``marinedata bench`` — load/validate the benchmark registry and report manifest coverage."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .benchmarks import BenchmarkRegistry, BenchmarksError, benchmarks_sha256


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        reg = BenchmarkRegistry.load(args.path)
    except BenchmarksError as exc:
        print(f"benchmarks.yaml: FAIL — {exc}", file=sys.stderr)
        return 1
    digest = benchmarks_sha256(reg)
    pending = reg.pending()
    verified = sum(1 for e in reg.benchmarks if e.split_verified)
    print(f"benchmarks {len(reg.benchmarks)} verified {verified}/{len(reg.benchmarks)}")
    print(f"benchmarks_sha256 {digest}")
    print(f"manifests present {len(reg.benchmarks) - len(pending)} pending {len(pending)}")
    uncovered = reg.uncovered()
    print(f"decon-exempt {len(reg.exemptions())} uncovered {len(uncovered)}")
    if args.strict and pending:
        print(f"pending: {', '.join(sorted(pending))}", file=sys.stderr)
    if args.strict and uncovered:
        print(
            f"uncovered (no manifest, no exemption): {', '.join(sorted(uncovered))}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_hash(args: argparse.Namespace) -> int:
    reg = BenchmarkRegistry.load(args.path)
    print(benchmarks_sha256(reg))
    return 0


def _cmd_manifests(args: argparse.Namespace) -> int:
    reg = BenchmarkRegistry.load(args.path)
    out = {
        "present": sorted(e.id for e in reg.benchmarks if e.id not in reg.pending()),
        "pending": sorted(reg.pending()),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


def _default_specs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "registry" / "ingest-specs"


def _cmd_manifest(args: argparse.Namespace) -> int:
    from .bench_manifest import (
        ManifestBuildError,
        build_manifest,
        iter_bucket_images,
        iter_upstream_images,
        resolve_source,
        write_manifest,
    )
    from .s3_client import client_from_rclone

    reg = BenchmarkRegistry.load(args.path)
    try:
        entry = reg.by_id(args.benchmark_id)
    except KeyError:
        print(f"unknown benchmark id: {args.benchmark_id}", file=sys.stderr)
        return 1
    out_path = reg.manifest_path(entry.id)
    client = client_from_rclone(args.remote, concurrent=True) if args.source != "upstream" else None
    source = resolve_source(args.source, entry, client, args.bucket) if client else "upstream"
    try:
        if source == "bucket":
            images = iter_bucket_images(client, args.bucket, entry, layout=args.layout)
            table, n_before = build_manifest(entry, images, out_path)
        else:
            with tempfile.TemporaryDirectory(prefix="marinedata-bench-") as tmp:
                images = iter_upstream_images(
                    entry, args.specs_dir, Path(tmp), max_bytes=args.max_bytes
                )
                table, n_before = build_manifest(entry, images, out_path)
    except ManifestBuildError as exc:
        print(f"{entry.id}: FAIL — {exc}", file=sys.stderr)
        return 1
    write_manifest(table, out_path)
    new = table.num_rows - n_before
    print(f"{entry.id}: source={source} rows={table.num_rows} new={new} -> {out_path}")
    return 0


def add_bench_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bench", help="Benchmark registry: validate, hash, manifest coverage")
    bsub = p.add_subparsers(dest="bench_command", required=True)

    def path_arg(q: argparse.ArgumentParser) -> None:
        q.add_argument("--path", type=Path, help="registry/benchmarks.yaml (default: repo copy)")

    q = bsub.add_parser("check", help="Validate the registry and report coverage")
    path_arg(q)
    q.add_argument("--strict", action="store_true", help="List pending benchmark ids on stderr")
    q.set_defaults(func=_cmd_check)

    q = bsub.add_parser("hash", help="Print benchmarks_sha256 only")
    path_arg(q)
    q.set_defaults(func=_cmd_hash)

    q = bsub.add_parser("manifests", help="List which benchmarks have a manifest vs. are pending")
    path_arg(q)
    q.add_argument("-o", "--output", help="Write JSON here instead of stdout")
    q.set_defaults(func=_cmd_manifests)

    q = bsub.add_parser("manifest", help="Build one benchmark's eval-image manifest parquet")
    path_arg(q)
    q.add_argument("benchmark_id")
    q.add_argument(
        "--source",
        choices=("bucket", "upstream", "auto"),
        default="auto",
        help="bucket: rs-storage-open parquet; upstream: ingest adapter stream; "
        "auto: pick by CHECKSUMS.sha256 (default)",
    )
    q.add_argument(
        "--layout",
        choices=("auto", "staged"),
        default="auto",
        help="bucket source only. auto: _stream parts when present, else staged images/. "
        "staged: metadata.parquet + loose images/ only, fetching just the eval-split objects "
        "(use when the eval split is a small slice of the tree)",
    )
    q.add_argument("--bucket", default="rs-storage-open")
    q.add_argument("--remote", default="rs-hel1", help="rclone remote name for bucket credentials")
    q.add_argument("--specs-dir", type=Path, default=_default_specs_dir())
    q.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Upstream source only: stop once fetched sample bytes reach this cap "
        "(resumable — rerun to continue past it)",
    )
    q.set_defaults(func=_cmd_manifest)
