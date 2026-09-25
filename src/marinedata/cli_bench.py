"""``marinedata bench`` — load/validate the benchmark registry and report manifest coverage."""

from __future__ import annotations

import argparse
import json
import sys
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
    if args.strict and pending:
        print(f"pending: {', '.join(sorted(pending))}", file=sys.stderr)
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
