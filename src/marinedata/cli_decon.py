"""``marinedata decon check`` — run the S0-S5 benchmark decontamination gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks import BenchmarkRegistry
from .decon import check, write_outputs


def _cmd_check(args: argparse.Namespace) -> int:
    from .dedup.corpus import iter_hf_images, iter_staged

    registry = BenchmarkRegistry.load(args.benchmarks_yaml)
    items = []
    for root in args.hf_images or []:
        items.extend(iter_hf_images(root))
    for spec in args.staged or []:
        label, _, root = spec.partition("=")
        items.extend(iter_staged(root, label))

    image_roots = {}
    for spec in args.benchmark_root or []:
        bid, _, root = spec.partition("=")
        image_roots[bid] = Path(root)

    result = check(
        args.release_dir,
        registry,
        items,
        manifests_root=args.manifests_root,
        image_roots=image_roots,
        embed_weights=args.embed_weights,
        embed_cache=args.embed_cache,
        workers=args.workers,
        dedup_crop=args.dedup_crop,
    )
    out_dir = args.out or (Path(args.release_dir) / "decon")
    write_outputs(result, registry, out_dir)
    print((out_dir / "overlap.md").read_text())
    if not result.ok:
        for failure in result.failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("decon check: PASS")
    return 0


def add_decon_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("decon", help="Benchmark decontamination gate (S0-S5)")
    dsub = p.add_subparsers(dest="decon_command", required=True)

    q = dsub.add_parser("check", help="Run the gate against a built release")
    q.add_argument("release_dir", type=Path, help="<out>/releases/<release>/ from `release build`")
    q.add_argument("--benchmarks-yaml", type=Path, default=None, dest="benchmarks_yaml")
    q.add_argument("--manifests-root", type=Path, default=None, dest="manifests_root")
    q.add_argument("--hf-images", action="append", dest="hf_images", help="HF `images` config dir")
    q.add_argument("--staged", action="append", dest="staged", help="label=root staged source tree")
    q.add_argument(
        "--benchmark-root",
        action="append",
        dest="benchmark_root",
        help="benchmark_id=root, the local staged tree its manifest was built from",
    )
    q.add_argument("--embed-weights", type=Path, default=None, dest="embed_weights")
    q.add_argument("--embed-cache", type=Path, default=None, dest="embed_cache")
    q.add_argument("--workers", type=int, default=4)
    q.add_argument(
        "--dedup-crop",
        action="store_true",
        dest="dedup_crop",
        help="Enable the S5 patch/crop stage (WP-10c's crop channel, D-T). Default OFF.",
    )
    q.add_argument("--out", type=Path, default=None, help="default: <release_dir>/decon")
    q.set_defaults(func=_cmd_check)
