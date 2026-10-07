"""``marinedata labelquality`` — WP-9 label-quality audit commands."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _cmd_features(args: argparse.Namespace) -> int:
    from .labelquality.features import extract

    shards = [Path(p) for p in sorted(glob.glob(str(Path(args.hf) / "data/images/*.parquet")))]
    run = extract(shards, Path(args.out), revision=args.revision, device=args.device)
    print(f"features={run.count} revision={run.revision} seconds={run.seconds}")
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    from .labelquality.pipeline import run_agreement

    res = run_agreement(Path(args.hf), Path(args.dhash_db), Path(args.out))
    print(json.dumps({"d0_sha_pairs": res["d0_sha_pairs"], "d0_clusters": res["d0_clusters"]}))
    return 0


def _cmd_confident(args: argparse.Namespace) -> int:
    from .labelquality.pipeline import run_confident

    res = run_confident(Path(args.hf), Path(args.groups), Path(args.features), Path(args.out))
    for task, t in res["tasks"].items():  # type: ignore[attr-defined]
        print(f"{task}: n={t['n']} flagged={t['flagged']} noise={t['noise']:.4f}")
    return 0


def add_labelquality_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire ``labelquality {features,agreement,confident}`` onto ``sub``."""
    p = sub.add_parser("labelquality", help="Label-quality audit (agreement, confident learning)")
    lq = p.add_subparsers(dest="labelquality_command", required=True)

    f = lq.add_parser("features", help="DINOv2 feature cache over an HF build's images")
    f.add_argument("--hf", required=True, help="HF build root (has data/images/)")
    f.add_argument("--out", required=True)
    f.add_argument("--revision", default=None, help="HF revision sha to pin")
    f.add_argument("--device", default="mps")
    f.set_defaults(func=_cmd_features)

    a = lq.add_parser("agreement", help="Cross-source agreement on identical / dHash-0 images")
    a.add_argument("--hf", required=True)
    a.add_argument("--dhash-db", required=True, help="dhash-pillow-<ver>.sqlite from the release")
    a.add_argument("--out", required=True)
    a.set_defaults(func=_cmd_agreement)

    c = lq.add_parser("confident", help="Out-of-fold probe + confident learning")
    c.add_argument("--hf", required=True)
    c.add_argument("--groups", required=True, help="WP-10 groups.parquet")
    c.add_argument("--features", required=True, help="feature cache dir")
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_confident)
