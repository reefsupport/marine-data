"""``marinedata release gate BUILD --patch-of v1.1.0``: the gates of a patch release.

Kept apart from :mod:`marinedata.cli_release_gate`, which only forwards here when ``--patch-of`` is
given: without it nothing changes. A patch adds configs to a published release (v1.1.1
``condition-masks``); the prior release is read from the Hub (anonymous file list with LFS sha256s
and the card, never the 20 GB tree) or offline (``--patch-prior-dir`` plus
``--patch-prior-tree-json``).
Every verdict ``release publish`` reads is merged into ``BUILD/GATES.json`` (see
:mod:`.gates.patch_runner`), so publish's own check passes without a ``required_for_publish``
override. The exit code is 1 when a gate failed or was skipped."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gates import GATES_FILE, gate_status, merge_gates_json, publish_verdict
from .gates.leak import rescan_gates_json
from .gates.patch_base import DEFAULT_REPO, load_patch_base
from .gates.patch_runner import PATCH_ALL, PatchOptions, run_patch_gate


def add_patch_options(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("patch releases (v1.1.1 condition-masks; docs/GATES.md)")
    group.add_argument(
        "--patch-of",
        default=None,
        metavar="RELEASE",
        help="the build is a patch of this published release (e.g. v1.1.0): run every publish "
        "gate against the patch and the release's Hub tree and card. Needs --v1-dir, "
        "--hidden-test and --staged-root; runs all gates",
    )
    group.add_argument("--patch-repo", default=DEFAULT_REPO, help="Hub dataset repo of RELEASE")
    group.add_argument(
        "--patch-prior-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="offline: DIR/README.md is RELEASE's card",
    )
    group.add_argument(
        "--patch-prior-tree-json",
        type=Path,
        default=None,
        metavar="FILE",
        help="offline: {path: lfs sha256} of RELEASE",
    )
    group.add_argument(
        "--split-map",
        type=Path,
        default=None,
        metavar="FILE",
        help="patch splits: the SPLIT_MAP.json that assigned the patch rows (default: the "
        "--release-dir's, else the build's)",
    )
    group.add_argument(
        "--staged-root", type=Path, default=None, metavar="DIR",
        help="patch masks: the staged source tree (CHECKSUMS.sha256, metadata.parquet, labels/)",
    )  # fmt: skip


def _options(args: argparse.Namespace, base) -> PatchOptions:
    return PatchOptions(
        base=base,
        flavour=args.flavour or "open",
        release_dir=Path(args.release_dir) if args.release_dir else None,
        v1_dir=args.v1_dir,
        hidden_test=tuple(args.hidden_test),
        staged_root=args.staged_root,
        split_map=args.split_map,
        extra_nc_ids=frozenset(args.nc_id),
        empty_threshold=args.empty_threshold,
        sample_rows=args.sample_rows,
        taxonomy_version=args.taxonomy_version,
    )


def cmd_release_gate_patch(args: argparse.Namespace) -> int:
    if args.build_dir is None or not Path(args.build_dir).is_dir():
        print("release gate --patch-of: pass the build directory", file=sys.stderr)
        return 2
    build = Path(args.build_dir)
    try:
        base = load_patch_base(
            args.patch_of,
            repo=args.patch_repo,
            prior_dir=args.patch_prior_dir,
            tree_json=args.patch_prior_tree_json,
        )
    except Exception as exc:  # network, missing card: the gates cannot run, so nothing passes
        print(
            f"release gate --patch-of: cannot read {args.patch_of}: "
            f"{type(exc).__name__}: {str(exc)[:160]}",
            file=sys.stderr,
        )
        return 1
    options = _options(args, base)
    path = Path(args.gates) if args.gates else build / GATES_FILE
    results = [run_patch_gate(name, build, options) for name in PATCH_ALL]
    try:
        merge_gates_json(path, results)
        late = rescan_gates_json(path, flavour=options.flavour, extra_nc_ids=options.extra_nc_ids)
        if late is not None:
            merge_gates_json(path, [late])
            results = [late if r.name == late.name else r for r in results]
    except (OSError, ValueError) as exc:
        print(f"release gate: cannot update GATES.json: {exc}", file=sys.stderr)
        return 1
    refused = []
    for result in results:
        status = gate_status(result.to_dict())
        print(f"gate {result.name}: {status} -> {path}")
        for text in result.failures[:10]:
            print(f"  {text}", file=sys.stderr)
        if status != "passed":
            refused.append(result.name)
    if path == build / GATES_FILE:
        _, problems = publish_verdict(build)
        print(f"publish verdict: {'BLOCKED' if problems else 'OK'} ({len(problems)} problem(s))")
        for line in problems[:10]:
            print(f"  {line}")
    return 1 if refused else 0
