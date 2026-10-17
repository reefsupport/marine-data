"""``marinedata release gate`` for a v1.1.0 build: G1 .. G9 into ``GATES.json`` (docs/GATES.md).

Kept apart from :mod:`marinedata.cli_release_gate` so the v1.0 gate path stays as it was: that
module only forwards here (:func:`wants_v11`). ``--v11``, any v1.1.0-only flag, or a v1.1.0 build
with no v1.0 gate flag runs :data:`V11_ALL`; ``--only GATE`` narrows it. Every result is merged
into ``BUILD_DIR/GATES.json`` (the other gates' entries are kept). The exit code is 1 when a gate
failed and, outside ``--rehearsal``, when a gate was skipped: a gate that did not look has not
passed. ``--rehearsal`` reaches only the gates that define it (splits_v11, decon), turns their
unsplit-only / missing-manifest failure into ``skipped``, and a skip never publishes. The v1.0
gates (licence, dedup, splits, load, card, croissant) run with ``--all`` and stay in the verdict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gates import GATES_FILE, gate_status, merge_gates_json, publish_verdict
from .gates.v11_runner import V11_ALL, V11Options, run_v11_gate


def add_v11_options(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("v1.1.0 gates (G1 join .. G9 labelmap; docs/GATES.md)")
    group.add_argument(
        "--v11",
        action="store_true",
        help="run G1..G9 on a v1.1.0 build (implied by the options below, and by a v1.1.0 "
        "build (task_layers/V11_TABLES.json) when no v1.0 gate flag is given)",
    )
    group.add_argument(
        "--only",
        action="append",
        choices=V11_ALL,
        metavar="GATE",
        help="run only this v1.1.0 gate (repeatable): " + ", ".join(V11_ALL),
    )
    group.add_argument(
        "--v1-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="the frozen v1.0 release directory (join, frozen, bytes_once, splits_v11, "
        "decon with --hidden-test-published-exception)",
    )
    group.add_argument(
        "--hidden-test",
        action="append",
        type=Path,
        default=[],
        metavar="MANIFEST",
        help="decon: hidden-test manifest (repeatable)",
    )
    group.add_argument(
        "--hidden-test-published-exception",
        action="store_true",
        help="decon: a hidden-test image already public in the v1.0 tree (--v1-dir) is exempt "
        "(counted as hidden_test_already_published); the others still fail as hidden_test_new. "
        "Benchmark overlaps are unchanged. Default off: any hidden-test overlap fails",
    )
    group.add_argument(
        "--benchmark",
        action="append",
        type=Path,
        default=[],
        metavar="MANIFEST",
        help="decon: external benchmark manifest (repeatable)",
    )
    group.add_argument(
        "--nc-id",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help="join, leak: one more non-open source id that must not appear (repeatable; the "
        "registry's own are always checked)",
    )
    group.add_argument(
        "--privacy-review",
        type=Path,
        default=None,
        metavar="FILE",
        help="privacy: the human review list of face detections (default: registry/privacy/"
        "review.yaml); a cleared image is counted, a withheld one still in the build fails",
    )
    group.add_argument(
        "--taxonomy-version",
        default=None,
        metavar="X.Y.Z",
        help="labelmap: taxonomy release the class ids must resolve in (default: registry/"
        "gates.yaml labelmap.taxonomy_version, else the current registry taxonomy)",
    )
    group.add_argument(
        "--rehearsal",
        action="store_true",
        help="splits_v11, decon: an unsplit-only / manifest-missing failure becomes 'skipped' "
        "(never 'passed'; release publish still refuses the build)",
    )


def wants_v11(args: argparse.Namespace, *, v10_selected: bool) -> bool:
    """True when this ``release gate`` call is for the v1.1.0 gates."""
    if (
        args.v11
        or args.only
        or args.hidden_test
        or getattr(args, "hidden_test_published_exception", False)
        or args.benchmark
        or args.nc_id
        or args.privacy_review
        or args.taxonomy_version
        or args.rehearsal
    ):
        return True
    if args.v1_dir and not v10_selected:  # with a v1.0 gate flag it feeds load, card and masks
        return True
    if v10_selected or not args.build_dir:
        return False
    return (Path(args.build_dir) / "task_layers" / "V11_TABLES.json").is_file()


def _options(args: argparse.Namespace) -> V11Options:
    return V11Options(
        flavour=args.flavour or "open",
        release_dir=Path(args.release_dir) if args.release_dir else None,
        v1_dir=args.v1_dir,
        hidden_test=tuple(args.hidden_test),
        hidden_test_published_exception=getattr(args, "hidden_test_published_exception", False),
        benchmarks=tuple(args.benchmark),
        rehearsal=args.rehearsal,
        extra_nc_ids=frozenset(args.nc_id),
        empty_threshold=args.empty_threshold,
        detector=args.detector,
        privacy_threshold=args.privacy_threshold,
        privacy_review=args.privacy_review,
        taxonomy_version=args.taxonomy_version,
    )


def cmd_release_gate_v11(args: argparse.Namespace) -> int:
    if args.build_dir is None:
        print(
            "release gate: pass the build directory: marinedata release gate BUILD_DIR --v11",
            file=sys.stderr,
        )
        return 2
    build = Path(args.build_dir)
    if not build.is_dir():
        print(f"release gate: not a directory: {build}", file=sys.stderr)
        return 1
    names = tuple(n for n in V11_ALL if not args.only or n in args.only)
    options = _options(args)
    path = Path(args.gates) if args.gates else build / GATES_FILE
    results = [run_v11_gate(name, build, options) for name in names]
    try:
        merge_gates_json(path, results)
    except (OSError, ValueError) as exc:
        print(f"release gate: cannot update GATES.json: {exc}", file=sys.stderr)
        return 1
    refused = []
    for result in results:
        status = gate_status(result.to_dict())
        print(f"gate {result.name}: {status} -> {path}")
        for text in result.failures[:10]:
            print(f"  {text}", file=sys.stderr)
        if status == "failed" or (status == "skipped" and not options.rehearsal):
            refused.append(result.name)
    if path == build / GATES_FILE:
        _, problems = publish_verdict(build)
        print(f"publish verdict: {'BLOCKED' if problems else 'OK'} ({len(problems)} problem(s))")
        for line in problems[:10]:
            print(f"  {line}")
    return 1 if refused else 0
