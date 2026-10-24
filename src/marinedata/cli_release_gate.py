# ruff: noqa: E501
"""``marinedata release gate``: run release gates on a build directory (an hf_export output) and
merge every result into ``<build>/GATES.json``; ``--metadata DIR`` runs the metadata gate (PA-P2.4)
over resolver outputs and merges it into ``--gates``. See docs/GATES.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli_release_gate_patch import add_patch_options, cmd_release_gate_patch
from .cli_release_gate_v11 import add_v11_options, cmd_release_gate_v11, wants_v11
from .gates import GATES_FILE, GateResult, gate_status, verdict, write_gates
from .gates.dedup import DEFAULT_PHASH_BITS
from .gates.load import SAMPLE_ROWS
from .gates.masks import DEFAULT_EMPTY_THRESHOLD
from .gates.privacy import DEFAULT_THRESHOLD
from .gates.runner import ALL, GateOptions, run_gate
from .gates.splits import TOLERANCE_PT

_HELP = {
    "licence": "lock source classes, row licences and attribution fit the flavour",
    "dedup": "no exact / perceptual (pHash) duplicate across sources or splits; writes phash.parquet",
    "splits": "groups stay in one split, ratios near the target, no train near-dup in val/test",
    "privacy": "EXIF GPS in images; people-row counts; --detector for faces/plates",
    "load": "datasets.load_dataset per config: schema, 50 rows per split, counts vs RELEASE.json",
    "card": "README.md parses as a dataset card; configs, counts and sizes match the release",
    "croissant": "croissant.json validates with mlcroissant (skipped when absent or not installed)",
}
_METADATA_HELP = (
    "metadata gate (PA-P2.4): DIR holds <source>/<version>/coverage.json from the resolvers; "
    "coverage floors and licence checks, merged into --gates (also alongside BUILD_DIR gates)"
)


def add_gate_parser(release_sub) -> None:
    p = release_sub.add_parser(
        "gate", help="Run release gates on a build directory and merge them into GATES.json"
    )
    p.add_argument(
        "build_dir", nargs="?", type=Path, help="hf_export output (data/, README.md, ...)"
    )
    p.add_argument("--all", action="store_true", help="run every gate")
    for name in ALL:
        if name == "masks":
            p.add_argument(
                "--masks",
                nargs="?",
                const=True,
                default=None,
                metavar="BUILD_DIR",
                help="mask configs line up with their images (a BUILD_DIR here is the legacy form)",
            )
        else:
            p.add_argument(f"--{name}", action="store_true", help=_HELP[name])
    p.add_argument(
        "--flavour", choices=("open", "nc"), default=None, help="default: RELEASE.json's"
    )
    p.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="where RELEASE.json lives (default: BUILD_DIR)",
    )
    p.add_argument(
        "--phash-bits",
        type=int,
        default=DEFAULT_PHASH_BITS,
        help="pHash Hamming limit (default: %(default)s)",
    )
    p.add_argument(
        "--split-tolerance",
        type=float,
        default=TOLERANCE_PT,
        metavar="PT",
        help="ratio tolerance in points (default: %(default)s)",
    )
    p.add_argument(
        "--empty-threshold",
        type=float,
        default=DEFAULT_EMPTY_THRESHOLD,
        metavar="RATE",
        help="masks: max share of empty masks per source (default: %(default)s)",
    )
    p.add_argument(
        "--detector", default=None, metavar="MODULE:ATTR", help="privacy: face/plate detector"
    )
    p.add_argument(
        "--privacy-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="privacy: detection score that fails a row (default: %(default)s)",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=SAMPLE_ROWS,
        help="load: rows read per split (default: %(default)s)",
    )
    p.add_argument("--metadata", default=None, metavar="DIR", help=_METADATA_HELP)
    p.add_argument(
        "--gates",
        default=None,
        metavar="FILE",
        help="GATES.json the --metadata result is merged into (created if absent; "
        "default: BUILD_DIR/GATES.json, else ./GATES.json)",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="exit 1 unless all eight gates in GATES.json ran and passed: an absent, "
        "skipped or failed gate (privacy, card and croissant included) refuses",
    )
    add_v11_options(p)
    add_patch_options(p)
    p.set_defaults(func=cmd_release_gate)


def _selected(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(ALL)
    chosen = {n for n in ALL if n != "masks" and getattr(args, n)}
    if args.masks not in (None, False):
        chosen.add("masks")
    return [n for n in ALL if n in chosen]


def _line(result: GateResult, path: Path) -> str:
    verdict_word = "SKIPPED" if result.skipped is not None else "PASS" if result.passed else "FAIL"
    rows = f" rows={result.counts['rows']}" if "rows" in result.counts else ""
    why = f" ({result.skipped})" if result.skipped else ""
    return f"release gate {result.name}: {verdict_word}{rows} failures={result.counts.get('failures_total', 0)}{why} -> {path}"


def cmd_metadata_gate(args: argparse.Namespace) -> int:
    """``release gate --metadata DIR``: the metadata gate over resolver outputs, merged into the
    GATES.json at ``--gates`` (default ``BUILD_DIR/GATES.json``, else ``./GATES.json``)."""
    from .gates import merge_gates_json
    from .gates.metadata import run_gate, source_dirs
    from .registry import Registry

    build = getattr(args, "build_dir", None)
    gates = args.gates or (Path(build) / GATES_FILE if build else GATES_FILE)
    directories = source_dirs(args.metadata)
    if not directories:
        print(
            f"release gate: no <source>/<version>/coverage.json under {args.metadata}",
            file=sys.stderr,
        )
        return 1
    result = run_gate(directories, Registry.load())
    merge_gates_json(gates, [result])
    for sid, info in sorted(result.counts["sources"].items()):
        checks = " ".join(
            f"{name}={c['level']}:{c['rate']:.4f}{'' if c['passed'] else '(MISS)'}"
            for name, c in info.get("checks", {}).items()
        )
        verdict_word = "PASS" if info.get("passed") else "FAIL"
        print(f"{verdict_word} {sid} [{info.get('version')}] rows={info.get('rows')} {checks}")
    for failure in result.failures:
        print(f"  - {failure}")
    print(f"metadata gate: {'PASS' if result.passed else 'FAIL'} -> {gates}")
    return 0 if result.passed else 1


def cmd_release_gate(args: argparse.Namespace) -> int:
    if args.patch_of:
        return cmd_release_gate_patch(args)
    if wants_v11(args, v10_selected=bool(args.require_all or args.metadata or _selected(args))):
        return cmd_release_gate_v11(args)
    metadata_rc = 0
    if args.metadata:
        metadata_rc = cmd_metadata_gate(args)
        if not _selected(args) and not args.require_all:
            return metadata_rc
    legacy = Path(args.masks) if isinstance(args.masks, str) else None
    if args.build_dir and legacy and args.build_dir != legacy:
        print("release gate: two different build directories given", file=sys.stderr)
        return 2
    build = args.build_dir or legacy
    names = tuple(ALL) if args.require_all else _selected(args)
    if build is None:
        print(
            "release gate: pass the build directory: marinedata release gate BUILD_DIR --all",
            file=sys.stderr,
        )
        return 2
    if not build.is_dir():
        print(f"release gate: not a directory: {build}", file=sys.stderr)
        return 1
    if not names and not args.require_all:
        print(
            "release gate: pick gates (--all, --licence, --dedup, --splits, --masks, ...)",
            file=sys.stderr,
        )
        return 2
    options = GateOptions(
        flavour=args.flavour,
        release_dir=args.release_dir,
        phash_bits=args.phash_bits,
        split_tolerance=args.split_tolerance,
        empty_threshold=args.empty_threshold,
        detector=args.detector,
        privacy_threshold=args.privacy_threshold,
        sample_rows=args.sample_rows,
        v1_dir=args.v1_dir,
    )
    failed = metadata_rc != 0
    for name in names:
        result = run_gate(name, build, options)
        try:
            path = write_gates(build, result)
        except (OSError, ValueError) as exc:
            print(f"release gate: cannot update GATES.json: {exc}", file=sys.stderr)
            return 1
        print(_line(result, path))
        for text in result.failures[:10]:
            print(f"  {text}", file=sys.stderr)
        failed = failed or gate_status(result.to_dict()) == "failed"
    if args.require_all:
        ok, problems = verdict(build, strict=True)
        for problem in problems:
            print(f"release gate --require-all: {problem}", file=sys.stderr)
        print(f"release gate --require-all: {'OK' if ok else 'REFUSED'} -> {build / GATES_FILE}")
        return 0 if ok and not failed else 1
    return 1 if failed else 0
