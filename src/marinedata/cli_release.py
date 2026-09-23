"""``marinedata release`` sub-command — the release-build CLI (WS-D S7k).

Split out of :mod:`marinedata.cli` for the same reason as :mod:`marinedata.cli_splitmap`.
The actual build lives in :mod:`marinedata.release`, which never touches the network;
this module's only job is resolving ``roots`` first — fetch a pinned staged tree per
admitted source (never listing — see ``fetchers_remote._is_pinned_staged_tree``), or
accept a ``--local <source_id>=<path>`` override for a tree that has not been uploaded to
the open bucket yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fetch import FetchError, fetch_sample
from .fetchers_remote import _is_pinned_staged_tree
from .gate import evaluate
from .registry import Registry
from .release import build_release

_RELEASE_FETCH_LIMIT = 10_000_000
"""Effectively "the whole pinned tree" — ``fetch_sample``'s ``limit`` exists to bound a
verification sample; a release build wants everything the manifest names."""


def _parse_local(specs: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for spec in specs or []:
        source_id, sep, path = spec.partition("=")
        if not sep or not path:
            raise ValueError(f"--local must be '<source_id>=<path>', got {spec!r}")
        overrides[source_id] = Path(path)
    return overrides


def _resolve_roots(registry: Registry, profile: str, local: dict[str, Path]) -> dict[str, Path]:
    """Every admitted source's local root: a ``--local`` override first, else a fetched
    pinned staged tree. A source with neither is silently absent from ``roots`` —
    :func:`marinedata.release.build_release` treats a task with no admitted source as
    skipped, not fatal, so a partial fetch still produces a partial release."""
    prof = registry.profile(profile)
    roots: dict[str, Path] = dict(local)
    for source in registry:
        if source.id in roots:
            continue
        if not evaluate(source, prof).allowed:
            continue
        if not _is_pinned_staged_tree(source):
            continue
        try:
            result = fetch_sample(source, limit=_RELEASE_FETCH_LIMIT)
        except FetchError as exc:
            print(f"release build: skipping {source.id}: {exc}", file=sys.stderr)
            continue
        roots[source.id] = result.root
    return roots


def _cmd_release_build(args: argparse.Namespace) -> int:
    registry = Registry.load()
    try:
        local = _parse_local(args.local)
    except ValueError as exc:
        print(f"release build: {exc}", file=sys.stderr)
        return 1
    roots = _resolve_roots(registry, args.profile, local)

    result = build_release(
        registry,
        release=args.release,
        split_map=args.split_map,
        roots=roots,
        out_dir=args.out,
        profile=args.profile,
    )

    print(
        f"release build: release={result.release} sources={len(result.sources)} "
        f"tasks={len(result.tasks)} skipped={len(result.skipped_tasks)} -> {result.out_dir}"
    )
    for task in result.tasks:
        print(f"  {task.task_id}: {len(task.rows)} images")
    for task_id, reason in result.skipped_tasks:
        print(f"  skipped {task_id}: {reason}", file=sys.stderr)
    return 0


def add_release_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``release`` sub-command onto ``sub`` (called from ``cli.build_parser``)."""
    p_release = sub.add_parser("release", help="Build a release from admitted sources")
    release_sub = p_release.add_subparsers(dest="release_command", required=True)

    p_build = release_sub.add_parser(
        "build", help="Build every registry task against a frozen SPLIT_MAP.json"
    )
    p_build.add_argument("--release", required=True, help="Release id")
    p_build.add_argument(
        "--split-map", dest="split_map", required=True, help="Path to the frozen SPLIT_MAP.json"
    )
    p_build.add_argument("--out", default=".", help="Output root (default: current directory)")
    p_build.add_argument(
        "--profile", default="research", help="Release profile to admit sources under"
    )
    p_build.add_argument(
        "--local",
        action="append",
        dest="local",
        metavar="SOURCE_ID=PATH",
        help="Use a local staged tree instead of fetching one; repeatable",
    )
    p_build.set_defaults(func=_cmd_release_build)
