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
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .checksums import file_digest
from .cli_splitmap import _parse_ratios
from .decon import DeconError
from .dedup.groups import DedupGateError
from .fetch import FetchError, cache_root, fetch_sample
from .fetchers_remote import _is_pinned_staged_tree
from .flavours import SPLIT_MAP_PROFILE, check_profile, flavour_spec, ships_in
from .gate import evaluate
from .manifest_identity import checksums_digest
from .neardup import NearDupConfig, NearDupError, default_workers, pil_version
from .registry import Registry
from .release import ReleaseSkipError, build_release, generate_split_map
from .splitmap import load_split_map
from .strata import DEFAULT_MIN_GROUPS

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


class ReleaseFetchError(Exception):
    """One or more admitted sources could not be fetched.

    Raised by :func:`_resolve_roots` instead of silently shrinking the release — a
    build must fail closed rather than quietly omit an admitted source (and any
    split-map groups or rows generated from it) because of a transient fetch error.
    """

    def __init__(self, failures: list[tuple[str, str]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{source_id}: {error}" for source_id, error in failures)
        super().__init__(f"failed to fetch {len(failures)} admitted source(s): {detail}")


def _resolve_roots(
    registry: Registry, profile: str, local: dict[str, Path], flavour: str | None = None
) -> dict[str, Path]:
    """Every admitted source's local root: a ``--local`` override first, else a fetched
    pinned staged tree. A fetch failure for an admitted source raises
    :class:`ReleaseFetchError` naming every failed source — the build must fail closed
    rather than silently produce a smaller release."""
    prof = registry.profile(profile)
    roots: dict[str, Path] = dict(local)
    failures: list[tuple[str, str]] = []
    for source in registry:
        if source.id in roots:
            continue
        if not evaluate(source, prof).allowed:
            continue
        if flavour is not None and not ships_in(registry, source.id, flavour):
            continue  # never fetch another flavour's bytes
        if not _is_pinned_staged_tree(source):
            continue
        try:
            result = fetch_sample(source, limit=_RELEASE_FETCH_LIMIT)
        except FetchError as exc:
            failures.append((source.id, str(exc)))
            continue
        roots[source.id] = result.root
    if failures:
        raise ReleaseFetchError(failures)
    return roots


def _cached_roots(
    registry: Registry,
    profile: str,
    local: dict[str, Path],
    only: set[str] | None,
    flavour: str | None = None,
) -> dict[str, Path]:
    """``--manifest-only`` roots: a ``--local`` override, else the source's already-staged
    tree under :func:`cache_root` — NEVER a fetch. ``only`` (from ``--sources-from``)
    restricts the admitted set to a published release's sources. Fails closed, naming
    every admitted source that is not staged locally."""
    prof = registry.profile(profile)
    roots = {k: v for k, v in local.items() if only is None or k in only}
    missing: list[str] = []
    for source in registry:
        if source.id in roots or (only is not None and source.id not in only):
            continue
        if not evaluate(source, prof).allowed or not _is_pinned_staged_tree(source):
            continue
        if flavour is not None and not ships_in(registry, source.id, flavour):
            continue
        cached = cache_root() / source.id
        if (cached / "metadata.parquet").is_file():
            roots[source.id] = cached
        else:
            missing.append(source.id)
    if missing:
        raise ValueError(
            "--manifest-only never fetches; not staged locally: "
            + ", ".join(sorted(missing))
            + " (pass --local or --sources-from)"
        )
    return roots


def _sources_from(path: str | None) -> set[str] | None:
    if path is None:
        return None
    return {entry["id"] for entry in json.loads(Path(path).read_text())["sources"]}


def _generate_global_split_map(
    registry: Registry,
    args: argparse.Namespace,
    local: dict[str, Path],
    split_map_path: Path,
    near_dup: NearDupConfig,
) -> int:
    """Generate the ONE frozen split map every flavour builds from (WP-R2d): enumerate the
    superset (open + nc sources under ``SPLIT_MAP_PROFILE``, never-released sources excluded),
    so an image or near-dup group has the same split in both repos."""
    try:
        ratios = _parse_ratios(args.ratios)
    except ValueError as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    skipped_sources: dict[str, str] = {}
    try:
        map_roots = (
            dict(local) if args.local_only else _resolve_roots(registry, SPLIT_MAP_PROFILE, local)
        )
        stats = generate_split_map(
            registry,
            out=split_map_path,
            roots=map_roots,
            profile=SPLIT_MAP_PROFILE,
            ratios=ratios,
            seed=args.seed,
            min_groups=args.min_groups,
            now=datetime.now(UTC).isoformat(),
            release=args.release,
            skipped=skipped_sources,
            near_dup=near_dup,
            upstream_test_off=args.no_honour_upstream_test or (),
        )
    except (NearDupError, ReleaseFetchError) as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    print(
        f"release: generated {split_map_path} (stratify=source) "
        f"merged_components={stats.merged_components} "
        f"merged_cross_partition={stats.merged_cross_partition} "
        f"near_dup_pairs={stats.near_dup_pairs} near_dup_unions={stats.near_dup_unions} "
        f"near_dup_max_component={stats.near_dup_max_component} "
        f"upstream_test_groups={stats.upstream_test_groups} "
        f"upstream_test_components={stats.upstream_test_components}"
    )
    for source_id, reason in sorted(skipped_sources.items()):
        print(f"  skipped {source_id}: {reason}", file=sys.stderr)
    return 0


def _cmd_release_split_map(args: argparse.Namespace) -> int:
    """``release split-map``: write the global split map once; both flavour builds read it."""
    registry = Registry.load()
    try:
        local = _parse_local(args.local)
    except ValueError as exc:
        print(f"release split-map: {exc}", file=sys.stderr)
        return 1
    split_map_path = Path(args.split_map)
    if load_split_map(split_map_path) is not None:
        print(
            f"release split-map: {split_map_path} already exists; never overwritten",
            file=sys.stderr,
        )
        return 1
    near_dup = NearDupConfig(cache_dir=cache_root() / "_dhash", workers=default_workers())
    return _generate_global_split_map(registry, args, local, split_map_path, near_dup)


def _cmd_release_build(args: argparse.Namespace) -> int:
    registry = Registry.load()
    flavour = getattr(args, "flavour", None)
    if flavour is None:
        print(
            "release build: --flavour open|nc is required (open = ungated repo, "
            "nc = gated non-commercial delta)",
            file=sys.stderr,
        )
        return 1
    if args.profile is None:
        args.profile = flavour_spec(flavour).profile
    try:
        check_profile(flavour, args.profile)
    except ValueError as exc:
        print(f"release build: {exc}", file=sys.stderr)
        return 1
    try:
        local = _parse_local(args.local)
    except ValueError as exc:
        print(f"release build: {exc}", file=sys.stderr)
        return 1
    digest = file_digest
    if args.sources_from and not args.manifest_only:
        print("release build: --sources-from needs --manifest-only", file=sys.stderr)
        return 1
    if args.manifest_only and args.generate_split_map:
        print("release build: --manifest-only needs a frozen --split-map", file=sys.stderr)
        return 1
    try:
        if args.local_only:
            roots = dict(local)  # offline / mini build: exactly the --local trees, never a fetch
        elif args.manifest_only:
            roots = _cached_roots(
                registry, args.profile, local, _sources_from(args.sources_from), flavour
            )
            digest = checksums_digest(registry, roots)
        else:
            roots = _resolve_roots(registry, args.profile, local, flavour)
    except ValueError as exc:
        print(f"release build: {exc}", file=sys.stderr)
        return 1
    except ReleaseFetchError as exc:
        for source_id, error in exc.failures:
            print(
                f"release build: fetch failed for admitted source {source_id}: {error}",
                file=sys.stderr,
            )
        return 1

    near_dup = NearDupConfig(cache_dir=cache_root() / "_dhash", workers=default_workers())
    split_map_path = Path(args.split_map)
    split_map_exists = load_split_map(split_map_path) is not None

    if split_map_exists and args.generate_split_map:
        print(
            f"release build: {split_map_path} already exists; --generate-split-map "
            "never overwrites a frozen map — use 'splitmap generate' to append",
            file=sys.stderr,
        )
        return 1

    if not split_map_exists and not args.generate_split_map:
        print(
            f"release build: {split_map_path} does not exist; pass --generate-split-map "
            "to create it, or point --split-map at an existing frozen map",
            file=sys.stderr,
        )
        return 1

    if not split_map_exists:
        rc = _generate_global_split_map(registry, args, local, split_map_path, near_dup)
        if rc:
            return rc

    # Fail closed on a Pillow mismatch (WS-D S49): dHash's LANCZOS resize is a Pillow
    # implementation detail, so a map's rule-A/rule-B near-dup exclusions are only valid
    # under the Pillow version that computed them. `--generate-split-map` just recorded
    # the running version above, so this only ever fires against a pre-existing, frozen
    # map built under a different Pillow.
    current_map = load_split_map(split_map_path)
    if current_map is not None and current_map.near_dup:
        saved_pillow = current_map.near_dup.get("pil_version")
        running_pillow = pil_version()
        if saved_pillow is not None and saved_pillow != running_pillow:
            print(
                f"release build: {split_map_path} was generated under Pillow "
                f"{saved_pillow}, but this environment has Pillow {running_pillow} — "
                "near-dup dHashes are not comparable across Pillow versions "
                "(LANCZOS resize output is a Pillow implementation detail); rule A "
                "exclusions would be silently wrong. Rebuild the split map with "
                "--generate-split-map under this Pillow, or run under Pillow "
                f"{saved_pillow}",
                file=sys.stderr,
            )
            return 1

    try:
        result = build_release(
            registry,
            release=args.release,
            split_map=args.split_map,
            roots=roots,
            out_dir=args.out,
            profile=args.profile,
            near_dup=near_dup,
            dedup_v2_groups=args.dedup_v2,
            decon=args.decon,
            dedup_crop=args.dedup_crop,
            split_v2=args.split_v2,
            tasks=args.tasks,
            v2=args.v2,
            tasklabels_root=args.tasklabels_root,
            digest=digest,
            flavour=flavour,
            allow_skip=args.allow_skip or (),
        )
    except (NearDupError, DedupGateError, DeconError, ReleaseSkipError) as exc:
        print(f"release build: {exc}", file=sys.stderr)
        return 1

    print(
        f"release build: release={result.release} sources={len(result.sources)} "
        f"tasks={len(result.tasks)} skipped={len(result.skipped_tasks)} "
        f"never_eval_excluded={result.never_eval_excluded} "
        f"never_eval_near_dup_excluded={len(result.never_eval_near_dup_excluded)} "
        f"(rows {result.never_eval_near_dup_rows}) -> {result.out_dir}"
    )
    for task in result.tasks:
        print(f"  {task.task_id}: {len(task.rows)} images")
    for task_id, reason in result.skipped_tasks:
        print(f"  skipped {task_id}: {reason}", file=sys.stderr)
    for config_id, n_images in sorted(result.task_layer_configs.items()):
        print(f"  task_layer {config_id}: {n_images} images")
    for exclusion in result.partial_abstain_excluded:
        labels = ", ".join(exclusion.abstaining_labels)
        print(
            f"  partial-abstain: {exclusion.source_id} excluded from "
            f"{exclusion.task_id} ({labels}; {exclusion.rows_dropped} rows dropped)",
            file=sys.stderr,
        )
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
        "--dedup-v2",
        dest="dedup_v2",
        type=Path,
        default=None,
        metavar="GROUPS_PARQUET",
        help="Run the WP-10 split-leak gate against this groups.parquet ('marinedata dedup "
        "run'); fails the build on leakage. Default OFF (v1 behaviour, byte-identical)",
    )
    p_build.add_argument(
        "--decon",
        action="store_true",
        help="Run the WP-12 benchmark decontamination gate ('marinedata decon check') "
        "against admitted sources; fails the build on any hit. Default OFF (v2 build)",
    )
    p_build.add_argument(
        "--dedup-crop",
        dest="dedup_crop",
        action="store_true",
        help="WP-10c: enable decon's S5 patch/crop stage (only takes effect with "
        "--decon). Default OFF (code default stays off, D-X)",
    )
    p_build.add_argument(
        "--split-v2",
        dest="split_v2",
        action="store_true",
        help="WP-11/12 P3: validate the split-v2 config (registry/splits/v2.yaml plus "
        "registry/benchmarks.yaml) and record its hashes in RELEASE.json. The full "
        "per-sample gate is deferred to the v2 build. Default OFF (D-X)",
    )
    p_build.add_argument(
        "--split-map",
        dest="split_map",
        required=True,
        help="Path to SPLIT_MAP.json — frozen if it exists. If it does not exist yet, "
        "pass --generate-split-map to create it here first (source-stratified) from the "
        "resolved staged trees; without the flag this refuses to run",
    )
    p_build.add_argument(
        "--generate-split-map",
        dest="generate_split_map",
        action="store_true",
        help="Required to create --split-map when it does not exist yet; refused if the "
        "path already exists (never overwrites a frozen map — use 'splitmap generate' "
        "to append)",
    )
    p_build.add_argument(
        "--ratios",
        default="70/15/15",
        help="'/'-separated, only used with --generate-split-map",
    )
    p_build.add_argument("--seed", type=int, default=0, help="Only used with --generate-split-map")
    p_build.add_argument(
        "--min-groups",
        type=int,
        default=DEFAULT_MIN_GROUPS,
        help="A stratum with fewer groups than this is train-only (default 3); only used "
        "with --generate-split-map",
    )
    p_build.add_argument(
        "--no-honour-upstream-test",
        action="append",
        dest="no_honour_upstream_test",
        metavar="SOURCE_ID",
        help="Opt a source out of 'an upstream-test group goes to our test split' (default on "
        "for every source with upstream test rows; '*' = all sources, repeatable); only used "
        "with --generate-split-map",
    )
    p_build.add_argument("--out", default=".", help="Output root (default: current directory)")
    p_build.add_argument(
        "--flavour",
        choices=("open", "nc"),
        default=None,
        help="REQUIRED. open = class-open sources (ungated repo reefsupport/marine-data); nc = the "
        "restricted-nc delta (gated repo reefsupport/marine-data-nc). Writes "
        "<out>/releases/<release>/<flavour>/",
    )
    p_build.add_argument(
        "--profile",
        default=None,
        help="Shipping profile to admit sources under (default: the flavour's own, "
        "registry/flavours.yaml)",
    )
    p_build.add_argument(
        "--local",
        action="append",
        dest="local",
        metavar="SOURCE_ID=PATH",
        help="Use a local staged tree instead of fetching one; repeatable",
    )
    p_build.add_argument(
        "--local-only",
        action="store_true",
        dest="local_only",
        help="Build from exactly the --local trees: never fetch another admitted source "
        "(mini / offline builds, incl. the split-map generation)",
    )
    p_build.add_argument(
        "--allow-skip",
        action="append",
        dest="allow_skip",
        metavar="SOURCE_ID",
        help="Leave this releasable source out of a flavour build on purpose (repeatable). "
        "Without it (or a registry release_skip_reason) a source with rows that the split map "
        "cannot cover fails the build; RELEASE.json keeps it under skipped_sources",
    )
    p_build.add_argument(
        "--tasks",
        choices=("v1", "v2"),
        default="v1",
        help="'v1' (default) builds only the registry's task manifests, byte-identical "
        "to before this flag existed (D-X). 'v2' additionally builds the 5 WP-8c "
        "task-layer configs (points/vqa/semseg/benthic-coarse/benthic-cover) from "
        "<--tasklabels-root>/_tasklabels/** into <out>/releases/<release>/task_layers/",
    )
    p_build.add_argument(
        "--tasklabels-root",
        dest="tasklabels_root",
        default=None,
        metavar="DIR",
        help="Directory holding _tasklabels/ (and _labelquality/) for --tasks v2; a path "
        "ending in _tasklabels means its parent. Default: <repo>/data, derived from the "
        "registry's location, never the cwd",
    )
    p_build.add_argument(
        "--v2",
        dest="v2",
        action="store_true",
        help="Preset (INT-core2): turns on --decon, --dedup-crop, --split-v2 and "
        "--tasks v2 together. Does not enable --dedup-v2 (needs an explicit "
        "groups.parquet path) and never runs captions (a separate `marinedata "
        "captions` step, WP-13). Default OFF, so an unflagged build is unchanged (D-X)",
    )
    p_build.add_argument(
        "--manifest-only",
        dest="manifest_only",
        action="store_true",
        help="D-X2: never fetch — roots come from already-staged trees in the local "
        "cache only, and every image sha256 comes from the source's pinned "
        "CHECKSUMS.sha256 (no image bytes read). Writes the same task TSVs / "
        "RELEASE.json a full build would",
    )
    p_build.add_argument(
        "--sources-from",
        dest="sources_from",
        metavar="RELEASE_JSON",
        help="With --manifest-only: admit only the sources a published RELEASE.json "
        "lists (e.g. to rebuild v1's task files under the current code)",
    )
    p_build.set_defaults(func=_cmd_release_build)

    p_map = release_sub.add_parser(
        "split-map",
        help="Generate the one global SPLIT_MAP.json (open + nc sources) both flavours build from",
    )
    p_map.add_argument("--release", required=True, help="Release id recorded in the map")
    p_map.add_argument("--split-map", dest="split_map", required=True, help="Map to create")
    p_map.add_argument("--ratios", default="70/15/15", help="'/'-separated train/val/test ratios")
    p_map.add_argument("--seed", type=int, default=0)
    p_map.add_argument("--min-groups", dest="min_groups", type=int, default=DEFAULT_MIN_GROUPS)
    p_map.add_argument(
        "--no-honour-upstream-test",
        dest="no_honour_upstream_test",
        action="append",
        metavar="SOURCE_ID",
        help="Opt a source out of the upstream-test rule (repeatable, '*' = all)",
    )
    p_map.add_argument("--local", action="append", metavar="ID=PATH", help="Staged tree override")
    p_map.add_argument(
        "--local-only", dest="local_only", action="store_true", help="Exactly the --local trees"
    )
    p_map.set_defaults(func=_cmd_release_split_map)
