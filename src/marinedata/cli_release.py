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
from .gate import evaluate
from .manifest_identity import checksums_digest
from .neardup import NearDupConfig, NearDupError, default_workers, pil_version
from .registry import Registry
from .release import build_release, generate_split_map
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


def _resolve_roots(registry: Registry, profile: str, local: dict[str, Path]) -> dict[str, Path]:
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
    registry: Registry, profile: str, local: dict[str, Path], only: set[str] | None
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


def _cmd_release_build(args: argparse.Namespace) -> int:
    registry = Registry.load()
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
        if args.manifest_only:
            roots = _cached_roots(registry, args.profile, local, _sources_from(args.sources_from))
            digest = checksums_digest(registry, roots)
        else:
            roots = _resolve_roots(registry, args.profile, local)
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
        # --generate-split-map: enumerate the resolved staged trees directly and generate
        # a fresh, source-stratified map at the path this same command then freezes
        # against — one command, pinned/local trees straight to a release.
        try:
            ratios = _parse_ratios(args.ratios)
        except ValueError as exc:
            print(f"release build: {exc}", file=sys.stderr)
            return 1
        skipped_sources: dict[str, str] = {}
        try:
            stats = generate_split_map(
                registry,
                out=split_map_path,
                roots=roots,
                profile=args.profile,
                ratios=ratios,
                seed=args.seed,
                min_groups=args.min_groups,
                now=datetime.now(UTC).isoformat(),
                release=args.release,
                skipped=skipped_sources,
                near_dup=near_dup,
            )
        except NearDupError as exc:
            print(f"release build: {exc}", file=sys.stderr)
            return 1
        print(
            f"release build: generated {split_map_path} (stratify=source) "
            f"merged_components={stats.merged_components} "
            f"merged_cross_partition={stats.merged_cross_partition} "
            f"near_dup_pairs={stats.near_dup_pairs} near_dup_unions={stats.near_dup_unions} "
            f"near_dup_max_component={stats.near_dup_max_component}"
        )
        for source_id, reason in sorted(skipped_sources.items()):
            print(f"  skipped {source_id}: {reason}", file=sys.stderr)

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
            digest=digest,
        )
    except (NearDupError, DedupGateError, DeconError) as exc:
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
    p_build.add_argument(
        "--tasks",
        choices=("v1", "v2"),
        default="v1",
        help="'v1' (default) builds only the registry's task manifests, byte-identical "
        "to before this flag existed (D-X). 'v2' additionally builds the 5 WP-8c "
        "task-layer configs (points/vqa/semseg/benthic-coarse/benthic-cover) from "
        "data/_tasklabels/** into <out>/releases/<release>/task_layers/",
    )
    p_build.add_argument(
        "--v2",
        dest="v2",
        action="store_true",
        help="Preset (INT-core2): turns on --decon, --dedup-crop, --split-v2 and "
        "--tasks v2 together. Does not enable --dedup-v2 (needs an explicit "
        "groups.parquet path). Default OFF, so an unflagged build is unchanged (D-X)",
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
