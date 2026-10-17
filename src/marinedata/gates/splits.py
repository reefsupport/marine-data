# ruff: noqa: E501
"""Splits gate (P1.6): groups stay in one split, ratios hit the target, no near-dup leaks to train.

* ``group_straddles_splits`` one ``split_group`` appears in two splits (any config);
* ``ratio_off_target``       a STRATUM whose share of distinct images per split (``images`` config, by
  ``image_sha256``) is more than 5 points (``--split-tolerance``) from its target. Why this
  statistic: the split map allocates image GROUPS stratified by source and co-assigns near-duplicate
  groups across a fork family (D-FORK), so the finest statistic it controls is a stratum's image
  split; annotation rows and per-config subsets are projections it cannot steer, so
  ``counts.ratio_by_config`` reports them (rows and distinct images per config) and never fails.
  A stratum is the registry family of a source (``licence_class.source_families()``), else the
  source id. Target, first match: the registry's entry for ``images``, the
  ``split_ratio.target_pct`` the build recorded in RELEASE.json (the ratios of the frozen split map
  it used), the registry default. ``images`` without all three splits is only checked when the
  registry names it. Left out, and never failing: a source the split map pins to its upstream test
  manifest (``pins.benchmarks`` of ``SPLIT_MAP.json``, in ``--release-dir`` else the build dir;
  ``ratio_pinned_sources``), an image with an ``upstream_split`` label (the upstream partition
  decided its split) and a stratum under ``ratio_min_images`` (100) images
  (``ratio_small_strata``);
* ``near_dup_leak``          a validation/test image within ``--phash-bits`` (pHash) of a train image;
* ``no_split_group``         no config carries ``split_group`` (a vacuous check fails).

The pHash table is ``<build>/phash.parquet`` (reused when it covers the build, else computed)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from ..licence_class import _registry_root
from . import GateResult, finish
from .dedup import DEFAULT_PHASH_BITS
from .phash import load_or_scan, near_pairs
from .shards import (
    SOURCE_COLUMNS,
    config_shards,
    find_split_map,
    iter_rows,
    norm_split,
    read_release,
    shard_columns,
    shard_split,
    source_split_counts,
    split_counts,
)
from .v11_common import LABEL_TABLES, table_rows

NAME = "splits"
TOLERANCE_PT = 5.0
DEFAULT_TARGET = {"train": 80.0, "validation": 10.0, "test": 10.0}
TARGETS_FILE = "split_ratio.yaml"
MAIN_SPLITS = ("train", "validation", "test")
IMAGES = "images"
RATIO_MIN_IMAGES = (
    100  # under it one image moves a share by a point or more: reported, never failed
)


def _target(raw: dict) -> dict[str, float]:
    return {norm_split(str(k)): float(v) for k, v in raw.items()}


def load_split_targets(root: str | Path | None = None) -> tuple[dict, dict]:
    """``(default, {config: target})`` from ``registry/split_ratio.yaml`` (percent per split)."""
    path = _registry_root(root) / TARGETS_FILE
    if not path.is_file():
        return dict(DEFAULT_TARGET), {}
    data = yaml.safe_load(path.read_text()) or {}
    default = _target(data.get("default") or DEFAULT_TARGET)
    return default, {str(c): _target(t) for c, t in (data.get("configs") or {}).items()}


def read_pinned_sources(
    build_dir: Path, release_dir: Path | None = None
) -> tuple[frozenset[str], str | None]:
    """``(pinned source ids, where the map was found)`` from ``SPLIT_MAP.json``.

    The ids are the keys of ``pins.benchmarks``: a benchmark label (the registry id, else the
    benchmark id) is the ``source_id`` of its rows. No map, an unreadable one or one without pins
    pins nothing, so no source is left out of the ratio; the second value is ``None``
    for no map, ``"unreadable"`` for one that does not parse, else ``"release_dir"``/``"build_dir"``."""
    found = find_split_map(build_dir, release_dir)
    if found is None:
        return frozenset(), None
    path, where = found
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return frozenset(), "unreadable"
    pins = data.get("pins") if isinstance(data, dict) else None
    block = pins.get("benchmarks") if isinstance(pins, dict) else None
    return (frozenset(str(k) for k in block) if isinstance(block, dict) else frozenset()), where


def _shares(sizes: Mapping[str, int], splits: Iterable[str] = ()) -> dict:
    """``{"n": total, split: percent}`` of per-split sizes, plus the named splits that are empty."""
    total = sum(sizes.values())
    return {
        "n": total,
        **{s: round(100.0 * sizes.get(s, 0) / max(total, 1), 1) for s in sorted({*sizes, *splits})},
    }


def _image_counts(files: Sequence[Path]) -> dict[str, int] | None:
    """Distinct ``image_sha256`` per split of one config; only that column is read. ``None`` when
    a shard lacks it."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    seen: dict[str, set[str]] = defaultdict(set)
    for path in files:
        shard = pq.ParquetFile(path)
        if "image_sha256" not in shard.schema_arrow.names:
            return None
        for batch in shard.iter_batches(batch_size=65536, columns=["image_sha256"]):
            seen[shard_split(path)].update(pc.unique(batch.column(0)).to_pylist())
    return {split: len(ids) for split, ids in sorted(seen.items())}


def _upstream_images(shards: Mapping[str, Sequence[Path]]) -> set[str]:
    """``image_sha256`` of every image with a non-empty ``upstream_split`` in a label table, as
    ``splits_v11`` reads them: the upstream partition decided its split, not the split map."""
    return {
        row.get("image_sha256")
        for table in LABEL_TABLES
        for row in table_rows(shards.get(table, []), ("image_sha256", "upstream_split"))
        if row.get("upstream_split")
    }


def _strata(
    files: Sequence[Path], pinned: frozenset[str], upstream: set[str], families: Mapping[str, str]
) -> tuple[dict[str, dict[str, set[str]]], dict[str, int]]:
    """``({stratum: {split: {image_sha256}}}, {why: images left out})`` over the ``images`` rows.
    A stratum is the source's registry family, else the source id."""
    strata: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    left: dict[str, set[str]] = defaultdict(set)
    for _, _, split, row in iter_rows(files, ("image_sha256", *SOURCE_COLUMNS), batch_size=4096):
        source = str(row.get("source_id") or row.get("source") or "?")
        sha = row.get("image_sha256")
        if source in pinned:
            left["pinned"].add(sha)
        elif sha in upstream:
            left["upstream_split"].add(sha)
        else:
            strata[families.get(source, source)][split].add(sha)
    return strata, {why: len(ids) for why, ids in sorted(left.items())}


def run_splits_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    phash_bits: int = DEFAULT_PHASH_BITS,
    tolerance: float = TOLERANCE_PT,
    registry_root: str | Path | None = None,
    families: Mapping[str, str] | None = None,
    ratio_min_images: int = RATIO_MIN_IMAGES,
) -> GateResult:
    build_dir = Path(build_dir)
    shards = config_shards(build_dir)
    if not shards:
        return finish(NAME, {}, [f"no_data: no data/<config>/*.parquet under {build_dir}"])
    release = read_release(build_dir, release_dir)
    default, per_config = load_split_targets(registry_root)
    recorded = ((release or {}).get("split_ratio") or {}).get("target_pct")
    counts: Counter = Counter()
    failures: list[str] = []

    def fail(kind: str, where: str, message: str) -> None:
        counts[f"failed_{kind}"] += 1
        failures.append(f"{where} {kind}: {message}")

    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for config, files in shards.items():
        if "split_group" not in shard_columns(files[0]):
            continue
        counts["group_configs"] += 1
        for _, _, split, row in iter_rows(files, ("split_group",)):
            group = row.get("split_group")
            if group:
                groups[str(group)].setdefault(split, config)
            else:
                counts["rows_without_group"] += 1
    if not counts["group_configs"]:
        fail(
            "no_split_group", "build", "no config carries split_group; group isolation is unchecked"
        )
    counts["split_groups"] = len(groups)
    for group, seen in sorted(groups.items()):
        if len(seen) > 1:
            where = ", ".join(f"{s} ({c})" for s, c in sorted(seen.items()))
            fail("group_straddles_splits", group, f"in {where}")

    pinned, pins_from = read_pinned_sources(build_dir, release_dir)
    used: dict[str, str] = {}
    target_of: dict[str, dict[str, float]] = {}
    pinned_rows: dict[str, dict] = {}
    by_config: dict[str, dict] = {}
    for config, files in shards.items():
        by_source = source_split_counts(files) if pinned else None
        if pinned and by_source is None:
            counts["ratio_unsourced_configs"] += 1  # no source column: pinned rows not listed
        held = {s: c for s, c in (by_source or {}).items() if s in pinned}
        if held:
            pinned_rows[config] = {
                s: {**dict.fromkeys(MAIN_SPLITS, 0), **c} for s, c in held.items()
            }
        if config in per_config:
            target_of[config], used[config] = per_config[config], "registry:config"
        elif recorded:
            target_of[config], used[config] = _target(recorded), "release:split_ratio"
        else:
            target_of[config], used[config] = default, "registry:default"
        rows, images = _shares(split_counts(files)), _image_counts(files)
        by_config[config] = {"rows": rows, **({"images": _shares(images)} if images else {})}
    present = {s for per in pinned_rows.values() for s in per}
    checked: dict[str, dict] = {}
    small: dict[str, int] = {}
    left_out: dict[str, int] = {}
    if IMAGES not in shards:
        counts["ratio_check"] = f"skipped: no {IMAGES} config"
    elif IMAGES not in per_config and not set(MAIN_SPLITS) <= set(split_counts(shards[IMAGES])):
        counts["ratio_skipped_configs"] += 1
    else:
        strata, left_out = _strata(shards[IMAGES], pinned, _upstream_images(shards), families or {})
        if not strata:  # every image is pinned or upstream-partitioned: nothing to hold to a target
            counts["ratio_all_excluded_configs"] += 1
        for stratum, by_split in sorted(strata.items()):
            sizes = {s: len(ids) for s, ids in by_split.items()}
            total = sum(sizes.values())
            if total < ratio_min_images:
                small[stratum] = total
                continue
            checked[stratum] = _shares(sizes, target_of[IMAGES])
            for split, want in sorted(target_of[IMAGES].items()):
                got = 100.0 * sizes.get(split, 0) / total
                if abs(got - want) > tolerance:
                    fail(
                        "ratio_off_target",
                        f"{IMAGES}/{split}",
                        f"{got:.1f}% of {total} images of {stratum}, target {want:g}% +/- {tolerance:g} ({used[IMAGES]})",
                    )
    records = [r for r in load_or_scan(build_dir) if r.phash is not None]
    if records:
        counts["near_dup_images"] = len(records)
        for i, j, dist in near_pairs([r.phash for r in records], phash_bits):
            a, b = records[i], records[j]
            if "train" in (a.split, b.split) and a.split != b.split and a.config == b.config:
                held, train = (b, a) if a.split == "train" else (a, b)
                fail(
                    "near_dup_leak",
                    f"{held.config}/{held.split} {held.image_sha256[:12]}",
                    f"hamming {dist} <= {phash_bits} to train image {train.image_sha256[:12]}",
                )
    else:
        counts["near_dup_check"] = "skipped: no image bytes"
    counts = {
        **counts,
        "phash_bits": phash_bits,
        "tolerance_pt": tolerance,
        "targets": used,
        "ratio_min_images": ratio_min_images,
        "ratio_strata": checked,
        "ratio_small_strata": small,
        "ratio_left_out_images": left_out,
        "ratio_by_config": by_config,
        "ratio_pins": {
            "map": pins_from,
            "sources": sorted(present),
            "absent": len(pinned - present),
        },
        "ratio_pinned_sources": pinned_rows,
    }
    return finish(NAME, counts, sorted(failures))
