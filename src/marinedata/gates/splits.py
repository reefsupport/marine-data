# ruff: noqa: E501
"""Splits gate (P1.6): groups stay in one split, ratios hit the target, no near-dup leaks to train.

* ``group_straddles_splits`` one ``split_group`` appears in two splits (any config);
* ``ratio_off_target``       a config whose train/validation/test share is more than 5 points
  (``--split-tolerance``) from its target. Target, first match: the registry's per-config
  entry, the ``split_ratio.target_pct`` the build recorded in RELEASE.json (the ratios of the
  frozen split map it used), the registry default (80/10/10). Configs without all three splits
  are only checked when the registry names them. The share is taken over the rows of the
  NON-pinned sources: a source the split map pins to its upstream test manifest
  (``pins.benchmarks`` of ``SPLIT_MAP.json``, in ``--release-dir`` else the build dir) has a
  structural share, so it is reported (``ratio_pinned_sources``) and never fails;
* ``near_dup_leak``          a validation/test image within ``--phash-bits`` (pHash) of a train image;
* ``no_split_group``         no config carries ``split_group`` (a vacuous check fails).

The pHash table is ``<build>/phash.parquet`` (reused when it covers the build, else computed)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..licence_class import _registry_root
from . import GateResult, finish
from .dedup import DEFAULT_PHASH_BITS
from .phash import load_or_scan, near_pairs
from .shards import (
    config_shards,
    find_split_map,
    iter_rows,
    norm_split,
    read_release,
    shard_columns,
    source_split_counts,
    split_counts,
)

NAME = "splits"
TOLERANCE_PT = 5.0
DEFAULT_TARGET = {"train": 80.0, "validation": 10.0, "test": 10.0}
TARGETS_FILE = "split_ratio.yaml"
MAIN_SPLITS = ("train", "validation", "test")


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
    pins nothing, so the ratio is taken over every row as before; the second value is ``None``
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


def _without(have: dict[str, int], held: dict[str, dict[str, int]]) -> dict[str, int]:
    """``have`` (rows per split) less the rows of the held (pinned) sources; a split that the
    exclusion empties is dropped, so the table then lacks it like a table that never had it."""
    if not held:
        return dict(have)
    left = {s: n - sum(c.get(s, 0) for c in held.values()) for s, n in have.items()}
    return {s: n for s, n in left.items() if n or not have[s]}


def _basis(held: dict[str, dict[str, int]]) -> str:
    """What a ratio finding counted: every row, or the non-pinned ones (and which were left out)."""
    if not held:
        return "rows"
    rows = sum(sum(c.values()) for c in held.values())
    names = sorted(held)
    shown = ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
    return f"non-pinned rows (excl. {rows} rows of pinned {shown})"


def run_splits_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    phash_bits: int = DEFAULT_PHASH_BITS,
    tolerance: float = TOLERANCE_PT,
    registry_root: str | Path | None = None,
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
    pinned_rows: dict[str, dict] = {}
    free_ratio: dict[str, dict] = {}
    for config, files in shards.items():
        by_source = source_split_counts(files) if pinned else None
        if pinned and by_source is None:
            counts["ratio_unsourced_configs"] += 1  # no source column: every row stays counted
        held = {s: c for s, c in (by_source or {}).items() if s in pinned}
        if held:
            pinned_rows[config] = {
                s: {**dict.fromkeys(MAIN_SPLITS, 0), **c} for s, c in held.items()
            }
        have = _without(split_counts(files), held)
        if config in per_config:
            target, used[config] = per_config[config], "registry:config"
        elif recorded:
            target, used[config] = _target(recorded), "release:split_ratio"
        else:
            target, used[config] = default, "registry:default"
        if config not in per_config and not set(MAIN_SPLITS) <= set(have):
            counts["ratio_skipped_configs"] += 1
            continue
        total = sum(have.values())
        if held and not total:  # every row belongs to a pinned source: nothing to hold to a target
            counts["ratio_all_pinned_configs"] += 1
            continue
        free_ratio[config] = {
            "rows": total,
            **{s: round(100.0 * have.get(s, 0) / total, 1) for s in sorted({*have, *target})},
        }
        for split, want in sorted(target.items()):
            got = 100.0 * have.get(split, 0) / total
            if abs(got - want) > tolerance:
                fail(
                    "ratio_off_target",
                    f"{config}/{split}",
                    f"{got:.1f}% of {total} {_basis(held)}, target {want:g}% +/- {tolerance:g} ({used[config]})",
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
        "ratio_pins": {"map": pins_from, "sources": sorted(pinned)},
        "ratio_pinned_sources": pinned_rows,
        "ratio_excluding_pinned": free_ratio,
    }
    return finish(NAME, counts, sorted(failures))
