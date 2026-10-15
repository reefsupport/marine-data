# ruff: noqa: E501
"""Splits gate (P1.6): groups stay in one split, ratios hit the target, no near-dup leaks to train.

* ``group_straddles_splits`` one ``split_group`` appears in two splits (any config);
* ``ratio_off_target``       a config whose train/validation/test share is more than 5 points
  (``--split-tolerance``) from its target. Target, first match: the registry's per-config
  entry, the ``split_ratio.target_pct`` the build recorded in RELEASE.json (the ratios of the
  frozen split map it used), the registry default (80/10/10). Configs without all three splits
  are only checked when the registry names them;
* ``near_dup_leak``          a validation/test image within ``--phash-bits`` (pHash) of a train image;
* ``no_split_group``         no config carries ``split_group`` (a vacuous check fails).

The pHash table is ``<build>/phash.parquet`` (reused when it covers the build, else computed)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..licence_class import _registry_root
from . import GateResult, finish
from .dedup import DEFAULT_PHASH_BITS
from .phash import load_or_scan, near_pairs
from .shards import config_shards, iter_rows, norm_split, read_release, shard_columns, split_counts

NAME = "splits"
TOLERANCE_PT = 5.0
DEFAULT_TARGET = {"train": 80.0, "validation": 10.0, "test": 10.0}
TARGETS_FILE = "split_ratio.yaml"


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

    used: dict[str, str] = {}
    for config, files in shards.items():
        have = split_counts(files)
        if config in per_config:
            target, used[config] = per_config[config], "registry:config"
        elif recorded:
            target, used[config] = _target(recorded), "release:split_ratio"
        else:
            target, used[config] = default, "registry:default"
        if config not in per_config and not {"train", "validation", "test"} <= set(have):
            counts["ratio_skipped_configs"] += 1
            continue
        total = sum(have.values())
        for split, want in sorted(target.items()):
            got = 100.0 * have.get(split, 0) / total
            if abs(got - want) > tolerance:
                fail(
                    "ratio_off_target",
                    f"{config}/{split}",
                    f"{got:.1f}% of {total} rows, target {want:g}% +/- {tolerance:g} ({used[config]})",
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
    counts = {**counts, "phash_bits": phash_bits, "tolerance_pt": tolerance, "targets": used}
    return finish(NAME, counts, sorted(failures))
