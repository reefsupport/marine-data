"""G6 splits (v1.1.0): the split columns keep the frozen v1.0 assignment and the split policy.

* ``unsplit``          a metadata row whose ``split`` is null or not train / validation / test /
  test_ood, or ``task_layers/V11_TABLES.json`` records ``unsplit_images > 0`` / ``allow_unsplit``
  (a ``--allow-unsplit`` build). With ``rehearsal`` a build whose only problem is that is
  ``skipped``, never ``passed``: ``release publish`` still refuses it. A missing report fails.
* ``metadata_missing`` ``V11_TABLES.json`` records an admitted source with staged images or task
  rows and no metadata row (PR-M: a narrow tree once vanished silently from ``metadata``);
* ``split_shard``      a row's ``split`` is not the split of its shard file;
* ``group_straddles``  one ``split_group`` in two splits (near-copies must not straddle);
* ``reassigned_v1``    an image of a v1.0 config in a split v1.0 did not give it (``v1_dir``). The
  one exception is a benchmark pin (D-STRADDLE: a pin wins over v1.0 continuity). When the
  release's ``SPLIT_MAP.json`` records ``pins.v1_continuity_overridden``, an image may sit in the
  split the map assigns its group if the record covers the group (listed in ``group_ids`` when it
  lists them; above 200 groups it only counts them). Such a group counts in
  ``pin_override_applied``, per source in ``pin_override_applied_<source>``; the groups of a source
  outside ``pins.benchmarks`` are component-mates (a pin moves its whole near-duplicate component)
  and also count in ``pin_override_mates``. Any other move fails, with the reason;
* ``override_unchanged``  a pin that was not applied: a listed group none of whose images moved
  while the map assigns it to another split, or (record that only counts groups) a pinned-source
  group with that fault, or more groups moved than the record counts. A group the pin recorded
  whose images already sit in the map's split is a no-op (``pin_override_noop``) and passes;
* ``ood_size`` / ``ood_share``  ``test_ood`` under the registry size guards (min images / groups) or
  over ``max_total_holdout_share`` of the release (registry/splits/v2.yaml ``size_guards``);
* ``ratio_off``        images no label row gives an ``upstream_split`` are not 70/15/15 within
  +/-5 points (checked from ``ratio_min_images`` images up).

The dHash Hamming bands (union <= 4, never-eval <= 8) are enforced when the split map is built
(``release split-map --inherit``); the build carries no hash, so this gate checks their outcome
(group and split columns), not the distances."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import GateResult, skip
from .v11_common import (
    LABEL_TABLES,
    MAIN_SPLITS,
    METADATA,
    Collector,
    shards,
    table_rows,
    v1_splits,
)

NAME = "splits_v11"
SPLITS = frozenset({*MAIN_SPLITS, "test_ood"})
TARGET_PCT = {"train": 70.0, "validation": 15.0, "test": 15.0}
TOLERANCE_PT = 5.0
OOD_GUARDS = {"min_images": 500, "min_groups": 5, "max_total_holdout_share": 0.25}
MAP_SPLIT = {"val": "validation"}  # SPLIT_MAP.json spells the validation split "val"


def read_report(build_dir: Path, release_dir: Path | None) -> dict | None:
    """``task_layers/V11_TABLES.json`` of the release dir (else of the build dir), or ``None``."""
    for base in (release_dir, build_dir):
        path = Path(base) / "task_layers" / "V11_TABLES.json" if base else None
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text())
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def read_split_map(build_dir: Path, release_dir: Path | None) -> dict | None:
    """``SPLIT_MAP.json`` (release dir, else build dir) as an object, ``None`` if unreadable."""
    for base in (release_dir, build_dir):
        path = Path(base) / "SPLIT_MAP.json" if base else None
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text())
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


@dataclass(frozen=True)
class MapPins:
    """What the gate reads of the split map: the ``pins.v1_continuity_overridden`` record (``None``
    when there is no map, it is unreadable or its ``pins`` block records no override), the
    sources of ``pins.benchmarks`` and the group assignments, spelled like the tables
    (``validation``)."""

    overridden: dict | None = None
    benchmarks: frozenset[str] = frozenset()
    assignments: Mapping[str, str] = field(default_factory=dict)

    def assigned(self, group: str) -> str | None:
        return self.assignments.get(group)


def map_pins(split_map: dict | None) -> MapPins:
    pins = split_map.get("pins") if isinstance(split_map, dict) else None
    block = pins.get("v1_continuity_overridden") if isinstance(pins, dict) else None
    if not isinstance(split_map, dict) or not isinstance(pins, dict) or not isinstance(block, dict):
        return MapPins()
    benchmarks, raw = pins.get("benchmarks"), split_map.get("assignments")
    return MapPins(
        overridden=block,
        benchmarks=frozenset(map(str, benchmarks))
        if isinstance(benchmarks, dict | list)
        else frozenset(),
        assignments={str(g): MAP_SPLIT.get(s, s) for g, s in raw.items()}
        if isinstance(raw, dict)
        else {},
    )


def read_overridden(build_dir: Path, release_dir: Path | None) -> dict | None:
    """``pins.v1_continuity_overridden`` of ``SPLIT_MAP.json`` (release dir, else build dir)."""
    return map_pins(read_split_map(build_dir, release_dir)).overridden


def _upstream_images(build: Path) -> set:
    found = set()
    for table in LABEL_TABLES:
        for row in table_rows(shards(build, table), ("image_sha256", "upstream_split")):
            if row.get("upstream_split"):
                found.add(row.get("image_sha256"))
    return found


def _source_of(row: dict, group: str) -> str | None:
    """The row's ``source_id``, else the prefix of its ``<source>/...`` group."""
    source = row.get("source_id")
    return str(source) if source else (group.split("/", 1)[0] if "/" in group else None)


def _unexplained(pins: MapPins, accepted: set[str] | None, group: str, split: str) -> str | None:
    """``None`` when a pin of the split map explains ``group`` leaving its v1.0 split for ``split``,
    else the (possibly empty) reason to append to the ``reassigned_v1`` line."""
    if pins.overridden is None:
        return ""
    if accepted is not None and group not in accepted:
        return "; its group is not in pins.v1_continuity_overridden.group_ids"
    if pins.assigned(group) != split:
        return f"; the split map assigns its group to {pins.assigned(group)}"
    return None


def _check_overrides(
    out: Collector,
    pins: MapPins,
    applied: Mapping[str, str | None],
    moved: set[str],
    v1_groups: Mapping[str, set[str]],
    group_source: Mapping[str, str | None],
) -> int:
    """``override_unchanged`` failures: a pin the map records that the build did not apply. Returns
    the no-op groups (``pin_override_noop``): recorded, with the images already in the map's split
    (or, when the record only counts groups, not seen moving: the build may not carry them)."""
    if pins.overridden is None:
        return 0
    if (listed := pins.overridden.get("group_ids")) is not None:
        noop = 0
        for group in sorted({str(g) for g in listed} - set(applied) - moved):
            if v1_groups.get(group) == {pins.assigned(group)}:
                noop += 1  # the map assigns it where its v1.0 images already sit
            else:
                out.fail(
                    "override_unchanged",
                    f"pins.v1_continuity_overridden names split_group {group[:24]}, "
                    "but no image of it left its v1.0 split",
                )
        return noop
    counted = int(pins.overridden.get("groups") or 0)
    if len(applied) > counted:
        out.fail(
            "override_unchanged",
            f"pins.v1_continuity_overridden counts {counted} group(s), "
            f"{len(applied)} left their v1.0 split",
        )
    for group, splits in sorted(v1_groups.items()):  # the groups are not listed: look at them all
        want = pins.assigned(group)
        if (
            group not in moved
            and group_source.get(group) in pins.benchmarks
            and want not in {None, *splits}
        ):
            out.fail(
                "override_unchanged",
                f"the split map assigns pinned split_group {group[:24]} to {want}, "
                f"but its v1.0 images stayed in {sorted(splits)}",
            )
    return max(counted - len(applied), 0)


def run_splits_v11_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    v1_dir: Path | None = None,
    rehearsal: bool = False,
    overridden: dict | None = None,
    split_map: dict | None = None,
    ood_guards: dict | None = None,
    ratio_min_images: int = 100,
    tolerance_pt: float = TOLERANCE_PT,
) -> GateResult:
    build, out = Path(build_dir), Collector()
    guards = {**OOD_GUARDS, **(ood_guards or {})}
    unsplit: list[str] = []
    report = read_report(build, release_dir)
    if report is None:
        out.fail(
            "no_report", "task_layers/V11_TABLES.json not found: --allow-unsplit not ruled out"
        )
    else:
        if report.get("allow_unsplit"):
            unsplit.append("V11_TABLES.json: allow_unsplit=true (the build used --allow-unsplit)")
        if int(report.get("unsplit_images") or 0) > 0:
            unsplit.append(f"V11_TABLES.json: unsplit_images={report['unsplit_images']}")
        for sid, c in sorted((report.get("metadata_by_source") or {}).items()):
            if (c.get("images") or c.get("task_rows")) and not (c.get("rows") or c.get("shared")):
                out.fail(
                    "metadata_missing",
                    f"V11_TABLES.json: {sid} has {c.get('images', 0)} staged image(s) and "
                    f"{c.get('task_rows', 0)} task row(s) but no metadata row",
                )
    pins = map_pins(read_split_map(build, release_dir) if split_map is None else split_map)
    if overridden is not None:  # an explicit record replaces the map's
        pins = replace(pins, overridden=overridden)
    listed = pins.overridden.get("group_ids") if pins.overridden is not None else None
    accepted = None if listed is None else {str(g) for g in listed}
    applied: dict[str, str | None] = {}  # group -> source: a benchmark pin explains its move
    refused: set[str] = set()  # groups with a moved image no pin explains
    moved: set[str] = set()  # groups with an image that left its v1.0 split
    v1_groups: dict[str, set[str]] = defaultdict(set)  # group -> splits of its v1.0 images
    group_source: dict[str, str | None] = {}
    v1 = v1_splits(Path(v1_dir)) if v1_dir else None
    if v1 is None:
        out.fail("no_v1_set", "no v1.0 tree (--v1-dir): a v1.0 reassignment cannot be ruled out")
    meta_files = shards(build, METADATA)
    if not meta_files:
        out.fail("no_metadata", "no metadata config: no split to check")
        return out.result(NAME)
    group_splits: dict = defaultdict(set)
    by_image: dict = {}
    for row in table_rows(meta_files, ("image_sha256", "split", "split_group", "source_id")):
        out.counts["metadata_rows"] += 1
        sha, split, where = row.get("image_sha256"), row.get("split"), row["_shard"]
        if split not in SPLITS:
            unsplit.append(f"metadata/{where} {str(sha)[:12]} has split {split!r}")
            continue
        by_image[sha] = split
        if split != row["_split"]:
            out.fail("split_shard", f"metadata/{where} {str(sha)[:12]} has split {split}")
        if row.get("split_group") is not None:
            group_splits[row["split_group"]].add(split)
        if v1 is not None and sha in v1:
            group = str(row.get("split_group"))
            source = _source_of(row, group)
            v1_groups[group].add(split)
            group_source.setdefault(group, source)
            if split not in v1[sha]:
                moved.add(group)
                why = _unexplained(pins, accepted, group, split)
                if why is None:
                    applied.setdefault(group, source)  # a benchmark pin moved this group
                else:
                    refused.add(group)
                    out.fail(
                        "reassigned_v1", f"{str(sha)[:12]}: v1.0 {sorted(v1[sha])} -> {split}{why}"
                    )
    for group, splits in sorted(group_splits.items(), key=lambda kv: str(kv[0])):
        if len(splits) > 1:
            out.fail("group_straddles", f"split_group {str(group)[:24]} is in {sorted(splits)}")
    applied = {g: s for g, s in applied.items() if g not in refused}
    noop = _check_overrides(out, pins, applied, moved, v1_groups, group_source)
    total, per_split = len(by_image), Counter(by_image.values())
    ood_groups = {g for g, s in group_splits.items() if "test_ood" in s}
    if per_split["test_ood"]:
        if per_split["test_ood"] < guards["min_images"]:
            out.fail(
                "ood_size", f"test_ood has {per_split['test_ood']} images < {guards['min_images']}"
            )
        if len(ood_groups) < guards["min_groups"]:
            out.fail("ood_size", f"test_ood has {len(ood_groups)} groups < {guards['min_groups']}")
        if per_split["test_ood"] / total > guards["max_total_holdout_share"]:
            out.fail("ood_share", f"test_ood is {per_split['test_ood'] / total:.1%} of the release")
    upstream = _upstream_images(build)
    free = Counter(s for sha, s in by_image.items() if sha not in upstream and s in TARGET_PCT)
    n_free = sum(free.values())
    if n_free >= ratio_min_images:
        for split, want in TARGET_PCT.items():
            got = 100.0 * free[split] / n_free
            if abs(got - want) > tolerance_pt:
                out.fail(
                    "ratio_off",
                    f"non-upstream {split}: {got:.1f}% vs {want:.0f}% +/-{tolerance_pt:g}",
                )
    extra = {"images": total, "non_upstream_images": n_free, "unsplit": len(unsplit)}
    if pins.overridden is not None:
        extra["v1_override_groups"] = extra["pin_override_applied"] = len(applied)
        extra["pin_override_noop"] = noop
        for source, n in sorted(Counter(map(str, applied.values())).items()):
            extra[f"pin_override_applied_{source}"] = n
        extra["pin_override_mates"] = sum(s not in pins.benchmarks for s in applied.values())
    extra.update({f"images_{k}": v for k, v in sorted(per_split.items())})
    if unsplit and rehearsal and not out.lines:
        return skip(NAME, f"rehearsal build: {unsplit[0]}", {**out.counts, **extra})
    for line in unsplit:
        out.fail("unsplit", line)
    return out.result(NAME, **extra)
