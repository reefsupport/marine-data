"""``marinedata splits check`` — run split v2 end to end and gate on §3.3's tolerances.

Orchestrates the four ``marinedata.splitv2`` modules: load the rules, assign OOD
holdouts (any-member-OOD), allocate the remaining ID pool, and check the achieved
ratios plus the "no group is split across two splits" invariant. Samples are read
from a parquet file with (at least) the columns
``sha256, split_group_id, source_id, never_eval, pinned_split`` plus the optional OOD
fields (``meow_realm``, ``meow_province``, ``depth_m``, ``platform``,
``capture_datetime``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .splitv2.allocate import achieved_by_stratum, allocate
from .splitv2.dp_province import (
    labelled_total,
    province_labelled_counts,
    resolve_tropical_province_rule,
    select_tropical_province,
)
from .splitv2.holdouts import Sample, assign_ood, share_issues
from .splitv2.rules import load_rules, rules_sha256


def build_strata(samples: list[Sample], never_eval: set[str]) -> dict[str, dict[str, int]]:
    strata: dict[str, dict[str, int]] = {}
    for s in samples:
        if s.source_id in never_eval:
            continue
        strata.setdefault(s.source_id, {}).setdefault(s.split_group_id, 0)
        strata[s.source_id][s.split_group_id] += 1
    return strata


def split_groups_ok(strata: dict[str, dict[str, int]], assignment: dict[str, str]) -> list[str]:
    """The invariant this whole package exists to protect: a group never spans splits."""
    seen: dict[str, str] = {}
    offenders = []
    for groups in strata.values():
        for gid in groups:
            split = assignment.get(gid)
            if gid in seen and seen[gid] != split:
                offenders.append(gid)
            seen[gid] = split
    return sorted(set(offenders))


def _cmd_check(args: argparse.Namespace) -> int:
    config = load_rules(args.rules)
    digest = rules_sha256(config)
    print(f"rules_sha256 {digest}")

    if not args.samples:
        print("no --samples given; rules loaded and validated only", file=sys.stderr)
        return 0

    import pyarrow.parquet as pq

    table = pq.read_table(args.samples)
    cols = table.to_pylist()
    samples = [
        Sample(
            sha256=row["sha256"],
            split_group_id=row["split_group_id"],
            source_id=row["source_id"],
            meow_realm=row.get("meow_realm"),
            meow_province=row.get("meow_province"),
            depth_m=row.get("depth_m"),
            platform=row.get("platform"),
            capture_datetime=row.get("capture_datetime"),
        )
        for row in cols
    ]
    never_eval = {row["source_id"] for row in cols if row.get("never_eval")}
    pinned = {row["split_group_id"]: row["pinned_split"] for row in cols if row.get("pinned_split")}

    dp_counts = province_labelled_counts(samples, config, never_eval)
    dp_choice = select_tropical_province(
        dp_counts,
        tropical_realms=config.tropical_realms,
        labelled_total=labelled_total(samples, never_eval),
        min_images=int(config.size_guards["min_images"]),
    )
    print(dp_choice.describe())
    config = resolve_tropical_province_rule(config, dp_choice)

    ood = assign_ood(samples, config)
    eligible = [
        s
        for s in samples
        if s.source_id not in never_eval and s.split_group_id not in ood.ood_groups
    ]
    strata = build_strata(eligible, never_eval)

    result = allocate(
        strata,
        config.ratios,
        seed=config.seed,
        pinned={g: s for g, s in pinned.items() if g not in ood.ood_groups},
        min_groups=config.min_groups,
        overshoot_cap=config.allocator["overshoot_cap"],
        repair_max_swaps=int(config.allocator["repair_max_swaps"]),
    )
    offenders = split_groups_ok(strata, result.assignment)
    eligible_total = sum(sum(g.values()) for g in strata.values())
    issues = share_issues(ood.outcomes, eligible_total + len(ood.ood_groups), config.size_guards)

    ok = True
    print(f"eligible ID images {eligible_total}, groups {sum(len(g) for g in strata.values())}")
    for outcome in ood.outcomes:
        print(
            f"  {outcome.name}: constructible={outcome.constructible} "
            f"images={outcome.n_images} groups={outcome.n_groups} {outcome.note}"
        )
    if offenders:
        ok = False
        print(
            f"FAIL: {len(offenders)} split groups span two splits: {offenders[:10]}",
            file=sys.stderr,
        )
    for issue in issues:
        ok = False
        print(f"FAIL: {issue}", file=sys.stderr)
    for name, achieved in sorted(achieved_by_stratum(strata, result.assignment).items()):
        total = sum(achieved.values())
        print(f"  {name}: {achieved} of {total}")

    print("OK" if ok else "FAIL", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


def add_splits_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("splits", help="split v2 (design §3): pools, OOD holdouts, allocator")
    sp = p.add_subparsers(dest="splits_cmd", required=True)

    check = sp.add_parser(
        "check", help="run split v2 and gate on ratio tolerance + group invariant"
    )
    check.add_argument("--rules", default="registry/splits/v2.yaml", type=Path)
    check.add_argument("--samples", type=Path, default=None, help="parquet of per-image records")
    check.set_defaults(func=_cmd_check)
