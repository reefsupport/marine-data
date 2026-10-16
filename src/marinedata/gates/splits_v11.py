"""G6 splits (v1.1.0): the split columns keep the frozen v1.0 assignment and the split policy.

* ``unsplit``          a metadata row whose ``split`` is null or not train / validation / test /
  test_ood, or ``task_layers/V11_TABLES.json`` records ``unsplit_images > 0`` / ``allow_unsplit``
  (a ``--allow-unsplit`` build). With ``rehearsal`` a build whose only problem is that is
  ``skipped``, never ``passed``: ``release publish`` still refuses it. A missing report fails.
* ``metadata_missing`` ``V11_TABLES.json`` records an admitted source with staged images or task
  rows and no metadata row (PR-M: a narrow tree once vanished silently from ``metadata``);
* ``split_shard``      a row's ``split`` is not the split of its shard file;
* ``group_straddles``  one ``split_group`` in two splits (near-copies must not straddle);
* ``reassigned_v1``    an image of a v1.0 config in a split v1.0 did not give it (``v1_dir``);
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


def _upstream_images(build: Path) -> set:
    found = set()
    for table in LABEL_TABLES:
        for row in table_rows(shards(build, table), ("image_sha256", "upstream_split")):
            if row.get("upstream_split"):
                found.add(row.get("image_sha256"))
    return found


def run_splits_v11_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    v1_dir: Path | None = None,
    rehearsal: bool = False,
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
    v1 = v1_splits(Path(v1_dir)) if v1_dir else None
    if v1 is None:
        out.fail("no_v1_set", "no v1.0 tree (--v1-dir): a v1.0 reassignment cannot be ruled out")
    meta_files = shards(build, METADATA)
    if not meta_files:
        out.fail("no_metadata", "no metadata config: no split to check")
        return out.result(NAME)
    group_splits: dict = defaultdict(set)
    by_image: dict = {}
    for row in table_rows(meta_files, ("image_sha256", "split", "split_group")):
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
        if v1 is not None and sha in v1 and split not in v1[sha]:
            out.fail("reassigned_v1", f"{str(sha)[:12]}: v1.0 {sorted(v1[sha])} -> {split}")
    for group, splits in sorted(group_splits.items(), key=lambda kv: str(kv[0])):
        if len(splits) > 1:
            out.fail("group_straddles", f"split_group {str(group)[:24]} is in {sorted(splits)}")
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
    extra.update({f"images_{k}": v for k, v in sorted(per_split.items())})
    if unsplit and rehearsal and not out.lines:
        return skip(NAME, f"rehearsal build: {unsplit[0]}", {**out.counts, **extra})
    for line in unsplit:
        out.fail("unsplit", line)
    return out.result(NAME, **extra)
