"""G6 splits_v11 and D-STRADDLE: a benchmark pin may move a group off its v1.0 split.

The synthetic map pins ``suim`` and ``uiis``; ``seaview`` is a component-mate (a pin moves the
whole near-duplicate component, so groups of other sources move with it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_gates_v11 import V1_CARD, failed, write_by_split, write_report, write_shard

from marinedata.gates.splits_v11 import run_splits_v11_gate

MOVED_A = ("suim/a", "suim", "train", "test")  # v1.0 train -> pinned to test
MOVED_B = ("suim/b", "suim", "train", "validation")  # the map spells validation "val"
MATE = ("seaview/d", "seaview", "train", "test")  # not a benchmark source: moved with its component
SAME = ("uiis/c", "uiis", "test", "test")  # recorded, but v1.0 already had it in the map's split
BOTH = ("uiis/f", "uiis", ("test", "train"), "test")  # v1.0 published the image in two splits
NEW = ("suim/e", "suim", None, "train")  # not a v1.0 image
MAP = {"suim/a": "test", "suim/b": "val", "seaview/d": "test", "uiis/c": "test", "uiis/f": "test"}


def scene(tmp_path: Path, images: list[tuple]) -> tuple[Path, Path]:
    """A v1.0 tree and a build; ``images``: (group, source, v1.0 split(s) or None, build split)."""
    v1_rows: dict[str, list[dict]] = {}
    meta = []
    for i, (group, source, was, split) in enumerate(images):
        sha = hashlib.sha256(f"pin{i}".encode()).hexdigest()
        for v1_split in (was,) if isinstance(was, str) else (was or ()):
            v1_rows.setdefault(v1_split, []).append({"image_sha256": sha})
        meta.append({"image_sha256": sha, "source_id": source, "split_group": group,
                     "split": split})  # fmt: skip
    for v1_split, rows in v1_rows.items():
        write_shard(tmp_path / "v1", "coralseg", v1_split, rows)
    (tmp_path / "v1" / "README.md").write_text(V1_CARD)
    build = tmp_path / "build"
    write_by_split(build, "metadata", meta, lambda r: r["split"])
    (build / "task_layers").mkdir()
    write_report(build)
    return build, tmp_path / "v1"


def pin_map(build: Path, *, counted: int, ids=None, assignments=None, benchmarks=("suim", "uiis")):
    block = {"images": counted, "groups": counted, "by_source": {}}
    if ids is not None:
        block["group_ids"] = list(ids)
    pins = {"benchmarks": {b: {"test": 1} for b in benchmarks}, "v1_continuity_overridden": block}
    (build / "SPLIT_MAP.json").write_text(
        json.dumps({"pins": pins, "assignments": MAP if assignments is None else assignments})
    )


def reasons(result) -> str:
    return " ".join(map(str, result.failures))


@pytest.mark.parametrize("listed", [True, False], ids=["group_ids", "count_only"])
def test_pin_overrides_in_the_maps_split_are_applied_and_counted_per_source(tmp_path, listed):
    build, v1 = scene(tmp_path, [MOVED_A, MOVED_B, MATE, NEW])
    ids = ["suim/a", "suim/b", "seaview/d"] if listed else None
    pin_map(build, counted=3, ids=ids)
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert result.passed, result.failures
    assert {k: v for k, v in result.counts.items() if k.startswith(("pin_", "v1_"))} == {
        "pin_override_applied": 3, "pin_override_applied_suim": 2,
        "pin_override_applied_seaview": 1, "pin_override_mates": 1,
        "pin_override_noop": 0, "v1_override_groups": 3,
    }  # fmt: skip


def test_recorded_groups_the_map_leaves_in_their_v1_split_are_noops(tmp_path):
    build, v1 = scene(tmp_path, [MOVED_A, SAME, BOTH])
    pin_map(build, counted=2, ids=["suim/a", "uiis/c"])  # listed: c did not move, map == v1.0
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert result.passed, result.failures
    assert result.counts["pin_override_applied"] == 1 and result.counts["pin_override_noop"] == 1
    pin_map(build, counted=3)  # count only: the 2 recorded groups that did not move are noops
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert result.passed, result.failures
    assert result.counts["pin_override_applied"] == 1 and result.counts["pin_override_noop"] == 2


def test_a_pin_the_build_did_not_apply_stays_override_unchanged(tmp_path):
    build, v1 = scene(tmp_path, [MOVED_A, SAME])
    wrong = {**MAP, "uiis/c": "train"}  # the map moves c to train; the build left it in test
    pin_map(build, counted=2, ids=["suim/a", "uiis/c"], assignments=wrong)
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "override_unchanged") == 1
    pin_map(build, counted=2, assignments=wrong)  # count only: the pinned source is checked
    unapplied = run_splits_v11_gate(build, v1_dir=v1)
    assert failed(unapplied, "override_unchanged") == 1
    assert "assigns pinned split_group uiis/c to train" in reasons(unapplied)
    pin_map(build, counted=0, assignments=MAP)  # one group moved, the record counts none
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "override_unchanged") == 1


def test_a_move_no_pin_explains_stays_reassigned_v1_with_the_reason(tmp_path):
    build, v1 = scene(tmp_path, [MOVED_A, MATE])
    pin_map(build, counted=2, assignments={**MAP, "suim/a": "val"})  # the map says val, build test
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert failed(result, "reassigned_v1") == 1
    assert "the split map assigns its group to validation" in reasons(result)
    pin_map(build, counted=2, ids=["suim/a"])  # seaview/d moved and is not in the record
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert failed(
        result, "reassigned_v1"
    ) == 1 and "not in pins.v1_continuity_overridden" in reasons(result)
    assert result.counts["pin_override_applied"] == 1
    pin_map(build, counted=2, assignments={"suim/a": "test"})  # the map does not know seaview/d
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "reassigned_v1") == 1


@pytest.mark.parametrize(
    "text",
    [
        None,
        "{not json",
        json.dumps({"assignments": MAP}),
        json.dumps({"pins": {}, "assignments": MAP}),
        json.dumps({"pins": {"benchmarks": {"suim": {}}}, "assignments": MAP}),
    ],
    ids=["no_map", "unreadable", "no_pins", "empty_pins", "no_override_record"],
)
def test_without_a_pin_record_every_v1_move_still_fails(tmp_path, text):
    build, v1 = scene(tmp_path, [MOVED_A, SAME])
    if text is not None:
        (build / "SPLIT_MAP.json").write_text(text)
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert failed(result, "reassigned_v1") == 1 and failed(result, "override_unchanged") == 0
    assert "pin_override_applied" not in result.counts and "assigns" not in reasons(result)
