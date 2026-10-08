from __future__ import annotations

import json

from _gates_build import Img, default_images, make_build

from marinedata.cli import main
from marinedata.gates import (
    REQUIRED,
    GateResult,
    read_gates,
    runner,
    summarize,
    verdict,
    write_gates,
)

ALL_GATES = {"licence", "dedup", "splits", "masks", "privacy", "load", "card", "croissant"}


def _gate(build, *flags: str) -> int:
    return main(["release", "gate", str(build), *flags])


def test_all_require_all_passes_on_a_clean_build_and_lists_every_gate(tmp_path):
    build = make_build(tmp_path)
    assert _gate(build, "--all", "--require-all") == 0
    gates = json.loads((build / "GATES.json").read_text())
    assert ALL_GATES | {"summary"} == set(gates)
    summary = gates["summary"]
    assert (summary["passed"], summary["failed"], summary["skipped"]) == (7, 0, 1)
    assert summary["required_for_publish"] == [*REQUIRED, "privacy"]
    assert gates["privacy"]["passed"] and gates["croissant"]["skipped"]


def test_a_failing_gate_exits_1_and_require_all_refuses(tmp_path):
    build = make_build(
        tmp_path, images=[*default_images(), Img(0, "test", group="gx")], masks=False
    )
    assert _gate(build, "--dedup") == 1
    assert read_gates(build)["dedup"]["passed"] is False
    assert _gate(build, "--dedup", "--require-all") == 1
    ok, problems = verdict(build)
    assert not ok and any(p.startswith("dedup: failed") for p in problems)


def test_require_all_needs_every_required_gate(tmp_path):
    build = make_build(tmp_path)
    assert _gate(build, "--require-all") == 1  # no GATES.json yet
    assert _gate(build, "--licence", "--require-all") == 1  # the other required gates are missing
    assert _gate(build, "--licence") == 0 and "licence" in read_gates(build)


def test_results_merge_into_gates_json(tmp_path):
    build = make_build(tmp_path)
    assert _gate(build, "--licence") == 0 and _gate(build, "--dedup", "--splits") == 0
    assert {"licence", "dedup", "splits", "summary"} == set(read_gates(build))


def test_legacy_masks_directory_form_still_works(tmp_path):
    build = make_build(tmp_path)
    assert main(["release", "gate", "--masks", str(build)]) == 0
    assert read_gates(build)["masks"]["passed"] is True
    assert _gate(build, "--masks") == 0


def test_usage_errors(tmp_path):
    build = make_build(tmp_path)
    assert main(["release", "gate"]) == 2
    assert _gate(build) == 2
    assert main(["release", "gate", str(build), "--masks", str(tmp_path)]) == 2
    assert main(["release", "gate", str(tmp_path / "missing"), "--all"]) == 1


def test_flags_reach_the_gates(tmp_path):
    build = make_build(tmp_path, images=[*default_images(), Img(0, "train", shift=6, group="gk")])
    assert _gate(build, "--dedup", "--phash-bits", "8") == 0
    build = make_build(tmp_path / "b", flavour=None)
    assert _gate(build, "--licence") == 1
    assert _gate(build, "--licence", "--flavour", "open") == 0


def test_a_gate_that_raises_is_a_failed_gate(monkeypatch, tmp_path):
    def boom(build, options):
        raise RuntimeError("boom")

    monkeypatch.setitem(runner.RUNNERS, "licence", boom)
    result = runner.run_gate("licence", tmp_path)
    assert not result.passed and "gate_error" in result.failures[0]


def test_summary_and_verdict_rules(tmp_path):
    gates = {n: {"passed": True} for n in REQUIRED} | {"privacy": {"passed": False, "skipped": "x"}}
    assert (
        summarize(gates)["skipped"] == 1
        and "privacy" not in summarize(gates)["required_for_publish"]
    )
    write_gates(tmp_path, GateResult("licence", True, {}, []))
    ok, problems = verdict(tmp_path)
    assert not ok and "dedup: missing" in problems
    write_gates(tmp_path, GateResult("dedup", False, {}, ["x"], skipped="no pixels"))
    assert any(p.startswith("dedup: skipped") for p in verdict(tmp_path)[1])
