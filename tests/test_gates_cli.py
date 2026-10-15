from __future__ import annotations

import json

import pytest
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


def _with_croissant(build) -> None:
    from marinedata import release_card, release_croissant

    doc = release_croissant.build_croissant(build, release_card.load_context(build))
    (build / "croissant.json").write_text(json.dumps(doc))


def test_all_require_all_passes_on_a_clean_build_and_lists_every_gate(tmp_path):
    pytest.importorskip("mlcroissant")
    build = make_build(tmp_path)
    _with_croissant(build)
    assert _gate(build, "--all", "--require-all") == 0
    gates = json.loads((build / "GATES.json").read_text())
    assert ALL_GATES | {"summary"} == set(gates)
    summary = gates["summary"]
    assert (summary["passed"], summary["failed"], summary["skipped"]) == (8, 0, 0)
    assert set(summary["required_for_publish"]) == ALL_GATES
    assert gates["privacy"]["passed"] and gates["croissant"]["passed"]


def test_require_all_refuses_a_build_without_croissant_json_that_plain_all_accepts(tmp_path):
    build = make_build(tmp_path)
    assert _gate(build, "--all") == 0
    assert _gate(build, "--all", "--require-all") == 1
    assert json.loads((build / "GATES.json").read_text())["croissant"]["skipped"]


def test_require_all_refuses_a_skipped_privacy_gate(tmp_path, capsys):
    """P1.13: the privacy gate has to run. A build with no image bytes skips it ("nothing to
    scan"), and under ``--require-all`` that skip is a failure, not a pass."""
    build = make_build(tmp_path, images=[], masks=False)  # no image bytes anywhere
    assert _gate(build, "--all", "--require-all") == 1
    gates = json.loads((build / "GATES.json").read_text())
    assert "no image bytes" in gates["privacy"]["skipped"]
    assert "release gate --require-all: privacy: skipped" in capsys.readouterr().err


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


def test_release_gate_help_builds_the_one_parser_and_lists_every_mode(capsys):
    """F2: gates and join each used to register `release gate`; argparse raised at build time."""
    with pytest.raises(SystemExit) as stop:
        main(["release", "gate", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--all", "--require-all", "--masks", "--metadata", "--gates", "--croissant"):
        assert flag in text, flag


def test_metadata_flag_writes_gates_out_next_to_the_e2e_build(tmp_path, monkeypatch):
    """F8: `release gate BUILD --metadata DIR --gates OUT` on the e2e fixture build; the
    metadata gate reads the fixture registry (the pipeline fixture patches `Registry.load`)."""
    from test_e2e_release import SOURCES, _pipeline
    from test_metadata_gate import _cov

    run = _pipeline(tmp_path, monkeypatch)
    resolved, out = tmp_path / "resolved", tmp_path / "metadata-GATES.json"
    for sid in SOURCES:
        _cov(resolved, sid, {}, n=15)
    argv = [
        "release",
        "gate",
        str(run.export_dir),
        "--metadata",
        str(resolved),
        "--gates",
        str(out),
    ]
    assert main(argv) == 0
    gates = json.loads(out.read_text())
    assert set(gates) == {"metadata", "summary"} and gates["metadata"]["passed"] is True
    _cov(resolved, "unregistered-source", {}, n=15)
    assert main(argv) == 1
    metadata = json.loads(out.read_text())["metadata"]
    assert metadata["passed"] is False
    assert any("unregistered-source: not in the registry" in f for f in metadata["failures"])
