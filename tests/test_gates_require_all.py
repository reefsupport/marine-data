"""``release gate --require-all`` means every gate, each one run and passed (P1.13).

An absent, skipped or failed gate refuses, privacy included: a gate that cannot run means the
build has not passed its gates. A skipped croissant gate is a failure here, while the plain
verdict (REQUIRED) does not ask for it.
"""

import json

import pytest

from marinedata.gates import verdict
from marinedata.gates.runner import ALL


def _write(tmp_path, **override):
    gates = {name: {"passed": True, "counts": {}, "failures": []} for name in ALL}
    gates.update(override)
    (tmp_path / "GATES.json").write_text(json.dumps(gates))
    return tmp_path


def test_every_gate_passed(tmp_path):
    assert verdict(_write(tmp_path), strict=True) == (True, [])


def test_a_missing_privacy_gate_blocks_the_strict_verdict_only(tmp_path):
    d = _write(tmp_path)
    gates = json.loads((d / "GATES.json").read_text())
    del gates["privacy"]
    (d / "GATES.json").write_text(json.dumps(gates))
    ok, problems = verdict(d, strict=True)
    assert not ok and problems == ["privacy: missing"]
    assert verdict(d)[0] is True


@pytest.mark.parametrize(
    ("gate", "reason"),
    [
        ("privacy", "no image bytes in this build"),
        ("privacy", "detector not installed"),
        ("card", "no card to check"),
    ],
)
def test_a_skipped_gate_fails_the_strict_verdict_whatever_the_reason(tmp_path, gate, reason):
    ok, problems = verdict(_write(tmp_path, **{gate: {"skipped": reason}}), strict=True)
    assert not ok and problems == [f"{gate}: skipped ({reason})"]


def test_a_skipped_croissant_gate_fails_the_strict_verdict(tmp_path):
    d = _write(tmp_path, croissant={"skipped": "mlcroissant not installed"})
    ok, problems = verdict(d, strict=True)
    assert not ok and problems[0].startswith("croissant: skipped")
    assert verdict(d)[0] is True
