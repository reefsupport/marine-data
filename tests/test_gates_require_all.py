"""``release gate --require-all`` means every gate, each required (P1.13).

Privacy may be skipped only when the build has no image bytes; a skipped croissant gate is a
failure here, while the plain verdict (REQUIRED) does not ask for it.
"""

import json

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


def test_privacy_may_skip_only_without_image_bytes(tmp_path):
    ok, _ = verdict(
        _write(tmp_path, privacy={"skipped": "no image bytes in this build"}), strict=True
    )
    assert ok
    ok, problems = verdict(
        _write(tmp_path, privacy={"skipped": "detector not installed"}), strict=True
    )
    assert not ok and "privacy: skipped" in problems[0]


def test_a_skipped_croissant_gate_fails_the_strict_verdict(tmp_path):
    d = _write(tmp_path, croissant={"skipped": "mlcroissant not installed"})
    ok, problems = verdict(d, strict=True)
    assert not ok and problems[0].startswith("croissant: skipped")
    assert verdict(d)[0] is True
