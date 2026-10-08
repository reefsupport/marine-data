"""``release build`` refuses the frozen v1.0 unless ``--allow-frozen`` (P1.13, P1.14 F1a).

v1.0 is frozen as the published artifact, not as code: building it again is allowed, but only
on purpose.
"""

from pathlib import Path

import pytest

from marinedata.cli import build_parser, main

LOCK = Path(__file__).resolve().parents[1] / "releases/marine-data/v1.0/release.lock.yaml"


def _argv(tmp_path, release_id, *extra):
    split_map = str(tmp_path / "S.json")
    base = ["release", "build", "--release", release_id, "--flavour", "open"]
    return [*base, "--split-map", split_map, "--out", str(tmp_path), *extra]


def _run(argv) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


@pytest.mark.parametrize("release_id", ["v1", "v1.0"])
def test_the_frozen_release_id_is_refused(release_id, tmp_path, capsys):
    assert _run(_argv(tmp_path, release_id)) == 1
    err = capsys.readouterr().err
    assert "frozen v1.0" in err and "--allow-frozen" in err
    assert not any(tmp_path.iterdir())


def test_the_v1_0_lock_is_refused_even_under_a_new_release_id(tmp_path, capsys):
    assert _run(_argv(tmp_path, "v1.1", "--lock", str(LOCK))) == 1
    err = capsys.readouterr().err
    assert "the v1.0 lock" in err and "--allow-frozen" in err


def test_allow_frozen_is_a_flag_of_release_build(tmp_path):
    argv = _argv(tmp_path, "v1.0", "--allow-frozen")
    assert build_parser().parse_args(argv).allow_frozen is True
    assert build_parser().parse_args(_argv(tmp_path, "v1.0")).allow_frozen is False
