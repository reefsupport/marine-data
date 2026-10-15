"""``marinedata --version`` (the release image's smoke check runs it)."""

from __future__ import annotations

import pytest

from marinedata.cli import main


def test_version_flag_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("marinedata ")
