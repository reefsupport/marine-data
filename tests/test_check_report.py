"""``marinedata check`` separates EXPECTED denials (the entry's own class says it cannot ship)
from UNEXPECTED ones (the class claims it ships, the gate refuses). Exit 1 only on the latter,
or on a malformed registry; there is no opt-out flag."""

from __future__ import annotations

import pytest

from marinedata import cli
from marinedata.check_report import build_report, expected_because, profile_ships_class
from marinedata.enums import AccessClass as AC
from marinedata.registry import Registry, RegistryError

REG = Registry.load()
DUO = REG.source("duo")  # denied under `research` by the no_derivatives flag


class _StubRegistry:
    """The slice of ``Registry`` that ``check`` reads, over an explicit list of sources."""

    def __init__(self, sources):
        self.root = REG.root
        self._sources = list(sources)

    def __iter__(self):
        return iter(self._sources)

    def profile(self, profile_id):
        return REG.profile(profile_id)


def _patch_registry(monkeypatch, sources):
    stub = _StubRegistry(sources)
    monkeypatch.setattr(cli.Registry, "load", classmethod(lambda cls, root=None: stub))


def test_real_registry_has_no_unexpected_denial_under_research(capsys):
    assert cli.main(["check", "--profile", "research"]) == 0
    out = capsys.readouterr().out
    assert "EXPECTED DENIALS" in out and "UNEXPECTED DENIALS (0)" in out
    report = build_report(REG, REG.profile("research"))
    assert report.expected and not report.unexpected


@pytest.mark.parametrize("cls", [AC.INTERNAL_ONLY, AC.UNKNOWN, AC.RESTRICTED_ND])
def test_denial_of_a_never_shipping_class_is_expected_everywhere(cls):
    src = DUO.model_copy(update={"access_class": cls})
    for pid in ("research", "ship-open", "ship-noncommercial", "pretrain-eu"):
        assert expected_because(src, REG.profile(pid))


def test_open_class_denial_is_unexpected_and_exits_one(monkeypatch, capsys):
    assert (
        expected_because(DUO.model_copy(update={"access_class": AC.OPEN}), REG.profile("research"))
        is None
    )
    _patch_registry(monkeypatch, [DUO.model_copy(update={"access_class": AC.OPEN})])
    assert cli.main(["check", "--profile", "research"]) == 1
    out = capsys.readouterr().out
    assert "UNEXPECTED DENIALS (1)" in out and "! duo" in out


def test_denial_of_an_internal_only_source_exits_zero(monkeypatch, capsys):
    _patch_registry(monkeypatch, [DUO])  # class internal-only, denied by the gate
    assert cli.main(["check", "--profile", "research"]) == 0
    assert "✗ duo" in capsys.readouterr().out


def test_restricted_nc_is_expected_only_under_a_profile_that_ships_no_nc():
    nc = DUO.model_copy(update={"access_class": AC.RESTRICTED_NC})
    assert expected_because(nc, REG.profile("research"))  # research ships nothing
    assert expected_because(nc, REG.profile("ship-open"))
    assert expected_because(nc, REG.profile("ship-noncommercial")) is None
    assert expected_because(nc, REG.profile("pretrain-eu")) is None  # ships, no class bar
    assert profile_ships_class(REG.profile("ship-noncommercial"), AC.RESTRICTED_NC)
    assert not profile_ships_class(REG.profile("research"), AC.OPEN)


def test_release_excluded_denial_is_expected():
    open_src = DUO.model_copy(update={"access_class": AC.OPEN})
    assert expected_because(open_src, REG.profile("research"), excluded=True) == "release-excluded"


def test_malformed_registry_exits_one(monkeypatch, capsys):
    def boom(cls, root=None):
        raise RegistryError("bad.yaml: invalid source entry")

    monkeypatch.setattr(cli.Registry, "load", classmethod(boom))
    assert cli.main(["check", "--profile", "research"]) == 1
    assert "registry invalid" in capsys.readouterr().err
