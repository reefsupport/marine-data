from __future__ import annotations

import pytest
from _gates_build import NC_LICENCE, make_build

from marinedata.gates.licence import run_licence_gate


class _Source:
    def __init__(self, per_row: bool) -> None:
        self.licence_per_row = per_row


class _Registry:
    def __init__(self, per_row=()) -> None:
        self._per_row = set(per_row)

    def source(self, source_id: str) -> _Source:
        return _Source(source_id in self._per_row)


def _run(build, registry=None, **kw):
    return run_licence_gate(build, registry=registry or _Registry(), **kw)


def _has(result, kind: str) -> bool:
    return any(f"{kind}:" in line for line in result.failures)


def _sources(*extra: tuple[str, str]) -> list[dict]:
    base = [{"id": "src-a", "access_class": "open"}, {"id": "src-b", "access_class": "open"}]
    return [*base, *({"id": i, "access_class": c} for i, c in extra)]


def test_clean_build_passes(tmp_path):
    result = _run(make_build(tmp_path))
    assert result.passed, result.failures
    assert result.counts["flavour"] == "open" and result.counts["rows_checked"] > 0


def test_no_flavour_fails_and_argument_fixes_it(tmp_path):
    build = make_build(tmp_path, flavour=None)
    assert _has(_run(build), "no_flavour") and not _run(build).passed
    assert _run(build, flavour="open").passed


def test_no_lock(tmp_path):
    assert _has(_run(make_build(tmp_path, sources=[])), "no_lock")


@pytest.mark.parametrize("access_class", ["restricted-nc", "internal-only"])
def test_source_class_outside_flavour(tmp_path, access_class):
    sources = _sources(("src-c", access_class))
    assert _has(_run(make_build(tmp_path, sources=sources)), "source_class")


def test_per_row_source_is_admitted_unless_internal(tmp_path):
    registry = _Registry(["src-c"])
    nc = make_build(tmp_path / "a", sources=_sources(("src-c", "restricted-nc")))
    assert _run(nc, registry).passed
    internal = make_build(tmp_path / "b", sources=_sources(("src-c", "internal-only")))
    assert _has(_run(internal, registry), "source_class")


@pytest.mark.parametrize(
    ("override", "kind"),
    [
        ({"license": None}, "null_licence"),
        ({"license": "  "}, "null_licence"),
        ({"license": NC_LICENCE}, "row_licence_outside_flavour"),
        ({"licence_class": "nc"}, "row_class_outside_flavour"),
        ({"attribution": ""}, "empty_attribution"),
        ({"source": "src-z"}, "source_not_in_lock"),
    ],
)
def test_row_failure_modes(tmp_path, override, kind):
    result = _run(make_build(tmp_path, extra_metadata=[override]))
    assert not result.passed and _has(result, kind), result.failures


def test_nc_flavour_accepts_nc_sources_and_rows(tmp_path):
    sources = [
        {"id": "src-a", "access_class": "restricted-nc"},
        {"id": "src-b", "access_class": "restricted-nc"},
    ]
    build = make_build(
        tmp_path, flavour="nc", sources=sources, licence=NC_LICENCE, licence_class="restricted-nc"
    )
    assert _run(build).passed
    assert not _run(build, flavour="open").passed
