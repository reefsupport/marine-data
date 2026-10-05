"""WP-R2: one licence truth. ``registry/sources/`` and ``registry/ingest-specs/`` describe the same
sources; the release path reads the first, the ingest path the second. They must never disagree on
``access_class`` or on the licence (a 'dry run' once shipped atlantis as open while its spec said
restricted-nc)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marinedata.licence_class import (
    SPEC_ALIASES,
    UNKNOWN,
    licence_class_of,
    source_class,
)
from marinedata.registry import Registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry"


def _raw_sources() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((REGISTRY / "sources").glob("*.yaml")):
        for e in (yaml.safe_load(path.read_text()) or {}).get("sources", []):
            if isinstance(e, dict) and "id" in e:
                out[e["id"]] = e
    return out


def _raw_specs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((REGISTRY / "ingest-specs").glob("*.yaml")):
        e = yaml.safe_load(path.read_text()) or {}
        if isinstance(e, dict) and "id" in e:
            out[e["id"]] = e
    return out


def _pairs() -> list[tuple[str, str]]:
    sources, specs = _raw_sources(), _raw_specs()
    pairs = [(sid, sid) for sid in sorted(set(sources) & set(specs))]
    pairs += [(sid, spec) for sid, spec in sorted(SPEC_ALIASES.items()) if sid in sources]
    return pairs


def _source_licence(entry: dict) -> str:
    lic = entry.get("licence")
    return str(lic.get("id") if isinstance(lic, dict) else lic or "")


def test_pairs_cover_the_overlap_and_atlantis_alias():
    pairs = _pairs()
    assert len(pairs) > 80
    assert ("atlantis-synthetic-depth", "atlantis") in pairs


def test_access_class_equal_in_both_registry_files():
    sources, specs = _raw_sources(), _raw_specs()
    bad = [
        f"{sid}/{spec}: sources={sources[sid].get('access_class')} "
        f"spec={specs[spec].get('access_class')}"
        for sid, spec in _pairs()
        if str(sources[sid].get("access_class")) != str(specs[spec].get("access_class"))
    ]
    assert not bad, bad


def test_licence_class_equal_in_both_registry_files():
    """Compared as a licence CLASS (free text differs: 'CC-BY-4.0 (Zenodo API)'); a side that
    does not parse to a class (NOASSERTION, research-only prose, per-row) makes no claim."""
    sources, specs = _raw_sources(), _raw_specs()
    bad = []
    for sid, spec in _pairs():
        a = licence_class_of(_source_licence(sources[sid]))
        b = licence_class_of(str(specs[spec].get("license") or ""))
        if UNKNOWN not in (a, b) and a != b:
            bad.append(
                f"{sid}/{spec}: sources={_source_licence(sources[sid])} -> {a}; "
                f"spec={specs[spec].get('license')} -> {b}"
            )
    assert not bad, bad


def test_licence_per_row_flag_equal_in_both_registry_files():
    sources, specs = _raw_sources(), _raw_specs()
    bad = [
        sid
        for sid, spec in _pairs()
        if bool(sources[sid].get("licence_per_row")) != bool(specs[spec].get("licence_per_row"))
    ]
    assert not bad, bad


def test_release_path_resolves_the_spec_class():
    """``registry.source(id)`` (what the release path reads) and ``source_class`` agree with the
    ingest spec, including the atlantis alias."""
    registry = Registry.load(REGISTRY)
    specs = _raw_specs()
    for sid, spec in _pairs():
        want = str(specs[spec]["access_class"])
        assert registry.source(sid).access_class.value == want, sid
        assert source_class(sid, root=REGISTRY) == want, sid
    assert registry.source("atlantis-synthetic-depth").access_class.value == "restricted-nc"


def test_spec_wins_over_a_stale_registry_source(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "ingest-specs").mkdir()
    (tmp_path / "sources" / "x.yaml").write_text(
        yaml.safe_dump({"sources": [{"id": "a", "access_class": "open"}]})
    )
    (tmp_path / "ingest-specs" / "a.yaml").write_text(
        yaml.safe_dump({"id": "a", "access_class": "restricted-nc"})
    )
    assert source_class("a", root=tmp_path) == "restricted-nc"
