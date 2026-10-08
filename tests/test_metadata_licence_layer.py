"""PA-P2.2: SampleRow v3, the per-row licence layer, the platform vocabulary, per-row verdicts."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from marinedata import sample_schema as ss
from marinedata.flavours import sample_ships, sample_verdict
from marinedata.metadata_norm.resolvers.common import finish_rows
from marinedata.metadata_norm.resolvers.licence import (
    LicenceDefaults,
    licence_columns,
    licence_defaults,
    licence_defaults_for,
    normalise_license,
)
from marinedata.metadata_norm.resolvers.schema import RESOLVED_COLUMNS, ResolvedRow
from marinedata.registry import Registry

PLATFORM_SOURCES = (
    "seaview-survey-imagery",
    "reef-support-benthic-own",
    "reef-support-bleaching",
    "noaa-pifsc-bleaching",
    "coralscapes",
    "suim",
    "uiis",
    "uiis10k",
    "usis10k",
    "roboflow-aquarium",
    "coralseg-ucsd-mosaics",
    "coralscop-masks-rs",
)


@pytest.fixture(scope="module")
def registry():
    return Registry.load()


def _row(**kw) -> ss.SampleRow:
    base = dict(
        sample_id="s/a",
        source_id="s",
        source_version="1",
        stem="a",
        image_path="images/a.png",
        image_sha256="a" * 64,
        image_bytes=10,
        image_format="png",
        license="CC-BY-4.0",
        attribution="Ann",
        fetch_date=dt.date(2026, 10, 8),
    )
    base.update(kw)
    return ss.SampleRow(**base)


def test_sample_row_v3_columns_and_older_tables_still_validate():
    assert ss.SCHEMA_VERSION == "3" and len(ss.FIELD_NAMES) == 52
    assert len(ss.V2_FIELD_NAMES) == 37 and len(ss.LEGACY_FIELD_NAMES) == 34
    row = _row(
        lat=1.0,
        lon=2.0,
        coord_precision_m=5.0,
        geo_source="exif",
        date=dt.date(2023, 5, 2),
        date_precision="day",
        year=2023,
        date_source="stem",
        site_id="S1",
        platform="diver-photo-quadrat",
        license_verified_on=dt.date(2026, 10, 1),
    )
    table = ss.to_table([row])
    ss.validate_table(table)
    assert ss.row_from_mapping(table.to_pylist()[0]) == row
    for names in (ss.V2_FIELD_NAMES, ss.LEGACY_FIELD_NAMES):
        old = table.select(list(names))
        ss.validate_table(old)
        assert ss.row_from_mapping(old.to_pylist()[0]).date is None


def test_year_only_rows_are_valid_and_inconsistent_dates_are_not():
    assert ss.validate_row(_row(year=2023, date_precision="year")) == []  # the Tetes shape
    assert any("date_precision" in e for e in ss.validate_row(_row(date=dt.date(2023, 5, 1))))
    assert any("date_precision" in e for e in ss.validate_row(_row(date_precision="week")))
    bad_year = _row(date=dt.date(2023, 5, 1), date_precision="day", year=2022)
    assert any("year" in e for e in ss.validate_row(bad_year))
    assert any("coord_precision_m" in e for e in ss.validate_row(_row(coord_precision_m=3.0)))


def test_platform_vocabulary_is_the_union_of_v3_and_legacy():
    assert {
        "diver-photo-quadrat",
        "diver-video",
        "aerial",
        "web-collected",
        "aquarium",
    } <= ss.PLATFORMS
    assert {"diver", "lander", "uav"} <= ss.PLATFORMS
    assert ss.validate_row(_row(platform="diver-video")) == []
    assert ss.validate_row(_row(platform="submarine"))


def test_normalise_license():
    assert normalise_license("CC BY-NC 4.0") == "CC-BY-NC-4.0"
    assert normalise_license("cc-by-4.0") == "CC-BY-4.0"
    assert normalise_license(" CC0-1.0 ") == "CC0-1.0"
    assert normalise_license("") is None and normalise_license(None) is None


def test_licence_defaults_come_from_the_registry_entry():
    d = licence_defaults("coralscapes")
    assert d.license and d.attribution and d.license_verified_by and d.license_verified_on
    cols = licence_columns({"upstream_id": "u-1"}, d)
    assert cols["license"] == d.license and cols["attribution"] == d.attribution
    assert cols["upstream_id"] == "u-1"


def test_per_row_override_replaces_defaults_even_with_null():
    d = LicenceDefaults(
        license="CC-BY-4.0",
        attribution="Registry attribution",
        rights_holder="Reg",
        per_row={"license": "lic", "attribution": "credit", "rights_holder": "owner"},
    )
    row = licence_columns({"lic": "CC BY-NC 4.0", "credit": "Ann (iNat)", "owner": "Ann"}, d)
    assert (row["license"], row["attribution"], row["rights_holder"]) == (
        "CC-BY-NC-4.0",
        "Ann (iNat)",
        "Ann",
    )
    blank = licence_columns({"lic": "  ", "credit": None, "owner": None}, d)
    assert blank["license"] is None and blank["attribution"] is None  # never the default
    plain = licence_columns({}, LicenceDefaults(license="CC0-1.0", attribution="x"))
    assert plain["license"] == "CC0-1.0" and plain["upstream_id"] is None
    assert licence_columns({}, None) == {}


def test_per_row_license_rejects_unknown_columns():
    with pytest.raises(ValueError, match="bogus"):
        licence_defaults_for(SimpleNamespace(per_row_license={"bogus": "x"}))


def _resolved(**kw) -> ResolvedRow:
    base = dict(
        sample_id="s/a",
        source_id="s",
        source_version="1",
        partition="p",
        stem="a",
        image_path="images/p/a.png",
        image_sha256="a" * 64,
    )
    base.update(kw)
    return ResolvedRow(**base)


def test_finish_rows_fills_platform_precision_and_licence():
    d = LicenceDefaults(license="CC-BY-4.0", attribution="Reg")
    day = dt.date(2023, 5, 2)
    rows = [
        _resolved(),
        _resolved(image_sha256="b" * 64, date=day, year=2023),
        _resolved(image_sha256="c" * 64, year=2023),  # Tetes: year only
        _resolved(image_sha256="d" * 64, platform="auv", date=day, date_precision="month"),
    ]
    out = finish_rows(
        rows,
        platform="diver-photo-quadrat",
        licence=d,
        staged_by_sha={"a" * 64: {"upstream_id": "u1"}},
    )
    assert [r.platform for r in out] == ["diver-photo-quadrat"] * 3 + ["auv"]
    assert [r.date_precision for r in out] == [None, "day", "year", "month"]
    assert out[2].date is None and out[2].year == 2023
    assert {r.license for r in out} == {"CC-BY-4.0"} and out[0].upstream_id == "u1"
    assert rows[0].platform is None  # inputs untouched
    assert {"date_precision", "license", "attribution", "upstream_id"} <= set(RESOLVED_COLUMNS)


def _registry(kinds: dict[str, tuple[bool, str]]):
    sources = {
        sid: SimpleNamespace(licence_per_row=per_row, access_class=cls)
        for sid, (per_row, cls) in kinds.items()
    }
    return SimpleNamespace(source=lambda sid: sources[sid])


def _sample(sid: str, **meta):
    return SimpleNamespace(source_id=sid, meta=meta)


def test_sample_verdict_per_row_and_null_licences():
    reg = _registry({"inat": (True, "restricted-nc"), "plain": (False, "open")})
    assert sample_verdict(reg, _sample("inat", license="CC-BY-4.0"), "open") is None
    out = "licence-outside-flavour"
    assert sample_verdict(reg, _sample("inat", license="CC-BY-NC-4.0"), "open") == out
    assert sample_verdict(reg, _sample("inat", license="CC-BY-NC-4.0"), "nc") is None
    assert sample_verdict(reg, _sample("inat", license="CC-BY-4.0"), "nc") == out
    assert sample_verdict(reg, _sample("inat", license=None), "nc") == "licence-null"
    assert sample_verdict(reg, _sample("plain", license=" "), "open") == "licence-null"
    assert sample_verdict(reg, _sample("plain"), "open") is None  # decided whole by the source
    assert sample_verdict(reg, _sample("plain", license=None), None) is None  # research build
    assert sample_ships(reg, _sample("inat", license="CC-BY-NC-4.0"), "open") is False


def test_registry_platforms_expectations_and_evidence(registry):
    for sid in PLATFORM_SOURCES:
        src = registry.source(sid)
        assert src.default_platform in ss.PLATFORMS and src.platform_evidence, sid
    assert registry.source("roboflow-aquarium").default_platform == "aquarium"
    seaview = registry.source("seaview-survey-imagery")
    assert (seaview.expect_geo, seaview.expect_date) == ("required", "required")
    noaa = registry.source("noaa-pifsc-bleaching")
    assert (noaa.expect_site, noaa.expect_date) == ("required", "none")
    own = registry.source("reef-support-benthic-own")
    assert (own.expect_site, own.expect_date) == ("required", "expected")
    caps = registry.source("coralscapes")
    assert (caps.expect_site, caps.expect_geo) == ("required", "none")
    for sid in ("uiis", "uiis10k", "usis10k", "suim"):
        src = registry.source(sid)
        assert {src.expect_geo, src.expect_date, src.expect_site, src.expect_depth} == {"none"}
