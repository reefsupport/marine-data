"""WP-2c: geography from upstream metadata — one fixture per join kind, the precision
rules and the centroid-bound rule."""

import pytest

from marinedata import geo_backfill as gb


def test_partition_join_maps_named_partition_to_documented_place():
    recs = gb.partition_place_join(
        "reef-support-benthic-own",
        [("SEAFLOWER_COURTOWN", "E9_T1_C9_Corr_26sep22_"), ("UNKNOWN_DIR", "x")],
    )
    assert len(recs) == 1  # an unmapped partition stays `none` (no row)
    r = recs[0]
    assert r.join_key == "SEAFLOWER_COURTOWN/E9_T1_C9_Corr_26sep22_"
    assert r.geo_precision == "source_centroid"
    assert (r.lat, r.lon) == (gb.PLACES["courtown"].lat, gb.PLACES["courtown"].lon)
    assert "partition SEAFLOWER_COURTOWN" in r.geo_source and "openstreetmap" in r.geo_source
    assert r.capture_datetime == "2022-09-26"


def test_whole_source_join_uses_the_star_mapping():
    (r,) = gb.partition_place_join("reefolution", [("default", "DSCN0774_ACR")])
    assert r.geo_precision == "source_centroid" and r.geo_source.startswith("source ->")
    assert r.capture_datetime is None


def test_unmapped_source_yields_nothing():
    assert (
        gb.partition_place_join("roboflow-coral-reef-bleach-detection-v2i", [("default", "a")])
        == []
    )


def test_site_table_join():
    sites = {"MAI-B4022": {"lat": 20.9, "lon": -156.7, "depth_m": 8.0}}
    recs = gb.table_join(
        [("default", "MAI-B4022_2019_13_24865"), ("default", "OAH-B3023_2019_26_32716")],
        gb.noaa_site_id,
        sites,
        "site",
        "fixture",
    )
    assert [(r.join_key, r.geo_precision, r.depth_m) for r in recs] == [
        ("default/MAI-B4022_2019_13_24865", "site", 8.0)
    ]


def test_image_table_join():
    table = {"10001001601": {"lat": -16.5, "lon": 145.8}}
    (r,) = gb.table_join(
        [("SEAVIEW_PAC_AUS", "10001001601")], lambda p, s: s, table, "image", "fixture"
    )
    assert r.geo_precision == "image" and r.geo_source.startswith("image 10001001601")


def test_table_join_refuses_centroid_precision():
    with pytest.raises(gb.GeoBackfillError):
        gb.table_join([], lambda p, s: s, {}, "source_centroid", "x")


@pytest.mark.parametrize(
    ("precision", "lat", "lon"),
    [("none", 1.0, 2.0), ("site", None, None), ("bogus", 1.0, 2.0), ("image", 95.0, 0.0)],
)
def test_precision_rules_reject_inconsistent_records(precision, lat, lon):
    with pytest.raises(gb.GeoBackfillError):
        gb.GeoRecord("k", lat, lon, precision, "src")


def test_centroid_bound_rule():
    ok = gb.bounded_region("kenya", [(-4.659, 39.218), (-1.747, 41.489)], "e")
    assert ok.extent_km <= gb.CENTROID_MAX_EXTENT_KM
    assert (ok.lat, ok.lon) == (pytest.approx(-3.203, abs=1e-3), pytest.approx(40.3535, abs=1e-3))
    with pytest.raises(gb.GeoBackfillError):  # Tayrona <-> San Andres: ~830 km
        gb.bounded_region("benthic-own", [(11.2667, -74.05), (12.5833, -81.7)], "e")
    assert all(gb.centroid_allowed(p.extent_km) for p in gb.PLACES.values())


def test_oversized_place_is_refused(monkeypatch):
    big = gb.Place("too big", 0.0, 0.0, 900.0, "e")
    monkeypatch.setitem(gb.PLACES, "courtown", big)
    with pytest.raises(gb.GeoBackfillError):
        gb.partition_place_join("reef-support-benthic-own", [("SEAFLOWER_COURTOWN", "s")])


@pytest.mark.parametrize(
    ("stem", "want"),
    [
        ("20220912_AnB_CB10_103_", "2022-09-12"),
        ("C10_BC_PM_T1_29nov24_CDaza_corr", "2024-11-29"),
        ("x_3ago23_y", "2023-08-03"),
        ("G0088299", None),
        ("DSCN0774_ACR", None),
    ],
)
def test_date_from_stem(stem, want):
    assert gb.date_from_stem(stem) == want


def test_noaa_site_id():
    assert gb.noaa_site_id("default", "FFS-B009_2019_06_51") == "FFS-B009"
    assert gb.noaa_site_id("default", "LaehouShallow2015IMG_2682_7264") == "LaehouShallow"


def test_write_and_load_roundtrip(tmp_path):
    recs = gb.partition_place_join(
        "reef-support-bleaching", [("UNAL_BLEACHING_TAYRONA", "a_1nov24")]
    )
    path = gb.write_backfill("reef-support-bleaching", recs, tmp_path)
    got = gb.load_backfill("reef-support-bleaching", tmp_path)
    assert path.name == "reef-support-bleaching.parquet"
    assert got["UNAL_BLEACHING_TAYRONA/a_1nov24"]["geo_precision"] == "source_centroid"
    assert gb.load_backfill("absent", tmp_path) == {}
    assert gb.write_backfill("absent", [], tmp_path) is None


def test_project_geo_coverage_uses_latest_version(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for ver in ("2026-08", "2026-09-24"):
        d = tmp_path / "staged" / "reefolution" / ver
        d.mkdir(parents=True)
        pq.write_table(
            pa.table({"stem": ["a", "b"], "partition": ["default"] * 2}), d / "metadata.parquet"
        )
    gb.write_backfill(
        "reefolution", gb.partition_place_join("reefolution", [("default", "a")]), tmp_path / "bf"
    )
    proj = gb.project_geo_coverage(tmp_path / "staged", tmp_path / "bf")
    assert proj["reefolution"] == {
        "version": "2026-09-24",
        "rows": 2,
        "image": 0,
        "site": 0,
        "source_centroid": 1,
        "dated": 0,
    }
