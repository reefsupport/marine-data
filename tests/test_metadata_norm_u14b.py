"""WP-U14b: geo/depth/time/platform/habitat extractors, split_group, fetch_date."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from marinedata.geo_meow import MeowFeature
from marinedata.metadata_norm import NormContext, geo, normalise
from marinedata.metadata_norm import defaults as dfl
from marinedata.metadata_norm.split_group import derive_split_group
from marinedata.metadata_norm.stage import earliest_modified
from marinedata.models import SplitGroupRule

UTC = dt.UTC


def _col(table, name):
    return table.column(name).to_pylist()


def test_utc_datetime_rules():
    assert geo.utc_datetime("2009-12-13T21:44:24Z") == dt.datetime(
        2009, 12, 13, 21, 44, 24, tzinfo=UTC
    )
    assert geo.utc_datetime("2020-01-02T03:00:00+02:00").hour == 1
    assert geo.utc_datetime("2023-10-16") == dt.datetime(2023, 10, 16, tzinfo=UTC)
    assert geo.utc_datetime("2023-10-16T10:00:00") is None  # naive time: no local-time guessing
    assert geo.utc_datetime("") is None and geo.utc_datetime("garbage") is None


def test_valid_position_rejects_range_nan_and_null_island():
    assert geo.valid_position(36.0, -122.0) == (36.0, -122.0)
    for bad in ((91, 0), (0, 181), (0, 0), (None, 1), (float("nan"), 1), ("x", "y")):
        assert geo.valid_position(*bad) is None


def test_platform_vocabulary():
    assert geo.platform_from("ROV") == "rov"
    assert geo.platform_from("Benthic crawler", "AUV Sentry") == "auv"
    assert geo.platform_from("Time-lapse camera") == "lander"
    assert geo.platform_from("Doc Ricketts") is None


FATHOM = {
    "uuid": "u1", "latitude": 36.067347, "longitude": -122.297455, "depthMeters": 1631.0,
    "timestamp": "2009-12-13T21:44:24Z", "imagingType": "ROV",
    "tags": [{"key": "platform", "value": "Doc Ricketts"}], "boundingBoxes": [],
}  # fmt: skip


def test_fathomnet_row_fields_and_provenance():
    ctx = NormContext(labels={"u1": FATHOM}, source_class="nc", per_row=True)
    t = normalise("fathomnet", "v", [{"stem": "u1", "partition": "default"}], ctx)
    row = t.to_pylist()[0]
    prov = dict(row["provenance"])
    assert (row["lat"], row["lon"], row["depth_m"], row["depth_source"]) == (
        36.067347,
        -122.297455,
        1631.0,
        "sensor",
    )
    assert row["capture_datetime"] == dt.datetime(2009, 12, 13, 21, 44, 24, tzinfo=UTC)
    assert row["platform"] == "rov" and prov["platform"] == "fathomnet-json:imagingType"
    assert prov["lat"].startswith("fathomnet-json") and prov["depth_m"].endswith("depthMeters")


def test_fathomnet_bad_depth_and_position_are_null():
    rec = {**FATHOM, "depthMeters": -5, "latitude": 200}
    out = geo.fathomnet_fields(rec)
    assert "depth_m" not in out and "lat" not in out and out["capture_datetime"]


INAT = {"photo_id": 1, "observation_uuid": "o1", "latitude": -27.3333233777, "longitude": 152.7666665241,
        "observed_on": "2023-10-16", "license": "CC-BY-NC"}  # fmt: skip


def test_inat_without_geoprivacy_column_is_generalised_never_exact():
    row = normalise("inat-marine", "v", [INAT], NormContext(per_row=True)).to_pylist()[0]
    assert (row["lat"], row["lon"]) == (-27.3, 152.8)
    assert (
        row["location_generalized"] is True
        and row["gps_precision_m"] == geo.INAT_GENERALISED_PRECISION_M
    )
    assert row["capture_datetime"] == dt.datetime(2023, 10, 16, tzinfo=UTC)
    assert "date only" in dict(row["provenance"])["capture_datetime"]


@pytest.mark.parametrize(
    "flags", [{"geoprivacy": "obscured"}, {"geoprivacy": "private"}, {"coordinates_obscured": True}]
)
def test_inat_obscured_or_private_coordinates_are_null(flags):
    row = normalise("inat-marine", "v", [{**INAT, **flags}], NormContext(per_row=True)).to_pylist()[
        0
    ]
    assert row["lat"] is None and row["lon"] is None and row["capture_datetime"]


def test_inat_open_flag_keeps_exact_position_and_accuracy():
    row = normalise(
        "inat-marine",
        "v",
        [{**INAT, "geoprivacy": "", "positional_accuracy": 12}],
        NormContext(per_row=True),
    ).to_pylist()[0]
    assert row["lat"] == pytest.approx(-27.3333233777) and row["gps_precision_m"] == 12.0
    assert row["location_generalized"] is False


EVENTS = {
    "images": {"img-1": "se-1"},
    "events": {
        "se-1": {"latitude": -8.5, "longitude": 119.5, "sample_date": "2019-05-04", "depth": 7}
    },
}


def test_mermaid_join_and_no_join():
    ctx = NormContext(events=EVENTS)
    rows = [{"stem": "img-1", "partition": "default"}, {"stem": "img-2", "partition": "default"}]
    t = normalise("mermaid-aws", "v", rows, ctx)
    assert _col(t, "lat") == [-8.5, None] and _col(t, "depth_m") == [7.0, None]
    assert _col(t, "depth_source") == ["site-nominal", None]
    assert normalise("mermaid-aws", "v", rows, NormContext()).column("lat").null_count == 2


def test_registry_defaults_validated_and_fallback_provenance():
    assert dfl.validated_defaults({"default_platform": "lander", "default_habitat": "brackish_water"}) == {
        "platform": "lander", "habitat": "brackish_water"}  # fmt: skip
    with pytest.raises(ValueError):
        dfl.validated_defaults({"default_platform": "submarine"})
    with pytest.raises(ValueError):
        dfl.validated_defaults({"default_habitat": "moon"})
    ctx = NormContext(default_platform="lander", default_habitat="brackish_water")
    row = normalise("brackishmot", "v", [{"stem": "a", "partition": "default"}], ctx).to_pylist()[0]
    prov = dict(row["provenance"])
    assert (row["platform"], row["habitat"]) == ("lander", "brackish_water")
    assert prov["platform"] == prov["habitat"] == "registry_default"


def test_real_registry_carries_the_known_constants():
    got = {
        s: dfl.validated_defaults(dfl.registry_entry(s))
        for s in ("brackishmot", "aris-didson-fish-td", "coralscop-masks-rs", "csiro-cots")
    }
    assert got["brackishmot"] == {"platform": "lander", "habitat": "brackish_water"}
    assert got["aris-didson-fish-td"] == {"habitat": "river"}
    assert got["coralscop-masks-rs"] == {"habitat": "coral_reef"} == got["csiro-cots"]


def test_staged_value_beats_registry_default():
    ctx = NormContext(default_platform="lander")
    row = normalise(
        "x", "v", [{"stem": "a", "partition": "default", "platform": "diver"}], ctx
    ).to_pylist()[0]
    assert row["platform"] == "diver" and dict(row["provenance"])["platform"] == "staged"


def test_meow_lookup_only_with_polygons_and_coordinates():
    square = ((0.0, 0.0), (0.0, 20.0), (20.0, 20.0), (20.0, 0.0), (0.0, 0.0))
    feat = MeowFeature("Realm", "Prov", "Eco", (square,))
    rows = [
        {"stem": "a", "partition": "default", "lat": 5.0, "lon": 5.0},
        {"stem": "b", "partition": "default"},
    ]
    t = normalise("x", "v", rows, NormContext(meow=(feat,)))
    assert _col(t, "meow_ecoregion") == ["Eco", None]
    assert normalise("x", "v", rows, NormContext()).column("meow_ecoregion").null_count == 2


def test_half_position_is_dropped():
    row = normalise(
        "x", "v", [{"stem": "a", "partition": "default", "lat": 5.0}], NormContext()
    ).to_pylist()[0]
    assert row["lat"] is None and row["lon"] is None


# -- split_group ----------------------------------------------------------------------


def test_split_group_rule_order():
    rule = SplitGroupRule(pattern=r"(site\d+)", match_field="stem", template="cs/{group}")
    vals = {
        "stem": "site3_img",
        "upstream_split": "train",
        "upstream_path": "a/seqX/1.png",
        "image_sha256": "ab" * 32,
    }
    assert derive_split_group("s", {}, vals, rule) == ("cs/site3", "registry-rule")
    assert derive_split_group("s", {"video_id": "v9", "site_id": "z"}, vals, None) == (
        "s/video_id:v9",
        "row:sequence",
    )
    assert derive_split_group("s", {"dive_id": "d1"}, vals, None) == ("s/dive_id:d1", "row:site")
    assert derive_split_group("s", {}, vals, None) == ("s/train/seqX", "upstream_split+folder")
    assert derive_split_group("s", {}, {**vals, "upstream_path": "train/1.png"}, None) == (
        "s/sha:" + "ab" * 32,
        "image_sha256",
    )
    assert derive_split_group("s", {"partition": "p"}, {"stem": "a"}, None) == (
        "s/p",
        "registry-fallback",
    )


def test_split_group_pattern_miss_falls_through_and_is_deterministic():
    rule = SplitGroupRule(pattern=r"(site\d+)", match_field="stem", template="cs/{group}")
    args = ("s", {}, {"stem": "nomatch", "image_sha256": "cd" * 32}, rule)
    assert (
        derive_split_group(*args)
        == derive_split_group(*args)
        == ("s/sha:" + "cd" * 32, "image_sha256")
    )


def test_normalised_rows_carry_split_group_and_rule():
    row = normalise(
        "brackishmot",
        "v",
        [
            {
                "stem": "a",
                "partition": "default",
                "upstream_path": "seq1/f.jpg",
                "upstream_split": "train",
            }
        ],
        NormContext(),
    ).to_pylist()[0]
    assert row["split_group"] == "brackishmot/train/seq1"
    assert dict(row["provenance"])["split_group"] == "upstream_split+folder"


# -- fetch_date -----------------------------------------------------------------------


def test_fetch_date_registry_beats_listing_and_ingest_json_beats_both():
    assert dfl.registry_fetch_date({"retrieved": "2026-09-19"}) == (
        dt.date(2026, 9, 19),
        "registry:retrieved",
    )
    assert dfl.registry_fetch_date({"fetched": dt.date(2026, 1, 2)})[1] == "registry:fetched"
    assert dfl.registry_fetch_date({}) == (None, "")
    fb = NormContext(
        fetch_date_fallback=dt.date(2026, 9, 1), fetch_date_origin="registry:retrieved"
    )
    r = normalise("x", "v", [{"stem": "a", "partition": "default"}], fb).to_pylist()[0]
    assert (
        r["fetch_date"] == dt.date(2026, 9, 1)
        and dict(r["provenance"])["fetch_date"] == "registry:retrieved"
    )
    both = NormContext(ingest={"fetch_date": "2026-09-30"}, fetch_date_fallback=dt.date(2026, 9, 1))
    r = normalise("x", "v", [{"stem": "a", "partition": "default"}], both).to_pylist()[0]
    assert (
        r["fetch_date"] == dt.date(2026, 9, 30)
        and dict(r["provenance"])["fetch_date"] == "INGEST.json"
    )


def test_earliest_modified_is_one_page_min():
    stamps = [dt.datetime(2026, 9, 20, tzinfo=UTC), dt.datetime(2026, 9, 19, 23, tzinfo=UTC)]
    calls = []

    def list_objects_v2(**kw):
        calls.append(kw)
        return {
            "Contents": [
                {"Key": f"k{i}", "LastModified": s, "Size": 1} for i, s in enumerate(stamps)
            ]
        }

    client = SimpleNamespace(list_objects_v2=list_objects_v2)
    assert earliest_modified(client, "sources/x/v/") == (
        dt.date(2026, 9, 19),
        "s3:earliest LastModified (first page)",
    )
    assert len(calls) == 1 and calls[0]["MaxKeys"] == 1000
    empty = SimpleNamespace(list_objects_v2=lambda **kw: {})
    assert earliest_modified(empty, "p/") == (None, "")
