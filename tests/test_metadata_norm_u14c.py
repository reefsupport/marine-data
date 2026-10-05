"""WP-U14c: next-5 extractors, instrument defaults, spatio-temporal split_group, near-dup union."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from marinedata.metadata_norm import NORMALISERS, NormContext, normalise
from marinedata.metadata_norm import defaults as dfl
from marinedata.metadata_norm.next5 import pingmapper_fields
from marinedata.metadata_norm.split_group import derive_split_group, spatiotemporal_group
from marinedata.splitmap import load_split_map, resolve_splits, rows_to_counts

UTC = dt.UTC
PING = (
    "PINGMapperv2.0_SegmentationModelsv1.0_ImageLabelPairs.zip#Bedpick_ImgLblPairs/images/"
    "Pearl_20220303_Solix_USM1_Rec00022_wcp_ss_port_00117_be.png"
)


def _row(table, i=0):
    return table.to_pylist()[i]


def test_instrument_vocabulary_is_controlled():
    assert dfl.validated_defaults({"default_instrument": "side-scan-sonar"}) == {
        "instrument": "side-scan-sonar"
    }
    with pytest.raises(ValueError):
        dfl.validated_defaults({"default_instrument": "gopro"})
    assert dfl.validated_defaults({}) == {}


def test_registry_instrument_defaults_name_their_evidence():
    got = {
        s: dfl.validated_defaults(dfl.registry_entry(s)).get("instrument")
        for s in (
            "aris-didson-fish-td",
            "pingmapper-sss-seg",
            "aqqua-baltic-holo",
            "nes-plankton-2022",
            "sonarsweep",
            "marineevt",
        )
    }
    assert got == {
        "aris-didson-fish-td": "imaging-sonar",
        "pingmapper-sss-seg": "side-scan-sonar",
        "aqqua-baltic-holo": "holographic-imager",
        "nes-plankton-2022": "ifcb",
        "sonarsweep": None,  # mixed simulated sonar + optical + textures: not evidenced
        "marineevt": None,
    }


def test_camera_is_filled_from_the_default_with_provenance_and_staged_wins():
    ctx = NormContext(default_instrument="ifcb")
    rows = [
        {"stem": "a", "partition": "default"},
        {"stem": "b", "partition": "default", "camera": "GoPro Hero 10"},
    ]
    t = normalise("nes-plankton-2022", "v", rows, ctx).to_pylist()
    assert (t[0]["camera"], dict(t[0]["provenance"])["camera"]) == ("ifcb", "registry_default")
    assert (t[1]["camera"], dict(t[1]["provenance"])["camera"]) == ("GoPro Hero 10", "staged")
    plain = normalise("x", "v", rows[:1], NormContext()).to_pylist()[0]
    assert plain["camera"] is None and "camera" not in dict(plain["provenance"])


def test_pingmapper_recording_date_is_date_only_utc():
    got = pingmapper_fields(PING)
    assert got["capture_datetime"] == dt.datetime(2022, 3, 3, tzinfo=UTC)
    assert "date only" in got["_prov"]["capture_datetime"]
    assert pingmapper_fields("Chick_20210519_Solix_USM1_Rec00008_wcp_ss_star_00073")
    assert pingmapper_fields("Pearl_20221345_Solix_USM1_Rec00022_x") == {}  # not a calendar date
    assert pingmapper_fields("Pearl_Solix_USM1_Rec00022_x") == {}
    assert pingmapper_fields(None) == {}


def test_pingmapper_normaliser_fills_time_only_and_names_the_rule():
    assert "pingmapper-sss-seg" in NORMALISERS
    rows = [
        {"stem": "a", "partition": "default", "upstream_id": PING},
        {"stem": "b", "partition": "default", "upstream_id": "no-date.png"},
    ]
    ctx = NormContext(default_instrument="side-scan-sonar")
    t = normalise("pingmapper-sss-seg", "v1.0.0", rows, ctx)
    a, b = _row(t, 0), _row(t, 1)
    assert a["capture_datetime"] == dt.datetime(2022, 3, 3, tzinfo=UTC)
    assert "date only" in dict(a["provenance"])["capture_datetime"]
    assert (a["lat"], a["lon"], a["depth_m"]) == (None, None, None)  # a site name is no position
    assert b["capture_datetime"] is None
    assert a["camera"] == b["camera"] == "side-scan-sonar"


@pytest.mark.parametrize(
    "source_id, upstream_id",
    [
        ("marineevt", "test/Causal/x/videos.zip#videos/14BS1DEa9Yfw/frames/frame_000012.jpg"),
        ("sonarsweep", "vfov12hfov60/blue_water_visual_degraded_1_118/cropped_depth_left.png"),
        ("nes-plankton-2022", "data/train-00000.parquet#12"),
        (
            "aqqua-baltic-holo",
            "AqQua_BalticSea_St15.zip#x/Finland_April2024_St15_image_1000_crop_1.tif",
        ),
    ],
)
def test_next_four_stage_no_geo_depth_or_time(source_id, upstream_id):
    """Evidence: bucket metadata.parquet heads (2026-10-05); nothing comes from folder names."""
    row = _row(
        normalise(source_id, "v", [{"stem": "a", "upstream_id": upstream_id}], NormContext())
    )
    assert all(row[k] is None for k in ("lat", "lon", "capture_datetime", "depth_m"))


def _values(**kw):
    return {
        "lat": 10.123,
        "lon": -20.347,
        "capture_datetime": dt.datetime(2020, 1, 2, 9, tzinfo=UTC),
        **kw,
    }


def test_spatiotemporal_group_rounds_to_001_deg_and_the_utc_date():
    assert spatiotemporal_group("fathomnet", _values()) == "fathomnet:10.12,-20.35:2020-01-02"
    near = _values(
        lat=10.1249, lon=-20.3451, capture_datetime=dt.datetime(2020, 1, 2, 23, tzinfo=UTC)
    )
    assert spatiotemporal_group("fathomnet", near) == spatiotemporal_group("fathomnet", _values())
    next_day = _values(capture_datetime=dt.datetime(2020, 1, 3, tzinfo=UTC))
    assert spatiotemporal_group("fathomnet", next_day) != spatiotemporal_group(
        "fathomnet", _values()
    )
    assert spatiotemporal_group("s", _values(lat=-0.004, lon=0.004)).startswith("s:0.00,0.00:")
    assert spatiotemporal_group("s", _values(capture_datetime=None)) is None
    assert spatiotemporal_group("s", _values(lat=None)) is None
    assert spatiotemporal_group("s", _values(lon=float("nan"))) is None


def test_split_group_chain_order_with_the_new_rule():
    v = _values(image_sha256="abc")
    assert derive_split_group("fathomnet", {}, v) == (
        "fathomnet:10.12,-20.35:2020-01-02",
        "spatiotemporal:0.01deg+utc_date",
    )
    # a sequence id on the row still wins; no time -> the old sha fallback
    assert derive_split_group("fathomnet", {"dive_id": "d7"}, v)[1] == "row:site"
    assert derive_split_group("coralvqa", {}, {"image_sha256": "abc"}) == (
        "coralvqa/sha:abc",
        "image_sha256",
    )


def test_normalised_rows_of_one_dive_day_share_a_group():
    when = dt.datetime(2020, 1, 2, 9, 30, tzinfo=UTC)
    rows = [
        {"stem": "a", "lat": 10.123, "lon": -20.347, "capture_datetime": when},
        {"stem": "b", "lat": 10.124, "lon": -20.346, "capture_datetime": when.replace(hour=14)},
        {"stem": "c", "lat": 10.123, "lon": -20.347},  # no time: its own sha/row fallback
    ]
    t = normalise("fathomnet-like", "v", rows, NormContext()).to_pylist()
    assert t[0]["split_group"] == t[1]["split_group"] == "fathomnet-like:10.12,-20.35:2020-01-02"
    assert t[2]["split_group"] != t[0]["split_group"]
    assert dict(t[0]["provenance"])["split_group"] == "spatiotemporal:0.01deg+utc_date"


RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_near_dup_pair_in_different_groups_lands_in_one_split(tmp_path: Path):
    """The release builder passes dHash near-dup links into ``rows_to_counts`` (union-find over
    split_group, like a shared digest): 3 rows, A and B near-identical, C unrelated."""
    rows = [("shaA", "s:1.00,1.00:2020-01-01", None), ("shaB", "s:2.00,2.00:2020-02-02", None),
            ("shaC", "s:3.00,3.00:2020-03-03", None)]  # fmt: skip
    counts, _, info = rows_to_counts(rows, links=[("shaA", "shaB")])
    assert info.near_dup_unions == 1
    assert info.canonical[rows[0][1]] == info.canonical[rows[1][1]] != info.canonical[rows[2][1]]
    unlinked, _, _ = rows_to_counts(rows)
    differ = 0
    for seed in range(30):
        out = tmp_path / f"map-{seed}.json"
        resolve_splits(out, counts, RATIOS, seed=seed, by="group", merge_canonical=info.canonical)
        m = load_split_map(out)
        assert m is not None and m.assignments[rows[0][1]] == m.assignments[rows[1][1]]
        out2 = tmp_path / f"free-{seed}.json"
        resolve_splits(out2, unlinked, RATIOS, seed=seed, by="group")
        f = load_split_map(out2)
        differ += f is not None and f.assignments[rows[0][1]] != f.assignments[rows[1][1]]
    assert differ > 0  # without the link the pair does straddle splits for some seeds
