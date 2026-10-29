"""Per-row metadata resolvers: tiny fixtures per source, the output files, and one network join."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.metadata_norm.resolvers import (
    OUTPUT_COLUMNS,
    RESOLVERS,
    coralscapes,
    noaa_pifsc,
    reef_support_own,
    resolve_source,
    resolver_for,
    run,
    seaview,
)
from marinedata.metadata_norm.resolvers.exif import exif_fields, exif_segment
from marinedata.metadata_norm.resolvers.run import parse_checksums
from marinedata.metadata_norm.resolvers.schema import coverage, write_outputs
from marinedata.metadata_norm.resolvers.staged import (
    ResolveContext,
    StagedImage,
    make_getter,
    read_rows,
    staged_images,
)
from marinedata.task_layers.s3_keyed import fetch_small


def img(partition: str, stem: str, ext: str = "png", row: dict | None = None) -> StagedImage:
    sha = hashlib.sha256(f"{partition}/{stem}".encode()).hexdigest()
    return StagedImage(partition, stem, f"images/{partition}/{stem}.{ext}", sha, row or {})


def ctx(source_id: str, tables=None, heads=None, platform=None) -> ResolveContext:
    tables, heads = tables or {}, heads or {}
    return ResolveContext(
        source_id,
        "v1",
        lambda rel: tables[rel],
        lambda rel, n: heads.get(rel, b"")[:n],
        platform,
    )


def jpeg(taken: str | None = None) -> bytes:
    """A tiny JPEG: JFIF only, or with an EXIF DateTime when ``taken`` is given."""
    buf = io.BytesIO()
    if taken:
        exif = Image.Exif()
        exif[306] = taken
        Image.new("RGB", (8, 8)).save(buf, "JPEG", exif=exif)
    else:
        Image.new("RGB", (8, 8)).save(buf, "JPEG")
    return buf.getvalue()


SURVEYS = (
    b"surveyid,surveydate,ocean,country,lat_start,lng_start,lat_end,lng_end\n"
    b"11111,20200315,ATL,Belize,16.0,-88.0,16.0,-88.0005\n"
    b"22222,notadate,PAC,Fiji,,,,\n"
)
QUADRATS = (
    b"surveyid,imageid,quadratid\n"
    b"11111,111110001,11111000101\n"
    b"11111,111110002,11111000201\n"
    b"22222,222220001,22222000101\n"
)
NO_REEF = b"quadratid,lat,lng\n"


def seaview_ctx() -> ResolveContext:
    tables = {seaview.SURVEYS: SURVEYS, seaview.QUADRATS: QUADRATS}
    tables.update({rel: NO_REEF for rel in seaview.REEFCOVER})
    tables[seaview.REEFCOVER[0]] = NO_REEF + b"11111000101,16.5,-88.5\n"
    return ctx("seaview-survey-imagery", tables)


def test_registry_keys_and_unknown_source():
    assert set(RESOLVERS) == {
        "seaview-survey-imagery",
        "noaa-pifsc-bleaching",
        "reef-support-benthic-own",
        "reef-support-bleaching",
        "coralscapes",
        "coralscapes-v2",
    }
    assert resolver_for("coralscapes") is coralscapes.resolve
    with pytest.raises(KeyError, match="no metadata resolver"):
        resolver_for("nope")


def test_seaview_quadrat_gps_survey_midpoint_and_unresolved():
    stems = [
        "11111000101",
        "11111000201",
        "11111000202",
        "11111000501",
        "99999000101",
        "notanid",
        "22222000101",
    ]
    res = seaview.resolve([img("ATL", s, "jpg") for s in stems], seaview_ctx())
    by = {r.stem: r for r in res.rows}
    assert len(res.rows) == len(stems)

    gps = by["11111000101"]
    assert (gps.lat, gps.lon, gps.coord_precision_m) == (
        16.5,
        -88.5,
        seaview.QUADRAT_GPS_PRECISION_M,
    )
    assert gps.geo_source.startswith("seaviewsurvey_reefcover_atlantic.csv")
    assert (gps.date, gps.year) == (dt.date(2020, 3, 15), 2020)
    assert gps.date_source == "seaviewsurvey_surveys.csv:surveydate"
    assert (gps.region, gps.survey_id, gps.quadrat_id) == ("ATL_Belize", "11111", "11111000101")
    assert gps.depth_m is None and gps.site_id is None

    mid = by["11111000201"]
    assert mid.lat == pytest.approx(16.0) and mid.lon == pytest.approx(-88.00025)
    assert mid.coord_precision_m == pytest.approx(seaview.SURVEY_PRECISION_FLOOR_M)
    assert "survey midpoint" in mid.geo_source and "quadrat absent" not in mid.geo_source

    absent = by[
        "11111000202"
    ]  # image 111110002 is inside the survey's range, the quadrat is not listed
    assert "quadrat absent from seaviewsurvey_quadrats.csv" in absent.geo_source
    assert (absent.lat, absent.date) == (mid.lat, mid.date)

    for stem in ("11111000501", "99999000101", "notanid"):
        assert by[stem].lat is None and by[stem].date is None
    assert by["11111000501"].survey_id == "11111" and by["99999000101"].survey_id is None
    no_geo = by["22222000101"]
    assert no_geo.lat is None and no_geo.date is None and no_geo.region == "PAC_Fiji"

    assert res.notes["rules"] == {
        "quadrat_gps": 1,
        "survey_midpoint": 2,
        "image_offset": 1,
        "survey_absent": 1,
        "bad_stem": 1,
        "no_survey_coordinates": 1,
    }
    assert sorted((u.stem, u.field) for u in res.unresolved) == sorted(
        [
            ("11111000501", "geo+date"),
            ("99999000101", "geo+date"),
            ("notanid", "geo+date"),
            ("22222000101", "geo"),
            ("22222000101", "date"),
        ]
    )


def test_noaa_site_and_year_from_stem():
    images = [
        img("bleach", "Maui_Kahekili_2019_12_345", row={"split_group": "reefs/Maui_Kahekili"}),
        img("bleach", "LaehouShallow2015IMG_2684_7313"),
        img("bleach", "Oahu_Kahe_2018_1_2", row={"split_group": "Oahu"}),
        img("bleach", "weird"),
    ]
    res = noaa_pifsc.resolve(images, ctx("noaa-pifsc-bleaching"))
    assert [(r.site_id, r.year) for r in res.rows] == [
        ("Maui_Kahekili", 2019),
        ("LaehouShallow", 2015),
        ("Oahu_Kahe", 2018),
        (None, None),
    ]
    assert {r.date_source for r in res.rows[:3]} == {noaa_pifsc.DATE_SOURCE}
    assert all(r.date is None and r.lat is None and r.coord_precision_m is None for r in res.rows)
    assert [(u.stem, u.field) for u in res.unresolved] == [("weird", "site+year")]
    assert res.notes == {"site_differs_from_split_group": 1}
    cov = coverage(res.rows)
    assert (cov["site_id"]["rate"], cov["year"]["rate"], cov["lat"]["rate"]) == (0.75, 0.75, 0.0)


def test_coralscapes_site_from_frame_name():
    images = [img("train", "site13_frame_0001"), img("val", "Site7_x"), img("val", "weird")]
    res = coralscapes.resolve(images, ctx("coralscapes", platform="diver"))
    assert [r.site_id for r in res.rows] == ["site13", "site7", None]
    assert res.rows[0].platform == "diver" and res.rows[0].date is None and res.rows[0].lat is None
    assert [(u.stem, u.field) for u in res.unresolved] == [("weird", "site")]


def test_own_stem_dates_and_stations():
    assert reef_support_own.stem_date("C10_BC_PM_T1_29nov24_CDaza_corr") == (
        dt.date(2024, 11, 29),
        "stem:ddmonyy token",
    )
    assert reef_support_own.stem_date("E0_T1_C10_Corr_30sep22") == (
        dt.date(2022, 9, 30),
        "stem:ddmonyy token",
    )
    assert reef_support_own.stem_date("20220912_AnB_CB10_103_x") == (
        dt.date(2022, 9, 12),
        "stem:YYYYMMDD token",
    )
    assert reef_support_own.stem_date("G0088299") is None
    assert reef_support_own.station_of(
        "UNAL_BLEACHING_TAYRONA", "C10_BC_PM_T1_29nov24_CDaza_corr"
    ) == (
        "BC",
        "C10_BC_PM_T1",
    )
    assert reef_support_own.station_of("SEAFLOWER_BOLIVAR", "20220912_AnB_CB10_103_x") == (
        "CB10",
        None,
    )
    assert reef_support_own.station_of("SEAFLOWER_COURTOWN", "E0_T1_C10_Corr_30sep22") == (
        "E0",
        None,
    )
    assert reef_support_own.station_of("TETES_PROVIDENCIA", "G0088299") == (None, None)


def test_own_resolve_stem_exif_and_campaign_year():
    tayrona = img("UNAL_BLEACHING_TAYRONA", "C10_BC_PM_T1_29nov24_CDaza_corr", "jpg")
    bolivar = img("SEAFLOWER_BOLIVAR", "20220912_AnB_CB10_103_x", "jpg")
    exif_hit = img("TETES_PROVIDENCIA", "G0088299", "jpg")
    stripped = img("TETES_PROVIDENCIA", "G0088300", "jpg")
    other = img("SEAFLOWER_BOLIVAR", "P9150320", "jpg")
    heads = {
        exif_hit.rel_path: jpeg("2023:05:27 10:11:12"),
        stripped.rel_path: jpeg(),
        other.rel_path: jpeg(),
    }
    res = reef_support_own.resolve(
        [tayrona, bolivar, exif_hit, stripped, other],
        ctx("reef-support-benthic-own", heads=heads, platform="diver"),
    )
    by = {r.stem: r for r in res.rows}

    t = by[tayrona.stem]
    assert (t.site_id, t.quadrat_id, t.date, t.year, t.region) == (
        "UNAL_BLEACHING_TAYRONA/BC",
        "C10_BC_PM_T1",
        dt.date(2024, 11, 29),
        2024,
        "UNAL_BLEACHING_TAYRONA",
    )
    assert t.date_source == "stem:ddmonyy token" and t.platform == "diver"
    b = by[bolivar.stem]
    assert (b.site_id, b.date) == ("SEAFLOWER_BOLIVAR/CB10", dt.date(2022, 9, 12))

    e = by["G0088299"]
    assert (e.date, e.site_id) == (dt.date(2023, 5, 27), "TETES_PROVIDENCIA")
    assert e.date_source.startswith("exif:")

    s = by["G0088300"]  # no date anywhere: year only, from the Labelbox dataset name
    assert s.date is None
    assert (s.year, s.date_source) == (2023, reef_support_own.CAMPAIGN_YEAR["TETES_PROVIDENCIA"][1])

    o = by["P9150320"]  # no date, no campaign hint: stays fully unresolved
    assert (o.date, o.year, o.date_source, o.site_id) == (None, None, None, "SEAFLOWER_BOLIVAR")

    assert sorted(u.stem for u in res.unresolved) == ["G0088300", "P9150320"]
    rules = res.notes["rules"]
    assert (rules["date_from_stem"], rules["date_from_exif"], rules["year_from_campaign"]) == (
        2,
        1,
        1,
    )


def test_exif_reads_a_date_and_skips_a_stripped_jpeg():
    assert exif_segment(jpeg()) is None
    assert exif_fields(jpeg()) == (None, None, None)
    day, point, tag = exif_fields(jpeg("2021:07:04 09:00:00"))
    assert day == dt.date(2021, 7, 4) and point is None and tag


def test_write_outputs_roundtrip_and_duplicate_sha_drop(tmp_path):
    images = [img("train", "site1_a"), img("train", "site2_b"), img("train", "weird")]
    res = coralscapes.resolve(images, ctx("coralscapes"))
    target = write_outputs(
        tmp_path,
        "coralscapes",
        "1.0",
        (*res.rows, res.rows[0]),
        res.unresolved,
        {"notes": {"k": 1}},
    )
    assert target == tmp_path / "coralscapes" / "1.0"
    table = pq.read_table(target / "samples.parquet")
    assert table.column_names == list(OUTPUT_COLUMNS) and table.num_rows == 3
    report = json.loads((target / "coverage.json").read_text())
    assert (report["n_rows"], report["n_duplicate_sha256_dropped"], report["n_unresolved"]) == (
        3,
        1,
        1,
    )
    assert report["columns"]["site_id"]["rate"] == pytest.approx(2 / 3, abs=1e-5)
    assert report["notes"] == {"k": 1}
    lines = (target / "unresolved.tsv").read_text().splitlines()
    assert lines[0].split("\t") == [
        "source_id",
        "version",
        "partition",
        "stem",
        "image_sha256",
        "field",
        "reason",
    ]
    assert lines[1].split("\t")[3:5] == ["weird", images[2].sha256]


def test_resolve_source_end_to_end_on_a_fake_tree(tmp_path):
    stems, parts = ["site13_frame_0001", "site7_frame_0002"], ["train", "val"]
    meta = io.BytesIO()
    pq.write_table(pa.table({"stem": stems, "partition": parts}), meta)
    sums = [f"{hashlib.sha256(meta.getvalue()).hexdigest()}  metadata.parquet"]
    sums += [
        f"{hashlib.sha256(s.encode()).hexdigest()}  images/{p}/{s}.png"
        for s, p in zip(stems, parts, strict=True)
    ]
    tree = "sources/coralscapes/1.0/"
    store = {
        tree + "CHECKSUMS.sha256": ("\n".join(sums) + "\n").encode(),
        tree + "metadata.parquet": meta.getvalue(),
    }
    target = resolve_source(
        "coralscapes",
        tmp_path / "out",
        version="1.0",
        cache_dir=tmp_path / "cache",
        fetch=lambda key, **kw: store[key],
    )
    rows = pq.read_table(target / "samples.parquet").to_pylist()
    assert sorted((r["stem"], r["site_id"], r["partition"]) for r in rows) == [
        ("site13_frame_0001", "site13", "train"),
        ("site7_frame_0002", "site7", "val"),
    ]
    assert json.loads((target / "coverage.json").read_text())["columns"]["site_id"]["rate"] == 1.0


def test_cmd_resolve_passes_arguments_through(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake(*args):
        seen["args"] = args
        return tmp_path

    monkeypatch.setattr(run, "resolve_source", fake)
    args = argparse.Namespace(source="coralscapes", out=str(tmp_path), version=None, cache=None)
    assert run.cmd_resolve(args) == 0
    assert seen["args"] == ("coralscapes", str(tmp_path), None, None)
    assert "samples.parquet" in capsys.readouterr().out


@pytest.mark.network
def test_seaview_join_on_the_real_tables(tmp_path):
    source, version = "seaview-survey-imagery", "2026-10-06"
    tree = f"sources/{source}/{version}/"
    sums = parse_checksums(fetch_small(tree + "CHECKSUMS.sha256").decode())
    get = make_getter(tree, sums, tmp_path)
    images = [
        i for i in staged_images(read_rows(get("metadata.parquet")), sums) if i.partition == "ATL"
    ][:300]
    atlantic = seaview.REEFCOVER[0]
    assert atlantic.endswith("_atlantic.csv") and len(images) == 300

    def only_atlantic(
        rel: str,
    ) -> bytes:  # the other four reefcover files are ~100 MB; ATL needs none of them
        return get(rel) if rel in (seaview.SURVEYS, seaview.QUADRATS, atlantic) else NO_REEF

    res = seaview.resolve(images, ResolveContext(source, version, only_atlantic, checksums=sums))
    assert not res.unresolved
    assert all(-90 <= r.lat <= 90 and -180 <= r.lon <= 180 and r.date is not None for r in res.rows)
    assert all(r.survey_id == r.stem[:5] and r.quadrat_id == r.stem for r in res.rows)
    assert res.notes["rules"].get("quadrat_gps", 0) >= 0.9 * len(images)
