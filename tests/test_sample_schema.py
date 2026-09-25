"""The canonical per-sample schema (D-K): fields, nulls, validator, parquet round-trip."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

pytest.importorskip("pyarrow")

from marinedata import sample_schema as ss


def _row(**kw) -> ss.SampleRow:
    base = dict(
        sample_id="src/a1",
        source_id="src",
        source_version="v1",
        stem="a1",
        image_path="images/a1.jpg",
        image_sha256="0" * 64,
        image_bytes=10,
        image_format="jpeg",
        license="CC-BY-4.0",
        attribution="Someone",
        fetch_date=dt.date(2026, 9, 25),
    )
    return ss.SampleRow(**{**base, **kw})


def test_wp2_fields_present():
    wanted = {
        "license",
        "attribution",
        "upstream_id",
        "upstream_url",
        "source_version",
        "fetch_date",
        "lineage_root_digest",
        "capture_datetime",
        "lat",
        "lon",
        "gps_precision_m",
        "depth_m",
        "depth_source",
        "platform",
        "camera",
        "meow_realm",
        "depth_zone",
        "habitat",
        "image_sha256",
        "width",
        "height",
        "split_hint",
        "label_refs",
    }
    assert wanted <= set(ss.FIELD_NAMES)
    assert [f.name for f in ss.arrow_schema()] == list(ss.FIELD_NAMES)


def test_valid_row_passes():
    assert (
        ss.validate_row(
            _row(
                depth_m=12.0,
                depth_source="sensor",
                depth_zone="shallow",
                lat=-18.1,
                lon=147.7,
                gps_precision_m=5.0,
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"license": ""}, "license is required"),
        ({"lat": 1.0}, "both set"),
        ({"depth_m": 50.0, "depth_source": "sensor", "depth_zone": "shallow"}, "disagrees"),
        ({"depth_m": 5.0}, "depth_source"),
        ({"platform": "submarine"}, "platform"),
        ({"stem": "a.1", "sample_id": "src/a.1"}, "WebDataset"),
        ({"image_path": "images/shard-00000.tar"}, "image_member"),
        ({"capture_datetime": dt.datetime(2020, 1, 1)}, "timezone"),
        ({"label_refs": ("masks/a.png",)}, "labels/"),
    ],
)
def test_violations(kw, needle):
    assert any(needle in e for e in ss.validate_row(_row(**kw)))


def test_depth_zones():
    assert [ss.depth_zone_for(d) for d in (None, -0.5, 10, 45, 200, 1000, 4500, 7000)] == [
        None,
        "shallow",
        "shallow",
        "mesophotic",
        "rariphotic",
        "bathyal",
        "abyssal",
        "hadal",
    ]


def test_parquet_round_trip_and_licence_filter(tmp_path):
    import pyarrow.parquet as pq

    rows = [
        _row(
            capture_datetime=dt.datetime(2021, 5, 1, 3, tzinfo=dt.timezone.utc),
            label_refs=("labels/image_labels.parquet",),
        ),
        dataclasses.replace(
            _row(), sample_id="src/b2", stem="b2", license="CC0-1.0", image_path="images/b2.jpg"
        ),
    ]
    path = tmp_path / "metadata.parquet"
    ss.write_samples(path, rows)
    table = pq.read_table(path)
    ss.validate_table(table)
    assert table.schema.metadata[b"marinedata.sample_schema"] == b"1"
    assert ss.licence_filter(table, ["CC0-1.0"]).column("stem").to_pylist() == ["b2"]
    back = ss.row_from_mapping(table.to_pylist()[0])
    assert back.label_refs == ("labels/image_labels.parquet",)


def test_every_row_needs_a_licence():
    with pytest.raises(ss.SampleSchemaError, match="license"):
        ss.validate_rows([_row(license=" ")])


def test_duplicate_sample_ids_rejected():
    with pytest.raises(ss.SampleSchemaError, match="duplicate"):
        ss.validate_rows([_row(), _row()])
