"""PA-P2.6: the HF ``points`` config carries geo/date/licence columns, joined by image_sha256."""

from __future__ import annotations

import datetime as dt
import inspect
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from marinedata import hf_export as hx
from marinedata.hf_export import build_layout, export
from marinedata.hf_parquet import ConfigSpec, ExportRow
from marinedata.metadata_norm.resolvers.join import (
    EXTRA_COLUMNS,
    POINTS_COLUMNS,
    join_layout,
    join_points_config,
    join_resolved,
    load_resolved,
)
from marinedata.metadata_norm.resolvers.schema import RESOLVED_COLUMNS, ResolvedRow, write_outputs
from marinedata.metadata_release import METADATA, METADATA_COLUMNS, METADATA_SPEC, refs_from_layout
from marinedata.sample_schema import FIELD_NAMES

SampleRow = getattr(hx, "SampleRow", None) or hx.ExportSampleRow
NEW = [name for name, _ in POINTS_COLUMNS]
SHA = {i: f"{i:064x}" for i in (1, 2, 3)}


def _sample(tmp_path: Path, i: int):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (i * 20,) * 3).save(buf, format="PNG")
    path = tmp_path / "src" / f"img{i}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return SampleRow(
        task_id="t",
        raw_split="test" if i == 2 else "train",
        image_sha256=SHA[i],
        source_id="s",
        sample_key=f"images/default/img{i}.png",
        image=path,
        label="HC",
    )


def _points() -> list[dict]:
    rows = []
    for i, n in ((1, 3), (2, 2), (3, 1)):  # image 3 stays unresolved
        for k in range(n):
            quadrat = "Q-own" if (i, k) == (1, 0) else None  # the config's own value wins
            rows.append(
                dict(image_sha256=SHA[i], source_id="s", ann_id=f"s:{i}:{k}", label_native="HC")
                | dict(x=str(0.1 * (k + 1)), y="0.5", quadrat_id=quadrat)
            )
    return rows


def _layout(tmp_path: Path, generalized: frozenset[str] = frozenset()) -> dict:
    layout = build_layout(
        {"task": [_sample(tmp_path, i) for i in SHA]}, task_layers={"points": _points()}
    )
    splits: dict[str, list[ExportRow]] = {}
    for ref in refs_from_layout(layout):
        known = dict(image_sha256=ref.sha256, source_id="s", source_version="1", license="X")
        values = {name: None for name, _ in METADATA_COLUMNS} | known
        values["location_generalized"] = ref.sha256 in generalized or None
        splits.setdefault(ref.split, []).append(ExportRow(values=values))
    return {**layout, METADATA: (METADATA_SPEC, splits)}


def _resolved(tmp_path: Path) -> dict:
    rows = [
        ResolvedRow(
            sample_id=f"s/img{i}",
            source_id="s",
            source_version="1",
            partition="",
            stem=f"img{i}",
            image_path=f"images/img{i}.png",
            image_sha256=SHA[i],
            lat=12.5 + i,
            lon=-70.25,
            coord_precision_m=10.0,
            depth_m=5.5,
            date=dt.date(2023, 5, i),
            date_precision="day",
            year=2023,
            site_id=f"S{i}",
            quadrat_id=f"QR{i}",
            survey_id="SV",
            region="Caribbean",
            platform="diver-photo-quadrat",
            license="CC-BY-4.0",
            license_url="https://example.org/by",
            attribution="Reg",
            rights_holder="Holder",
            license_verified_on=dt.date(2026, 10, 1),
        )
        for i in (1, 2)
    ]
    write_outputs(tmp_path / "res", "s", "1", rows, [])
    return load_resolved(tmp_path / "res")


def _points_of(layout: dict) -> dict[str, dict]:
    return {r.values["ann_id"]: r.values for rows in layout["points"][1].values() for r in rows}


def _shards(root: Path, config: str) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if config in p.relative_to(root).parts)


def test_points_config_carries_the_columns_when_metadata_dir_is_given(tmp_path):
    layout, resolved = _layout(tmp_path), _resolved(tmp_path)
    joined = join_layout(layout, resolved)
    rows = _points_of(joined)
    assert len(rows) == 6 and rows.keys() == _points_of(layout).keys()  # no row added
    one = rows["s:1:1"]
    assert {c: one[c] for c in NEW} == dict(
        lat=13.5, lon=-70.25, date="2023-05-01", date_precision="day", coord_precision_m=10.0,
        depth_m=5.5, platform="diver-photo-quadrat", site_id="S1", quadrat_id="QR1",
        survey_id="SV", region="Caribbean", license="CC-BY-4.0",
        license_url="https://example.org/by", attribution="Reg", rights_holder="Holder",
    )  # fmt: skip
    assert rows["s:1:0"]["quadrat_id"] == "Q-own"  # fill-only: a value the config holds stays
    assert rows["s:2:0"]["site_id"] == "S2" and rows["s:2:1"]["lat"] == 14.5
    assert all(rows["s:3:0"][c] is None for c in NEW)  # unresolved image: columns null
    # one fill rule and no other config touched: join_points_config alone leaves them as they were
    only_points = join_points_config(layout, resolved)
    assert {k: v for k, v in only_points.items() if k != "points"} == {
        k: v for k, v in layout.items() if k != "points"
    }
    assert join_resolved(layout, resolved)[METADATA][1] != layout[METADATA][1]  # metadata widened


def test_points_columns_are_null_without_metadata_dir_and_the_schema_is_stable(tmp_path):
    layout, resolved = _layout(tmp_path), _resolved(tmp_path)
    plain, joined = join_layout(layout), join_layout(layout, resolved)
    assert plain[METADATA] == layout[METADATA]  # no flag: the metadata config is unchanged
    assert plain["points"][0].columns == joined["points"][0].columns
    kinds = dict(plain["points"][0].columns)
    assert [kinds[c] for c in ("lat", "lon", "coord_precision_m", "depth_m")] == ["double"] * 4
    assert all(
        kinds[c] == "string" for c in NEW if c not in ("lat", "lon", "coord_precision_m", "depth_m")
    )
    assert all(v[c] is None for v in _points_of(plain).values() for c in NEW if c != "quadrat_id")
    export(plain, tmp_path / "a")
    export(joined, tmp_path / "b")
    a, b = _shards(tmp_path / "a", "points"), _shards(tmp_path / "b", "points")
    assert a and [p.relative_to(tmp_path / "a") for p in a] == [
        p.relative_to(tmp_path / "b") for p in b
    ]
    for pa_, pb_ in zip(a, b, strict=True):  # same schema (names and types) with and without
        assert pq.read_schema(pa_).remove_metadata() == pq.read_schema(pb_).remove_metadata()
        assert set(NEW) <= set(pq.read_schema(pa_).names)
    for pa_, pb_ in zip(
        _shards(tmp_path / "a", "task"), _shards(tmp_path / "b", "task"), strict=True
    ):
        assert pq.read_table(pa_).to_pylist() == pq.read_table(pb_).to_pylist()  # images untouched


def test_points_columns_are_sample_row_v3_names(tmp_path):
    assert len(NEW) == 15 and len(set(NEW)) == 15
    assert set(NEW) <= set(FIELD_NAMES) and set(NEW) <= set(RESOLVED_COLUMNS)
    assert dict(POINTS_COLUMNS).items() <= dict(EXTRA_COLUMNS).items()  # same types as `metadata`
    layout = _layout(tmp_path)
    added = set(join_layout(layout)["points"][0].columns) - set(layout["points"][0].columns)
    assert {n for n, _ in added} <= set(FIELD_NAMES)


def test_position_is_withheld_for_location_sensitive_and_generalized_images(tmp_path):
    resolved = _resolved(tmp_path)
    sensitive = _points_of(join_layout(_layout(tmp_path), resolved, frozenset({"s"})))
    assert all(
        r[c] is None for r in sensitive.values() for c in ("lat", "lon", "coord_precision_m")
    )
    assert sensitive["s:1:1"]["site_id"] == "S1" and sensitive["s:1:1"]["date"] == "2023-05-01"
    generalized = _points_of(join_layout(_layout(tmp_path, frozenset({SHA[2]})), resolved))
    assert generalized["s:2:0"]["lat"] is None and generalized["s:2:0"]["site_id"] == "S2"
    assert generalized["s:1:1"]["lat"] == 13.5


def test_legacy_sha256_keyed_points_rows_are_joined():
    spec = ConfigSpec("points", (("sha256", "string"), ("source_id", "string")))
    row = ExportRow(values={"sha256": SHA[1], "source_id": "s"})
    out = join_points_config(
        {"points": (spec, {"test": [row]})}, {SHA[1]: {"site_id": "S9", "year": 2020}}
    )
    (joined,) = out["points"][1]["test"]
    assert joined.values["site_id"] == "S9" and "year" not in joined.values


def test_hf_export_main_joins_the_points_config_with_and_without_the_flag():
    src = inspect.getsource(hx.main)
    assert "load_resolved(args.metadata_dir) if args.metadata_dir else {}" in src
    assert "join_layout(layout, resolved" in src
