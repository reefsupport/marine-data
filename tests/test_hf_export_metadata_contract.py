"""P2.3 contract: ``hf_export --metadata-dir`` only adds columns (PA-P2.2)."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from marinedata import hf_export as hx
from marinedata.hf_export import build_layout, export
from marinedata.hf_parquet import ExportRow
from marinedata.metadata_norm.resolvers.join import EXTRA_COLUMNS, join_resolved, load_resolved
from marinedata.metadata_norm.resolvers.schema import ResolvedRow, write_outputs
from marinedata.metadata_release import METADATA, METADATA_COLUMNS, METADATA_SPEC, refs_from_layout

SampleRow = getattr(hx, "SampleRow", None) or hx.ExportSampleRow


def _png(path: Path, shade: int) -> Path:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buf, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


def _row(tmp_path: Path, i: int, **kw):
    image = _png(tmp_path / "src" / f"img{i}.png", i * 20)
    base = dict(
        task_id="t",
        raw_split="train",
        image_sha256=f"{i:064x}",
        source_id="s",
        sample_key=f"images/default/img{i}.png",
        image=image,
    )
    base.update(kw)
    return SampleRow(**base)


def _fixture(tmp_path: Path):
    mask = _png(tmp_path / "src" / "mask3.png", 7)
    rows = [
        _row(tmp_path, 1, label="HC"),
        _row(tmp_path, 2, label="HC", raw_split="test"),
        _row(
            tmp_path,
            3,
            label="HC",
            raw_split="probe",
            mask=mask,
            mask_values="0=unlabelled,1=HC",
            mask_class_map='{"0": null, "1": "HC"}',
        ),
    ]
    layout = build_layout({"task": rows})
    splits: dict[str, list[ExportRow]] = {}
    for ref in refs_from_layout(layout):
        values = {name: None for name, _ in METADATA_COLUMNS}
        known = dict(
            image_sha256=ref.sha256, source_id="s", source_version="1", license="CC-BY-4.0"
        )
        values.update({k: v for k, v in known.items() if k in values})
        splits.setdefault(ref.split, []).append(ExportRow(values=values))
    return {**layout, METADATA: (METADATA_SPEC, splits)}


def _resolved(tmp_path: Path) -> Path:
    rows = [
        ResolvedRow(
            sample_id=f"s/img{i}",
            source_id="s",
            source_version="1",
            partition="",
            stem=f"img{i}",
            image_path=f"images/img{i}.png",
            image_sha256=f"{i:064x}",
            lat=12.5 + i,
            lon=-70.25,
            date=dt.date(2023, 5, i),
            date_precision="day",
            year=2023,
            site_id=f"S{i}",
            platform="diver-photo-quadrat",
            license="CC-BY-4.0",
            attribution="Reg",
            license_verified_on=dt.date(2026, 10, 1),
        )
        for i in (1, 2)  # image 3 stays unresolved: its row is still exported, columns null
    ]
    write_outputs(tmp_path / "res", "s", "1", rows, [])
    return tmp_path / "res"


def _shards(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def test_metadata_dir_adds_columns_and_nothing_else(tmp_path):
    layout = _fixture(tmp_path)
    resolved = load_resolved(_resolved(tmp_path))
    export(layout, tmp_path / "a")
    export(join_resolved(layout, resolved), tmp_path / "b")
    a, b = _shards(tmp_path / "a"), _shards(tmp_path / "b")
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    meta_a: dict[str, dict] = {}
    meta_b: dict[str, dict] = {}
    for pa_, pb_ in zip(a, b, strict=True):
        ta, tb = pq.read_table(pa_), pq.read_table(pb_)
        if "metadata" in pa_.relative_to(tmp_path / "a").parts:
            meta_a.update({r["image_sha256"]: r for r in ta.to_pylist()})
            meta_b.update({r["image_sha256"]: r for r in tb.to_pylist()})
        else:  # images / task / masks: same rows, same image and mask bytes
            assert ta.to_pylist() == tb.to_pylist(), pa_.name
    assert meta_a.keys() == meta_b.keys() and len(meta_a) == 3
    new = {n for n, _ in EXTRA_COLUMNS} - {n for n, _ in METADATA_COLUMNS}
    cols_a, cols_b = set(next(iter(meta_a.values()))), set(next(iter(meta_b.values())))
    assert cols_b - cols_a == new and cols_a <= cols_b
    for sha, row in meta_a.items():  # fill-only: a non-null value never changes
        assert all(meta_b[sha][c] == v for c, v in row.items() if v is not None)
    one = meta_b[f"{1:064x}"]
    assert (one["site_id"], one["date"], one["date_precision"], one["year"]) == (
        "S1",
        "2023-05-01",
        "day",
        2023,
    )
    assert meta_b[f"{3:064x}"]["site_id"] is None


def test_no_resolved_rows_leaves_the_layout_unchanged(tmp_path):
    layout = _fixture(tmp_path)
    assert join_resolved(layout, {}) == layout


def test_position_is_never_filled_for_location_sensitive_sources(tmp_path):
    layout = _fixture(tmp_path)
    joined = join_resolved(layout, load_resolved(_resolved(tmp_path)), frozenset({"s"}))
    values = [r.values for rows in joined[METADATA][1].values() for r in rows]
    assert all(v.get("lat") is None and v.get("lon") is None for v in values)
    assert {v["site_id"] for v in values} == {"S1", "S2", None}
