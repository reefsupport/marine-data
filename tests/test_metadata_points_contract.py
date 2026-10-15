"""PA-P1.9b (P2.6 Open 1): `metadata` and `points` agree on position and date for every image."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
from test_points_config_metadata import SHA, _layout

from marinedata.hf_export import export
from marinedata.metadata_norm.resolvers.join import join_layout
from marinedata.metadata_release import METADATA

CONTRACT = ("lat", "lon", "date", "date_precision")
RESOLVED = {  # the resolver output (samples.parquet) is the truth; image 3 is unresolved
    SHA[1]: dict(lat=10.5, lon=20.25, date="2020-05-17", date_precision="day"),
    SHA[2]: dict(lat=-3.0, lon=7.5, date="2021", date_precision="year"),
}
STAGED = {SHA[1]: dict(lat=-1.0, lon=-2.0, date="1999-01-01", date_precision="day")}


def _stage(layout: dict) -> dict:
    spec, splits = layout[METADATA]
    staged = {
        split: [
            replace(r, values={**r.values, **STAGED.get(r.values["image_sha256"], {})})
            for r in rows
        ]
        for split, rows in splits.items()
    }
    return {**layout, METADATA: (spec, staged)}


def _by_image(out: Path, config: str) -> dict[str, set[tuple]]:
    seen: dict[str, set[tuple]] = {}
    for path in sorted((out / "data" / config).glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            seen.setdefault(row["image_sha256"], set()).add(tuple(row.get(c) for c in CONTRACT))
    return seen


def test_metadata_and_points_agree_and_the_resolver_wins_over_a_staged_value(tmp_path):
    export(join_layout(_stage(_layout(tmp_path)), RESOLVED), tmp_path / "out")
    meta, points = _by_image(tmp_path / "out", "metadata"), _by_image(tmp_path / "out", "points")
    assert meta.keys() & points.keys() == set(SHA.values())
    for sha in SHA.values():
        assert len(meta[sha]) == 1 and meta[sha] == points[sha], sha
    assert meta[SHA[1]] == {(10.5, 20.25, "2020-05-17", "day")}  # the staged value lost
    assert meta[SHA[3]] == {(None, None, None, None)}
