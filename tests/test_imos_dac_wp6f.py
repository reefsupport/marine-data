"""WP-6f: D-AC imos-auv selection solver on a synthetic listing (no network)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bim():
    spec = importlib.util.spec_from_file_location("bim6f", ROOT / "scripts/build_imos_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _listing() -> list[dict]:
    """3 campaigns x dives of different lengths x 2 depth bands x 2 regions."""
    rows = []
    for camp, n_dives, frames in (("A", 2, 900), ("B", 3, 400), ("C", 1, 60)):
        for d in range(n_dives):
            for seq in range(frames):
                rows.append(
                    {
                        "key": f"IMOS/AUV/{camp}/r{d}/i{d}_gtif/PR_{camp}{d}_{seq:05d}_LC16.tif",
                        "campaign": camp,
                        "dive": f"r{d}",
                        "seq": seq,
                        "depth_band": "0-20m" if seq % 3 else "20-40m",
                        "region": "Bassian" if d % 2 else "Cape Howe",
                    }
                )
    return rows


def _annotated(rows: list[dict]) -> set[str]:
    # every 7th frame of campaign A dive 0 (incl. frames the thinning would drop)
    return {
        r["key"].rsplit("/", 1)[-1][:-4]
        for r in rows
        if r["campaign"] == "A" and r["dive"] == "r0" and r["seq"] % 7 == 0
    }


def test_solve_cap_is_the_largest_cap_under_target(bim):
    cells = {("a",): (10, 50), ("b",): (0, 5), ("c",): (0, 100)}
    cap = bim.solve_cap(cells, 60)

    def total(c):
        return sum(a + min(t, max(0, c - a)) for a, t in cells.values())

    assert total(cap) <= 60 < total(cap + 1)
    assert bim.solve_cap(cells, 10**6) == 100  # everything fits -> cap = largest cell
    assert bim.solve_cap({("a",): (70, 5)}, 60) == 0  # annotated alone over target


def test_select_frames_follows_dac_order(bim):
    rows, target = _listing(), 400
    ann = _annotated(rows)
    sel, st = bim.select_frames(rows, ann, target, thin=10)
    keys = {r["key"] for r in sel}
    # (1) every annotated frame is kept, uncapped, whatever its seq
    assert {bim.stem(k) for k in keys} >= ann and st["annotated"] == len(ann)
    # (2) every non-annotated frame kept is 1-in-10 along its dive's track
    assert all(r["seq"] % 10 == 0 for r in sel if not r["squidle_annotated"])
    # (3) total under target; the solved cap is maximal
    assert st["selected"] == len(sel) <= target
    by_cell: dict = {}
    for r in sel:
        by_cell.setdefault(bim.cell_of(r), []).append(r)
    cap = st["cap"]
    assert all(
        sum(not r["squidle_annotated"] for r in v)
        <= max(0, cap - sum(r["squidle_annotated"] for r in v))
        for v in by_cell.values()
    )
    bigger, _ = bim.select_frames(rows, ann, target + 10**6, thin=10)
    assert len(bigger) > len(sel)
    # (4) within a cell the capped frames are the lowest sha256(key) of the thinned pool
    cell = ("A", "0-20m", "Cape Howe")
    pool = [
        r["key"]
        for r in rows
        if bim.cell_of(r) == cell and r["seq"] % 10 == 0 and bim.stem(r["key"]) not in ann
    ]
    got = sorted(r["key"] for r in by_cell[cell] if not r["squidle_annotated"])
    want = sorted(sorted(pool, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[: len(got)])
    assert got == want and 0 < len(got) < len(pool)


def test_select_frames_is_listing_order_independent(bim):
    rows = _listing()
    ann = _annotated(rows)
    a, _ = bim.select_frames(rows, ann, 400)
    random.Random(7).shuffle(rows)
    b, _ = bim.select_frames(rows, ann, 400)
    assert [r["key"] for r in a] == [r["key"] for r in b]


def test_meow_region_point_in_polygon_with_hole(bim, tmp_path):
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
    gj = {
        "features": [
            {
                "properties": {"ecoregion": "Sq"},
                "geometry": {"type": "Polygon", "coordinates": [square, hole]},
            },
            {
                "properties": {"ecoregion": "Far"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[20, 20], [30, 20], [30, 30], [20, 20]]]],
                },
            },
        ]
    }
    p = tmp_path / "meow.geojson"
    p.write_text(json.dumps(gj))
    meow = bim.load_meow(p)
    assert bim.meow_region(2, 2, meow) == "Sq"
    assert bim.meow_region(5, 5, meow) == "unassigned"  # in the hole
    assert bim.meow_region(22, 28, meow) == "Far"


def test_build_refuses_an_incomplete_cache(bim, tmp_path):
    (tmp_path / "_progress.json").write_text(
        json.dumps({"complete": False, "dives_cached": 3, "dives_total": 9})
    )
    with pytest.raises(SystemExit, match="incomplete"):
        bim.build(tmp_path, tmp_path / "m.parquet", tmp_path / "meow.geojson", 300_000, 10)
