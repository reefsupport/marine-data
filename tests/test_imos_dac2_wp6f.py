"""D-AC2 (WP-6f-b): every annotated IMOS frame + <= N unannotated, total <= T, named by rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bim():
    spec = importlib.util.spec_from_file_location("bim6fb", ROOT / "scripts/build_imos_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n_cells: int = 6, per_cell: int = 400) -> list[dict]:
    """``per_cell`` frames per campaign cell; seq % 10 == 0 survive thinning (40 per cell)."""
    return [
        {"key": f"c{c}/d/c{c}_f{i:05d}_LC16.tif", "campaign": f"c{c}", "dive": "d", "seq": i,
         "depth_band": "0-20", "region": "R"}
        for c in range(n_cells) for i in range(per_cell)
    ]  # fmt: skip


def _ann(rows: list[dict], bim, per_cell: dict[int, int]) -> set[str]:
    """Annotate frames with seq % 10 == 5 (never thinning survivors) in the given cells."""
    out: set[str] = set()
    for c, n in per_cell.items():
        keys = [r["key"] for r in rows if r["campaign"] == f"c{c}" and r["seq"] % 10 == 5][:n]
        out |= {bim.stem(k) for k in keys}
    return out


def _check(bim, sel, stats, annotated, target, umax):
    got_ann = {bim.stem(r["key"]) for r in sel if r["squidle_annotated"]}
    assert got_ann == annotated  # annotated frames are never dropped
    n_un = sum(not r["squidle_annotated"] for r in sel)
    assert n_un == stats["unannotated"] <= umax
    assert len(sel) <= max(target, len(annotated))
    return n_un


def test_unannotated_cap_binds(bim):
    rows = _rows()
    annotated = _ann(rows, bim, {0: 30})
    sel, st = bim.select_frames(rows, annotated, target=370, thin=10, unannotated_max=150)
    n_un = _check(bim, sel, st, annotated, 370, 150)
    assert st["budget"] == 30 + 150
    # maximal: one more unit of cap would overshoot the unannotated budget
    cells = {(f"c{c}", "0-20", "R"): (30 if c == 0 else 0, 40) for c in range(6)}
    assert bim.solve_cap(cells, st["budget"] + 0) == st["cap"]
    assert bim.solve_cap(cells, 10**9) > st["cap"] and n_un > 150 - 6


def test_total_cap_binds(bim):
    rows = _rows()
    annotated = _ann(rows, bim, {0: 35, 1: 35, 2: 35})  # 105 annotated
    sel, st = bim.select_frames(rows, annotated, target=200, thin=10, unannotated_max=150)
    _check(bim, sel, st, annotated, 200, 150)
    assert st["budget"] == 200 and len(sel) <= 200 and st["unannotated"] <= 95


def test_annotated_over_target_keeps_all_and_adds_none(bim):
    rows = _rows()
    annotated = _ann(rows, bim, {c: 40 for c in range(6)})  # 240 > 200
    sel, st = bim.select_frames(rows, annotated, target=200, thin=10, unannotated_max=150)
    assert st["unannotated"] == 0 and st["annotated"] == 240 and len(sel) == 240


def test_unannotated_fill_least_annotated_cells_first(bim):
    rows = _rows(n_cells=2)
    annotated = _ann(rows, bim, {0: 40})
    sel, _st = bim.select_frames(rows, annotated, target=370, thin=10, unannotated_max=30)
    per = {
        c: sum(r["campaign"] == c and not r["squidle_annotated"] for r in sel) for c in ("c0", "c1")
    }
    assert per["c1"] >= per["c0"] and per["c1"] + per["c0"] <= 30


def test_dac_default_unchanged(bim):
    rows = _rows()
    annotated = _ann(rows, bim, {0: 30})
    sel, st = bim.select_frames(rows, annotated, target=100, thin=10)
    assert st["budget"] == 100 and st["unannotated_max"] is None and len(sel) <= 100


def test_rule_names_and_caps(bim):
    assert bim.RULES["DAC2"] == {
        "target": 370_000, "unannotated_max": 150_000, "name": "imos-auv-DAC2.parquet"
    }  # fmt: skip
    assert "imos-auv-DAC2.parquet" in (ROOT / "registry/ingest-specs/imos-auv.yaml").read_text()
