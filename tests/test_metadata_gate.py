"""PA-P2.4: the metadata gate (coverage floors + licence checks) and its GATES.json merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.gates import GateResult, merge_gates_json
from marinedata.gates.metadata import check_source
from marinedata.registry import Registry


def _src(sid: str = "s", **kw):
    base = dict(
        id=sid,
        licence_per_row=False,
        access_class="open",
        expect_geo="none",
        expect_date="none",
        expect_site="none",
        expect_depth="none",
        per_row_license=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cov(
    root: Path,
    sid: str,
    rates: dict[str, float],
    n: int = 100,
    licence: bool = True,
    version: str = "1",
):
    cols = {c: {"non_null": round(r * n), "rate": r} for c, r in rates.items()}
    if licence:
        cols["license"] = {"non_null": n, "rate": 1.0}
        cols["attribution"] = {"non_null": n, "rate": 1.0}
    d = root / sid / version
    d.mkdir(parents=True)
    report = {"source_id": sid, "version": version, "n_rows": n, "columns": cols}
    (d / "coverage.json").write_text(json.dumps(report))
    return d


def test_floors_required_expected_none(tmp_path):
    src = _src(expect_geo="required", expect_date="expected", expect_site="required")
    reg = SimpleNamespace(source=lambda sid: src)
    d = _cov(
        tmp_path, "s", {"lat": 0.995, "lon": 0.995, "date": 0.94, "site_id": 0.98, "depth_m": 0.0}
    )
    info, failures = check_source(d, reg)
    assert info["checks"]["geo"]["passed"] and info["checks"]["date"]["passed"]
    assert not info["checks"]["site"]["passed"]
    assert failures == ["s: site coverage 0.9800 below the required floor 0.99"]
    assert info["checks"]["depth"]["floor"] is None and info["checks"]["depth"]["passed"]


def test_gate_result_caps_failures_and_merges_into_gates_json(tmp_path):
    many = GateResult("metadata", False, {"n": 1}, [f"f{i}" for i in range(80)])
    assert len(many.failures) == 50
    path = tmp_path / "GATES.json"
    path.write_text(json.dumps({"masks": {"passed": True, "counts": {}, "failures": []}}))
    assert set(merge_gates_json(path, [many])) == {"masks", "metadata"}
    assert json.loads(path.read_text())["metadata"]["passed"] is False
    merge_gates_json(path, [GateResult("metadata", True)])
    data = json.loads(path.read_text())
    assert data["metadata"]["passed"] is True and "masks" in data


def test_licence_checks_on_rows(tmp_path):
    reg = SimpleNamespace(source=lambda sid: _src(sid))
    d = _cov(tmp_path, "s", {})
    table = pa.table({"license": ["CC-BY-4.0", None, "CC-BY-4.0"], "attribution": ["a", "a", ""]})
    pq.write_table(table, d / "samples.parquet")
    info, failures = check_source(d, reg)
    assert info["licence"]["checked_by"] == "rows"
    assert "s: 1 rows with a null license" in failures
    assert "s: 1 rows without attribution" in failures


def test_coverage_without_licence_columns_checks_registry_defaults(tmp_path):
    d = _cov(tmp_path, "coralscapes", {"site_id": 1.0}, n=2074, licence=False, version="1.0")
    info, failures = check_source(d, Registry.load())
    assert failures == []
    assert info["licence"]["licence_checked"] is False and info["licence"]["defaults_ok"] is True


def test_cli_gate_over_resolver_output(tmp_path, capsys):
    from marinedata.cli_release import _cmd_release_gate

    res, gates = tmp_path / "res", tmp_path / "GATES.json"
    _cov(res, "coralscapes", {"site_id": 1.0}, n=2074, version="1.0")
    assert _cmd_release_gate(argparse.Namespace(metadata=str(res), gates=str(gates))) == 0
    assert json.loads(gates.read_text())["metadata"]["passed"] is True
    assert "PASS coralscapes" in capsys.readouterr().out
    _cov(res, "seaview-survey-imagery", {"lat": 0.2, "lon": 0.2, "date": 1.0}, version="2026-10-06")
    assert _cmd_release_gate(argparse.Namespace(metadata=str(res), gates=str(gates))) == 1
    metadata = json.loads(gates.read_text())["metadata"]
    assert metadata["passed"] is False
    assert any(f.startswith("seaview-survey-imagery: geo coverage") for f in metadata["failures"])
