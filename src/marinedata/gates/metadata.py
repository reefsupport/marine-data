"""The metadata gate (PA-P2.4): per-source coverage floors plus the per-row licence checks.

Inputs are the resolver outputs ``<dir>/<source>/<version>/coverage.json`` (and ``samples.parquet``
when it sits beside it) and the registry's ``expect_geo / expect_date / expect_site /
expect_depth``:
``required`` (floor 0.99), ``expected`` (floor 0.90), ``none`` (not checked). A missing column is
coverage 0. Licence checks: no null ``license``, ``attribution`` on every row, and a row's licence
class never differing from a non-per-row source's registry class (a stricter row would be dropped
silently). ``samples.parquet`` gives row-level checks; ``coverage.json`` alone gives the null checks
when it carries the licence columns, else only the registry defaults are checked
(``licence_checked``
is then false and the source is listed under ``licence_unchecked``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..licence_class import coerce_class, resolve_row_class
from ..metadata_norm.resolvers.licence import licence_defaults_for
from . import GateResult

NAME = "metadata"
FLOORS = {"required": 0.99, "expected": 0.90}
EXPECTATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("geo", "expect_geo", ("lat", "lon")),
    ("date", "expect_date", ("date",)),
    ("site", "expect_site", ("site_id",)),
    ("depth", "expect_depth", ("depth_m",)),
)


def source_dirs(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/coverage.json"))


def _licence_from_rows(path: Path, source, failures: list[str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    src_class = coerce_class(source.access_class)
    classes: Counter = Counter()
    null_licence = no_attribution = mismatch = 0
    for rec in pq.read_table(path).to_pylist():
        lic, attribution = rec.get("license"), rec.get("attribution")
        if not (lic and str(lic).strip()):
            null_licence += 1
        if not (attribution and str(attribution).strip()):
            no_attribution += 1
        cls = resolve_row_class(src_class, lic, per_row=bool(source.licence_per_row))
        classes[cls] += 1
        if not source.licence_per_row and cls != src_class:
            mismatch += 1
    for n, what in (
        (null_licence, "rows with a null license"),
        (no_attribution, "rows without attribution"),
        (mismatch, f"rows whose licence class differs from the registry class {src_class!r}"),
    ):
        if n:
            failures.append(f"{source.id}: {n} {what}")
    shipped = {"open": classes.get("open", 0), "nc": classes.get("restricted-nc", 0)}
    return {
        "licence_checked": True,
        "checked_by": "rows",
        "null_license": null_licence,
        "null_attribution": no_attribution,
        "class_mismatch": mismatch,
        "rows_by_flavour": {**shipped, "dropped": sum(classes.values()) - sum(shipped.values())},
    }


def _licence_from_coverage(cov: dict, source, failures: list[str]) -> dict[str, Any]:
    cols, n = cov.get("columns", {}), int(cov.get("n_rows", 0))
    if "license" not in cols or "attribution" not in cols:
        defaults = licence_defaults_for(source)
        ok = bool(defaults.license) and bool(defaults.attribution)
        if not ok and not source.per_row_license:
            failures.append(f"{source.id}: the registry default license/attribution is null")
        return {
            "licence_checked": False,
            "defaults_ok": ok,
            "reason": "coverage.json predates the licence columns; registry defaults checked",
        }
    null_licence = n - int(cols["license"]["non_null"])
    no_attribution = n - int(cols["attribution"]["non_null"])
    for count, what in ((null_licence, "null license"), (no_attribution, "no attribution")):
        if count:
            failures.append(f"{source.id}: {count} rows with {what}")
    return {
        "licence_checked": True,
        "checked_by": "coverage",
        "null_license": null_licence,
        "null_attribution": no_attribution,
    }


def check_source(directory: Path, registry) -> tuple[dict[str, Any], list[str]]:
    cov = json.loads((directory / "coverage.json").read_text())
    sid, failures = cov["source_id"], []
    try:
        source = registry.source(sid)
    except Exception:  # an id the registry does not know: one failure, no further checks
        info = {"version": cov.get("version"), "rows": cov.get("n_rows")}
        return info, [f"{sid}: not in the registry"]
    info: dict[str, Any] = {"version": cov.get("version"), "rows": cov.get("n_rows"), "checks": {}}
    for name, attr, columns in EXPECTATIONS:
        level = getattr(source, attr)
        rate = min(float(cov["columns"].get(c, {}).get("rate", 0.0)) for c in columns)
        floor = FLOORS.get(level)
        ok = floor is None or rate >= floor
        info["checks"][name] = {"level": level, "rate": rate, "floor": floor, "passed": ok}
        if not ok:
            failures.append(f"{sid}: {name} coverage {rate:.4f} below the {level} floor {floor}")
    rows = directory / "samples.parquet"
    info["licence"] = (
        _licence_from_rows(rows, source, failures)
        if rows.is_file()
        else _licence_from_coverage(cov, source, failures)
    )
    return info, failures


def run_gate(directories: Sequence[Path], registry) -> GateResult:
    sources: dict[str, Any] = {}
    failures: list[str] = []
    for directory in directories:
        info, bad = check_source(directory, registry)
        info["passed"] = not bad
        sources[directory.parent.name] = info
        failures += bad
    unchecked = sorted(
        sid for sid, i in sources.items() if not i.get("licence", {}).get("licence_checked", True)
    )
    counts = {"sources": sources, "n_sources": len(sources), "licence_unchecked": unchecked}
    return GateResult(NAME, not failures and bool(sources), counts, failures)
