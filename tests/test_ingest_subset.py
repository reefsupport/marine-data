"""SPEC-w3: measured/subset blocks on ingest specs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from marinedata.ingest_source import IngestSpec
from marinedata.ingest_subset import (
    SUBSET_ITEMS,
    check_spec,
    parse_measured,
    parse_subset,
    subset_required,
    transfer_hours,
)

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "registry" / "ingest-specs"

BASE = {
    "id": "x",
    "adapter": "hf",
    "license": "CC-BY-4.0",
    "attribution": "A",
    "params": {"repo": "a/b"},
}
MEASURED = {"items": 10, "bytes": 1000, "method": "hf tree", "date": "2026-09-25"}


def test_spec_with_measured_and_subset_loads(tmp_path):
    raw = {
        **BASE,
        "measured": MEASURED,
        "subset": {"target_items": 5, "stratify": ["class"], "rule": "cap"},
    }
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(raw))
    spec = IngestSpec.load(p)
    assert spec.measured["items"] == 10
    assert spec.subset["target_items"] == 5


def test_unknown_keys_still_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({**BASE, "subsett": {}}))
    with pytest.raises(ValueError, match="unknown spec keys"):
        IngestSpec.load(p)


def test_subset_required_thresholds():
    assert not subset_required(SUBSET_ITEMS, 10**12)
    assert subset_required(SUBSET_ITEMS + 1, 0)
    assert subset_required(None, 10**12 + 1)
    assert not subset_required(None, None)


def test_transfer_hours():
    assert transfer_hours(None, 200) is None
    assert transfer_hours(720 * 10**9, 200) == pytest.approx(1.0)


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_measured({"items": 1})
    with pytest.raises(ValueError):
        parse_measured({**MEASURED, "items": -1})
    with pytest.raises(ValueError):
        parse_subset({"target_items": 0, "stratify": ["a"], "rule": "r"})
    with pytest.raises(ValueError):
        parse_subset({"target_items": 5, "stratify": [], "rule": "r"})
    assert parse_subset(None) is None


def test_check_spec_flags_missing_subset_and_oversized_target():
    big = {**BASE, "measured": {**MEASURED, "items": SUBSET_ITEMS + 1}}
    assert any("no subset block" in p for p in check_spec(big))
    over = {
        **BASE,
        "measured": MEASURED,
        "subset": {"target_items": 11, "stratify": ["a"], "rule": "r"},
    }
    assert any("> measured items" in p for p in check_spec(over))
    assert check_spec({**BASE, "measured": MEASURED}) == []
    assert check_spec(BASE) == ["no measured block"]
    nobytes = {**BASE, "measured": {**MEASURED, "bytes": None}}
    assert any("bytes_basis" in p for p in check_spec(nobytes))


def test_zero_item_dry_run_is_needs_adapter():
    spec = importlib.util.spec_from_file_location(
        "spec_dryrun", ROOT / "scripts" / "spec_dryrun.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.zero_item_kind("salmon-cage", {}) == "video"
    assert mod.zero_item_kind("new", {"measured": {"zero_item_kind": "video"}}) == "video"
    assert mod.zero_item_kind("new", {}) == "unsupported-format"


def _w3_specs():
    return [
        p
        for p in sorted(SPECS.glob("*.yaml"))
        if "measured" in (yaml.safe_load(p.read_text()) or {})
    ]


@pytest.mark.parametrize("path", _w3_specs(), ids=lambda p: p.stem)
def test_every_measured_spec_is_consistent(path):
    raw = yaml.safe_load(path.read_text())
    assert raw["id"] == path.stem
    assert check_spec(raw) == []
    if not str(raw["adapter"]).startswith("needs:"):
        IngestSpec.load(path)  # runnable specs must load with the real schema
