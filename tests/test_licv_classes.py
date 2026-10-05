"""LICV 2026-10-05: two sources to restricted-nc, four to unknown (fail closed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.licence_class import NC, UNKNOWN, source_class

REGISTRY = Path(__file__).resolve().parents[1] / "registry"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("coralvqa", NC),
        ("atlantis", NC),
        ("noaa-dsc-csv", UNKNOWN),
        ("noaa-oceaneyes", UNKNOWN),
        ("rf100-coral-lwptl", UNKNOWN),
        ("viame-public", UNKNOWN),
    ],
)
def test_licv_2026_10_05_classes(source, expected):
    assert source_class(source, root=REGISTRY) == expected
