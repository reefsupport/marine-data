"""The real vendored MEOW polygon set (registry/geo/meow-2026-09-25.parquet, see
registry/geo/SOURCES.md for provenance/licence). Requires the optional `geo` extra
(shapely) to decode the WKB geometry column — skipped otherwise, same convention as
other optional-dependency tests in this repo."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("shapely")

from marinedata.geo_meow import classify, load_meow_polygons

MEOW_PATH = Path(__file__).resolve().parents[1] / "registry" / "geo" / "meow-2026-09-25.parquet"

pytestmark = pytest.mark.skipif(not MEOW_PATH.is_file(), reason="MEOW parquet not vendored")


@pytest.fixture(scope="module")
def polygons():
    return load_meow_polygons(MEOW_PATH)


def test_vendored_set_covers_the_full_spalding_taxonomy(polygons):
    """The brief's exact counts: 232 ecoregions, 62 provinces, 12 realms."""
    assert len(polygons) == 232
    assert len({f.province for f in polygons}) == 62
    assert len({f.realm for f in polygons}) == 12


@pytest.mark.parametrize(
    ("name", "lat", "lon", "realm", "ecoregion"),
    [
        (
            "Great Barrier Reef",
            -18.0,
            147.0,
            "Central Indo-Pacific",
            "Central and Southern Great Barrier Reef",
        ),
        ("Belize", 17.0, -87.9, "Tropical Atlantic", "Western Caribbean"),
        ("Red Sea", 20.0, 38.5, "Western Indo-Pacific", "Northern and Central Red Sea"),
        ("Raja Ampat", -0.5, 130.5, "Central Indo-Pacific", "Papua"),
    ],
)
def test_known_reef_points_classify_correctly(polygons, name, lat, lon, realm, ecoregion):
    result = classify(lat, lon, polygons)
    assert result.realm == realm, name
    assert result.ecoregion == ecoregion, name
