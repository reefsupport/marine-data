"""Point-in-polygon MEOW lookup: pure ray-casting, no vendored polygon set required."""

from __future__ import annotations

import json

from marinedata import geo_meow as gm

SQUARE = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]  # (lon, lat)


def _feature(**kw) -> gm.MeowFeature:
    base = dict(realm="Tropical Atlantic", province="P", ecoregion="E", rings=(SQUARE,))
    return gm.MeowFeature(**{**base, **kw})


def test_point_inside_ring_matches():
    result = gm.classify(lat=5.0, lon=5.0, polygons=[_feature()])
    assert result == gm.MeowResult("Tropical Atlantic", "P", "E")


def test_point_outside_every_ring_is_all_none():
    result = gm.classify(lat=50.0, lon=50.0, polygons=[_feature()])
    assert result == gm.MeowResult(None, None, None)


def test_first_matching_feature_wins():
    inner_square = [(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0)]
    inner = gm.MeowFeature("R2", "P2", "E2", rings=(inner_square,))
    result = gm.classify(lat=5.0, lon=5.0, polygons=[_feature(), inner])
    assert result.ecoregion == "E"


def test_load_polygons_reads_polygon_and_multipolygon(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"REALM": "R", "PROVINCE": "P", "ECOREGION": "Eco"},
                "geometry": {"type": "Polygon", "coordinates": [[*SQUARE, SQUARE[0]]]},
            },
            {
                "type": "Feature",
                "properties": {"REALM": "R2", "PROVINCE": "P2", "ECOREGION": "Eco2"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[(20.0, 20.0), (20.0, 30.0), (30.0, 30.0), (30.0, 20.0)]]],
                },
            },
        ],
    }
    path = tmp_path / "meow.geojson"
    path.write_text(json.dumps(geojson))
    features = gm.load_polygons(path)
    assert len(features) == 2
    assert gm.classify(5.0, 5.0, features).ecoregion == "Eco"
    assert gm.classify(25.0, 25.0, features).ecoregion == "Eco2"


def test_file_sha256_is_stable(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"meow")
    digest = gm.file_sha256(path)
    assert digest == gm.file_sha256(path)
    assert len(digest) == 64
