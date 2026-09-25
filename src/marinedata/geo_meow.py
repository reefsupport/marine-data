"""Point-in-polygon lookup against the TNC/WWF Marine Ecoregions of the World (MEOW,
Spalding et al. 2007): realm / province / ecoregion for one ``(lat, lon)``.

**v1 status (2026-09-25, WP-2):** the global MEOW polygon set is not yet vendored into
this repo. ``docs/TAXONOMY.md`` already flagged this (Phase 4, "vendor the shapefile" —
VLIZ's WFS/geometry endpoints 404 and OBIS ``/area`` does not carry MEOW geometry). None
of the eight v1 sources carry per-image GPS (checked directly: no source's staged
``metadata.parquet`` has a location column and a sample of ``reef-support-benthic-own``
JPEGs carries no EXIF at all — GoPro stills, EXIF stripped upstream), so a real MEOW
polygon set would classify zero v1 rows today. Rather than spend the vendoring effort on
data that classifies nothing this pass, this module ships the lookup mechanism — pure
Python ray-casting, no ``shapely`` dependency — proven against a fixture polygon in
``tests/test_geo_meow.py``, so a future ingest that carries real coordinates (or the real
MEOW GeoJSON landing in ``$SP/wp2/`` or committed under ``registry/geo/``) plugs in with
no further code change. See the WP-2 report for the "needs Yohan" flag on sourcing the
polygon file (TNC's public ArcGIS `MEOW` layer or a Zenodo mirror — direct download only,
D-E: no accounts, no gated services).

GeoJSON shape expected: a ``FeatureCollection`` of polygon/multipolygon features whose
``properties`` carry ``REALM`` / ``PROVINCE`` / ``ECOREGION`` (the field names in the
canonical Spalding shapefile attribute table).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

Ring = Sequence[tuple[float, float]]
"""A linear ring: ``(lon, lat)`` pairs, GeoJSON coordinate order."""


@dataclass(frozen=True)
class MeowFeature:
    realm: str | None
    province: str | None
    ecoregion: str | None
    rings: tuple[Ring, ...]
    """Outer ring per polygon part (holes are ignored — coastline noise, never a reef)."""


@dataclass(frozen=True)
class MeowResult:
    realm: str | None
    province: str | None
    ecoregion: str | None


def _point_in_ring(lon: float, lat: float, ring: Ring) -> bool:
    """Standard even-odd ray-casting test; ``ring`` need not be closed."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_at_lat = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_at_lat:
                inside = not inside
    return inside


def _rings_from_geometry(geometry: dict) -> tuple[Ring, ...]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return (tuple(map(tuple, coords[0])),) if coords else ()
    if kind == "MultiPolygon":
        return tuple(tuple(map(tuple, poly[0])) for poly in coords if poly)
    return ()


def load_polygons(path: Path) -> tuple[MeowFeature, ...]:
    """Load a MEOW ``FeatureCollection`` GeoJSON file into lookup-ready features."""
    data = json.loads(Path(path).read_text())
    features = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        rings = _rings_from_geometry(feat.get("geometry") or {})
        if rings:
            features.append(
                MeowFeature(
                    realm=props.get("REALM"),
                    province=props.get("PROVINCE"),
                    ecoregion=props.get("ECOREGION"),
                    rings=rings,
                )
            )
    return tuple(features)


def classify(lat: float, lon: float, polygons: Sequence[MeowFeature]) -> MeowResult:
    """The first feature whose polygon contains ``(lat, lon)``; all-``None`` if none does
    (open ocean gaps and near-shore gaps are both real in the Spalding coastal/shelf
    bioregionalisation — a miss is not an error)."""
    for feature in polygons:
        if any(_point_in_ring(lon, lat, ring) for ring in feature.rings):
            return MeowResult(feature.realm, feature.province, feature.ecoregion)
    return MeowResult(None, None, None)


def file_sha256(path: Path) -> str:
    """sha256 of a vendored polygon file, for the provenance record the brief asks for."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
