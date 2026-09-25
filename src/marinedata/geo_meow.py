"""Point-in-polygon lookup against the TNC/WWF Marine Ecoregions of the World (MEOW,
Spalding et al. 2007): realm / province / ecoregion for one ``(lat, lon)``.

**v2 status (2026-09-25, WP-2b):** the global MEOW polygon set (232 ecoregions, 62
provinces, 12 realms — the exact Spalding et al. 2007 attribute counts) is now vendored
at ``registry/geo/meow-2026-09-25.parquet`` (simplified to a 0.01° tolerance, WKB
geometry column). Provenance, licence text and sha256 are in ``registry/geo/SOURCES.md``.
Load it with :func:`load_polygons_parquet` (needs the optional ``geo`` extra —
``shapely`` — only at load time; the ray-casting lookup itself stays pure Python). The
GeoJSON loader below is kept for any future mirror that ships that format instead.

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


def load_polygons_parquet(path: Path) -> tuple[MeowFeature, ...]:
    """Load the vendored MEOW parquet (WKB geometry + ``realm``/``province``/``ecoregion``
    columns — see ``registry/geo/SOURCES.md``). Requires the optional ``geo`` extra
    (``shapely``) for WKB decoding only; the returned :class:`MeowFeature` rings feed the
    same dependency-free :func:`classify` as the GeoJSON path."""
    import pandas as pd
    from shapely import from_wkb
    from shapely.geometry import mapping

    df = pd.read_parquet(path)
    features = []
    for row in df.itertuples(index=False):
        geometry = mapping(from_wkb(row.wkb))
        rings = _rings_from_geometry(geometry)
        if rings:
            features.append(
                MeowFeature(
                    realm=row.realm, province=row.province, ecoregion=row.ecoregion, rings=rings
                )
            )
    return tuple(features)


def load_meow_polygons(path: Path) -> tuple[MeowFeature, ...]:
    """Dispatch on file suffix: ``.parquet`` -> :func:`load_polygons_parquet`, anything
    else -> the GeoJSON :func:`load_polygons`."""
    return load_polygons_parquet(path) if Path(path).suffix == ".parquet" else load_polygons(path)


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
