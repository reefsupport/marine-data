"""WP-2c: per-sample geography from *upstream metadata* (not EXIF).

WP-2b proved there is no EXIF GPS anywhere in v1 (0/69,600) and the originals on the
bucket are EXIF-stripped too (JFIF, no APP1 — re-checked for ``reef-support-benthic-own``
in WP-2c). Geography therefore has to come from what upstream *says* about where a
sample was taken. Every source's answer — where the coordinates live, their precision,
the join key and the evidence URL — is in ``docs/geo-provenance.md``; this module turns
the answers that yield coordinates into ``registry/geo/backfill/<source>.parquet``:

``join_key -> lat, lon, depth_m, geo_precision, geo_source, capture_datetime, platform, camera``

``join_key`` is ``"<partition>/<stem>"`` — the same ``(partition, stem)`` pair
:func:`marinedata.metadata_release.build_rows` already recovers from the cached file path
and every staged ``metadata.parquet`` (D-D) carries.

Precision (decided, WP-2c brief): ``image`` > ``site`` > ``source_centroid`` > ``none``.

* ``image`` — upstream states a position for this very image (per-image table join).
* ``site`` — upstream names the survey site/station and publishes that site's position.
* ``source_centroid`` — a *collection-level* centroid: the whole source, or one upstream
  partition of it that names a single documented place. Allowed ONLY when the
  collection's documented extent is <= :data:`CENTROID_MAX_EXTENT_KM` across
  (:func:`centroid_allowed`). A partition is a subset of its source, so a bounded
  partition keeps the same error bound the source-level rule exists to guarantee.
* ``none`` — no row is written; the release leaves lat/lon null.

Location-sensitive rounding (0.1 deg) is applied downstream in ``build_rows``, on top of
whatever precision this module yields.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GEO_PRECISIONS = ("image", "site", "source_centroid", "none")
CENTROID_MAX_EXTENT_KM = 500.0
BACKFILL_ROOT = Path(__file__).resolve().parents[2] / "registry" / "geo" / "backfill"
BACKFILL_COLUMNS = (
    "join_key",
    "lat",
    "lon",
    "depth_m",
    "geo_precision",
    "geo_source",
    "capture_datetime",
    "platform",
    "camera",
)


class GeoBackfillError(ValueError):
    """A join would claim more precision, or a wider centroid, than the rules allow."""


@dataclass(frozen=True)
class Place:
    """A documented point for a named place, with the evidence for it."""

    name: str
    lat: float
    lon: float
    extent_km: float
    evidence: str


@dataclass(frozen=True)
class GeoRecord:
    join_key: str
    lat: float | None
    lon: float | None
    geo_precision: str
    geo_source: str
    depth_m: float | None = None
    capture_datetime: str | None = None
    platform: str | None = None
    camera: str | None = None

    def __post_init__(self) -> None:
        if self.geo_precision not in GEO_PRECISIONS:
            raise GeoBackfillError(f"unknown geo_precision {self.geo_precision!r}")
        has_pos = self.lat is not None and self.lon is not None
        if (self.geo_precision == "none") == has_pos:
            raise GeoBackfillError(
                f"{self.join_key}: precision {self.geo_precision!r} vs position {has_pos}"
            )
        if has_pos and not (-90 <= self.lat <= 90 and -180 <= self.lon <= 180):
            raise GeoBackfillError(f"{self.join_key}: position out of range")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def max_extent_km(points: Sequence[tuple[float, float]]) -> float:
    """Largest pairwise great-circle distance — the 'across' in the centroid rule."""
    return max(
        (haversine_km(*a, *b) for i, a in enumerate(points) for b in points[i + 1 :]), default=0.0
    )


def centroid_allowed(extent_km: float) -> bool:
    return extent_km <= CENTROID_MAX_EXTENT_KM


def bounded_region(name: str, endpoints: Sequence[tuple[float, float]], evidence: str) -> Place:
    """A region documented by its extreme points; its centroid is the endpoint mean.
    Raises if the region is wider than the centroid rule allows."""
    extent = max_extent_km(endpoints)
    if not centroid_allowed(extent):
        raise GeoBackfillError(f"{name}: {extent:.0f} km across > {CENTROID_MAX_EXTENT_KM:.0f} km")
    lat = sum(p[0] for p in endpoints) / len(endpoints)
    lon = sum(p[1] for p in endpoints) / len(endpoints)
    return Place(name, round(lat, 4), round(lon, 4), round(extent, 1), evidence)


# --- documented places (evidence fetched 2026-09-25; see docs/geo-provenance.md) ----------

_OSM = "https://www.openstreetmap.org/"
_WP = "https://en.wikipedia.org/wiki/"
PLACES: Mapping[str, Place] = {
    "tayrona": Place(
        "Tayrona National Natural Park, Colombia",
        11.2667,
        -74.05,
        35.0,
        _WP + "Tayrona_National_Natural_Park",
    ),
    "cayo-bolivar": Place(
        "Cayo Bolivar, Courtown Cays, Colombia", 12.3996, -81.4745, 9.6, _OSM + "way/20257555"
    ),
    "courtown": Place(
        "Cayos del Este Sudeste (Courtown Cays), Colombia",
        12.4237,
        -81.4825,
        9.6,
        _OSM + "relation/10757870",
    ),
    "providencia": Place(
        "Providencia Island, Colombia",
        13.3489,
        -81.3747,
        32.0,
        _WP + "Providencia_Island,_Colombia",
    ),
    "kenya-coast": bounded_region(
        "Kenyan coast (Vanga to Kiunga)",
        [(-4.6590, 39.2180), (-1.7470, 41.4893)],
        "https://reefolution.org/ ('restored corals ... on the Kenyan coast'); endpoints "
        + _OSM
        + "way/257169285 (Vanga), "
        + _OSM
        + "node/44933213 (Kiunga)",
    ),
}

PARTITION_PLACES: Mapping[str, Mapping[str, str]] = {
    # source_id -> {staged partition (or "*" for the whole source) -> PLACES key}
    "reef-support-benthic-own": {
        "UNAL_BLEACHING_TAYRONA": "tayrona",
        "SEAFLOWER_BOLIVAR": "cayo-bolivar",
        "SEAFLOWER_COURTOWN": "courtown",
        "TETES_PROVIDENCIA": "providencia",
    },
    "reef-support-bleaching": {"UNAL_BLEACHING_TAYRONA": "tayrona"},
    "reefolution": {"*": "kenya-coast"},
}


# --- capture date from upstream filenames -------------------------------------------------

_MONTHS = {
    **{
        m: i
        for i, m in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
        )
    },
    **{
        m: i
        for i, m in enumerate(
            ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"], 1
        )
    },
}
_YMD = re.compile(r"(?<!\d)(20[0-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_DMONY = re.compile(r"(?<![a-z\d])(\d{1,2})([a-z]{3})(\d{2})(?![a-z\d])", re.I)


def date_from_stem(stem: str) -> str | None:
    """An ISO date (``YYYY-MM-DD``, date precision only — never a fake midnight) from an
    upstream filename: ``20220912_...`` or ``..._26sep22_...`` (English or Spanish month)."""
    m = _YMD.search(stem)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    for m in _DMONY.finditer(stem):
        mon = _MONTHS.get(m[2].lower())
        day = int(m[1])
        if mon and 1 <= day <= 31:
            return f"20{m[3]}-{mon:02d}-{day:02d}"
    return None


# --- the join kinds -----------------------------------------------------------------------


def partition_place_join(source_id: str, keys: Iterable[tuple[str, str]]) -> list[GeoRecord]:
    """Collection-level join: staged partition (or the whole source) -> documented place."""
    mapping = PARTITION_PLACES.get(source_id, {})
    out = []
    for partition, stem in keys:
        place_key = mapping.get(partition) or mapping.get("*")
        if place_key is None:
            continue
        place = PLACES[place_key]
        if not centroid_allowed(place.extent_km):
            raise GeoBackfillError(
                f"{source_id}/{partition}: {place.name} is {place.extent_km} km across"
            )
        scope = "source" if partition not in mapping else f"partition {partition}"
        out.append(
            GeoRecord(
                f"{partition}/{stem}",
                place.lat,
                place.lon,
                "source_centroid",
                f"{scope} -> {place.name} ({place.extent_km:g} km across); {place.evidence}",
                capture_datetime=date_from_stem(stem),
            )
        )
    return out


def table_join(
    keys: Iterable[tuple[str, str]],
    key_fn: Callable[[str, str], str | None],
    table: Mapping[str, Mapping[str, float | None]],
    precision: str,
    evidence: str,
) -> list[GeoRecord]:
    """Site- or image-level join: ``key_fn(partition, stem)`` -> an upstream id looked up
    in ``table`` (id -> {lat, lon, depth_m}). ``precision`` is ``site`` or ``image``."""
    if precision not in ("site", "image"):
        raise GeoBackfillError(f"table_join precision must be site|image, got {precision!r}")
    out = []
    for partition, stem in keys:
        upstream_id = key_fn(partition, stem)
        hit = table.get(upstream_id) if upstream_id else None
        if not hit or hit.get("lat") is None or hit.get("lon") is None:
            continue
        out.append(
            GeoRecord(
                f"{partition}/{stem}",
                float(hit["lat"]),
                float(hit["lon"]),
                precision,
                f"{precision} {upstream_id}; {evidence}",
                depth_m=hit.get("depth_m"),
                capture_datetime=date_from_stem(stem),
            )
        )
    return out


NOAA_SITE_RE = re.compile(r"^([A-Z]{3}-?[A-Z]?\d{3,4}[A-Z]?|[A-Z][A-Za-z]+?(?:Shallow|Deep))")


def noaa_site_id(partition: str, stem: str) -> str | None:
    """``MAI-B4022_2019_13_24865`` -> ``MAI-B4022``;
    ``LaehouShallow2015IMG_...`` -> ``LaehouShallow``.
    The coordinates for these ESD site ids live in NCEI accession 0269246 (unreachable
    2026-09-25) — the join is ready, the site table is not."""
    m = NOAA_SITE_RE.match(stem)
    return m[1] if m else None


# --- build / load --------------------------------------------------------------------------


def records_for_source(source_id: str, staged_metadata: Path) -> list[GeoRecord]:
    import pyarrow.parquet as pq

    if source_id not in PARTITION_PLACES:
        return []
    names = pq.read_schema(staged_metadata).names
    t = pq.read_table(staged_metadata, columns=[c for c in ("partition", "stem") if c in names])
    stems = t["stem"].to_pylist()
    # WP-6-schema staging has no `partition` column; build_rows then sees "default"
    parts = t["partition"].to_pylist() if "partition" in names else ["default"] * len(stems)
    return partition_place_join(source_id, zip(parts, stems, strict=True))


def write_backfill(
    source_id: str, records: Sequence[GeoRecord], root: Path = BACKFILL_ROOT
) -> Path | None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return None
    root.mkdir(parents=True, exist_ok=True)
    cols = {
        c: [getattr(r, c) for r in sorted(records, key=lambda r: r.join_key)]
        for c in BACKFILL_COLUMNS
    }
    types = {"lat": pa.float64(), "lon": pa.float64(), "depth_m": pa.float64()}
    table = pa.table({c: pa.array(v, type=types.get(c, pa.string())) for c, v in cols.items()})
    path = root / f"{source_id}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def load_backfill(source_id: str, root: Path = BACKFILL_ROOT) -> dict[str, dict]:
    """``join_key -> record`` for one source; ``{}`` when the source has no backfill."""
    import pyarrow.parquet as pq

    path = root / f"{source_id}.parquet"
    if not path.is_file():
        return {}
    return {r["join_key"]: r for r in pq.read_table(path).to_pylist()}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m marinedata.geo_backfill")
    parser.add_argument(
        "--staged-root", type=Path, required=True, help="<root>/<source>/<version>/metadata.parquet"
    )
    parser.add_argument("--out", type=Path, default=BACKFILL_ROOT)
    args = parser.parse_args(argv)
    written = {}
    for meta in sorted(args.staged_root.glob("*/*/metadata.parquet")):
        source_id = meta.parent.parent.name
        recs = records_for_source(source_id, meta)
        if recs and write_backfill(source_id, recs, args.out):
            written[source_id] = len(recs)
    print(json.dumps(written, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def project_geo_coverage(staged_root: Path, backfill_root: Path = BACKFILL_ROOT) -> dict:
    """WP-2c v2 projection: per staged source (latest version dir), the share of rows the
    backfill positions, by precision, plus capture-date coverage."""
    import pyarrow.parquet as pq

    out: dict[str, dict] = {}
    for src_dir in sorted(p for p in staged_root.iterdir() if p.is_dir()):
        versions = sorted(v for v in src_dir.iterdir() if (v / "metadata.parquet").is_file())
        if not versions:
            continue
        meta = versions[-1] / "metadata.parquet"
        names = pq.read_schema(meta).names
        t = pq.read_table(meta, columns=[c for c in ("partition", "stem") if c in names])
        stems = t["stem"].to_pylist()
        parts = t["partition"].to_pylist() if "partition" in names else ["default"] * len(stems)
        bf = load_backfill(src_dir.name, backfill_root)
        hits = [bf.get(f"{p}/{s}") for p, s in zip(parts, stems, strict=True)]
        prec = {
            k: sum(1 for h in hits if h and h["geo_precision"] == k) for k in GEO_PRECISIONS[:3]
        }
        out[src_dir.name] = {
            "version": versions[-1].name,
            "rows": len(stems),
            **prec,
            "dated": sum(1 for h in hits if h and h["capture_datetime"]),
        }
    return out
