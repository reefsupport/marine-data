"""Seaview: join a staged image's 11-digit quadrat id to the upstream tables inside the staged tree.

The id is ``<survey 5><image 4><quadrat 2>`` digits (the JPG stem). Three tables under
``labels/points/tabular-data/`` carry the metadata (all read as single anonymous GETs,
sha256-checked against the tree's CHECKSUMS):

* ``seaviewsurvey_surveys.csv``: per survey (transect) date, ocean/country and start/end
  coordinates. It has no depth column, so ``depth_m`` stays null.
* ``seaviewsurvey_quadrats.csv``: survey / image / quadrat ids only (no coordinates).
* ``seaviewsurvey_reefcover_<region>.csv``: per-quadrat GPS (``lat``, ``lng``) for the quadrats that
  were annotated.

Precision, stated honestly: a quadrat found in a reefcover table gets its own GPS with
``coord_precision_m`` = :data:`QUADRAT_GPS_PRECISION_M` (a nominal figure; the upstream tables state
none). Any other image gets the great-circle midpoint of its survey's start/end with
``coord_precision_m`` = half the start-end distance (the farthest a quadrat on a straight transect
can be), never below :data:`SURVEY_PRECISION_FLOOR_M`.

Unresolved: an image whose survey is absent from the surveys table, and an image whose image number
lies outside its survey's image range in the quadrats table (the id-offset anomaly).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .common import Resolution, miss, row_for
from .schema import ResolvedRow, Unresolved
from .staged import ResolveContext, StagedImage

TABLES = "labels/points/tabular-data/"
SURVEYS = TABLES + "seaviewsurvey_surveys.csv"
QUADRATS = TABLES + "seaviewsurvey_quadrats.csv"
REEFCOVER_REGIONS = (
    "atlantic",
    "indianocean",
    "pacificaustralia",
    "pacifichawaii",
    "southeastasia",
)
REEFCOVER = tuple(f"{TABLES}seaviewsurvey_reefcover_{r}.csv" for r in REEFCOVER_REGIONS)
QUADRAT_GPS_PRECISION_M = 10.0
SURVEY_PRECISION_FLOOR_M = 100.0
QUADRAT_ID = re.compile(r"^\d{11}$")
EARTH_RADIUS_M = 6371008.8


@dataclass(frozen=True)
class Survey:
    survey_id: str
    date: dt.date | None
    region: str
    start: tuple[float, float] | None
    end: tuple[float, float] | None


@dataclass(frozen=True)
class QuadratIndex:
    survey_of: dict[str, str]
    image_range: dict[str, tuple[str, str]]


def _rows(data: bytes) -> Iterator[dict[str, str]]:
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline=""))
    header = next(reader)
    for row in reader:
        yield dict(zip(header, row, strict=False))


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _point(lat: str | None, lon: str | None) -> tuple[float, float] | None:
    a, b = _float(lat), _float(lon)
    return (a, b) if a is not None and b is not None and abs(a) <= 90 and abs(b) <= 180 else None


def parse_surveys(data: bytes) -> dict[str, Survey]:
    out: dict[str, Survey] = {}
    for r in _rows(data):
        try:
            day = dt.datetime.strptime(r["surveydate"], "%Y%m%d").date()
        except (KeyError, ValueError):
            day = None
        out[r["surveyid"]] = Survey(
            r["surveyid"],
            day,
            f"{r.get('ocean', '')}_{r.get('country', '')}",
            _point(r.get("lat_start"), r.get("lng_start")),
            _point(r.get("lat_end"), r.get("lng_end")),
        )
    return out


def parse_quadrats(data: bytes, wanted: frozenset[str]) -> QuadratIndex:
    survey_of: dict[str, str] = {}
    lo: dict[str, str] = {}
    hi: dict[str, str] = {}
    for r in _rows(data):
        sid, img, quad = r["surveyid"], r["imageid"], r["quadratid"]
        if quad in wanted:
            survey_of[quad] = sid
        if sid not in lo or img < lo[sid]:
            lo[sid] = img
        if sid not in hi or img > hi[sid]:
            hi[sid] = img
    return QuadratIndex(survey_of, {s: (lo[s], hi[s]) for s in lo})


def parse_reefcover(data: bytes, wanted: frozenset[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for r in _rows(data):
        pt = _point(r.get("lat"), r.get("lng"))
        if r["quadratid"] in wanted and pt is not None:
            out[r["quadratid"]] = pt
    return out


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    bx, by = math.cos(p2) * math.cos(dl), math.cos(p2) * math.sin(dl)
    lat = math.atan2(math.sin(p1) + math.sin(p2), math.hypot(math.cos(p1) + bx, by))
    lon = math.radians(a[1]) + math.atan2(by, math.cos(p1) + bx)
    return math.degrees(lat), (math.degrees(lon) + 540.0) % 360.0 - 180.0


def survey_position(s: Survey) -> tuple[float, float, float] | None:
    """``(lat, lon, precision_m)`` of a survey: start/end midpoint, half-length precision."""
    if s.start is None:
        return None
    if s.end is None:
        return (*s.start, SURVEY_PRECISION_FLOOR_M)
    lat, lon = midpoint(s.start, s.end)
    return lat, lon, max(round(haversine_m(s.start, s.end) / 2), SURVEY_PRECISION_FLOOR_M)


def _unresolvable(
    img: StagedImage, sid: str, surveys: dict[str, Survey], qidx: QuadratIndex
) -> tuple[str, str] | None:
    """``(rule, reason)`` when the image cannot be tied to a survey row, else ``None``."""
    if not QUADRAT_ID.match(img.stem):
        return "bad_stem", "stem is not an 11-digit quadrat id"
    if sid not in surveys:
        return (
            "survey_absent",
            f"survey {sid} absent from seaviewsurvey_surveys.csv and seaviewsurvey_quadrats.csv",
        )
    listed = qidx.survey_of.get(img.stem)
    if listed is not None and listed != sid:
        return (
            "survey_mismatch",
            f"quadrats table puts {img.stem} in survey {listed}, stem says {sid}",
        )
    if listed is None:
        lo, hi = qidx.image_range.get(sid, ("", ""))
        if not lo or not lo <= img.stem[:9] <= hi:
            return "image_offset", (
                f"image {img.stem[:9]} outside survey {sid} image range {lo}..{hi} in "
                "seaviewsurvey_quadrats.csv (id offset from the annotation ids)"
            )
    return None


def resolve(images: Sequence[StagedImage], ctx: ResolveContext) -> Resolution:
    surveys = parse_surveys(ctx.get(SURVEYS))
    wanted = frozenset(i.stem for i in images)
    qidx = parse_quadrats(ctx.get(QUADRATS), wanted)
    gps: dict[str, tuple[float, float, str]] = {}
    for rel in REEFCOVER:
        found = parse_reefcover(ctx.get(rel), wanted - gps.keys())
        gps = {**gps, **{q: (*pt, rel.rsplit("/", 1)[-1]) for q, pt in found.items()}}
    rows: list[ResolvedRow] = []
    bad: list[Unresolved] = []
    rules: dict[str, int] = {}

    def tally(rule: str) -> None:
        rules[rule] = rules.get(rule, 0) + 1

    for img in images:
        sid = img.stem[:5]
        problem = _unresolvable(img, sid, surveys, qidx)
        if problem:
            tally(problem[0])
            bad.append(miss(img, "geo+date", problem[1]))
            rows.append(row_for(img, ctx, survey_id=sid if sid in surveys else None))
            continue
        s = surveys[sid]
        note = (
            ""
            if img.stem in qidx.survey_of
            else " [quadrat absent from seaviewsurvey_quadrats.csv; survey id from stem]"
        )
        if img.stem in gps:
            lat, lon, name = gps[img.stem]
            geo = (lat, lon, QUADRAT_GPS_PRECISION_M, f"{name}:lat,lng (quadrat GPS){note}")
            tally("quadrat_gps")
        elif (pos := survey_position(s)) is not None:
            geo = (
                pos[0],
                pos[1],
                float(pos[2]),
                f"seaviewsurvey_surveys.csv:lat/lng_start,end (survey midpoint){note}",
            )
            tally("survey_midpoint")
        else:
            geo = (None, None, None, None)
            tally("no_survey_coordinates")
            bad.append(miss(img, "geo", f"survey {sid} has no usable start coordinates"))
        if s.date is None:
            bad.append(miss(img, "date", f"survey {sid} has no parseable surveydate"))
        rows.append(
            row_for(
                img,
                ctx,
                lat=geo[0],
                lon=geo[1],
                coord_precision_m=geo[2],
                geo_source=geo[3],
                date=s.date,
                year=s.date.year if s.date else None,
                date_source=f"seaviewsurvey_surveys.csv:surveydate{note}" if s.date else None,
                survey_id=sid,
                quadrat_id=img.stem,
                region=s.region,
                platform=ctx.default_platform,
            )
        )
    tables = {rel: ctx.checksums.get(rel) for rel in (SURVEYS, QUADRATS, *REEFCOVER)}
    return Resolution(tuple(rows), tuple(bad), {"rules": rules, "tables_sha256": tables})
