"""Reef Support's own Colombian photo sets (benthic-own, bleaching): site, date, platform.

Tetes rows carry year precision only (PA-P2.2): ``year=2023``, ``date=null`` and
``date_precision=year``, because their only date source is the Labelbox ``dataset_name``
``Tetes_Providencia_May23`` (a campaign month, no day). ``date_precision`` is filled for every
resolver by ``common.finish_rows`` (``day`` when ``date`` is set, ``year`` when only ``year`` is).

Per partition (the Labelbox project folder) the stem carries the station and, mostly, the date:

* ``UNAL_BLEACHING_TAYRONA``: ``C10_BC_PM_T1_29nov24_CDaza_corr`` -> bay ``BC``, quadrat
  ``C10_BC_PM_T1``, date ``29nov24``.
* ``SEAFLOWER_BOLIVAR``: ``20220912_AnB_CB10_103_`` -> date ``20220912``, station ``CB10``.
* ``SEAFLOWER_COURTOWN``: ``E0_T1_C10_Corr_30sep22`` -> station ``E0``, date ``30sep22``.
* ``TETES_PROVIDENCIA``: GoPro / camera names (``G0088299``) carry neither; the date falls back to
  EXIF ``DateTimeOriginal`` read from a ranged GET of the first 64 KB of the JPEG. The staged JPEGs
  are re-encoded without an EXIF segment, so none resolves. ``year`` (never ``date``) then comes
  from the Labelbox dataset name ``Tetes_Providencia_May23`` (``CAMPAIGN_YEAR``) and the image stays
  in unresolved.tsv.

``site_id`` is ``<partition>/<station>``, or just ``<partition>`` when the stem has no station.
``platform`` is the registry entry's ``default_platform`` (null when the entry has none). The
Tayrona sidecar ``labelbox_segments_report.csv`` lists segment classes only, so it adds no field
here. No source table carries coordinates; EXIF GPS is used only when the fallback read finds it.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from .common import Resolution, miss, row_for
from .exif import exif_fields
from .schema import ResolvedRow, Unresolved
from .staged import EXIF_WINDOW, ResolveContext, StagedImage

MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
DMY = re.compile(r"(?<![0-9])(\d{1,2})(" + "|".join(MONTHS) + r")(\d{2})(?![0-9])", re.IGNORECASE)
YMD = re.compile(r"(?<![0-9])(20\d{2})(\d{2})(\d{2})(?![0-9])")
TAYRONA = re.compile(
    r"^(?P<cell>[A-Za-z]\d+)_(?P<bay>[A-Za-z]+)_(?P<pt>[A-Za-z]+)_(?P<tr>T\d+)(?:_|$)"
)
STATION = {
    "SEAFLOWER_BOLIVAR": re.compile(r"^\d{8}_[A-Za-z]+_(?P<st>[A-Za-z]+\d+)_"),
    "SEAFLOWER_COURTOWN": re.compile(r"^(?P<st>E\d+)_T\d+_"),
}
JPEG = {".jpg", ".jpeg"}
# Evidence: labels/exports/TETES_PROVIDENCIA/export-result.ndjson, data_row.details.dataset_name is
# "Tetes_Providencia_May23" (project "Tetes-Providencia-May23"). Campaign month only, so
# year precision.
CAMPAIGN_YEAR = {
    "TETES_PROVIDENCIA": (
        2023,
        "labelbox dataset_name Tetes_Providencia_May23 (campaign May 2023; year only)",
    ),
}
LAST_YEAR = dt.date.today().year


def stem_date(stem: str) -> tuple[dt.date, str] | None:
    """The first valid calendar date written in ``stem`` and the rule that found it."""
    for m in YMD.finditer(stem):
        try:
            d = dt.date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
        if d.year <= LAST_YEAR:
            return d, "stem:YYYYMMDD token"
    for m in DMY.finditer(stem):
        try:
            d = dt.date(2000 + int(m[3]), MONTHS[m[2].lower()], int(m[1]))
        except ValueError:
            continue
        if d.year <= LAST_YEAR:
            return d, "stem:ddmonyy token"
    return None


def station_of(partition: str, stem: str) -> tuple[str | None, str | None]:
    """``(station, quadrat_id)`` parsed from the stem; ``None`` where the stem has none."""
    if partition == "UNAL_BLEACHING_TAYRONA":
        m = TAYRONA.match(stem)
        return (m["bay"], f"{m['cell']}_{m['bay']}_{m['pt']}_{m['tr']}") if m else (None, None)
    rule = STATION.get(partition)
    m = rule.match(stem) if rule else None
    return (m["st"], None) if m else (None, None)


def _exif(img: StagedImage, ctx: ResolveContext):
    if ctx.get_range is None or PurePosixPath(img.rel_path).suffix.lower() not in JPEG:
        return None, None, None
    return exif_fields(ctx.get_range(img.rel_path, EXIF_WINDOW))


def resolve(images: Sequence[StagedImage], ctx: ResolveContext) -> Resolution:
    rows: list[ResolvedRow] = []
    bad: list[Unresolved] = []
    counts = {
        "date_from_stem": 0,
        "date_from_exif": 0,
        "exif_gps": 0,
        "year_from_campaign": 0,
        "site_differs_from_split_group": 0,
    }
    for img in images:
        station, quadrat = station_of(img.partition, img.stem)
        group = str(img.row.get("split_group") or "")
        if station and group and group.rsplit("/", 1)[-1] != station:
            counts["site_differs_from_split_group"] += 1
        found = stem_date(img.stem)
        day, source, point = (found[0], found[1], None) if found else (None, None, None)
        if found:
            counts["date_from_stem"] += 1
        else:
            exif_day, point, tag = _exif(img, ctx)
            if exif_day is not None:
                day, source = exif_day, f"exif:{tag} (ranged GET, first 64 KB)"
                counts["date_from_exif"] += 1
        year, year_source = (day.year, source) if day else (None, None)
        if day is None:
            campaign = CAMPAIGN_YEAR.get(img.partition)
            if campaign:
                year, year_source = campaign
                counts["year_from_campaign"] += 1
            why = "no day-level date in the stem or EXIF" + (
                "; year only, from the Labelbox dataset name" if campaign else ""
            )
            bad.append(miss(img, "date", why))
        if point is not None:
            counts["exif_gps"] += 1
        rows.append(
            row_for(
                img,
                ctx,
                lat=point[0] if point else None,
                lon=point[1] if point else None,
                geo_source="exif:GPS (ranged GET, first 64 KB)" if point else None,
                date=day,
                year=year,
                date_source=year_source,
                site_id=f"{img.partition}/{station}" if station else img.partition,
                quadrat_id=quadrat,
                region=img.partition,
                platform=ctx.default_platform,
            )
        )
    return Resolution(
        tuple(rows),
        tuple(bad),
        {"rules": counts, "exif_requests": ctx.ledger.requests, "exif_bytes": ctx.ledger.bytes},
    )
