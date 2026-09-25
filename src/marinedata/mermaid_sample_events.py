"""WP-2d: MERMAID's own sample-event geography, paged from the public
``/v1/summarysampleevents/`` endpoint (no auth, verified 2026-09-25 by the manager and
again here).

This does **not** currently join to ``mermaid-aws`` (the community image-classification
corpus): its images carry an ``image_id`` (a classifier-side UUID, see
``mermaid_confirmed_annotations.parquet``), and this endpoint's rows carry a
``sample_event_id``/``site_id``/``project_id`` — nothing links the two without
``GET /v1/images/<id>/``, which 401s (login required, skipped per D-E). See
``docs/geo-provenance.md`` for the full trail.

It is fetched and cached anyway, because it is a real, reusable, ready-to-join site
table (mirroring how ``geo_backfill.noaa_site_id`` was built ready before the NOAA site
table was reachable): any future MERMAID source that DOES carry a ``site_id`` or
``sample_event_id`` per image can join against :func:`site_table` immediately.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

API_BASE = "https://api.datamermaid.org/v1/summarysampleevents/"
DEFAULT_RATE_LIMIT_HZ = 2.0
_USER_AGENT = "reef-support-marine-data/1.0 (open dataset registry; WP-2d)"


@dataclass(frozen=True)
class SampleEventSite:
    sample_event_id: str
    site_id: str
    site_name: str | None
    project_id: str
    lat: float
    lon: float


def parse_page(payload: Mapping) -> list[SampleEventSite]:
    """One page's ``results`` -> records, skipping rows with no site position."""
    out = []
    for row in payload.get("results", []):
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat is None or lon is None:
            continue
        out.append(
            SampleEventSite(
                sample_event_id=row["sample_event_id"],
                site_id=row["site_id"],
                site_name=row.get("site_name"),
                project_id=row["project_id"],
                lat=float(lat),
                lon=float(lon),
            )
        )
    return out


def _default_page_fetcher(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def page_through(
    fetch_page: Callable[[str], Mapping],
    start_url: str = API_BASE,
    *,
    rate_limit_hz: float = DEFAULT_RATE_LIMIT_HZ,
    cache_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_pages: int | None = None,
) -> list[SampleEventSite]:
    """Follow ``next`` until exhausted (or ``max_pages``), at most ``rate_limit_hz``
    requests/sec. Each raw page is cached as ``page-NNNNN.json`` when ``cache_dir`` is
    given, so a re-run never re-fetches. ``fetch_page`` is injected so tests exercise the
    real paging/caching/rate-limit logic against a fixture, with no network."""
    url: str | None = start_url
    out: list[SampleEventSite] = []
    n = 0
    while url:
        if n and rate_limit_hz > 0:
            sleep(1.0 / rate_limit_hz)
        payload = fetch_page(url)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"page-{n:05d}.json").write_text(json.dumps(payload))
        out.extend(parse_page(payload))
        url = payload.get("next")
        n += 1
        if max_pages is not None and n >= max_pages:
            break
    return out


def site_table(records: Iterable[SampleEventSite]) -> dict[str, dict]:
    """``site_id -> {lat, lon}`` for :func:`marinedata.geo_backfill.table_join`. A site's
    position is fixed, so the first sample event seen for it wins."""
    out: dict[str, dict] = {}
    for r in records:
        out.setdefault(r.site_id, {"lat": r.lat, "lon": r.lon})
    return out


def fetch_live(
    cache_dir: Path,
    *,
    rate_limit_hz: float = DEFAULT_RATE_LIMIT_HZ,
    max_pages: int | None = None,
) -> list[SampleEventSite]:
    """The real, network-hitting entry point (not exercised by tests)."""
    return page_through(
        _default_page_fetcher,
        cache_dir=cache_dir,
        rate_limit_hz=rate_limit_hz,
        max_pages=max_pages,
    )
