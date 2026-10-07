"""Next-5 source extractors (WP-U14c): only what a staged field proves.

Inspected 2026-10-05 (``sources/<id>/<version>/metadata.parquet``, 1000-row heads, plus one
labels file for marineevt); the geo/depth/time evidence per source:

* ``pingmapper-sss-seg``: ``upstream_id`` ends in
  ``<site>_<YYYYMMDD>_<unit>_Rec<NNNNN>_wcp_ss_<side>_<chunk>`` (a PINGMapper recording name), so
  the survey date is staged. Date only, no position (a lake or river name, never coordinates).
* ``marineevt``: frame jpgs of videos; ``upstream_id`` is
  ``<split>/<task>/videos.zip#videos/<video id>/frames/frame_N.jpg``. No geo, depth or time; the
  QA json under ``labels/files`` is not joined to frames (``label_refs`` empty). none.
* ``sonarsweep``: simulated sonar sweeps, degraded optical frames and USD texture assets
  (``LIAS_OCEAN``). Synthetic, so none.
* ``nes-plankton-2022``: ``upstream_id`` is ``data/train-NNNNN.parquet#<row>``; the IFCB sample id
  (``D<YYYYMMDD>T<HHMMSS>_IFCB<nnn>``) is not staged, so there is no timestamp to parse. none.
* ``aqqua-baltic-holo``: ``upstream_id`` carries ``Finland_April2024_St15`` (a month and a
  station code); the per-station csv with date and location is not staged (bucket tree: images,
  INGEST, LICENSE, CHECKSUMS, metadata.parquet only). A month is not a ``capture_datetime``. none.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

_DATE = r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
_RECORDING = re.compile(rf"(?:^|[_/#])(?P<date>{_DATE})_(?:[A-Za-z0-9]+_){{1,3}}Rec\d+")


def pingmapper_fields(upstream_id: Any) -> dict[str, Any]:
    """``capture_datetime`` (UTC midnight, date only) from a PINGMapper recording name."""
    m = _RECORDING.search(str(upstream_id or ""))
    if not m:
        return {}
    try:
        day = dt.datetime.strptime(m["date"], "%Y%m%d").replace(tzinfo=dt.UTC)
    except ValueError:
        return {}
    return {
        "capture_datetime": day,
        "_prov": {"capture_datetime": "pingmapper recording name YYYYMMDD (date only)"},
    }
