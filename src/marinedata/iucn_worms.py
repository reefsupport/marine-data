"""WP-2b: IUCN Red List category via WoRMS, keyed by AphiaID.

WoRMS (`marinespecies.org`) carries IUCN Red List Category as one of several
"biology/conservation" attributes exposed by `AphiaAttributesByAphiaID`. This module
resolves that one attribute for every AphiaID in WP-7's taxonomy table
(`registry/taxonomy/worms-2026-09-25.parquet`, read-only — WP-7 owns that file) and
writes `registry/conservation/iucn-via-worms-2026-09-25.parquet`.

Rate limit: WoRMS' terms ask for polite use; this module caps at 2 requests/second
(`RATE_LIMIT_HZ`) and caches every raw response to `cache_dir` so a re-run never
re-fetches an AphiaID it already has.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

IUCN_ATTRIBUTE_NAME = "IUCN Red List Category"
RATE_LIMIT_HZ = 2.0
API_URL = "https://www.marinespecies.org/rest/AphiaAttributesByAphiaID/{aphia_id}"

# WoRMS' own category codes -> the two we gate on. Kept as a module constant so the
# location-sensitivity gate (metadata_release.is_location_sensitive) can import it
# without re-deriving the string set.
SENSITIVE_CATEGORIES = frozenset({"CR", "EN"})

# WoRMS' `measurementValue` for this attribute comes back as the IUCN's full English
# category name (verified live, 2026-09-25: "Critically Endangered", "Least Concern",
# etc — never the bare "CR"/"EN" code). Normalize to the standard IUCN abbreviation so
# `category` is comparable against SENSITIVE_CATEGORIES regardless of which form a
# given WoRMS record happens to use.
_CATEGORY_TO_CODE = {
    "critically endangered": "CR",
    "endangered": "EN",
    "vulnerable": "VU",
    "near threatened": "NT",
    "least concern": "LC",
    "data deficient": "DD",
    "not evaluated": "NE",
    "extinct": "EX",
    "extinct in the wild": "EW",
}


def _normalize_category(raw: str) -> str:
    return _CATEGORY_TO_CODE.get(raw.strip().lower(), raw.strip())


def _extract_category(attributes: list[dict]) -> str | None:
    """Walk the (nested) AphiaAttributes tree for the IUCN category leaf value,
    normalized to the standard abbreviation (see ``_CATEGORY_TO_CODE``)."""
    stack = list(attributes or [])
    while stack:
        node = stack.pop()
        if node.get("measurementType") == IUCN_ATTRIBUTE_NAME:
            value = node.get("measurementValue")
            if value:
                return _normalize_category(str(value))
        children = node.get("children") or []
        stack.extend(children)
    return None


def fetch_iucn_category(
    aphia_id: int, *, cache_dir: Path, timeout: float = 15.0
) -> str | None:
    """One AphiaID -> its IUCN category, or ``None`` if WoRMS has no such attribute.
    Cached to ``cache_dir/{aphia_id}.json`` so a retried run makes zero new requests
    for AphiaIDs it already resolved."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{aphia_id}.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text())
    else:
        url = API_URL.format(aphia_id=aphia_id)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 204:  # WoRMS: no attributes at all for this AphiaID
                raw = b"[]"
            else:
                raise
        payload = json.loads(raw or b"[]")
        cache_file.write_text(json.dumps(payload))
    return _extract_category(payload)


def build_iucn_table(
    aphia_ids: list[int],
    *,
    cache_dir: Path,
    rate_limit_hz: float = RATE_LIMIT_HZ,
) -> list[dict]:
    """Fetch (or read from cache) every id in ``aphia_ids``, sleeping between live
    network calls only (cache hits are free) to stay under ``rate_limit_hz``."""
    import datetime as dt

    rows = []
    min_interval = 1.0 / rate_limit_hz
    for aphia_id in aphia_ids:
        cache_file = cache_dir / f"{aphia_id}.json"
        was_cached = cache_file.is_file()
        category = fetch_iucn_category(aphia_id, cache_dir=cache_dir)
        if not was_cached:
            time.sleep(min_interval)
        rows.append(
            {
                "aphia_id": aphia_id,
                "category": category,
                "retrieved_at": dt.datetime.now(dt.UTC).date().isoformat(),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(prog="python -m marinedata.iucn_worms")
    parser.add_argument("--worms", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    taxonomy = pd.read_parquet(args.worms, columns=["aphia_id"])
    aphia_ids = sorted(set(taxonomy["aphia_id"].dropna().astype(int).tolist()))
    rows = build_iucn_table(aphia_ids, cache_dir=args.cache_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(json.dumps({"aphia_ids": len(aphia_ids), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
