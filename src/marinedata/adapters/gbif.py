"""``gbif-occurrence-media`` adapter (WP-6e-B, D-AB): in-situ GBIF occurrence images,
excluding iNaturalist, at most ``per_species_cap`` (100) per species, via the ANONYMOUS
occurrence search API (bulk downloads need an account: never used).

Enumeration: one ``facet=datasetKey`` call over the marine clade ``taxon_keys``, then
every non-excluded dataset in sorted key order, paged ``limit=300``; a dataset over the
100k offset window is split by ``year`` facets. The first StillImage per occurrence is
the item; lat/lon go to schema fields, date/licence/taxon/ids to inline labels."""

from __future__ import annotations

import datetime as dt
import urllib.parse
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Any

from . import BaseAdapter, RemoteItem
from ._http import get_json
from .manifest import RowJoinMixin

API = "https://api.gbif.org/v1/occurrence/search"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
OFFSET_WINDOW = 100_000
_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/tiff": ".tif",
    "image/webp": ".webp",
}


def media_ext(media: Mapping[str, Any]) -> str:
    fmt = str(media.get("format") or "").lower()
    if fmt in _EXT:
        return _EXT[fmt]
    tail = urllib.parse.urlparse(str(media.get("identifier") or "")).path.lower()
    for ext in (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".gif"):
        if tail.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ".tif" if ext == ".tiff" else ext
    return ".jpg"


def occurrence_row(rec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The first StillImage of an occurrence as a manifest row, or ``None``."""
    media = next(
        (
            m
            for m in rec.get("media") or []
            if m.get("type") == "StillImage" and m.get("identifier")
        ),
        None,
    )
    if media is None:
        return None
    ds, gid = rec.get("datasetKey"), rec.get("gbifID") or rec.get("key")
    return {
        "key": f"{ds}/{gid}{media_ext(media)}",
        "url": str(media["identifier"]),
        "lat": rec.get("decimalLatitude"),
        "lon": rec.get("decimalLongitude"),
        "event_date": rec.get("eventDate"),
        "license": media.get("license") or rec.get("license"),
        "rights_holder": media.get("rightsHolder") or rec.get("rightsHolder"),
        "creator": media.get("creator"),
        "species": rec.get("species") or rec.get("scientificName"),
        "species_key": rec.get("speciesKey") or rec.get("taxonKey"),
        "gbif_id": gid,
        "occurrence_id": rec.get("occurrenceID"),
        "dataset_key": ds,
        "basis_of_record": rec.get("basisOfRecord"),
        "depth_m": rec.get("depth"),
    }


class GbifOccurrenceMediaAdapter(RowJoinMixin, BaseAdapter):
    name = "gbif-occurrence-media"

    def __init__(self, params: Mapping[str, Any]) -> None:
        super().__init__(params)
        self._rows = {}
        self.params.setdefault("field_columns", {"lat": "lat", "lon": "lon", "depth_m": "depth_m"})
        self.params.setdefault(
            "label_columns",
            [
                "species",
                "species_key",
                "event_date",
                "license",
                "rights_holder",
                "creator",
                "gbif_id",
                "occurrence_id",
                "dataset_key",
                "basis_of_record",
            ],
        )

    def _query(self, **extra: Any) -> list[tuple[str, Any]]:
        q: list[tuple[str, Any]] = [("mediaType", "StillImage")]
        q += [
            ("basisOfRecord", b)
            for b in self.params.get("basis_of_record")
            or ["HUMAN_OBSERVATION", "MACHINE_OBSERVATION"]
        ]
        q += [("taxonKey", k) for k in self.params["taxon_keys"]]
        return q + list(extra.items())

    def _get(self, query: list[tuple[str, Any]]) -> Mapping[str, Any]:
        return get_json(f"{API}?{urllib.parse.urlencode(query)}")  # type: ignore[return-value]

    def _facet(self, field: str, **extra: Any) -> dict[str, int]:
        res = self._get(self._query(facet=field, facetLimit=100000, limit=0, **extra))
        facets = res.get("facets") or [{}]
        return {str(c["name"]): int(c["count"]) for c in facets[0].get("counts", [])}

    def resolve_version(self) -> str:
        return str(self.params.get("snapshot") or f"api-{dt.date.today():%Y%m%d}")

    def _partitions(self, ds: str, n: int) -> Iterator[dict[str, Any]]:
        if n < OFFSET_WINDOW:
            yield {"datasetKey": ds}
            return
        for year in sorted(self._facet("year", datasetKey=ds)):
            yield {"datasetKey": ds, "year": year}

    def list_items(self) -> Iterator[RemoteItem]:
        exclude = set(self.params.get("exclude_dataset_keys") or [INAT_DATASET])
        cap = int(self.params.get("per_species_cap", 100))
        limit = int(self.params.get("page_limit", 300))
        max_items = self.params.get("max_items")
        datasets = sorted(k for k in self._facet("datasetKey") if k not in exclude)
        per_species: Counter = Counter()
        n_out = 0
        for ds in datasets[: self.params.get("max_datasets") or None]:
            counts = self._facet("datasetKey", datasetKey=ds)
            for part in self._partitions(ds, counts.get(ds, 0)):
                offset = 0
                while offset < OFFSET_WINDOW:
                    page = self._get(self._query(limit=limit, offset=offset, **part))
                    for rec in page.get("results") or []:
                        row = occurrence_row(rec)
                        sp = row and row["species_key"]
                        if row is None or sp is None or per_species[sp] >= cap:
                            continue
                        per_species[sp] += 1
                        self._rows[row["key"]] = row
                        yield RemoteItem(key=row["key"], url=row["url"])
                        n_out += 1
                        if max_items and n_out >= int(max_items):
                            return
                    if page.get("endOfRecords", True):
                        break
                    offset += limit
