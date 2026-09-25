"""FathomNet REST API (``adapter: fathomnet``, WP-6d-A).

``GET <endpoint>/images?limit=&offset=[&concept=]`` returns a Spring-style page
(``content``/``pageNumber``/``pageSize``/``totalItems``/``totalPages``); each entry
carries the image ``url``, an upstream ``sha256`` (verified on fetch, same as any
other :class:`RemoteItem`) and per-annotation ``boundingBoxes`` with a per-box
``annotationLicense`` and ``concept`` name. There is no per-image top-level licence
field, so the licence recorded per sample is the set of distinct box licences (or
``unlabelled`` when an image carries no boxes).

The full corpus is ~481k images (2026-09-25); ``params.max_items`` bounds how many
are enumerated (default 2000, a first-pass subset per the queue's own scoping note)
so a dry-run stays fast and polite; ``params.full: true`` lifts the cap (D-AB: FathomNet
is fetched in full). ``params.concept`` filters to one FathomNet concept. Paginated at
<= 5 req/s (``_MIN_INTERVAL_S``).

WP-6e-A: the per-image licence also goes into ``fields["license"]`` (so it lands in
``metadata.parquet``, D-C), and items are deduped by the upstream sha256 — within
FathomNet, and against ``params.exclude_sha256`` (files of sha256s: ``.txt`` one per
line, or ``.parquet`` with an ``image_sha256``/``sha256`` column), which is how the
223,863 NOAA GFISHER frames already staged via noaa-gfisher/seamapd21 are skipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import BaseAdapter, Decoded, Fetched, RemoteItem
from ._http import get_json

_MIN_INTERVAL_S = 0.2  # <= 5 req/s
_DEFAULT_PAGE = 100
_DEFAULT_MAX_ITEMS = 2000


class FathomNetAdapter(BaseAdapter):
    name = "fathomnet"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._meta: dict[str, dict[str, Any]] = {}
        self._last_request = 0.0
        self.duplicates = 0
        self.excluded = 0

    def _exclude_set(self) -> set[str]:
        out: set[str] = set()
        for raw in self.params.get("exclude_sha256") or []:
            path = Path(str(raw)).expanduser()
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                t = pq.read_table(path)
                col = "image_sha256" if "image_sha256" in t.column_names else "sha256"
                out.update(str(v).lower() for v in t[col].to_pylist() if v)
            else:
                out.update(ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip())
        return out

    @property
    def _endpoint(self) -> str:
        return str(self.params.get("endpoint", "https://database.fathomnet.org/api")).rstrip("/")

    def _get(self, path: str) -> Any:
        wait = _MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        return get_json(f"{self._endpoint}{path}")

    def resolve_version(self) -> str:
        if self.params.get("version"):
            return str(self.params["version"])
        first = self._page(0, min(_DEFAULT_PAGE, int(self.params.get("max_items") or 100)))
        uuids = ",".join(sorted(e["uuid"] for e in first["content"]))
        digest = hashlib.sha256(uuids.encode()).hexdigest()[:12]
        self._first_page = first
        return f"fathomnet-{digest}"

    def _page(self, offset: int, limit: int) -> dict[str, Any]:
        qs = {"limit": str(limit), "offset": str(offset)}
        if self.params.get("concept"):
            qs["concept"] = str(self.params["concept"])
        return self._get(f"/images?{urllib.parse.urlencode(qs)}")

    def list_items(self) -> Iterator[RemoteItem]:
        cap = (
            sys.maxsize
            if self.params.get("full")
            else int(self.params.get("max_items") or _DEFAULT_MAX_ITEMS)
        )
        exclude = self._exclude_set()
        seen: set[str] = set()
        page_size = int(self.params.get("page_size") or _DEFAULT_PAGE)
        offset = 0
        yielded = 0
        page = getattr(self, "_first_page", None)
        while yielded < cap:
            limit = min(page_size, cap - yielded)
            body = page if page is not None and offset == 0 else self._page(offset, limit)
            page = None
            entries = body.get("content") or []
            if not entries:
                return
            for entry in entries:
                sha = str(entry.get("sha256") or "").lower()
                if sha and sha in exclude:
                    self.excluded += 1
                    continue
                if sha and sha in seen:
                    self.duplicates += 1
                    continue
                seen.add(sha)
                boxes = entry.get("boundingBoxes") or []
                concepts = sorted({b["concept"] for b in boxes if b.get("concept")})
                licences = sorted(
                    {b["annotationLicense"] for b in boxes if b.get("annotationLicense")}
                )
                self._meta[entry["uuid"]] = {
                    "fields": {
                        "lat": entry.get("latitude"),
                        "lon": entry.get("longitude"),
                        "depth_m": entry.get("depthMeters"),
                        "capture_datetime": entry.get("timestamp") or entry.get("createdTimestamp"),
                        "license": ";".join(licences) or None,
                    },
                    "labels": {
                        "concepts": ";".join(concepts) or "unlabelled",
                        "licence": ";".join(licences) or "unlabelled",
                    },
                    "raw": entry,
                }
                yield RemoteItem(
                    key=f"{entry['uuid']}.jpg",
                    url=str(entry["url"]),
                    sha256=entry.get("sha256"),
                )
                yielded += 1
                if yielded >= cap:
                    return
            offset += len(entries)
            if int(body.get("totalItems") or 0) and offset >= int(body["totalItems"]):
                return

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        uuid = fetched.item.key.rsplit(".", 1)[0]
        meta = self._meta.get(uuid, {"fields": {}, "labels": {}, "raw": {}})
        for decoded in super().decode(fetched):
            yield Decoded(
                decoded.upstream_id,
                decoded.data,
                decoded.suffix,
                decoded.upstream_url,
                decoded.split_hint,
                {**decoded.fields, **{k: v for k, v in meta["fields"].items() if v is not None}},
                {**decoded.labels, **meta["labels"]},
                {**decoded.label_files, f"{uuid}.json": json.dumps(meta["raw"]).encode()},
            )
