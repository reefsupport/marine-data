"""FathomNet REST API (``adapter: fathomnet``, WP-6d-A).

``GET <endpoint>/images?page=&size=[&concept=]`` returns a Spring-style page
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
_MAX_PAGE = 2000  # measured: size=2000 answers in ~4 s


class FathomNetAdapter(BaseAdapter):
    name = "fathomnet"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._meta: dict[str, dict[str, Any]] = {}
        self.upstream_mismatch = 0  # D-AG: kept, counted for the datasheet
        self._last_request = 0.0
        self.duplicates = 0
        self.excluded = 0

    def _exclude_set(self) -> set[str]:
        out: set[str] = set()
        from .manifest import _resolve

        pin = self.params.get("exclude_sha256_pin")
        for raw in self.params.get("exclude_sha256") or []:
            # a local path, or an https URL (lists > 5 MB live on S3 under _manifest/)
            path = (
                _resolve(str(raw), pin)
                if str(raw).startswith("http")
                else Path(str(raw)).expanduser()
            )
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

    def _cap(self) -> int:
        if self.params.get("full"):
            return sys.maxsize
        return int(self.params.get("max_items") or _DEFAULT_MAX_ITEMS)

    def _page_size(self) -> int:
        # one size for every request: Spring numbers pages by size, so page 1 at size 1000
        # after page 0 at size 100 would skip items 100..999
        return max(1, min(int(self.params.get("page_size") or _DEFAULT_PAGE), _MAX_PAGE))

    def resolve_version(self) -> str:
        if self.params.get("version"):
            return str(self.params["version"])
        first = self._page(0, self._page_size())
        # digest of the first 100 uuids, so the version does not depend on page_size
        uuids = ",".join(sorted(e["uuid"] for e in first["content"][:_DEFAULT_PAGE]))
        digest = hashlib.sha256(uuids.encode()).hexdigest()[:12]
        self._first_page = first
        return f"fathomnet-{digest}"

    def _page(self, number: int, size: int) -> dict[str, Any]:
        # WP-6f: /api/images is a Spring Pageable (page/size). It silently IGNORES
        # limit/offset and returns page 0 every time, which made a capped run re-read page 0
        # ~4,812 times (totalItems / 100) whenever page 0 held a duplicate sha256.
        qs = {"page": str(number), "size": str(size)}
        if self.params.get("concept"):
            qs["concept"] = str(self.params["concept"])
        body = self._get(f"/images?{urllib.parse.urlencode(qs)}")
        got = body.get("pageNumber")
        if got is not None and int(got) != number:
            raise RuntimeError(f"fathomnet: asked for page {number}, server returned page {got}")
        return body

    def list_items(self) -> Iterator[RemoteItem]:
        cap = self._cap()
        exclude = self._exclude_set()
        seen: set[str] = set()
        size = self._page_size()
        number = 0
        yielded = 0
        page = getattr(self, "_first_page", None)
        while yielded < cap:
            reuse = page is not None and number == 0 and len(page.get("content") or []) <= size
            body = page if reuse else self._page(number, size)
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
                    sha256=sha or None,
                )
                yielded += 1
                if yielded >= cap:
                    return
            number += 1
            if int(body.get("totalPages") or 0) and number >= int(body["totalPages"]):
                return

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        uuid = fetched.item.key.rsplit(".", 1)[0]
        meta = self._meta.get(uuid, {"fields": {}, "labels": {}, "raw": {}})
        declared = str(meta["raw"].get("sha256") or "").lower() or None
        for decoded in super().decode(fetched):
            # D-AG: FathomNet's declared sha256 differs from the served bytes for ~13% of
            # images; keep them, record the declared hash and whether it matched.
            dag: dict[str, Any] = {}
            if declared:
                match = declared == hashlib.sha256(decoded.data).hexdigest()
                self.upstream_mismatch += not match
                dag = {"upstream_sha256": declared, "upstream_sha256_match": match}
            yield Decoded(
                decoded.upstream_id,
                decoded.data,
                decoded.suffix,
                decoded.upstream_url,
                decoded.split_hint,
                {
                    **decoded.fields,
                    **{k: v for k, v in meta["fields"].items() if v is not None},
                    **dag,
                },
                {**decoded.labels, **meta["labels"], **{k: str(v).lower() for k, v in dag.items()}},
                {**decoded.label_files, f"{uuid}.json": json.dumps(meta["raw"]).encode()},
            )
