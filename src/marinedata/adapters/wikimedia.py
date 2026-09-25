"""Wikimedia Commons category walk (``adapter: commons-api``, WP-6e-A).

Enumerates the files of ``params.root_category`` plus every subcategory whose title
contains ``params.subcat_contains`` (default ``nderwater``), breadth-first to
``params.max_depth`` (default 6), through the MediaWiki API
(``list=categorymembers`` for subcategories, ``generator=categorymembers`` +
``prop=imageinfo`` with ``extmetadata`` for files). Every request carries a descriptive
User-Agent (Wikimedia's UA policy) and the API is paced at <= 2 req/s.

Per file: ``license`` (``LicenseShortName``) and ``attribution`` (``Artist``, HTML
stripped) go into ``metadata.parquet``, with GPS and ``DateTimeOriginal`` when present;
the description, the Commons category it was found in and the file page go to labels.
The version pins the sorted (title, sha1) set, so a re-run against a changed category
gets a new version instead of silently mixing.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import BaseAdapter, Decoded, Fetched, RemoteItem
from ._http import HashingReader, open_url

USER_AGENT = (
    "ReefSupportMarineData/1.0 (https://reef.support; open marine imagery dataset build) "
    "marinedata-ingest/commons-api"
)
_MIN_INTERVAL_S = 0.5  # <= 2 req/s
_TAG = re.compile(r"<[^>]+>")
_META = "LicenseShortName|Artist|DateTimeOriginal|GPSLatitude|GPSLongitude|ImageDescription"


def plain(value: Any, limit: int = 2000) -> str:
    return html.unescape(_TAG.sub("", str(value or ""))).strip()[:limit]


def _num(value: Any) -> float | None:
    try:
        return float(plain(value))
    except ValueError:
        return None


class CommonsAdapter(BaseAdapter):
    name = "commons-api"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._meta: dict[str, dict[str, Any]] = {}
        self._files: list[RemoteItem] | None = None
        self._last = 0.0
        self.categories: list[str] = []

    @property
    def _api(self) -> str:
        return str(self.params.get("api", "https://commons.wikimedia.org/w/api.php"))

    def _pace(self) -> None:
        wait = float(self.params.get("min_interval_s", _MIN_INTERVAL_S)) - (
            time.monotonic() - self._last
        )
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, query: dict[str, str]) -> dict[str, Any]:
        self._pace()
        qs = urllib.parse.urlencode({"action": "query", "format": "json", **query})
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        with open_url(f"{self._api}?{qs}", headers=hdrs) as resp:
            return json.loads(resp.read())

    def _paged(self, query: dict[str, str]) -> Iterator[dict[str, Any]]:
        cont: dict[str, str] = {}
        while True:
            body = self._get({**query, **cont})
            yield body
            if "continue" not in body:
                return
            cont = {k: str(v) for k, v in body["continue"].items()}

    def _walk_categories(self) -> list[str]:
        root = "Category:" + str(self.params.get("root_category", "Underwater photographs"))
        needle = str(self.params.get("subcat_contains", "nderwater"))
        depth_cap = int(self.params.get("max_depth", 6))
        seen, order, todo = {root}, [root], deque([(root, 0)])
        while todo:
            cat, depth = todo.popleft()
            if depth >= depth_cap:
                continue
            q = {"list": "categorymembers", "cmtitle": cat, "cmtype": "subcat", "cmlimit": "500"}
            for body in self._paged(q):
                for m in (body.get("query") or {}).get("categorymembers", []):
                    t = str(m["title"])
                    if needle in t and t not in seen:
                        seen.add(t)
                        order.append(t)
                        todo.append((t, depth + 1))
        return order

    def _collect(self) -> list[RemoteItem]:
        if self._files is not None:
            return self._files
        self.categories = self._walk_categories()
        found: dict[str, RemoteItem] = {}
        for cat in self.categories:
            q = {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|size|sha1|mime|extmetadata",
                "iiextmetadatafilter": _META,
            }
            for body in self._paged(q):
                for page in ((body.get("query") or {}).get("pages") or {}).values():
                    info = (page.get("imageinfo") or [{}])[0]
                    key = str(page["title"]).split(":", 1)[-1].replace(" ", "_")
                    if not info.get("url") or key in found:
                        continue  # first category that lists a file wins
                    em = info.get("extmetadata") or {}
                    val = {k: (em.get(k) or {}).get("value") for k in _META.split("|")}
                    self._meta[key] = {
                        "fields": {
                            "license": plain(val["LicenseShortName"], 200) or None,
                            "attribution": plain(val["Artist"], 500) or None,
                            "capture_datetime": plain(val["DateTimeOriginal"], 40) or None,
                            "lat": _num(val["GPSLatitude"]),
                            "lon": _num(val["GPSLongitude"]),
                        },
                        "labels": {
                            "description": plain(val["ImageDescription"]),
                            "commons_category": cat.split(":", 1)[-1],
                            "commons_page": str(info.get("descriptionurl") or ""),
                            "sha1": str(info.get("sha1") or ""),
                        },
                    }
                    found[key] = RemoteItem(key=key, url=str(info["url"]), size=info.get("size"))
        self._files = [found[k] for k in sorted(found)]
        return self._files

    def resolve_version(self) -> str:
        files = self._collect()
        if self.params.get("version"):
            return str(self.params["version"])
        sig = "\n".join(f"{i.key}\t{self._meta[i.key]['labels']['sha1']}" for i in files)
        return f"commons-{hashlib.sha256(sig.encode()).hexdigest()[:12]}"

    def list_items(self) -> Iterator[RemoteItem]:
        yield from self._collect()

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        if item.parts or Path(item.key).suffix.lower() in {".webm", ".ogv", ".mp4"}:
            return super().fetch(item, tmp_dir)
        self._pace()  # upload.wikimedia.org 429s bursts too (3/53 in the live smoke)
        url = item.url.split("?utm_source=", 1)[0]  # imageinfo appends tracking params
        return Fetched(
            item, stream=HashingReader(open_url(url, headers={"User-Agent": USER_AGENT}))
        )

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        meta = self._meta.get(fetched.item.key, {"fields": {}, "labels": {}})
        extra = {k: v for k, v in meta["fields"].items() if v not in (None, "")}
        labels = {k: v for k, v in meta["labels"].items() if v}
        for d in super().decode(fetched):
            yield Decoded(
                d.upstream_id,
                d.data,
                d.suffix,
                d.upstream_url or fetched.item.url,
                d.split_hint,
                {**d.fields, **extra},
                {**d.labels, **labels},
                d.label_files,
            )
