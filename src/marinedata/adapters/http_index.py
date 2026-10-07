"""Generic listable-index adapter (``adapter: http-index``, WP-6d-A).

Three interchangeable list sources (many ``needs_adapter`` rows collapse onto
whichever fits their host — see ``docs/ingest-howto.md``):

* ``params.index_url``: an Apache/nginx-style HTML directory index, walked
  recursively (``<a href>`` links; entries ending in ``/`` are subdirectories)
  up to ``params.depth`` (default 3), same host only. Covers plain
  site-navigation pages, NCEI accession trees, DRUM/BODC listings and ERDDAP
  ``files/`` pages when the host actually serves static HTML (a JS-rendered
  shell enumerates zero links and the caller's D-R4 rule downgrades it, rather
  than this adapter guessing).
* ``params.list_url``: a CSV or newline-delimited TXT of URLs (ERDDAP
  image-list CSVs, NCEI/BODC manifest exports). ``params.url_column`` names or
  indexes the URL column for CSV; a bare TXT file is one URL per line.
* ``params.urls``: an explicit small list (same shape as the ``http`` adapter),
  for a manifest already resolved by hand.

``params.version`` is required (an index/listing carries no version signal of
its own, same rule as the plain-URL ``http`` adapter).
"""

from __future__ import annotations

import csv
import io
import re
import urllib.parse
from collections.abc import Iterator
from pathlib import PurePosixPath

from . import BaseAdapter, RemoteItem
from ._http import open_url

_HREF = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_SKIP = {"../", "..", "/", "?", "#"}


class HttpIndexAdapter(BaseAdapter):
    name = "http-index"

    def resolve_version(self) -> str:
        if not self.params.get("version"):
            raise ValueError("http-index adapter: params.version is required")
        return str(self.params["version"])

    def _fetch_text(self, url: str) -> str:
        with open_url(url) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _walk_index(self) -> Iterator[str]:
        root = str(self.params["index_url"])
        max_depth = int(self.params.get("depth", 3))
        host = urllib.parse.urlparse(root).netloc
        seen: set[str] = set()
        stack: list[tuple[str, int]] = [(root if root.endswith("/") else root + "/", 0)]
        while stack:
            url, depth = stack.pop()
            if url in seen:
                continue
            seen.add(url)
            try:
                html = self._fetch_text(url)
            except Exception:
                continue
            for href in _HREF.findall(html):
                if href in _SKIP or href.startswith(("?", "#", "mailto:")):
                    continue
                child = urllib.parse.urljoin(url, href)
                if urllib.parse.urlparse(child).netloc != host:
                    continue
                if not child.startswith(root.rsplit("/", 1)[0]):
                    continue
                if child.endswith("/"):
                    if depth < max_depth:
                        stack.append((child, depth + 1))
                else:
                    yield child

    def _list_url_rows(self) -> Iterator[dict[str, str] | str]:
        text = self._fetch_text(str(self.params["list_url"]))
        if "," in text.splitlines()[0] if text.splitlines() else False:
            reader = csv.DictReader(io.StringIO(text))
            yield from reader
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield line

    def list_items(self) -> Iterator[RemoteItem]:
        if self.params.get("index_url"):
            for url in self._walk_index():
                path = urllib.parse.urlparse(url).path
                yield RemoteItem(key=str(PurePosixPath(path).name) or url, url=url)
            return
        if self.params.get("list_url"):
            col = self.params.get("url_column")
            for row in self._list_url_rows():
                if isinstance(row, str):
                    url = row
                elif col:
                    url = row[col]
                else:
                    url = next(iter(row.values()))
                url = url.strip()
                if not url:
                    continue
                path = urllib.parse.urlparse(url).path
                yield RemoteItem(key=str(PurePosixPath(path).name) or url, url=url)
            return
        for entry in self.params.get("urls") or []:
            e = {"url": entry} if isinstance(entry, str) else dict(entry)
            path = urllib.parse.urlparse(e["url"]).path
            yield RemoteItem(
                key=str(e.get("key") or PurePosixPath(path).name),
                url=e["url"],
                size=e.get("size"),
                md5=e.get("md5"),
                sha256=e.get("sha256"),
            )
