"""A PANGAEA publication series: parent DOI -> child datasets -> images (``pangaea-series``).

A "collection" (dataset publication series, e.g. one cruise's OFOS profiles) has no
tab export of its own — ``?format=textfile`` answers HTTP 400 "not available for
collection data sets". Its children come from PANGAEA's search API,
``<search_url>?q=incollection:<id>&count=<n>&offset=<k>`` (JSON ``totalCount`` +
``results[].URI``), the same call ``pangaeapy`` uses; the count is checked against
``totalCount`` so a short page can never silently drop profiles.

Each child is then read exactly like ``adapter: pangaea`` (:func:`.pangaea.table_items`),
with the child's numeric id as the key prefix (``<child>/<file>``) because profiles
reuse camera filenames. A child that is itself a collection (textfile 400) is walked
recursively up to ``max_depth``; a DOI with no children is read as a plain dataset, so
a spec that names one profile still works. A child refused as login-gated is skipped
with a warning; if every child is refused, the refusal is raised.
"""

from __future__ import annotations

import json
import logging
import urllib.error
from collections.abc import Iterator

from . import RemoteItem
from ._http import AccessRefused, open_url
from .pangaea import _DOI_ID, PangaeaAdapter, read_table, table_items

log = logging.getLogger(__name__)
_PAGE = 500


class PangaeaSeriesAdapter(PangaeaAdapter):
    name = "pangaea-series"

    @property
    def _search_url(self) -> str:
        return str(
            self.params.get("search_url", "https://www.pangaea.de/advanced/search.php")
        ).rstrip("/")

    def children(self, numeric_id: str) -> list[str]:
        """Numeric ids of the collection's direct children (empty for a plain dataset)."""
        out: list[str] = []
        total = offset = 0
        while True:
            url = f"{self._search_url}?q=incollection:{numeric_id}&count={_PAGE}&offset={offset}"
            with open_url(url) as resp:
                page = json.loads(resp.read().decode("utf-8"))
            total = int(page.get("totalCount") or 0)
            results = page.get("results") or []
            for r in results:
                m = _DOI_ID.search(str(r.get("URI", "")))
                if m and m.group(1) not in out:
                    out.append(m.group(1))
            offset += len(results)
            if not results or offset >= total:
                break
        if len(out) < total:
            raise RuntimeError(
                f"{self.name}: PANGAEA.{numeric_id} lists {total} children, got {len(out)}"
            )
        return out

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_numeric_id"):
            self.resolve_version()
        kids = self.children(self._numeric_id)
        if not kids:  # a plain dataset, not a collection
            yield from table_items(self._table(), self._numeric_id, self._download_base)
            return
        yield from self._walk(kids, depth=1)

    def _walk(self, kids: list[str], depth: int) -> Iterator[RemoteItem]:
        refused: AccessRefused | None = None
        read = 0
        for kid in kids:
            try:
                rows = read_table(self._endpoint, f"10.1594/PANGAEA.{kid}")
            except AccessRefused as exc:
                log.warning("%s: child PANGAEA.%s refused (%s); skipped", self.name, kid, exc)
                refused = exc
                continue
            except urllib.error.HTTPError as exc:
                grandkids = self.children(kid) if exc.code == 400 else []
                if not grandkids or depth >= int(self.params.get("max_depth", 3)):
                    raise
                read += 1
                yield from self._walk(grandkids, depth + 1)
                continue
            read += 1
            yield from table_items(rows, kid, self._download_base, key_prefix=f"{kid}/")
        if refused is not None and read == 0:
            raise refused
