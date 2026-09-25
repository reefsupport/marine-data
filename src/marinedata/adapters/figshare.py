"""Public figshare article/collection file API (``adapter: figshare``, WP-6d-A).

``params.article`` -> ``GET <endpoint>/articles/<id>`` (optionally
``/versions/<version>`` when ``params.version`` is set) for that article's
``files`` (``name``, ``size``, ``download_url``, ``computed_md5``).
``params.collection`` -> paginates ``GET <endpoint>/collections/<id>/articles``
and unions every member article's files. An embargoed or confidential article
raises :class:`AccessRefused` (D-E: no login/terms).
"""

from __future__ import annotations

from collections.abc import Iterator

from . import AccessRefused, BaseAdapter, RemoteItem
from ._http import get_json


class FigshareAdapter(BaseAdapter):
    name = "figshare"

    @property
    def _endpoint(self) -> str:
        return str(self.params.get("endpoint", "https://api.figshare.com/v2")).rstrip("/")

    def _article(self, article_id: int) -> dict:
        version = self.params.get("version")
        path = f"/articles/{article_id}" + (f"/versions/{version}" if version else "")
        art = get_json(f"{self._endpoint}{path}")
        assert isinstance(art, dict)
        if art.get("is_embargoed") or art.get("is_confidential"):
            raise AccessRefused(
                f"https://figshare.com/articles/{article_id}",
                "embargoed or confidential: request access on the article page",
            )
        return art

    def _article_ids(self) -> Iterator[int]:
        if self.params.get("article"):
            yield int(self.params["article"])
            return
        collection = int(self.params["collection"])
        page = 1
        page_size = 100
        while True:
            batch = get_json(
                f"{self._endpoint}/collections/{collection}/articles"
                f"?page={page}&page_size={page_size}"
            )
            assert isinstance(batch, list)
            if not batch:
                return
            for entry in batch:
                yield int(entry["id"])
            if len(batch) < page_size:
                return
            page += 1

    def resolve_version(self) -> str:
        if self.params.get("article") and not self.params.get("collection"):
            art = self._article(int(self.params["article"]))
            self._articles = [art]
            return str(self.params.get("version") or art.get("version") or f"article-{art['id']}")
        self._articles = [self._article(aid) for aid in self._article_ids()]
        collection = self.params.get("collection")
        return str(
            self.params.get("version") or f"collection-{collection}-{len(self._articles)}"
        )

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_articles"):
            self.resolve_version()
        for art in self._articles:
            for f in art.get("files") or []:
                yield RemoteItem(
                    key=f"{art['id']}/{f['name']}",
                    url=str(f["download_url"]),
                    size=int(f["size"]) if f.get("size") is not None else None,
                    md5=f.get("computed_md5") or None,
                )
