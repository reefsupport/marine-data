"""CSIRO Data Access Portal (DAP) collection file listing (``adapter: csiro_dap``).

``GET <endpoint>/collections/<id>/data`` returns the WHOLE file listing for the
collection in one response — verified empirically for the deepseagrass collection
(72,004 files, no ``page``/``pageSize`` needed, no ``Link`` pagination header either).
Each entry's ``filename`` already carries the full upload path
(``<split>/<class>/<name>.jpg``); combined with ``params.format: imagefolder`` the
existing generic decode path (``decode.py:_group``) turns the immediate parent folder
into the ``label``, same as any other image-folder classification source.

Anonymous GET, no login (D-E). Each file's own ``link.href`` (the stable
``/collections/<id>/data/<fileId>`` URL, not the query-string ``presignedLink`` that
already carries a 48 h expiry) 302-redirects to a freshly presigned S3 URL on every
request, so it never goes stale between ``enumerate()`` and a later ``fetch()``.
"""

from __future__ import annotations

from collections.abc import Iterator

from . import BaseAdapter, RemoteItem
from ._http import get_json


class CsiroDapAdapter(BaseAdapter):
    name = "csiro_dap"

    @property
    def _endpoint(self) -> str:
        return str(self.params.get("endpoint", "https://data.csiro.au/dap/ws/v2")).rstrip("/")

    @property
    def _collection(self) -> str:
        return str(self.params["collection"])

    def resolve_version(self) -> str:
        body = get_json(f"{self._endpoint}/collections/{self._collection}/data")
        assert isinstance(body, dict)
        self._files = list(body.get("file") or [])
        return str(self.params.get("version") or f"dap-{self._collection}-{len(self._files)}")

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_files"):
            self.resolve_version()
        for f in self._files:
            yield RemoteItem(
                key=str(f["filename"]),
                url=str(f["link"]["href"]),
                size=int(f["fileSize"]) if f.get("fileSize") is not None else None,
            )
