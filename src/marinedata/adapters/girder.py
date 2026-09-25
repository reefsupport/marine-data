"""Public Girder REST item/folder listing (``adapter: girder``, WP-6d-A).

Walks ``GET <api>/folder?parentId=<id>&parentType=collection|folder`` (child
folders) and ``GET <api>/item?folderId=<id>`` (files) from ``params.collection``
(or ``params.folder`` to start deeper), and downloads each item from
``<api>/item/<id>/download``. No auth: every request is anonymous, matching a
public Girder collection's own access control (``public: true``).
"""

from __future__ import annotations

from collections.abc import Iterator

from . import BaseAdapter, RemoteItem
from ._http import get_json


class GirderAdapter(BaseAdapter):
    name = "girder"

    @property
    def _api(self) -> str:
        return str(self.params.get("api", "https://data.kitware.com/api/v1")).rstrip("/")

    def resolve_version(self) -> str:
        return str(self.params.get("version") or "girder-pinned")

    def _folders(self, parent_id: str, parent_type: str) -> list[dict]:
        body = get_json(f"{self._api}/folder?parentId={parent_id}&parentType={parent_type}&limit=0")
        assert isinstance(body, list)
        return body

    def _items(self, folder_id: str) -> list[dict]:
        body = get_json(f"{self._api}/item?folderId={folder_id}&limit=0")
        assert isinstance(body, list)
        return body

    def _walk(self, folder_id: str, parent_type: str, prefix: str) -> Iterator[RemoteItem]:
        for item in self._items(folder_id):
            name = str(item["name"])
            yield RemoteItem(
                key=f"{prefix}{name}" if prefix else name,
                url=f"{self._api}/item/{item['_id']}/download",
                size=item.get("size"),
            )
        for folder in self._folders(folder_id, "folder"):
            yield from self._walk(folder["_id"], "folder", f"{prefix}{folder['name']}/")

    def list_items(self) -> Iterator[RemoteItem]:
        root_id = str(self.params.get("folder") or self.params["collection"])
        root_type = "folder" if self.params.get("folder") else "collection"
        for folder in self._folders(root_id, root_type):
            yield from self._walk(folder["_id"], "folder", f"{folder['name']}/")
        if self.params.get("folder"):
            yield from (
                RemoteItem(
                    key=str(item["name"]),
                    url=f"{self._api}/item/{item['_id']}/download",
                    size=item.get("size"),
                )
                for item in self._items(root_id)
            )
