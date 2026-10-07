"""Public Seafile share-link directory listing (``adapter: seafile-share``, WP-6d-A).

``GET <share>/api/v2.1/share-links/<token>/dirents/?path=<path>`` lists one
directory of a public share link (``dirent_list``: dicts with ``is_dir`` and
either ``file_path``/``file_name`` or ``folder_path``/``folder_name``); walked
recursively. Each file downloads from ``<share>/files/?p=<url-quoted path>&dl=1``
(``params.share``, e.g. ``https://seafile.example.org/d/<token>/``).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator

from . import BaseAdapter, RemoteItem
from ._http import get_json


class SeafileAdapter(BaseAdapter):
    name = "seafile-share"

    @property
    def _share(self) -> str:
        return str(self.params["share"]).rstrip("/")

    @property
    def _token(self) -> str:
        return self._share.rsplit("/", 1)[-1]

    @property
    def _api_base(self) -> str:
        parsed = urllib.parse.urlparse(self._share)
        return f"{parsed.scheme}://{parsed.netloc}"

    def resolve_version(self) -> str:
        return str(self.params.get("version") or f"share-{self._token}")

    def _walk(self, path: str) -> Iterator[dict]:
        qs = urllib.parse.urlencode({"path": path})
        body = get_json(f"{self._api_base}/api/v2.1/share-links/{self._token}/dirents/?{qs}")
        assert isinstance(body, dict)
        for entry in body.get("dirent_list") or []:
            if entry.get("is_dir"):
                yield from self._walk(entry["folder_path"])
            else:
                yield entry

    def list_items(self) -> Iterator[RemoteItem]:
        for entry in self._walk("/"):
            path = entry["file_path"]
            url = f"{self._share}/files/?p={urllib.parse.quote(path)}&dl=1"
            yield RemoteItem(
                key=path.lstrip("/"),
                url=url,
                size=entry.get("size"),
            )
