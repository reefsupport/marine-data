"""Public Google Drive files, no login (``adapter: gdrive-public``, WP-6d-A).

Every entry in ``params.files`` (``[{id, name}, ...]``) or a single
``params.file_id``/``params.drive_link`` (a ``.../d/<id>/...`` or
``...?id=<id>`` URL, as several catalog pages give it) is fetched from
``https://drive.google.com/uc?export=download&id=<id>``. A file past Drive's
virus-scan size threshold answers with an HTML confirmation page instead of
bytes; the page's own ``confirm=`` download link is followed once (no cookie
jar — the link Drive embeds is self-contained). A "quota exceeded" or sign-in
page raises :class:`AccessRefused` so the row is routed to needs_yohan (D-E:
never work around a permission wall), never retried as a bug.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from . import AccessRefused, BaseAdapter, Fetched, RemoteItem
from ._http import HashingReader, download, open_url

_ID_IN_PATH = re.compile(r"/d/([\w-]{20,})")
_ID_IN_QS = re.compile(r"[?&]id=([\w-]{20,})")
_CONFIRM_HREF = re.compile(r'href="(/uc\?export=download[^"]*confirm=[^"]+)"')
_QUOTA = re.compile(r"quota.{0,20}exceeded|sign in|too many users", re.IGNORECASE)


def extract_file_id(url: str) -> str | None:
    m = _ID_IN_PATH.search(url) or _ID_IN_QS.search(url)
    return m.group(1) if m else None


class GDriveAdapter(BaseAdapter):
    name = "gdrive-public"

    @property
    def _endpoint(self) -> str:
        return str(self.params.get("endpoint", "https://drive.google.com")).rstrip("/")

    def resolve_version(self) -> str:
        return str(self.params.get("version") or "gdrive-pinned")

    def _entries(self) -> list[dict[str, str]]:
        if self.params.get("files"):
            return [dict(f) for f in self.params["files"]]
        file_id = self.params.get("file_id") or extract_file_id(
            str(self.params.get("drive_link") or self.params.get("landing") or "")
        )
        if not file_id:
            raise ValueError("gdrive-public adapter: no file id (params.file_id/files/drive_link)")
        return [{"id": file_id, "name": self.params.get("name") or f"{file_id}.bin"}]

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        resp = open_url(item.url)
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            body = resp.read().decode("utf-8", errors="replace")
            resp.close()
            if _QUOTA.search(body):
                raise AccessRefused(item.url, "Drive quota/permission wall: needs a human export")
            m = _CONFIRM_HREF.search(body)
            if not m:
                raise AccessRefused(item.url, "Drive confirm page had no resolvable download link")
            confirm_url = self._endpoint + m.group(1).replace("&amp;", "&")
            dest = Path(tmp_dir) / "fetch" / item.key
            sha, _md5, size = download(confirm_url, dest)
            return Fetched(item, path=dest, sha256=sha, size=size)
        return Fetched(item, stream=HashingReader(resp))

    def list_items(self) -> Iterator[RemoteItem]:
        for entry in self._entries():
            yield RemoteItem(
                key=str(entry["name"]),
                url=f"{self._endpoint}/uc?export=download&id={entry['id']}",
                size=entry.get("size"),
                md5=entry.get("md5"),
            )
