"""Public Google Drive files, no login (``adapter: gdrive-public``, WP-6d-A).

Every entry in ``params.files`` (``[{id, name}, ...]``) or a single
``params.file_id``/``params.drive_link`` (a ``.../d/<id>/...`` or
``...?id=<id>`` URL, as several catalog pages give it) is fetched from
``https://drive.google.com/uc?export=download&id=<id>``. A file past Drive's
virus-scan size threshold answers with an HTML confirmation page instead of
bytes. The current interstitial posts (GET) a ``<form>`` to
``drive.usercontent.google.com/download`` with hidden ``id``/``export``/
``confirm``/``uuid`` inputs; that form is parsed with the stdlib
``html.parser`` and resubmitted (no cookie jar — the fields Drive embeds are
self-contained). Older pages that only carry a bare ``confirm=`` anchor are
still handled as a fallback. A "quota exceeded" or sign-in page raises
:class:`AccessRefused` so the row is routed to needs_yohan (D-E: never work
around a permission wall), never retried as a bug.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

from . import AccessRefused, BaseAdapter, Fetched, RemoteItem
from ._http import HashingReader, download, open_url

_ID_IN_PATH = re.compile(r"/d/([\w-]{20,})")
_ID_IN_QS = re.compile(r"[?&]id=([\w-]{20,})")
_CONFIRM_HREF = re.compile(r'href="(/uc\?export=download[^"]*confirm=[^"]+)"')
_QUOTA = re.compile(r"quota.{0,20}exceeded|sign in|too many users", re.IGNORECASE)
_ABS_URL = re.compile(r"^https?://", re.IGNORECASE)


class _ConfirmFormParser(HTMLParser):
    """Grabs the first ``<form>``'s ``action`` and its hidden ``<input>`` fields."""

    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}
        self._in_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag == "form" and self.action is None:
            self._in_form = True
            self.action = attrs_d.get("action")
        elif tag == "input" and self._in_form:
            if (attrs_d.get("type") or "").lower() != "hidden":
                return
            name = attrs_d.get("name")
            if name:
                self.fields[name] = attrs_d.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_form = False


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

    def _extract_confirm_url(self, body: str) -> str | None:
        """The real download URL behind Drive's virus-scan interstitial, or ``None``.

        Tries the current usercontent form first (absolute ``action`` + hidden
        ``id``/``export``/``confirm``/``uuid`` inputs), then falls back to the
        legacy bare ``confirm=`` anchor some pages still carry.
        """
        parser = _ConfirmFormParser()
        parser.feed(body)
        if parser.action and parser.fields:
            action = parser.action.replace("&amp;", "&")
            if not _ABS_URL.match(action):
                action = self._endpoint + (action if action.startswith("/") else f"/{action}")
            return f"{action}?{urlencode(parser.fields)}"
        m = _CONFIRM_HREF.search(body)
        if m:
            return self._endpoint + m.group(1).replace("&amp;", "&")
        return None

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        resp = open_url(item.url)
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            body = resp.read().decode("utf-8", errors="replace")
            resp.close()
            if _QUOTA.search(body):
                raise AccessRefused(item.url, "Drive quota/permission wall: needs a human export")
            confirm_url = self._extract_confirm_url(body)
            if confirm_url is None:
                raise AccessRefused(item.url, "Drive confirm page had no resolvable download link")
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
