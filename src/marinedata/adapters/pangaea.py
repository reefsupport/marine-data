"""One PANGAEA dataset's tab-export file list (``adapter: pangaea``, WP-6d-A).

``GET <endpoint>/<doi>?format=textfile`` returns a ``/* ... */`` metadata header
(ending on a line that is exactly ``*/``) followed by a tab-separated table: a
header row, then one data row per event/sample. Three image-column shapes occur
(WP-6j, surveyed across the OFOS/lander specs):

* ``IMAGE ...`` — a filename; per the dataset's own ``Files:`` line it downloads from
  ``https://download.pangaea.de/dataset/<numeric-id>/files/<filename>``. A sibling
  ``IMAGE ... (Hash)`` column (when present) is the file's md5, verified on fetch.
* ``Binary`` — the same filename convention under a generic name (e.g. the CVAP
  lander boxes, PANGAEA.961314), with an optional ``Binary (Hash)`` sibling.
* ``URL image`` — a full URL (older OFOS profiles on ``hs.pangaea.de``); used verbatim.

Rows repeat an image once per annotation (one row per box), so items are deduped by
key. A dataset with none of these columns enumerates zero items — the caller's D-R4
zero-item downgrade turns that into ``needs_adapter``, not a false ok (this adapter
never guesses a URL shape it can't verify from the table itself).
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterator
from pathlib import PurePosixPath

from . import BaseAdapter, RemoteItem
from ._http import open_url

_DOI_ID = re.compile(r"PANGAEA\.(\d+)", re.IGNORECASE)
_NAME_PREFIXES = ("IMAGE", "Binary")  # filename columns, files under <download_base>/<id>/files/
_URL_PREFIX = "URL IMAGE"  # full-URL column


def read_table(endpoint: str, doi: str) -> list[list[str]]:
    """The tab export as rows (header first). Raises ``HTTPError`` 400 for a collection."""
    with open_url(f"{endpoint}/{doi}?format=textfile") as resp:
        text = resp.read().decode("utf-8", errors="replace")
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "*/") + 1
    except StopIteration:
        start = 0
    return [ln.split("\t") for ln in lines[start:] if ln.strip()]


def image_column(header: list[str]) -> tuple[int, str, int | None] | None:
    """``(column, kind, hash_column)``; ``kind`` is ``"name"`` or ``"url"``.

    The parameter names are matched case-sensitively: PANGAEA spells the binary
    parameters ``IMAGE``/``Binary``, and an ordinary ``Image no/name`` column (SO268
    analysis-ready profiles) precedes the real ``IMAGE`` column and must not win.
    """
    for prefix in _NAME_PREFIXES:
        col = next((i for i, h in enumerate(header) if _is_param(h, prefix)), None)
        if col is not None:
            hash_col = next(
                (i for i, h in enumerate(header) if h.startswith(prefix) and "(HASH)" in h.upper()),
                None,
            )
            return col, "name", hash_col
    col = next(
        (i for i, h in enumerate(header) if h.upper().startswith(_URL_PREFIX) and "(" not in h),
        None,
    )
    return (col, "url", None) if col is not None else None


def _is_param(header: str, prefix: str) -> bool:
    return "(" not in header and (header == prefix or header.startswith(prefix + " "))


def table_items(
    rows: list[list[str]], numeric_id: str, download_base: str, key_prefix: str = ""
) -> Iterator[RemoteItem]:
    """One :class:`RemoteItem` per distinct image in a tab export (see module doc)."""
    found = image_column(rows[0]) if rows else None
    if found is None:
        return
    col, kind, hash_col = found
    seen: set[str] = set()
    for row in rows[1:]:
        value = row[col].strip() if col < len(row) else ""
        if not value:
            continue
        if kind == "url":
            url = value
            name = PurePosixPath(urllib.parse.urlsplit(value).path).name
        else:
            name = value
            url = f"{download_base}/{numeric_id}/files/{urllib.parse.quote(name)}"
        key = f"{key_prefix}{name}"
        if key in seen:
            continue
        seen.add(key)
        md5 = row[hash_col].strip() if hash_col is not None and hash_col < len(row) else ""
        yield RemoteItem(key=key, url=url, md5=md5 or None)


class PangaeaAdapter(BaseAdapter):
    name = "pangaea"

    @property
    def _endpoint(self) -> str:
        return str(self.params.get("endpoint", "https://doi.pangaea.de")).rstrip("/")

    @property
    def _download_base(self) -> str:
        return str(self.params.get("download_base", "https://download.pangaea.de/dataset")).rstrip(
            "/"
        )

    @property
    def _doi(self) -> str:
        return str(self.params["doi"])

    def resolve_version(self) -> str:
        m = _DOI_ID.search(self._doi)
        if not m:
            raise ValueError(f"{self.name} adapter: can't find a numeric id in doi {self._doi!r}")
        self._numeric_id = m.group(1)
        return str(self.params.get("version") or f"PANGAEA.{self._numeric_id}")

    def _table(self) -> list[list[str]]:
        return read_table(self._endpoint, self._doi)

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_numeric_id"):
            self.resolve_version()
        yield from table_items(self._table(), self._numeric_id, self._download_base)
