"""One PANGAEA dataset's tab-export file list (``adapter: pangaea``, WP-6d-A).

``GET <endpoint>/<doi>?format=textfile`` returns a ``/* ... */`` metadata header
(ending on a line that is exactly ``*/``) followed by a tab-separated table: a
header row, then one data row per event/sample. A dataset that publishes images
names them in an ``IMAGE ...`` column (not a full URL) and, per the dataset's own
``Files:`` line, every filename downloads from
``https://download.pangaea.de/dataset/<numeric-id>/files/<filename>``; a sibling
``IMAGE ... (Hash)`` column (when present) is the file's md5, verified on fetch.

A dataset with no ``IMAGE`` column enumerates zero items — the caller's D-R4
zero-item downgrade turns that into ``needs_adapter``, not a false ok (this
adapter never guesses a URL shape it can't verify from the table itself).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from . import BaseAdapter, RemoteItem
from ._http import open_url

_DOI_ID = re.compile(r"PANGAEA\.(\d+)", re.IGNORECASE)


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
            raise ValueError(f"pangaea adapter: can't find a numeric id in doi {self._doi!r}")
        self._numeric_id = m.group(1)
        return str(self.params.get("version") or f"PANGAEA.{self._numeric_id}")

    def _table(self) -> list[list[str]]:
        with open_url(f"{self._endpoint}/{self._doi}?format=textfile") as resp:
            text = resp.read().decode("utf-8", errors="replace")
        lines = text.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip() == "*/") + 1
        except StopIteration:
            start = 0
        return [ln.split("\t") for ln in lines[start:] if ln.strip()]

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_numeric_id"):
            self.resolve_version()
        rows = self._table()
        if not rows:
            return
        header = rows[0]
        image_col = next(
            (i for i, h in enumerate(header) if h.upper().startswith("IMAGE") and "(" not in h),
            None,
        )
        if image_col is None:
            return
        hash_col = next(
            (
                i
                for i, h in enumerate(header)
                if h.upper().startswith("IMAGE") and "(HASH)" in h.upper()
            ),
            None,
        )
        for row in rows[1:]:
            if image_col >= len(row) or not row[image_col].strip():
                continue
            name = row[image_col].strip()
            md5 = row[hash_col].strip() if hash_col is not None and hash_col < len(row) else None
            yield RemoteItem(
                key=name,
                url=f"{self._download_base}/{self._numeric_id}/files/{name}",
                md5=md5 or None,
            )
