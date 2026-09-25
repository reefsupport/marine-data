"""Plain HTTP(S) files and Zenodo records (``adapter: http`` / ``adapter: zenodo``).

Zenodo: ``params.zenodo_record`` -> ``GET <endpoint>/api/records/<id>``; the record must
be ``access_right: open`` (else :class:`AccessRefused`). Record ids are immutable, so the
version is the record's ``metadata.version`` (or ``record-<id>``) and every file's md5
from the record is verified on fetch.

Plain HTTP: ``params.urls`` is a list of URLs or ``{url, key, size, md5, sha256}`` dicts
and ``params.version`` is REQUIRED (a bare URL carries no version signal).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from pathlib import PurePosixPath

from . import AccessRefused, BaseAdapter, RemoteItem
from ._http import get_json


class HttpAdapter(BaseAdapter):
    name = "http"

    def _zenodo(self) -> dict:
        base = str(self.params.get("endpoint", "https://zenodo.org")).rstrip("/")
        rec = get_json(f"{base}/api/records/{int(self.params['zenodo_record'])}")
        assert isinstance(rec, dict)
        access = (rec.get("metadata") or {}).get("access_right", "open")
        if access != "open":
            raise AccessRefused(
                f"{base}/records/{self.params['zenodo_record']}",
                f"access_right={access}: request access on the record page",
            )
        return rec

    def resolve_version(self) -> str:
        if "zenodo_record" in self.params:
            self._record = self._zenodo()
            meta = self._record.get("metadata") or {}
            return str(
                self.params.get("version")
                or meta.get("version")
                or f"record-{self.params['zenodo_record']}"
            )
        if not self.params.get("version"):
            raise ValueError("http adapter: params.version is required for plain URLs")
        return str(self.params["version"])

    def list_items(self) -> Iterator[RemoteItem]:
        if "zenodo_record" in self.params:
            if not hasattr(self, "_record"):
                self.resolve_version()
            for f in self._record.get("files") or []:
                checksum = str(f.get("checksum") or "")
                yield RemoteItem(
                    key=str(f["key"]),
                    url=str((f.get("links") or {})["self"]),
                    size=int(f["size"]) if f.get("size") is not None else None,
                    md5=checksum.split(":", 1)[1] if checksum.startswith("md5:") else None,
                )
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
