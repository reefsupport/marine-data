"""Plain HTTP(S) files and Zenodo records (``adapter: http`` / ``adapter: zenodo``).

Zenodo: ``params.zenodo_record`` -> ``GET <endpoint>/api/records/<id>``; the record must
be ``access_right: open`` (else :class:`AccessRefused`). Record ids are immutable, so the
version is the record's ``metadata.version`` (or ``record-<id>``) and every file's md5
from the record is verified on fetch.

Plain HTTP: ``params.urls`` is a list of URLs or ``{url, key, size, md5, sha256}`` dicts
and ``params.version`` is REQUIRED (a bare URL carries no version signal).

WP-6k (D-AH 4): a ``.zip`` item is never spooled to local disk, whatever its size. Its
central directory is read over HTTP Range (:func:`~._range.open_remote_zip`) and each
member is expanded into its own item, read with one further ranged GET
(:func:`~._range.read_member`, Deflate64-safe) straight into memory.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import urllib.parse
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from . import SPOOLED, STREAMABLE, AccessRefused, BaseAdapter, Fetched, RemoteItem, suffix_of
from ._http import HashingReader, get_json
from ._range import RemoteFile, open_remote_zip, read_member

log = logging.getLogger(__name__)


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

    # -- WP-6k: zip members over HTTP Range, never spooled (D-AH 4) ---------------------
    def enumerate(self) -> Iterator[RemoteItem]:
        for item in super().enumerate():
            if suffix_of(item.key) == ".zip" and not self.is_label(item.key):
                yield from self._expand_zip(item)
            else:
                yield item

    def _expand_zip(self, item: RemoteItem) -> Iterator[RemoteItem]:
        if item.size is None:
            raise ValueError(f"{item.key}: a remote zip needs a declared size (D-AH 4)")
        zf, rf = open_remote_zip(item.url, item.size)
        if not hasattr(self, "_zip_members"):
            self._zip_members: dict[str, tuple[zipfile.ZipFile, RemoteFile, zipfile.ZipInfo]] = {}
            self._zip_lock = threading.Lock()
        infos = [
            i
            for i in zf.infolist()
            if not i.is_dir()
            and "__MACOSX" not in i.filename
            # same decodable-suffix-or-label filter BaseAdapter.enumerate() applies to
            # top-level items: a stray README/LICENSE/script inside the zip must not
            # crash the whole source (D-AH 4).
            and (suffix_of(i.filename) in STREAMABLE + SPOOLED or self.is_label(i.filename))
        ]
        log.info(
            "%s: remote zip %d entries (%d range requests, %d B read so far)",
            item.key, len(infos), rf.requests, rf.fetched,
        )  # fmt: skip
        for info in sorted(infos, key=lambda i: i.filename):
            key = f"{item.key}#{info.filename}"
            self._zip_members[key] = (zf, rf, info)
            yield RemoteItem(key=key, url=item.url, size=info.file_size)

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        member = getattr(self, "_zip_members", {}).get(item.key)
        if member is None:
            return super().fetch(item, tmp_dir)
        zf, rf, info = member
        with self._zip_lock:  # one cache window per archive: serialise member reads
            data = read_member(zf, rf, info)
        if suffix_of(info.filename) in SPOOLED:
            # this member itself needs on-disk random access (video/rar/nested zip/parquet):
            # spool just the member, bounded by its own size — the container is still never
            # spooled whole (D-AH 4).
            dest = tmp_dir / "fetch" / PurePosixPath(info.filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return Fetched(item, path=dest, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        return Fetched(item, stream=HashingReader(io.BytesIO(data)))
