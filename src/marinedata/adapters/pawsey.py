"""Per-file HTTPS listings that don't fit the S3 ``bucket`` adapter (WP-6d-A).

``adapter: pawsey-portal``: a public Pawsey/Nectar Swift object-storage
container. Swift's own listing convention is ``GET <container_url>?format=json``
(paginated via ``marker=<last-object-name>``), returning ``[{name, bytes, hash,
last_modified}, ...]``; each object then downloads from
``<container_url>/<name>``. A container whose host has migrated to a JS-only
front end (observed for storage.pawsey.org.au 2026-09-25 — the plain listing
now 200s with an SPA shell, not JSON) answers zero items here, which the
caller's D-R4 rule correctly turns into ``needs_adapter``/``needs_yohan``
rather than a false ok.

``adapter: frdr-https``: FRDR (and similar repositories) that expose no plain
enumerable file index publish a per-file HTTPS list some other way (a resolved
manifest, or a small hand-verified set); ``params.urls`` takes that list, same
shape as the ``http`` adapter's plain-URL mode. ``params.version`` is required
for both (neither listing convention carries a version of its own).
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator
from pathlib import PurePosixPath

from . import BaseAdapter, RemoteItem
from ._http import open_url


class PawseyPortalAdapter(BaseAdapter):
    name = "pawsey-portal"

    def resolve_version(self) -> str:
        if not self.params.get("version"):
            raise ValueError("pawsey-portal adapter: params.version is required")
        return str(self.params["version"])

    def list_items(self) -> Iterator[RemoteItem]:
        base = str(self.params["container_url"]).rstrip("/")
        prefix = self.params.get("prefix", "")
        marker = ""
        while True:
            qs = {"format": "json"}
            if prefix:
                qs["prefix"] = prefix
            if marker:
                qs["marker"] = marker
            try:
                with open_url(f"{base}?{urllib.parse.urlencode(qs)}") as resp:
                    if "json" not in resp.headers.get("Content-Type", ""):
                        return
                    entries = json.loads(resp.read())
            except Exception:
                return
            if not entries:
                return
            for entry in entries:
                name = str(entry["name"])
                yield RemoteItem(
                    key=name,
                    url=f"{base}/{name}",
                    size=entry.get("bytes"),
                    md5=entry.get("hash"),
                )
            marker = entries[-1]["name"]


class FrdrHttpsAdapter(BaseAdapter):
    name = "frdr-https"

    def resolve_version(self) -> str:
        if not self.params.get("version"):
            raise ValueError("frdr-https adapter: params.version is required")
        return str(self.params["version"])

    def list_items(self) -> Iterator[RemoteItem]:
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
