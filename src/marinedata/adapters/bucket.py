"""Anonymous public S3/GCS bucket listing (``adapter: bucket``).

Reuses :func:`marinedata.s3_listing.list_prefix` (unsigned ``ListObjectsV2`` XML; GCS's
XML API answers the same ``list-type=2`` request at ``storage.googleapis.com``). A bucket
has no version, so the version is the digest of the listing itself (``list-<12 hex>``):
if upstream adds or changes an object, the next ingest lands under a new version.

Params: ``endpoint`` (``s3.amazonaws.com``, ``s3.<region>.amazonaws.com``,
``storage.googleapis.com``, or ``http://127.0.0.1:<port>`` in tests), ``bucket``,
``prefix``, ``version``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

from .. import s3_listing
from . import BaseAdapter, RemoteItem

_MD5 = re.compile(r"^[0-9a-f]{32}$")


class BucketAdapter(BaseAdapter):
    name = "bucket"

    def _plan(self) -> s3_listing.S3Plan:
        p = self.params
        return cast(
            s3_listing.S3Plan,
            SimpleNamespace(
                endpoint=str(p["endpoint"]),
                bucket=str(p["bucket"]),
                prefix=str(p.get("prefix", "")),
            ),
        )

    def resolve_version(self) -> str:
        self._listing = sorted(s3_listing.list_prefix(self._plan()))
        digest = s3_listing.listing_digest(self._listing)
        return str(self.params.get("version") or f"list-{digest[:12]}")

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_listing"):
            self.resolve_version()
        plan = self._plan()
        for key, size, etag in self._listing:
            if key.endswith("/"):
                continue
            yield RemoteItem(
                key=key,
                url=s3_listing._object_url(plan, key),
                size=size,
                md5=etag if _MD5.match(etag) else None,
            )
