"""A pinned staged tree on ``rs-storage-open`` says where to fetch it.

``fetchers_remote._s3_endpoint`` defaults the endpoint to ``s3.amazonaws.com`` and ``fetch_s3``
takes the prefix from ``access.params.prefix`` (default ``""``). Neither reads ``access.uri``. Six
pinned sources named only ``uri: s3://rs-storage-open/...``, so the release VM requested
``https://rs-storage-open.s3.amazonaws.com/CHECKSUMS.sha256`` and got HTTP 404.

Every source on the open bucket with ``method: s3``, a ``checksums.root_digest`` and
``layout: staged-tree`` (the sources ``fetchers_remote._is_pinned_staged_tree`` routes to the
manifest fetch) must therefore carry ``bucket``, ``endpoint`` and ``prefix`` itself. The prefix is
the path of its own ``uri``, ending in ``/`` whether or not the ``uri`` does. No network: the
registry is read from disk.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from marinedata.enums import AccessMethod
from marinedata.fetchers_remote import _is_pinned_staged_tree
from marinedata.registry import Registry

BUCKET = "rs-storage-open"
ENDPOINT = "hel1.your-objectstorage.com"

SOURCES = sorted(
    (
        source
        for source in Registry.load()
        if source.access.method == AccessMethod.S3
        and (source.access.uri or "").startswith(f"s3://{BUCKET}/")
        and _is_pinned_staged_tree(source)
    ),
    key=lambda source: source.id,
)


def test_the_guard_covers_the_pinned_staged_trees() -> None:
    assert SOURCES, "no pinned staged tree on rs-storage-open: the guard below checks nothing"


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.id)
def test_pinned_staged_tree_declares_bucket_endpoint_and_prefix(source) -> None:
    params = source.access.params
    uri_path = urlparse(source.access.uri).path.lstrip("/")
    assert params.get("bucket") == BUCKET, f"{source.id}: params.bucket"
    assert params.get("endpoint") == ENDPOINT, f"{source.id}: params.endpoint"
    assert params.get("prefix") == uri_path.rstrip("/") + "/", f"{source.id}: params.prefix"
