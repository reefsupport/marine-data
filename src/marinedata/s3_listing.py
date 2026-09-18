"""S3 REST listing and stem-grouping for an anonymous points source (D3a2).

Split out of :mod:`marinedata.ingest_s3` purely to keep that module under its
400-line cap (same reason :mod:`marinedata.staging_finish` exists). Plain
``urllib`` + ``xml.etree`` only — the package adds no boto3/requests dependency —
so this reads the raw, unsigned ``ListObjectsV2`` XML response the same way any
anonymous client would (``?list-type=2&prefix=...``, no credentials, no SDK).
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from .checksums import DOWNLOAD_USER_AGENT
from .ingest import IngestError

_LIST_TIMEOUT = 60


@dataclass(frozen=True)
class S3Plan:
    """Per-source facts for a points corpus living under one anonymous S3 prefix.

    Deliberately carries no ``version``, ``listing_digest`` or ``expected_*``
    field: those are per-listing facts a live bucket produces fresh on every
    list, never plan constants a maintainer hand-edits (2026-09-18-D3a2 brief,
    which overrides the earlier design note that had them here).
    """

    bucket: str
    prefix: str
    endpoint: str
    image_suffix: str
    sidecar_suffixes: tuple[str, ...]
    annotations_key: str
    stem_pattern: re.Pattern[str]
    partition: str
    annotated_only: bool
    license_text: str
    points_schema_id: str


def _base_url(endpoint: str) -> str:
    """``endpoint`` as a request origin. A bare host (production:
    ``s3.amazonaws.com``) is assumed HTTPS; a value already carrying a scheme
    (tests: ``http://127.0.0.1:<port>``) is used verbatim — a local
    ``http.server`` has no TLS."""
    return endpoint if "://" in endpoint else f"https://{endpoint}"


def _object_url(plan: S3Plan, key: str) -> str:
    """Path-style object URL — works against any host, including a bare
    IP:port that cannot resolve a bucket subdomain the way virtual-hosted-style
    addressing needs."""
    return f"{_base_url(plan.endpoint)}/{plan.bucket}/{urllib.parse.quote(key, safe='/')}"


def _listing_url(plan: S3Plan, *, continuation_token: str | None) -> str:
    params = {"list-type": "2", "prefix": plan.prefix}
    if continuation_token:
        params["continuation-token"] = continuation_token
    return f"{_base_url(plan.endpoint)}/{plan.bucket}?{urllib.parse.urlencode(params)}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(elem: ET.Element, name: str) -> str | None:
    for child in elem:
        if _local_name(child.tag) == name:
            return child.text
    return None


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem if _local_name(child.tag) == name]


def list_prefix(plan: S3Plan) -> Iterator[tuple[str, int, str]]:
    """Yield ``(key, size, etag)`` for every object under ``plan.prefix``,
    paging through ``ListObjectsV2`` via ``continuation-token`` until
    ``IsTruncated`` is ``false``. Namespace-agnostic parsing: a real S3 response
    carries the ``http://s3.amazonaws.com/doc/2006-03-01/`` xmlns, a minimal
    test fixture may not — both parse the same way here.
    """
    token: str | None = None
    while True:
        request = urllib.request.Request(
            _listing_url(plan, continuation_token=token),
            headers={"User-Agent": DOWNLOAD_USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=_LIST_TIMEOUT) as response:
            root = ET.fromstring(response.read())
        for contents in _children(root, "Contents"):
            key = _child_text(contents, "Key") or ""
            size = int(_child_text(contents, "Size") or "0")
            etag = (_child_text(contents, "ETag") or "").strip('"')
            yield key, size, etag
        if (_child_text(root, "IsTruncated") or "false").lower() != "true":
            return
        token = _child_text(root, "NextContinuationToken")


def listing_digest(entries: Iterable[tuple[str, int, str]]) -> str:
    """sha256 over the SORTED ``key\\tsize\\tetag\\n`` lines — order-independent,
    so two listings of an unchanged bucket (pages can arrive in a different
    order under concurrent writes) digest identically."""
    lines = sorted(f"{key}\t{size}\t{etag}\n" for key, size, etag in entries)
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def group_by_stem(
    entries: Iterable[tuple[str, int, str]], plan: S3Plan
) -> dict[str, dict[str, tuple[str, int, str]]]:
    """Group listed keys (minus ``plan.annotations_key``) by stem, keyed by
    which suffix each carries — the image suffix or a recognised sidecar.
    Raises on a key that is neither: an unrecognised object under the prefix is
    a corpus change this plan does not know how to stage, never silently
    skipped. Longest-suffix-first so e.g. ``_thumbnail.png`` is never mistaken
    for the plain ``.png`` image suffix.
    """
    known = sorted((plan.image_suffix, *plan.sidecar_suffixes), key=len, reverse=True)
    groups: dict[str, dict[str, tuple[str, int, str]]] = {}
    for key, size, etag in entries:
        if key == plan.annotations_key:
            continue
        if not key.startswith(plan.prefix):
            raise IngestError(f"{key!r} is outside prefix {plan.prefix!r}")
        remainder = key[len(plan.prefix) :]
        suffix = next((s for s in known if remainder.endswith(s)), None)
        if suffix is None:
            raise IngestError(f"{key!r}: no known suffix among {known} and not the annotations key")
        stem = remainder[: -len(suffix)]
        if not plan.stem_pattern.fullmatch(stem):
            raise IngestError(
                f"{key!r}: stem {stem!r} does not match {plan.stem_pattern.pattern!r}"
            )
        groups.setdefault(stem, {})[suffix] = (key, size, etag)
    return groups


def select_stems(
    groups: Mapping[str, tuple[str, int, str]],
    annotated_stems: Iterable[str],
    *,
    annotated_only: bool,
    cap_bytes: int | None,
) -> list[str]:
    """Stems to stage, lexicographic, image bytes cumulative under
    ``cap_bytes`` (``None`` -> every eligible stem). ``groups`` here is stem ->
    its IMAGE ``(key, size, etag)`` only — the image-suffix slice of
    :func:`group_by_stem`'s result, never the sidecar entries.
    """
    wanted = set(annotated_stems)
    eligible = sorted(stem for stem in groups if not annotated_only or stem in wanted)
    if cap_bytes is None:
        return eligible
    selected: list[str] = []
    total = 0
    for stem in eligible:
        size = groups[stem][1]
        if total + size > cap_bytes:
            break
        selected.append(stem)
        total += size
    return selected
