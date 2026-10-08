"""Bounded sample fetching.

The purpose is verification, not bulk download. Roughly 100 items per source is enough
to prove that a declared layout matches reality — which is the failure mode synthetic
fixtures cannot catch. A fixture proves ``ImageMaskPairLoader`` works; it says nothing
about whether *this dataset* is actually laid out as ``images/`` + ``masks/``.

That distinction is not academic. Coralscapes was declared here as image/mask
directories; it is in fact HuggingFace parquet with ``image`` and ``label`` columns. The
loader was fine and the declaration was wrong, and only real data reveals which.

Fetchers **normalise** into the local layout the source declares, so the fetcher owns
source-specific unpacking and the loader stays generic.
"""

# ``marinedata.fetch`` is the one import path for every name below, so the imports that only
# re-export are deliberate (the old single module defined all of them at top level).
# ruff: noqa: F401

from __future__ import annotations

import json
import time  # part of the surface: tests patch ``fetch.time.sleep``
import urllib.parse
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from ..enums import AccessMethod
from ..models import Source
from .archive import (
    _check_tar_member_bounds,
    _check_zip_member_bounds,
    _extract_archive,
    _extract_remote_tar,
    _extract_remote_zip,
    _nested_zip_window,
    _safe_extract_path,
)
from .clients import _fetch_fathomnet_client, _fetch_s3, _unimplemented_client
from .constants import (
    _ARCHIVE_SUFFIXES,
    DATASETS_SERVER,
    DEFAULT_LIMIT,
    MAX_COMPRESSION_RATIO,
    MAX_DOWNLOAD_WITHOUT_RANGE,
    MAX_GET_BYTES,
    MAX_MEMBER_BYTES,
    MAX_TOTAL_EXTRACT_BYTES,
    USER_AGENT,
)
from .core import FetchError, FetchNotSupported, FetchResult, IntegrityError, cache_root
from .huggingface import _class_dir, _fetch_hf, _fetch_huggingface
from .integrity import _backoff, _check_complete, _retrying_read, _write, get_to_file, sample_digest
from .ranged import _get_range, _head, _RemoteFile
from .transport import (
    _get,
    _get_json,
    _get_stream,
    _require_digest_for_plaintext_http,
    _verify_digest,
)


def _fetch_http(source: Source, root: Path, limit: int) -> FetchResult:
    """Retrieve a single declared sample — a bare file, or a small archive extracted
    in place.

    Params:
        sample_url: a small, directly-downloadable file or archive (``.zip``, ``.tar``,
            ``.tar.gz``, ...). Required — we deliberately do not crawl arbitrary pages
            looking for data.

    Most multi-file layouts (``image-mask-pairs``, ``coco-json``, ``yolo-txt``) need
    more than one file, which is why this extracts archives rather than only writing
    the raw bytes: a bare-file fetch cannot satisfy any layout needing an ``images/`` +
    ``masks/`` pair, no matter how correct the URL is.

    A ``.zip`` over ``MAX_DOWNLOAD_WITHOUT_RANGE`` is read via HTTP Range when the host
    supports it — only the central directory and the first ``limit`` members are ever
    transferred, regardless of how large the archive is — and refused outright when it
    does not, rather than silently downloaded in full: a "bounded verification sample"
    that pulls 26 GB to read 100 images is not bounded. A zip small enough to download
    safely is downloaded whole even when Range is available: a few hundred tiny ranged
    reads (roughly two per member — a local-header read, then the payload) is *slower*
    than one bulk transfer once the whole thing already fits comfortably in memory.

    Params:
        nested_archive: name of a member, stored uncompressed inside the outer zip,
            to read as a zip of its own rather than extracting the outer zip's direct
            members — e.g. #DeOlhoNosCorais publishes one 5.15GB zip containing two
            multi-GB zips-of-zips; the real image/mask pairs are one level deeper than
            Zenodo's own file listing reaches. Requires Range support on the outer
            file (the nested member is windowed, never downloaded to find). Applies
            to ``sample_url`` only.
        extra_sample_url: a second file or archive, extracted into the same ``root``
            alongside ``sample_url`` — for a source split across two independent
            downloads, e.g. UIEB's paired raw/reference halves ship as separate
            Kaggle archives with no single file containing both. Point the loader's
            own ``images_dir``/``masks_dir`` at whatever each archive's real internal
            folder is named; nothing here renames them.
        fetch_style: ``"manifest"`` dispatches to :func:`fetch_pangaea_manifest`
            instead of the archive logic below — for a source whose real images are
            individually addressable URLs inside a small metadata manifest, rather
            than embedded in the archive itself (e.g. Heron Reef's PANGAEA export).
        tar_start_offset: byte offset to start reading a remote ``.tar`` from,
            instead of 0 — a tar has no index, so reaching a section that sits deep
            inside a huge archive (e.g. DeepFish's Segmentation/ folder, the last
            ~1.4% of a 7.6GB tar behind Classification/ and Localization/) would
            otherwise mean reading every header sequentially from the start, which
            is many thousands of round trips. Found once by binary-searching the
            remote file for a real tar header near a candidate byte offset (see
            ``_extract_remote_tar``'s docstring for why this only works for tar,
            never a compressed variant) and pinned here so every later fetch skips
            straight to it. Requires Range support; applies to ``sample_url`` only.
    """
    if str(source.access.params.get("fetch_style", "")) == "manifest":
        from ..fetchers_remote import fetch_pangaea_manifest

        return fetch_pangaea_manifest(source, root, limit)

    sample_url = str(source.access.params.get("sample_url") or "")
    if not sample_url:
        raise FetchNotSupported(
            f"{source.id}: no `sample_url` declared. Add one pointing at a small "
            f"downloadable sample, or fetch the data manually."
        )
    _require_digest_for_plaintext_http(source.id, sample_url, source.access.sha256)
    nested_archive = source.access.params.get("nested_archive")
    tar_start_offset = int(source.access.params.get("tar_start_offset", 0) or 0)
    count = _fetch_one(
        source.id,
        sample_url,
        root,
        limit,
        nested_archive=nested_archive,
        tar_start_offset=tar_start_offset,
        expected_sha256=source.access.sha256,
    )

    extra_url = source.access.params.get("extra_sample_url")
    if extra_url:
        _require_digest_for_plaintext_http(source.id, str(extra_url), source.access.sha256)
        count += _fetch_one(source.id, str(extra_url), root, limit)

    return FetchResult(source.id, root, count, "http", truncated=True)


def _fetch_one(
    source_id: str,
    url: str,
    root: Path,
    limit: int,
    *,
    nested_archive: object = None,
    tar_start_offset: int = 0,
    expected_sha256: str | None = None,
) -> int:
    """Retrieve one declared file or archive into ``root``. The single-URL body of
    :func:`_fetch_http`, factored out so a second URL (``extra_sample_url``) can reuse
    the exact same archive/size/Range handling rather than a parallel copy of it.

    ``expected_sha256`` is checked only against a payload fetched *in full* (a bare
    file, or an archive small enough to download whole) — the Range-read branches below
    never hold the complete artefact in memory, so there is nothing there to check a
    whole-file digest against. See :func:`_verify_digest`.
    """
    name = Path(urllib.parse.urlparse(url).path).name or "sample.bin"

    if nested_archive:
        size, supports_range = _head(url)
        if not (supports_range and size):
            raise FetchNotSupported(
                f"{source_id}: nested_archive is set but {url} does not support "
                f"Range requests (or its size could not be determined) — reading a "
                f"member out of it without downloading the whole outer archive needs "
                f"Range support."
            )
        with zipfile.ZipFile(_RemoteFile(url, size, get_range=_get_range)) as outer:
            offset, member_size = _nested_zip_window(outer, str(nested_archive))
        inner = _RemoteFile(
            url, size, window_offset=offset, window_size=member_size, get_range=_get_range
        )
        return _extract_remote_zip(inner, root, limit, label=f"{name}:{nested_archive}")

    if name.lower().endswith(".zip"):
        size, supports_range = _head(url)
        if size and size <= MAX_DOWNLOAD_WITHOUT_RANGE:
            payload = _get(url)
            _verify_digest(source_id, url, payload, expected_sha256)
            return _extract_archive(payload, root, name, limit)
        if supports_range and size:
            return _extract_remote_zip(
                _RemoteFile(url, size, get_range=_get_range), root, limit, label=name
            )
        if size and size > MAX_DOWNLOAD_WITHOUT_RANGE:
            raise FetchNotSupported(
                f"{source_id}: {url} is a {size / 1e9:.1f} GB zip and the host does "
                f"not support Range requests, so a bounded sample cannot be read from it "
                f"without downloading the whole archive. Find a smaller published subset, "
                f"or fetch it manually."
            )
        payload = _get(url)
        _verify_digest(source_id, url, payload, expected_sha256)
        return _extract_archive(payload, root, name, limit)

    if name.lower().endswith(".tar"):
        size, supports_range = _head(url)
        if size and size <= MAX_DOWNLOAD_WITHOUT_RANGE and not tar_start_offset:
            payload = _get(url)
            _verify_digest(source_id, url, payload, expected_sha256)
            return _extract_archive(payload, root, name, limit)
        if supports_range and size:
            remote = _RemoteFile(
                url,
                size,
                window_offset=tar_start_offset,
                window_size=size - tar_start_offset,
                get_range=_get_range,
            )
            return _extract_remote_tar(remote, root, limit, label=name)
        if size and size > MAX_DOWNLOAD_WITHOUT_RANGE:
            raise FetchNotSupported(
                f"{source_id}: {url} is a {size / 1e9:.1f} GB tar and the host does "
                f"not support Range requests, so a bounded sample cannot be read from "
                f"it without downloading the whole archive. Find a smaller published "
                f"subset, or fetch it manually."
            )
        payload = _get(url)
        _verify_digest(source_id, url, payload, expected_sha256)
        return _extract_archive(payload, root, name, limit)

    payload = _get(url)
    _verify_digest(source_id, url, payload, expected_sha256)
    if name.lower().endswith(_ARCHIVE_SUFFIXES):
        return _extract_archive(payload, root, name, limit)
    _write(root / name, payload)
    return 1


_API_CLIENTS: Mapping[str, Callable[[Source, Path, int], FetchResult]] = MappingProxyType(
    {
        "fathomnet": _fetch_fathomnet_client,
        "obis": _unimplemented_client("obis"),
        "allen-coral-atlas": _unimplemented_client("allen-coral-atlas"),
        "copernicus-marine": _unimplemented_client("copernicus-marine"),
        "coralnet": _unimplemented_client("coralnet"),
        "atlantis": _unimplemented_client("atlantis"),
    }
)

# Clients in `_API_CLIENTS` that actually fetch rather than raise `NotImplementedError`.
# `auto_fetchable` consults this instead of calling the client, so it stays a predicate
# and never raises.
_IMPLEMENTED_API_CLIENTS: frozenset[str] = frozenset({"fathomnet"})


def _fetch_api(source: Source, root: Path, limit: int) -> FetchResult:
    """Dispatches ``method: api`` sources to the client their ``access.params.client``
    names.

    Six ``api`` sources exist in the registry today — fathomnet, obis,
    allen-coral-atlas, copernicus-globcolour and (once re-valued) coralnet and
    atlantis-synthetic-depth — each backed by a genuinely different upstream API.
    Dispatch is explicit on ``client`` so a source can never be silently routed to
    another source's client; there is no default and no fallback.
    """
    client = source.access.params.get("client")
    if client is None:
        raise FetchError(
            f"{source.id}: access.params.client is not set — known clients: "
            f"{', '.join(sorted(_API_CLIENTS))}"
        )
    fetcher = _API_CLIENTS.get(str(client))
    if fetcher is None:
        raise FetchError(
            f"{source.id}: unknown api client '{client}' — known clients: "
            f"{', '.join(sorted(_API_CLIENTS))}"
        )
    return fetcher(source, root, limit)


_FETCHERS = {
    AccessMethod.HUGGINGFACE: _fetch_hf,
    AccessMethod.HTTP: _fetch_http,
    AccessMethod.ZENODO: _fetch_http,
    AccessMethod.S3: _fetch_s3,
    AccessMethod.API: _fetch_api,
}


def auto_fetchable(source: Source) -> bool:
    """Whether ``fetch_sample`` could run without a human — no network required to ask.

    Gated, request-only and scrape sources need a human regardless of access method,
    which is exactly what ``fetch_sample`` itself checks before actually fetching; this
    exposes that same static answer for reporting (``marinedata doctor``) without
    reaching for the network. For ``method: api`` this also checks that the declared
    ``access.params.client`` is a known, implemented client — a client that is only
    registered as a ``NotImplementedError`` stub is not auto-fetchable. This is a pure
    predicate: unlike ``_fetch_api`` it never raises on a missing or unknown client.
    """
    if source.access.gated or source.access.method not in _FETCHERS:
        return False
    if source.access.method is AccessMethod.API:
        client = source.access.params.get("client")
        return client is not None and str(client) in _IMPLEMENTED_API_CLIENTS
    return True


def fetch_sample(
    source: Source,
    *,
    limit: int = DEFAULT_LIMIT,
    root: Path | None = None,
    force: bool = False,
) -> FetchResult:
    """Fetch a bounded verification sample.

    Args:
        source: Registry entry to fetch.
        limit: Maximum items. Kept small on purpose — this verifies layout, not scale.
        root: Destination. Defaults to ``<cache>/<source_id>``.
        force: Re-fetch even if a sample is already cached.

    Raises:
        FetchNotSupported: gated, request-only, S3 and scrape sources need a human.
        FetchError: the fetch failed or produced nothing.
    """
    target = root or (cache_root() / source.id)
    marker = target / "_fetch.json"

    if marker.is_file() and not force:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        cached_items = int(existing.get("items", 0))
        cached_truncated = bool(existing.get("truncated", True))
        # A truncated cache recorded fewer items than the source actually has. It only
        # satisfies a request that asks for no more than it already holds — a bigger or
        # unbounded (release) ``limit`` must re-fetch rather than silently hand back a
        # partial tree as if it were complete (the coralscop-masks-rs release bug: a
        # bounded 5-image verification sample was reused for a 10,000,000-item release
        # fetch, so ``metadata.parquet``'s 38,928 rows outran the images on disk).
        if not cached_truncated or cached_items >= limit:
            return FetchResult(
                source.id,
                target,
                cached_items,
                str(existing.get("method", "cache")),
                truncated=cached_truncated,
            )

    from ..fetchers_remote import fetch_staged_root, staged_root_params

    # A pinned staged tree declared in `access.params.staged_*` roots the source regardless of its
    # fetch method (atlantis stays `api`/`atlantis`; its release root is the staged tree).
    fetcher = (
        fetch_staged_root
        if staged_root_params(source) is not None
        else _FETCHERS.get(source.access.method)
    )
    if fetcher is None:
        raise FetchNotSupported(
            f"{source.id}: access method '{source.access.method.value}' cannot be "
            f"fetched automatically. "
            + (
                "It is gated — a human must accept the terms first."
                if source.access.gated
                else "Retrieve it manually and point the loader at the local path."
            )
        )

    target.mkdir(parents=True, exist_ok=True)
    result = fetcher(source, target, limit)
    marker.write_text(json.dumps(result.manifest(), indent=2), encoding="utf-8")
    return result


get_bytes = _get
"""Public alias for :func:`_get` — a retrying GET returning the whole response body.
For :mod:`marinedata.ingest`, which needs the full archive rather than a sample."""

extract_archive = _extract_archive
"""Public alias for :func:`_extract_archive` — extract an in-memory archive to a
directory, bounded to ``limit`` files (``0`` means every member)."""

get_stream = _get_stream
"""Public alias for :func:`_get_stream` — a retrying GET that streams straight to a
file (D2a), for :mod:`marinedata.ingest_parquet`, which downloads ~450 MB shards."""
