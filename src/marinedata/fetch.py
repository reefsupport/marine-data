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

from __future__ import annotations

import hashlib
import http.client
import json
import os
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from .checksums import CHUNK_SIZE, DOWNLOAD_USER_AGENT, stream_into
from .enums import AccessMethod
from .models import Source

DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_LIMIT = 100
USER_AGENT = DOWNLOAD_USER_AGENT
"""Alias, not a second literal (D3a2 tidy) — see :data:`marinedata.checksums.
DOWNLOAD_USER_AGENT`."""


class FetchError(Exception):
    """A sample could not be retrieved."""


class FetchNotSupported(FetchError):
    """No fetcher exists for this access method.

    Raised rather than silently returning nothing, so "we cannot automate this" is
    distinguishable from "this dataset is empty". Gated and request-only sources land
    here by design — they require a human to accept terms, which is exactly the
    ``contract_gated`` situation the licence model already tracks.
    """


def cache_root() -> Path:
    """Cache location. ``MARINEDATA_CACHE`` overrides."""
    env = os.environ.get("MARINEDATA_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "marinedata"


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    root: Path
    items: int
    method: str
    truncated: bool
    """True when the source has more items than were fetched — always, for a sample."""

    def manifest(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "items": self.items,
            "method": self.method,
            "truncated": self.truncated,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Bounded verification sample. NOT the full dataset."
                if self.truncated
                else "Complete fetch of the source, not a bounded sample."
            ),
        }


def _get(
    url: str,
    *,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    retries: int = 3,
) -> bytes:
    """GET with retries on transient faults.

    Retrying is not optional at this scale. Fetching a few hundred images means a few
    hundred connections, and some fraction of those will be dropped by the far end
    mid-transfer — the first real multi-source build died on
    ``http.client.RemoteDisconnected``, which is neither an ``HTTPError`` nor a
    ``URLError`` and so escaped the original handler entirely.

    4xx is not retried: a 404 will still be a 404 — except 429, which is the server
    asking a concurrent fetch to slow down. 5xx and connection-level faults (timeouts
    included) are retried with exponential backoff (:func:`_backoff`).
    """
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            # RemoteDisconnected, BadStatusLine, IncompleteRead, ConnectionReset, and
            # socket timeouts all land here.
            last = exc
        if attempt < retries - 1:
            time.sleep(_backoff(attempt))
    raise FetchError(f"failed after {retries} attempt(s) for {url}: {last}")


def _backoff(attempt: int) -> float:
    """Seconds to wait after failed attempt ``attempt`` (0-based): 1.5, 3, 6, 12, … capped
    at 30. The first two steps equal the old linear schedule, so a default 3-attempt
    ``_get`` waits exactly as long as it always did."""
    return min(30.0, 1.5 * (2**attempt))


def _get_stream(url: str, dest: Path, *, timeout: int = 60, retries: int = 3) -> int:
    """GET ``url``, writing the response body straight to ``dest`` in
    :data:`marinedata.checksums.CHUNK_SIZE` pieces rather than buffering it whole in
    memory the way :func:`_get` does (D2a: an HF parquet shard is ~450 MB, and every
    shard would otherwise be held in RAM at once).

    Same retry policy as :func:`_get`: a 4xx other than 429 raises immediately; a 5xx or a
    connection-level fault retries a fresh GET from the start, so ``dest`` is truncated
    and rewritten on each attempt rather than appended to. Returns the number of bytes
    written.
    """
    last: Exception | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                dest.open("wb") as handle,
            ):
                written = 0
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    written += len(chunk)
            return written
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(_backoff(attempt))
    raise FetchError(f"failed after {retries} attempt(s) for streaming GET of {url}: {last}")


def _get_json(url: str, *, timeout: int = 60) -> dict:
    try:
        return json.loads(_get(url, timeout=timeout))
    except json.JSONDecodeError as exc:
        raise FetchError(f"non-JSON response from {url}") from exc


def _class_dir(value: object, names: list[str] | None) -> str:
    """Directory name for a class label, resolving ClassLabel integers to their names.

    A label of ``74`` becomes ``Sparisoma_viride`` when the feature schema supplies the
    names. Without this the on-disk vocabulary is a set of integers that no crosswalk can
    match and no reviewer can read.
    """
    if names is not None and isinstance(value, int) and 0 <= value < len(names):
        raw = str(names[value])
    else:
        raw = str(value)
    # Directory-safe, but preserve the name — it IS the label.
    return raw.replace("/", "_").replace("\\", "_").strip() or "unlabelled"


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fetch_huggingface(source: Source, root: Path, limit: int) -> FetchResult:
    """Pull rows via the HF datasets-server and materialise them locally.

    Params (on ``access.params``):
        hf_id: ``owner/name``. Falls back to parsing ``access.uri``.
        config, split: default ``"default"`` / ``"train"``
        image_column: column holding the image. Default ``"image"``
        mask_column: optional column holding a segmentation mask.
        label_column: optional column holding a class label (written as directories).
    """
    params = dict(source.access.params)
    hf_id = str(params.get("hf_id") or "")
    if not hf_id and source.access.uri:
        hf_id = urllib.parse.urlparse(source.access.uri).path.removeprefix("/datasets/").strip("/")
    if not hf_id:
        raise FetchError(f"{source.id}: cannot determine the HuggingFace dataset id")

    config = str(params.get("config", "default"))
    split = str(params.get("split", "train"))
    image_col = str(params.get("image_column", "image"))
    mask_col = params.get("mask_column")
    label_col = params.get("label_column")
    audio_col = params.get("audio_column")

    # The rows endpoint caps `length` at 100 — which is the sample size we want anyway.
    query = urllib.parse.urlencode(
        {
            "dataset": hf_id,
            "config": config,
            "split": split,
            "offset": 0,
            "length": min(limit, 100),
        }
    )
    payload = _get_json(f"{DATASETS_SERVER}/rows?{query}")
    if "error" in payload:
        raise FetchError(f"{source.id}: datasets-server error — {payload['error']}")

    # ClassLabel columns come back as INTEGERS; the human-readable names live in the
    # feature schema. Materialising the integer as a directory name produces label
    # vocabularies like {"13", "74", "96"} — structurally valid, semantically useless,
    # and impossible to crosswalk. Resolve them here.
    class_names: dict[str, list[str]] = {}
    for feature in payload.get("features", []):
        ftype = feature.get("type", {})
        if ftype.get("_type") == "ClassLabel" and ftype.get("names"):
            class_names[feature["name"]] = list(ftype["names"])

    rows = payload.get("rows", [])
    if not rows:
        raise FetchError(f"{source.id}: datasets-server returned no rows for {hf_id}")

    count = 0
    for index, entry in enumerate(rows):
        row = entry.get("row", {})

        if audio_col:
            # Audio rows expose a list of encodings; take the first playable one.
            media = row.get(audio_col)
            if isinstance(media, list) and media:
                media = media[0]
            audio_src = media.get("src") if isinstance(media, dict) else None
            if not audio_src:
                continue
            label = (
                _class_dir(row.get(label_col), class_names.get(str(label_col)))
                if label_col and row.get(label_col) is not None
                else "unlabelled"
            )
            suffix = Path(urllib.parse.urlparse(audio_src).path).suffix or ".wav"
            _write(root / label / f"{index:05d}{suffix}", _get(audio_src))
            count += 1
            continue

        image = row.get(image_col)
        src = image.get("src") if isinstance(image, dict) else None
        if not src:
            continue

        stem = f"{index:05d}"
        if label_col and row.get(label_col) is not None:
            # Class-per-directory at the root — the `image-folder` layout.
            _write(
                root / _class_dir(row[label_col], class_names.get(str(label_col))) / f"{stem}.png",
                _get(src),
            )
        else:
            # Parallel images/ + masks/ — the `image-mask-pairs` layout.
            _write(root / "images" / f"{stem}.png", _get(src))

        if mask_col:
            mask = row.get(mask_col)
            mask_src = mask.get("src") if isinstance(mask, dict) else None
            if mask_src:
                _write(root / "masks" / f"{stem}.png", _get(mask_src))
        count += 1

    if count == 0:
        available = sorted(rows[0].get("row", {}))
        wanted_col = str(audio_col) if audio_col else image_col
        present = wanted_col in available
        reason = (
            f"column '{wanted_col}' exists but carries no resolvable media `src` "
            f"(the upstream datasets-server may be failing to post-process this dataset)"
            if present
            else f"no column named '{wanted_col}'"
        )
        raise FetchError(f"{source.id}: {reason} in {hf_id}. Columns: {available}")

    return FetchResult(source.id, root, count, "huggingface", truncated=True)


_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip")

MAX_DOWNLOAD_WITHOUT_RANGE = 200 * 1024 * 1024
"""A full-body GET is only safe up to this size. CoralVQA's real image archive is a
single 26.7 GB zip with no smaller subset published anywhere — downloading that just to
read 100 verification images would be absurd, and it is exactly the shape of "large
upstream archive, needs a bounded sample" that this registry expects to hit again."""


def _head(url: str, *, timeout: int = 30) -> tuple[int | None, bool]:
    """Content-Length and whether the server actually honours Range, probed with a
    real 1-byte ranged GET rather than trusted from the ``Accept-Ranges`` header.

    Zenodo (confirmed 2026-08-28, deolhonoscorais) returns 206 to a real Range request
    while never sending ``Accept-Ranges`` on either HEAD or GET — trusting the header
    alone wrongly refused every large Zenodo zip as "no Range support", when the
    server was honouring Range the whole time. A live probe is the only reliable
    signal; it costs one extra small request, once, per fetch.

    Follows redirects like any other urllib request (HF's resolve URLs 302 to a signed
    CDN URL; the redirected response is what carries the real headers). Returns
    ``(None, False)`` on any failure — callers treat that as "cannot be range-read" and
    fall back to a plain full download when the size is small enough to make that safe.
    """
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 206:
                total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                return (int(total) if total.isdigit() else None, True)
            length = response.headers.get("Content-Length")
            return (int(length) if length is not None else None, False)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return (None, False)


def _get_range(url: str, start: int, end: int, *, timeout: int = 60, retries: int = 3) -> bytes:
    """Ranged GET, ``bytes=start-end`` inclusive. Raises rather than silently returning
    the wrong bytes if the server does not honour the range (some proxies strip the
    header and return 200 with the whole body) — a caller parsing a zip's central
    directory from what it thinks is a 4 KB slice would otherwise get garbage from the
    front of a 26 GB file and fail in a confusing way far from the actual cause.
    """
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise FetchError(
                        f"{url} did not honour the Range request (status "
                        f"{response.status}) — cannot safely read a slice of it"
                    )
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise FetchError(f"HTTP {exc.code} for ranged GET of {url}") from exc
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"failed after {retries} attempt(s) for ranged GET of {url}: {last}")


class _RemoteFile:
    """A read-only, seekable view over a remote resource, fetched lazily via HTTP Range
    requests — lets :class:`zipfile.ZipFile` read a multi-gigabyte archive's central
    directory and individual members without ever downloading the whole thing.

    ``zipfile`` reads a central directory near the *end* of the file and then seeks
    around to each member's local header, so this needs real seek support, not just
    sequential reads — a plain streaming GET could not do this.

    ``window_offset``/``window_size`` present a byte range of a larger remote resource
    as if it were its own zero-based file — used to read a zip nested *uncompressed*
    (``ZIP_STORED``) inside an outer zip without downloading either one. This only
    works because STORED bytes are the nested file's real bytes verbatim; a DEFLATEd
    nested archive cannot be windowed this way, since compressed bytes are not
    arbitrarily seekable into.
    """

    def __init__(
        self, url: str, size: int, *, window_offset: int = 0, window_size: int | None = None
    ) -> None:
        self._url = url
        self._window_offset = window_offset
        self._size = size if window_size is None else window_size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1:
            offset += self._pos
        elif whence == 2:
            offset += self._size
        self._pos = max(0, offset)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size - 1 if size is None or size < 0 else min(self._pos + size, self._size) - 1
        data = _get_range(self._url, self._window_offset + self._pos, self._window_offset + end)
        self._pos += len(data)
        return data


def _nested_zip_window(outer: zipfile.ZipFile, member_name: str) -> tuple[int, int]:
    """Byte range of a member stored *uncompressed* inside an already-open zip.

    Returns ``(data_offset, size)`` — where the member's own bytes begin within the
    outer file, and how many there are. Only valid when the member's
    ``compress_type`` is ``ZIP_STORED``: those bytes are the member's real content
    verbatim, so if the member is itself a zip, this range can be opened as one
    without extracting it first. Raises for anything else — a DEFLATEd nested archive
    cannot be windowed this way, and guessing would produce a file that mysteriously
    fails to parse instead of a clear error naming the actual cause.
    """
    import struct

    info = outer.getinfo(member_name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise FetchError(
            f"'{member_name}' is compressed (not stored) inside its outer archive — "
            f"it cannot be read as a nested zip without extracting the whole thing "
            f"first, which defeats the point of a bounded sample."
        )

    # header_offset points at the LOCAL file header, not the data — the local header
    # can carry different filename/extra-field lengths than the central directory
    # entry does, so its size has to be read, not assumed.
    fp = outer.fp
    fp.seek(info.header_offset)
    header = fp.read(30)
    if header[:4] != b"PK\x03\x04":
        raise FetchError(f"'{member_name}': malformed local file header in outer archive")
    filename_len, extra_len = struct.unpack("<HH", header[26:30])
    data_offset = info.header_offset + 30 + filename_len + extra_len
    return data_offset, info.compress_size


def _extract_remote_zip(file_obj: _RemoteFile, root: Path, limit: int, *, label: str) -> int:
    """Extract up to ``limit`` members from a remote zip without downloading it.

    ``file_obj`` is whatever :class:`_RemoteFile` (or compatible) view the caller
    already built — a whole remote archive, or a window into one nested inside
    another. The counterpart to :func:`_extract_archive` for archives too large to
    pull in full — see ``MAX_DOWNLOAD_WITHOUT_RANGE``.
    """
    resolved_root = root.resolve()
    count = 0
    with zipfile.ZipFile(file_obj) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members[:limit] if limit else members:
            target = _safe_extract_path(resolved_root, member.filename)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            count += 1

    if count == 0:
        raise FetchError(f"archive at {label} contained no extractable files")
    return count


def _extract_remote_tar(file_obj: _RemoteFile, root: Path, limit: int, *, label: str) -> int:
    """Extract up to ``limit`` members from a remote **uncompressed** tar without
    downloading it.

    Unlike zip, tar has no central directory to read upfront — the only way to know
    what is in it is to read forward from the start, one 512-byte header block at a
    time. ``tarfile.TarFile`` is iterable exactly this way, and critically,
    :meth:`TarFile.getmembers` must never be called here: it exhausts the *whole*
    stream to build its list, which for a multi-GB archive means downloading all of
    it — the exact cost this function exists to avoid. Iterating the archive object
    itself and stopping after ``limit`` members only ever reads as far as the last
    member's data, because :class:`_RemoteFile` treats an unread gap as a cheap
    ``seek`` (no request) rather than bytes that must be fetched.

    Compressed tars (``.tar.gz``/``.tar.bz2``/``.tar.xz``) are not supported this way:
    decompression is inherently sequential from the very start, so a compressed
    stream cannot skip past an unwanted member's data without decompressing it
    anyway, which defeats the point of a bounded sample.
    """
    resolved_root = root.resolve()
    count = 0
    with tarfile.open(fileobj=file_obj, mode="r:") as archive:
        for member in archive:
            if limit and count >= limit:
                break
            if not member.isfile():
                continue
            target = _safe_extract_path(resolved_root, member.name)
            if target is None:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())
            count += 1

    if count == 0:
        raise FetchError(f"archive at {label} contained no extractable files")
    return count


def _safe_extract_path(resolved_root: Path, member_name: str) -> Path | None:
    """Resolve an archive member's target path, refusing to leave ``resolved_root``.

    Archive members are attacker-influenced input in general (here, a maintainer-chosen
    URL, but the extraction code itself does not know that) — an absolute path or a
    ``../`` traversal in a member name must not write outside the sample directory.
    Returns ``None`` for a member that would escape, so the caller can skip it.

    ``resolved_root`` must already be ``.resolve()``d by the caller, once, before the
    extraction loop. Resolving it fresh on every call here — comparing an *unresolved*
    root against a *resolved* target — used to reject every single member as "escaping"
    whenever the destination sat under a symlink, which on macOS ``/tmp`` always does:
    caught by a live verification run against a real archive, silently invisible to
    every synthetic-fixture test because pytest's ``tmp_path`` is never symlinked.
    """
    target = (resolved_root / member_name).resolve()
    if target != resolved_root and resolved_root not in target.parents:
        return None
    return target


def _extract_archive(payload: bytes, root: Path, name: str, limit: int) -> int:
    """Extract a small archive in place, bounded to ``limit`` files.

    Handles ``.zip`` and ``.tar`` (optionally ``.gz``/``.bz2``/``.xz``) — the two
    conventions every dataset host in this registry actually uses. Bounded because a
    verification sample is meant to stay small even when the upstream archive is not.
    """
    import io

    resolved_root = root.resolve()
    count = 0
    lower = name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            for member in members[:limit] if limit else members:
                target = _safe_extract_path(resolved_root, member.filename)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                count += 1
    else:
        mode = "r:*"  # autodetect gz/bz2/xz/plain from the stream itself
        with tarfile.open(fileobj=io.BytesIO(payload), mode=mode) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            for member in members[:limit] if limit else members:
                target = _safe_extract_path(resolved_root, member.name)
                if target is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(extracted.read())
                count += 1

    if count == 0:
        raise FetchError(f"archive '{name}' contained no extractable files")
    return count


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
        from .fetchers_remote import fetch_pangaea_manifest

        return fetch_pangaea_manifest(source, root, limit)

    sample_url = str(source.access.params.get("sample_url") or "")
    if not sample_url:
        raise FetchNotSupported(
            f"{source.id}: no `sample_url` declared. Add one pointing at a small "
            f"downloadable sample, or fetch the data manually."
        )
    nested_archive = source.access.params.get("nested_archive")
    tar_start_offset = int(source.access.params.get("tar_start_offset", 0) or 0)
    count = _fetch_one(
        source.id,
        sample_url,
        root,
        limit,
        nested_archive=nested_archive,
        tar_start_offset=tar_start_offset,
    )

    extra_url = source.access.params.get("extra_sample_url")
    if extra_url:
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
) -> int:
    """Retrieve one declared file or archive into ``root``. The single-URL body of
    :func:`_fetch_http`, factored out so a second URL (``extra_sample_url``) can reuse
    the exact same archive/size/Range handling rather than a parallel copy of it.
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
        with zipfile.ZipFile(_RemoteFile(url, size)) as outer:
            offset, member_size = _nested_zip_window(outer, str(nested_archive))
        inner = _RemoteFile(url, size, window_offset=offset, window_size=member_size)
        return _extract_remote_zip(inner, root, limit, label=f"{name}:{nested_archive}")

    if name.lower().endswith(".zip"):
        size, supports_range = _head(url)
        if size and size <= MAX_DOWNLOAD_WITHOUT_RANGE:
            return _extract_archive(_get(url), root, name, limit)
        if supports_range and size:
            return _extract_remote_zip(_RemoteFile(url, size), root, limit, label=name)
        if size and size > MAX_DOWNLOAD_WITHOUT_RANGE:
            raise FetchNotSupported(
                f"{source_id}: {url} is a {size / 1e9:.1f} GB zip and the host does "
                f"not support Range requests, so a bounded sample cannot be read from it "
                f"without downloading the whole archive. Find a smaller published subset, "
                f"or fetch it manually."
            )
        return _extract_archive(_get(url), root, name, limit)

    if name.lower().endswith(".tar"):
        size, supports_range = _head(url)
        if size and size <= MAX_DOWNLOAD_WITHOUT_RANGE and not tar_start_offset:
            return _extract_archive(_get(url), root, name, limit)
        if supports_range and size:
            remote = _RemoteFile(
                url, size, window_offset=tar_start_offset, window_size=size - tar_start_offset
            )
            return _extract_remote_tar(remote, root, limit, label=name)
        if size and size > MAX_DOWNLOAD_WITHOUT_RANGE:
            raise FetchNotSupported(
                f"{source_id}: {url} is a {size / 1e9:.1f} GB tar and the host does "
                f"not support Range requests, so a bounded sample cannot be read from "
                f"it without downloading the whole archive. Find a smaller published "
                f"subset, or fetch it manually."
            )
        return _extract_archive(_get(url), root, name, limit)

    payload = _get(url)
    if name.lower().endswith(_ARCHIVE_SUFFIXES):
        return _extract_archive(payload, root, name, limit)
    _write(root / name, payload)
    return 1


def _fetch_hf(source: Source, root: Path, limit: int) -> FetchResult:
    """Dispatch between the parquet dataset API and a plain repo file tree.

    Both live behind ``method: huggingface``; ``fetch_style: files`` selects the tree.
    """
    if str(source.access.params.get("fetch_style", "rows")) == "files":
        from .fetchers_remote import fetch_hf_files

        return fetch_hf_files(source, root, limit)
    return _fetch_huggingface(source, root, limit)


def _fetch_s3(source: Source, root: Path, limit: int) -> FetchResult:
    from .fetchers_remote import fetch_s3

    return fetch_s3(source, root, limit)


def _fetch_fathomnet_client(source: Source, root: Path, limit: int) -> FetchResult:
    from .fetchers_remote import fetch_fathomnet

    return fetch_fathomnet(source, root, limit)


def _unimplemented_client(client: str) -> Callable[[Source, Path, int], FetchResult]:
    """One stub factory shared by every registered-but-unbuilt api client.

    Each of the bespoke clients below (obis, allen-coral-atlas, copernicus-marine,
    coralnet, atlantis) is a separate workstream (master brief WS-D: 7 non-bulk
    sources) — this raises rather than silently reusing another source's client.
    """

    def _fetch(source: Source, root: Path, limit: int) -> FetchResult:
        raise NotImplementedError(
            f"{source.id}: api client '{client}' is known but not yet implemented — "
            "building it is a separate workstream (master brief WS-D: 7 non-bulk sources)."
        )

    return _fetch


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

    fetcher = _FETCHERS.get(source.access.method)
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


def sample_digest(root: Path) -> str:
    """Stable digest of a fetched sample, for reproducibility in test reports.

    Hashes file *content*, not just name and size — a same-size edit (a relabelled mask,
    a re-exported image) used to pass undetected, since two files of equal size hashed
    identically regardless of their bytes. These are bounded verification samples
    (~100 items, per ``fetch_sample``'s ``limit``), so reading them in full costs
    nothing worth trading correctness for.

    Bytes reach the hash through :func:`marinedata.checksums.stream_into`, the one
    chunked reader in this package — the same value as before, since the same bytes
    arrive in the same order, but a sample holding one oversized file no longer has
    to fit in memory.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "_fetch.json"):
        digest.update(path.name.encode())
        stream_into(digest, path)
    return "sha256:" + digest.hexdigest()[:16]


get_bytes = _get
"""Public alias for :func:`_get` — a retrying GET returning the whole response body.
For :mod:`marinedata.ingest`, which needs the full archive rather than a sample."""

extract_archive = _extract_archive
"""Public alias for :func:`_extract_archive` — extract an in-memory archive to a
directory, bounded to ``limit`` files (``0`` means every member)."""

get_stream = _get_stream
"""Public alias for :func:`_get_stream` — a retrying GET that streams straight to a
file (D2a), for :mod:`marinedata.ingest_parquet`, which downloads ~450 MB shards."""
