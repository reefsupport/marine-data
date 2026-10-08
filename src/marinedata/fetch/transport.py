"""Whole-body and streaming GETs, and the digest gate (a declared ``sha256`` is checked
against the bytes that arrived; plaintext ``http://`` without one is refused up front)."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..checksums import CHUNK_SIZE
from ..s3_auth import S3Creds
from .constants import MAX_GET_BYTES, USER_AGENT
from .core import FetchError, FetchNotSupported, IntegrityError
from .integrity import _backoff, _check_complete, _retrying_read


def _get(
    url: str,
    *,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    max_bytes: int = MAX_GET_BYTES,
    creds: S3Creds | None = None,
) -> bytes:
    """GET with retries on transient faults (:func:`_retrying_read`).

    Retrying is not optional at this scale: fetching a few hundred images means a few
    hundred connections and some fraction are dropped by the far end mid-transfer.

    Read incrementally and bounded by ``max_bytes`` rather than one bare
    ``response.read()`` — a ``Content-Length`` header can under-report or be absent while
    the body keeps streaming, so the only reliable cap is one enforced against the bytes
    actually received. A body shorter than its declared length is retried."""

    def consume(response: Any) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise FetchError(
                    f"{url}: response exceeded {max_bytes} bytes — refused rather than "
                    f"read further (a cap on bytes actually received, not on Content-Length)"
                )
            chunks.append(chunk)
        _check_complete(response, total)
        return b"".join(chunks)

    return _retrying_read(
        url, consume, timeout=timeout, headers=headers, retries=retries, creds=creds
    )


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


def _verify_digest(source_id: str, url: str, payload: bytes, expected_sha256: str | None) -> None:
    """Raise if ``payload`` does not hash to a declared ``sha256``.

    Only called where ``payload`` is the *whole* fetched artefact — a bare file, or an
    archive downloaded in full before extraction. A Range-read sample never has the
    whole artefact's bytes in memory (that is the entire point of Range reading), so it
    cannot be checked against a whole-file digest without downloading the whole file,
    which defeats the point of a bounded sample; that gap is real and is not silently
    hidden — see ``_fetch_http``'s plaintext-http guard, which requires a digest be
    declared up front specifically because it cannot always be checked after the fact.
    """
    if not expected_sha256:
        return
    expected = expected_sha256.removeprefix("sha256:").lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise IntegrityError(
            f"{source_id}: declared sha256 does not match the fetched bytes at {url} — "
            f"expected {expected}, got {actual}. The bytes on disk are not the bytes "
            f"the licence was verified against."
        )


def _require_digest_for_plaintext_http(
    source_id: str, url: str, expected_sha256: str | None
) -> None:
    """``http://`` carries no transport-layer integrity — anything between this process
    and the server can alter the bytes in flight with nothing here the wiser. TLS is
    the usual defence; the fallback for a host that does not offer it is a digest to
    check the bytes against after the fact. A plaintext source with neither is exactly
    the D2 quiet skip — "a checksum that is absent rather than wrong" — so this fails
    before ever making the request, by name and reason, rather than fetching something
    unverifiable and calling it done.
    """
    if url.startswith("http://") and not expected_sha256:
        raise FetchNotSupported(
            f"{source_id}: {url} is plaintext http:// with no declared `sha256` — "
            f"nothing vouches for what actually arrives. Declare `access.sha256`, or "
            f"switch to an https:// URL if the host serves one."
        )
