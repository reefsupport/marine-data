"""Transfer integrity: the completeness check, bounded retries with backoff, ``.part``
renames that never leave a partial file under its final name, and the sample digest."""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..checksums import CHUNK_SIZE, stream_into
from ..s3_auth import S3Creds, auth_headers_for
from .constants import MAX_GET_BYTES, USER_AGENT
from .core import FetchError, IntegrityError


def _check_complete(response: Any, received: int) -> None:
    """Raise ``IncompleteRead`` when the body is shorter than its ``Content-Length``.

    ``HTTPResponse.read(n)`` hands back a short chunk without complaint when the server
    closes early (only the no-argument ``read()`` raises), so a chunked read must check
    for itself or a truncated transfer looks like a finished one."""
    headers = getattr(response, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None and str(declared).isdigit() and int(declared) != received:
        raise http.client.IncompleteRead(b"", int(declared) - received)


def _retrying_read(
    url: str,
    consume: Callable[[Any], Any],
    *,
    timeout: int,
    headers: dict[str, str] | None,
    retries: int,
    creds: S3Creds | None,
) -> Any:
    """Open ``url`` and hand the response to ``consume``, retrying transient faults.

    4xx is not retried: a 404 will still be a 404 — except 429, which is the server asking
    a concurrent fetch to slow down. 5xx and connection-level faults (``RemoteDisconnected``,
    ``BadStatusLine``, ``IncompleteRead``, resets, timeouts) are retried with exponential
    backoff (:func:`_backoff`), at most ``retries`` attempts.

    With ``creds`` the request is SigV4-signed afresh on every attempt (only for the
    credentials' own endpoint). A 401/403 on a signed request drops to anonymous for the
    rest of the call without using up an attempt: signed first, anonymous as the fallback,
    never the reverse."""
    last: Exception | None = None
    signed = creds is not None
    attempt = 0
    while attempt < retries:
        auth = auth_headers_for(url, creds) if signed else {}
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, **(headers or {}), **auth}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return consume(response)
        except urllib.error.HTTPError as exc:
            last = exc
            if auth and exc.code in (401, 403):
                signed = False
                continue
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(_backoff(attempt))
        attempt += 1
    raise FetchError(f"failed after {retries} attempt(s) for {url}: {last}")


def get_to_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    label: str | None = None,
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    max_bytes: int = MAX_GET_BYTES,
    creds: S3Creds | None = None,
) -> str:
    """Stream ``url`` to ``dest`` through ``<dest>.part`` and return the sha256 hex digest.

    The ``.part`` file is renamed onto ``dest`` only after the whole body arrived and — when
    ``expected_sha256`` is given — hashes to it, so ``dest`` never holds partial or
    unverified bytes. A truncated transfer is retried (at most ``retries``, backoff); a
    digest mismatch raises :class:`IntegrityError` naming ``label``/the URL and both
    digests, removes the ``.part`` and is not retried."""
    expected = expected_sha256.removeprefix("sha256:").lower() if expected_sha256 else None
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def consume(response: Any) -> str:
        digest = hashlib.sha256()
        total = 0
        try:
            with part.open("wb") as out:
                while chunk := response.read(CHUNK_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"{url}: response exceeded {max_bytes} bytes — refused")
                    digest.update(chunk)
                    out.write(chunk)
            _check_complete(response, total)
        except BaseException:
            part.unlink(missing_ok=True)  # a partial transfer is never left behind as a file
            raise
        return digest.hexdigest()

    actual = _retrying_read(
        url, consume, timeout=timeout, headers=headers, retries=retries, creds=creds
    )
    if expected is not None and actual != expected:
        part.unlink(missing_ok=True)
        raise IntegrityError(
            f"{label or url}: downloaded with sha256 {actual}, expected {expected} — "
            f"the transfer is corrupt or the object changed under a pinned digest"
        )
    os.replace(part, dest)
    return actual


def _backoff(attempt: int) -> float:
    """Seconds to wait after failed attempt ``attempt`` (0-based): 1.5, 3, 6, 12, … capped
    at 30. The first two steps equal the old linear schedule, so a default 3-attempt
    ``_get`` waits exactly as long as it always did."""
    return min(30.0, 1.5 * (2**attempt))


def _write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through ``<path>.part`` and a rename, so ``path`` is
    either absent or complete — a killed run never leaves a half-written file under its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        part.write_bytes(payload)
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


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
        # Relative to `root`, not just the basename: otherwise moving `a/img.jpg` to
        # `b/img.jpg` (a declared-layout change) leaves the digest unchanged.
        digest.update(path.relative_to(root).as_posix().encode())
        stream_into(digest, path)
    return "sha256:" + digest.hexdigest()[:16]
