"""HTTP Range reads: the capability probe, a ranged GET, and a seekable remote file."""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from .constants import USER_AGENT
from .core import FetchError


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

    ``get_range`` swaps the ranged reader (default: this module's :func:`_get_range`); the
    ``marinedata.fetch`` facade passes its own, so assigning ``marinedata.fetch._get_range``
    still steers every ranged read.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        window_offset: int = 0,
        window_size: int | None = None,
        get_range: Callable[..., bytes] | None = None,
    ) -> None:
        self._url = url
        self._get_range = get_range or _get_range
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
        data = self._get_range(
            self._url, self._window_offset + self._pos, self._window_offset + end
        )
        self._pos += len(data)
        return data
