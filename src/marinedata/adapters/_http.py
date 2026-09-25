"""Anonymous HTTP for adapters: no credentials ever, streamed bodies, bounded retries.

D-E: open sources only. Nothing here reads ``HF_TOKEN``, ``GITHUB_TOKEN`` or any other
credential, and :func:`open_url` refuses a caller-supplied ``Authorization`` header, so an
adapter cannot accidentally reach a gated resource with Yohan's identity.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from ..checksums import DOWNLOAD_USER_AGENT
from ..concurrency import RetriesExhausted
from ._throttle import THROTTLE_TRIES, ThrottleExhausted, is_hf_host, retry_after_s, throttle_for

CHUNK = 1 << 20
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


# INT-ingest7 (WP-6j Open 1): hs.pangaea.de answers bursts of 503 under load (PS96 smoke:
# 2 of 3 attempts died on it). A PANGAEA host gets 8 tries per request and a longer wait —
# exponential 2, 4, … s, never below its Retry-After — each wait capped at 120 s, before
# RetriesExhausted lets D-AF count the item missing. Other hosts keep 4 tries / 30 s.
_SLOW_5XX_DOMAINS = ("pangaea.de",)
SLOW_5XX_TRIES = 8
SLOW_5XX_CAP_S = 120.0


def is_slow_5xx_host(host: str) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in _SLOW_5XX_DOMAINS)


def backoff_s(attempt: int, exc: BaseException | None, *, slow: bool) -> float:
    """Seconds to wait after failed ``attempt`` (0-based), before the jitter."""
    if not slow:
        return float(min(2**attempt, 30))
    hdrs = exc.headers if isinstance(exc, urllib.error.HTTPError) else None
    asked = retry_after_s(hdrs) if hdrs is not None else None
    # Retry-After is a floor, never a ceiling: hs.pangaea.de sends 2-7 s on its
    # "loading from tape" 503 while the recall itself takes minutes.
    return min(max(asked or 0.0, float(2 ** (attempt + 1))), SLOW_5XX_CAP_S)


class AccessRefused(RuntimeError):
    """The resource needs a login, a terms click-through or a gate (D-E): never bypass.
    ``url`` + ``needs`` go on the "needs Yohan" list verbatim."""

    def __init__(self, url: str, needs: str) -> None:
        super().__init__(f"{url}: {needs}")
        self.url = url
        self.needs = needs


def open_url(
    url: str, *, headers: dict[str, str] | None = None, retries: int = 4, timeout: int = 120
) -> IO[bytes]:
    """GET ``url`` and return the open response (caller closes). 401/403 -> AccessRefused."""
    hdrs = {"User-Agent": DOWNLOAD_USER_AGENT, **(headers or {})}
    if any(k.lower() in {"authorization", "cookie"} for k in hdrs):
        raise ValueError("adapters are anonymous-only (D-E): no Authorization/Cookie headers")
    # INT-ingest5c: a non-HF host's 429 is paced per item (see ``_throttle``); an HF
    # host's 429 keeps the D-AA path (retried like a 5xx, then the source aborts).
    host = urllib.parse.urlsplit(url).hostname or ""
    throttle = None if is_hf_host(host) else throttle_for(host)
    slow = is_slow_5xx_host(host)
    if slow:
        retries = SLOW_5XX_TRIES
    last: Exception | None = None
    attempt = throttled = 0
    while attempt < retries:
        if throttle is not None:
            throttle.wait_turn()
        try:
            req = urllib.request.Request(url, headers=hdrs)
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AccessRefused(url, f"HTTP {exc.code}: login/terms/gate required") from exc
            if exc.code not in _RETRY_STATUS:
                raise
            last = exc
            if exc.code == 429 and throttle is not None:
                throttled += 1
                throttle.on_429(retry_after_s(exc.headers), throttled)  # may raise HostThrottled
                if throttled >= THROTTLE_TRIES:
                    raise ThrottleExhausted(
                        f"GET {url} failed after {throttled} throttled attempts: {exc}"
                    ) from exc
                continue  # the host-wide wait happens in wait_turn()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
        else:
            if throttle is not None:
                throttle.on_success()
            return resp
        attempt += 1
        if attempt < retries:  # INT-ingest7: no dead wait after the last try
            time.sleep(backoff_s(attempt - 1, last, slow=slow) + random.uniform(0, 1))  # jitter
    raise RetriesExhausted(f"GET {url} failed after {retries} attempts: {last}") from last


def get_json(url: str) -> object:
    with open_url(url, headers={"Accept": "application/json"}) as resp:
        return json.loads(resp.read())


def get_json_pages(url: str) -> Iterator[object]:
    """Follow RFC 5988 ``Link: <...>; rel="next"`` pagination (HF tree, GitHub)."""
    next_url: str | None = url
    while next_url:
        with open_url(next_url, headers={"Accept": "application/json"}) as resp:
            body = json.loads(resp.read())
            match = _LINK_NEXT.search(resp.headers.get("Link", "") or "")
        yield body
        next_url = match.group(1) if match else None


class HashingReader:
    """Wrap a stream; sha256 (and md5) everything read through it, byte count too."""

    def __init__(self, raw: IO[bytes]) -> None:
        self.raw = raw
        self.sha = hashlib.sha256()
        self.md5 = hashlib.md5(usedforsecurity=False)
        self.nbytes = 0

    def read(self, n: int = -1) -> bytes:
        data = self.raw.read(n)
        self.sha.update(data)
        self.md5.update(data)
        self.nbytes += len(data)
        return data

    def drain(self) -> None:
        while self.read(CHUNK):
            pass


class ConcatReader:
    """Read several URLs back-to-back as one file-like stream.

    WP-6d-B multipart-tar (D-D): ``<name>.tar.gz.aa``/``.ab``/... parts are concatenated
    in order with no full local copy — each part is opened only when the previous one is
    exhausted, and closed immediately after. tarfile's own streaming reader (``r|*``)
    already loops on short reads, so returning less than ``n`` mid-transition is fine;
    only an empty read means every part is exhausted.
    """

    def __init__(self, urls: tuple[str, ...]) -> None:
        self._urls = list(urls)
        self._current: IO[bytes] | None = None

    def _advance(self) -> bool:
        if self._current is not None:
            self._current.close()
            self._current = None
        if not self._urls:
            return False
        self._current = open_url(self._urls.pop(0))
        return True

    def read(self, n: int = -1) -> bytes:
        if self._current is None and not self._advance():
            return b""
        data = self._current.read(n)
        while not data and self._advance():
            data = self._current.read(n)
        return data

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None


def download(url: str, dest: Path) -> tuple[str, str, int]:
    """Stream ``url`` to ``dest`` (via ``.part`` + rename). Returns (sha256, md5, size)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with open_url(url) as resp, part.open("wb") as out:
        reader = HashingReader(resp)
        shutil.copyfileobj(reader, out, CHUNK)  # type: ignore[arg-type]
    part.replace(dest)
    return reader.sha.hexdigest(), reader.md5.hexdigest(), reader.nbytes
