"""Anonymous HTTP for adapters: no credentials ever, streamed bodies, bounded retries.

D-E: open sources only. Nothing here reads ``HF_TOKEN``, ``GITHUB_TOKEN`` or any other
credential, and :func:`open_url` refuses a caller-supplied ``Authorization`` header, so an
adapter cannot accidentally reach a gated resource with Yohan's identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from ..checksums import DOWNLOAD_USER_AGENT

CHUNK = 1 << 20
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


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
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=hdrs), timeout=timeout
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AccessRefused(url, f"HTTP {exc.code}: login/terms/gate required") from exc
            if exc.code not in _RETRY_STATUS:
                raise
            last = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


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


def download(url: str, dest: Path) -> tuple[str, str, int]:
    """Stream ``url`` to ``dest`` (via ``.part`` + rename). Returns (sha256, md5, size)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with open_url(url) as resp, part.open("wb") as out:
        reader = HashingReader(resp)
        shutil.copyfileobj(reader, out, CHUNK)  # type: ignore[arg-type]
    part.replace(dest)
    return reader.sha.hexdigest(), reader.md5.hexdigest(), reader.nbytes
