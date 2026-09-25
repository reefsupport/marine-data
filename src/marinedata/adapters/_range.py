"""HTTP Range reads for adapters (WP-6i): resume a dropped stream; read a remote zip.

* :class:`ResumableStream` — a forward-only body that survives a dropped connection.
  HF's CDN (``us.aws.cdn.hf.co``, xet-bridge) closes a connection left idle ~30 s
  (WP-6i, measured on a TreeOfLife-10M shard: idle 10 s / 20 s fine, 40 s / 60 s -> the
  next reads return only the 1.4-6.9 MiB already in the socket buffer, then EOF).
  ``http.client`` reports that as a clean EOF — ``read(n)`` returns ``b""`` and never
  raises ``IncompleteRead`` — so a streaming tar reader fails with "unexpected end of
  data". Here a read that ends before ``Content-Length`` (or raises a transport error)
  reopens the ORIGINAL url with ``Range: bytes=<pos>-`` (a fresh signed redirect) and
  carries on; the resumed response must be a 206 starting at ``pos``.
* :class:`RemoteFile` — a read-only, seekable file over Range requests, so
  :class:`zipfile.ZipFile` reads a remote zip's central directory and only the members it
  is asked for. No local copy of the archive ever exists (D-L budget stays per member).

Both go through :func:`~marinedata.adapters._http.open_url` (anonymous, D-E; retries).
"""

from __future__ import annotations

import bisect
import http.client
import io
import logging
import re
import tarfile
import time
import zipfile
from collections.abc import Callable, Iterator
from typing import IO, Any

from ..concurrency import RetriesExhausted
from ._http import CHUNK, AccessRefused, open_url
from .zipread import read_member as _read_zip_member

log = logging.getLogger(__name__)

OpenFn = Callable[..., IO[bytes]]
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
_TRANSPORT = (http.client.HTTPException, OSError)  # IncompleteRead, reset, timeout, TLS


def _range_start(resp: Any) -> int | None:
    """First byte of a 206 response (``None`` if the server ignored the Range header)."""
    if getattr(resp, "status", None) != 206:
        return None
    m = _CONTENT_RANGE.match(resp.headers.get("Content-Range") or "")
    return int(m.group(1)) if m else None


class ResumableStream(io.RawIOBase):
    """Forward-only HTTP body; reconnects with ``Range`` after a premature end.

    ``max_stalls`` bounds consecutive resumes that deliver no byte; any progress resets
    the count, so a 32 GB shard may be resumed any number of times as long as it moves.
    Exhausted -> :class:`RetriesExhausted` (the runner's transport-failure status).
    A ``RawIOBase``, so ``io.BufferedReader`` / pyarrow can wrap it.
    """

    def __init__(
        self, url: str, *, max_stalls: int = 5, timeout: int = 120,
        open_fn: OpenFn | None = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self.url = url
        self._open: OpenFn = open_fn or open_url
        self._max_stalls = max_stalls
        self._timeout = timeout
        self._stalls = 0
        self.pos = 0
        self.resumes = 0
        self._resp: Any = self._open(url, timeout=timeout)
        cl = self._resp.headers.get("Content-Length")
        self.length: int | None = int(cl) if cl and cl.isdigit() else None

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            return b"".join(iter(lambda: self.read(CHUNK), b""))
        while n:
            err: BaseException | None = None
            try:
                data = self._resp.read(n)
            except _TRANSPORT as exc:
                data, err = b"", exc
            if data:
                self.pos += len(data)
                self._stalls = 0
                return data
            if err is None and (self.length is None or self.pos >= self.length):
                return b""  # a true end of body
            self._resume(err)
        return b""

    def _resume(self, cause: BaseException | None) -> None:
        self._stalls += 1
        why = f"{type(cause).__name__}: {cause}" if cause else "short read (connection closed)"
        if self._stalls > self._max_stalls:
            raise RetriesExhausted(
                f"GET {self.url}: no progress after {self._max_stalls} resumes at byte "
                f"{self.pos}/{self.length} ({why})"
            ) from cause
        self._close_resp()
        self.resumes += 1
        log.warning(
            "resume %s at byte %d/%s after %s (resume %d)",
            self.url, self.pos, self.length, why, self.resumes,
        )  # fmt: skip
        if self._stalls > 1:
            time.sleep(min(2**self._stalls, 30))
        resp = self._open(self.url, headers={"Range": f"bytes={self.pos}-"}, timeout=self._timeout)
        if _range_start(resp) != self.pos:
            resp.close()
            raise RetriesExhausted(
                f"GET {self.url}: cannot resume at byte {self.pos} (server ignored Range)"
            )
        self._resp = resp

    def _close_resp(self) -> None:
        resp = getattr(self, "_resp", None)  # absent if the first open raised
        try:
            if resp is not None:
                resp.close()
        except _TRANSPORT:
            pass

    def close(self) -> None:
        if not self.closed:
            self._close_resp()
        super().close()


class RemoteFile:
    """Seekable, read-only view of ``url`` (``size`` bytes) over HTTP Range requests.

    One contiguous cache window: :meth:`load` pulls ``[start, end)`` in ONE request (a
    zip member's local header + data), plain :meth:`read` misses fetch ``block`` bytes.
    The redirect target (HF: a signed CDN url, ~1 h) is reused until it is refused,
    then the original url is resolved again — one HF resolve per hour, not per member.
    """

    def __init__(
        self, url: str, size: int, *, block: int = CHUNK, tries: int = 4,
        open_fn: OpenFn | None = None,
    ) -> None:  # fmt: skip
        self.url, self.size, self.block, self.tries = url, size, block, tries
        self._open: OpenFn = open_fn or open_url
        self._direct: str | None = None
        self._pos = 0
        self._start = 0
        self._buf = b""
        self.requests = 0
        self.fetched = 0

    # -- file protocol used by zipfile ------------------------------------------------
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self._pos + n)
        if end <= self._pos:
            return b""
        if not (self._start <= self._pos and end <= self._start + len(self._buf)):
            self.load(self._pos, max(end, min(self.size, self._pos + self.block)))
        lo = self._pos - self._start
        data = self._buf[lo : lo + end - self._pos]
        self._pos += len(data)
        return data

    def close(self) -> None:
        self._buf = b""

    # -- range fetch ------------------------------------------------------------------
    def load(self, start: int, end: int) -> None:
        """Cache bytes ``[start, end)`` with one ranged GET (bounded retries)."""
        last: BaseException | None = None
        for attempt in range(self.tries):
            try:
                self._buf, self._start = self._get(start, end), start
                return
            except _TRANSPORT as exc:  # dropped mid-body / reset: ask again
                last = exc
                self._direct = None
                if attempt + 1 < self.tries:
                    time.sleep(min(2**attempt, 30))
        raise RetriesExhausted(f"GET {self.url} [{start}, {end}): {last}") from last

    def _get(self, start: int, end: int) -> bytes:
        hdrs = {"Range": f"bytes={start}-{end - 1}"}
        try:
            resp = self._open(self._direct or self.url, headers=hdrs)
        except AccessRefused:
            if self._direct is None:
                raise  # the original url refuses: a real gate (D-E), never retried here
            self._direct = None  # an expired signed redirect: resolve it again
            resp = self._open(self.url, headers=hdrs)
        with resp:
            if _range_start(resp) != start:
                raise RetriesExhausted(f"GET {self.url}: server ignored Range {hdrs['Range']}")
            if self._direct is None and hasattr(resp, "geturl"):
                self._direct = resp.geturl()
            data = resp.read()
        self.requests += 1
        self.fetched += len(data)
        if len(data) != end - start:
            raise http.client.IncompleteRead(data, end - start - len(data))
        return data


def open_remote_zip(url: str, size: int, *, open_fn: OpenFn | None = None) -> tuple[
    zipfile.ZipFile, RemoteFile
]:  # fmt: skip
    """``ZipFile`` over a :class:`RemoteFile`: the tail (end records, 64 KiB comment max)
    is one request and the central directory is one more — no member bytes are read."""
    rf = RemoteFile(url, size, open_fn=open_fn)
    rf.load(max(0, size - (1 << 16) - 22 - 20 - 56), size)
    return zipfile.ZipFile(rf), rf  # type: ignore[arg-type]


def member_span(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    """``[local header, next local header or central directory)`` for ``info``."""
    offsets = getattr(zf, "_wp6i_offsets", None)
    if offsets is None:
        offsets = sorted({i.header_offset for i in zf.infolist()} | {zf.start_dir})
        zf._wp6i_offsets = offsets  # type: ignore[attr-defined]
    nxt = offsets[bisect.bisect_right(offsets, info.header_offset)]
    return info.header_offset, nxt


def read_member(zf: zipfile.ZipFile, rf: RemoteFile, info: zipfile.ZipInfo) -> bytes:
    """One ranged GET for the member's span, then the shared member reader decompresses
    + CRC-checks it from that cached window — Deflate64 (method 9) included (D-AH 2/4)."""
    start, end = member_span(zf, info)
    rf.load(start, end)
    return _read_zip_member(zf, info)


def iter_tar_stream(
    fileobj: IO[bytes],
    *,
    member_ok: Callable[[str], bool],
    max_bytes: int,
    resume_after: str | None = None,
) -> Iterator[tuple[str, bytes]]:
    """WP-6n: stream a ``.tar``/``.tar.gz``/``.tgz`` container member-by-member with no
    on-disk copy. Unlike a zip (WP-6k), a tar has no central directory, so it cannot be
    range-read; ``fileobj`` (the raw HTTP body) is consumed sequentially, once, through
    ``tarfile.open(mode="r|*")`` (gz/bz2/xz auto-detected from content). ``member_ok`` is
    the caller's existing image/label member filter; a member whose header declares more
    than ``max_bytes`` is skipped without ever being read — the same per-item size cap
    other adapters apply, just enforced from the header instead of a completed download.

    Resume (D-D): a tar cannot seek, so a caller resuming after a crash reopens the url
    from byte 0 and passes the checkpointed ``resume_after`` member name; every member up
    to and including it is re-read off the wire here and discarded, never re-yielded --
    that redownload of the already-uploaded prefix is the accepted cost of a tar having no
    index (a zip resume needs no redownload, WP-6k, because its central directory lets a
    later member be range-read directly).
    """
    skipping = resume_after is not None
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if skipping:
                if member.name == resume_after:
                    skipping = False
                continue
            if not member_ok(member.name):
                continue
            if member.size > max_bytes:
                log.warning(
                    "tar member %s: %d B over the %d B cap, skipped",
                    member.name, member.size, max_bytes,
                )  # fmt: skip
                continue
            handle = tar.extractfile(member)
            yield member.name, (handle.read() if handle is not None else b"")
