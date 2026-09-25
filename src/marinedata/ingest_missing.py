"""D-AF (2026-09-25 charter): per-item skip in the ``ingest-source`` runner.

A dead item is skipped and recorded in ``MISSING.tsv`` (key, url, status, first_seen) in
the staged tree instead of aborting the whole source:

* fetch: HTTP 404/410, or retries exhausted on a 5xx/connection failure;
* decode: any failure *before the item wrote anything* to the staged tree (a container
  that fails half-way has already written rows, so it still aborts — nothing to roll back).

The source still aborts (:class:`TooManyMissing`) when skipped > 5% of attempted items
after >= 200 attempts, when 50 items in a row fail (host down), or when every attempted
item failed (a small source must never stage an empty version).

Never skipped, so they propagate exactly as before D-AF: 401/403 ``AccessRefused`` (D-E,
``needs-yohan``), any other 4xx, 429 after its retries (a throttle, not a dead item — D-AA
cools HF down on it), a declared-digest mismatch (pin violation), the disk floor / temp
cap, and non-``Exception`` interrupts.

``first_seen`` survives a kill-and-resume: every skip is also appended to a journal in the
work dir, and a re-run of the same version keeps the first timestamp for that key.
"""

from __future__ import annotations

import datetime as dt
import urllib.error
from collections.abc import Callable
from pathlib import Path

from .adapters import AccessRefused, DigestMismatch
from .adapters._throttle import HostThrottled, ThrottleExhausted
from .concurrency import RetriesExhausted
from .s3_upload import DiskFloorError, TempCapError

MISSING_FILE = "MISSING.tsv"
COLUMNS = ("key", "url", "status", "first_seen")
DEAD_HTTP = frozenset({404, 410})
MAX_FRACTION = 0.05
MIN_ATTEMPTS = 200
MAX_CONSECUTIVE = 50


class TooManyMissing(RuntimeError):
    """D-AF abort: the source is failing as a whole, not item by item."""


def _never_skip(exc: BaseException) -> bool:
    return not isinstance(exc, Exception) or isinstance(
        exc,
        (
            AccessRefused,
            DigestMismatch,
            DiskFloorError,
            TempCapError,
            TooManyMissing,
            HostThrottled,
        ),
    )


def _http_code(exc: BaseException | None) -> int | None:
    return exc.code if isinstance(exc, urllib.error.HTTPError) else None


def fetch_status(exc: BaseException) -> str | None:
    """MISSING.tsv status for a failed fetch, or ``None`` when it must abort the source."""
    if _never_skip(exc):
        return None
    code = _http_code(exc)
    if code is not None:
        return f"http-{code}" if code in DEAD_HTTP else None
    if isinstance(exc, ThrottleExhausted):  # INT-ingest5c: non-HF host, paced 6 tries
        return "throttled-429"
    if isinstance(exc, RetriesExhausted):
        cause = _http_code(exc.__cause__)
        if cause in DEAD_HTTP:
            return f"http-{cause}"
        if cause == 429 or "HTTP Error 429" in str(exc):
            return None
        return "retries-exhausted"
    return None


def decode_status(exc: BaseException, *, wrote: bool) -> str | None:
    """MISSING.tsv status for a failed decode, or ``None`` when it must abort the source."""
    if wrote or _never_skip(exc):
        return None
    return fetch_status(exc) or f"decode-error:{type(exc).__name__}"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cell(value: str) -> str:
    return " ".join(str(value).split("\t")).replace("\n", " ").replace("\r", " ")


class MissingLedger:
    """Counts attempts, records skips, and enforces D-AF's abort thresholds."""

    def __init__(self, journal: Path, *, now: Callable[[], str] = _utc_now) -> None:
        self.journal, self._now = journal, now
        self.rows: list[tuple[str, str, str, str]] = []
        self.attempted = 0
        self.consecutive = 0
        self._first_seen: dict[str, str] = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) == len(COLUMNS):
                    self._first_seen.setdefault(parts[0], parts[3])

    def ok(self) -> None:
        self.attempted += 1
        self.consecutive = 0
        self._check()

    def skip(self, key: str, url: str, status: str) -> None:
        self.attempted += 1
        self.consecutive += 1
        seen = self._first_seen.setdefault(key, self._now())
        row = (_cell(key), _cell(url), _cell(status), seen)
        self.rows.append(row)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as fh:
            fh.write("\t".join(row) + "\n")
        self._check()

    def _check(self) -> None:
        n = len(self.rows)
        last = f" (last: {self.rows[-1][0]} {self.rows[-1][2]})" if n else ""
        if self.consecutive >= MAX_CONSECUTIVE:
            raise TooManyMissing(
                f"D-AF: {self.consecutive} consecutive item failures — host down?{last}"
            )
        if self.attempted >= MIN_ATTEMPTS and n > MAX_FRACTION * self.attempted:
            raise TooManyMissing(
                f"D-AF: {n}/{self.attempted} items skipped (> {MAX_FRACTION:.0%}){last}"
            )

    def finish(self) -> None:
        if self.attempted and len(self.rows) == self.attempted:
            last = self.rows[-1]
            raise TooManyMissing(
                f"D-AF: all {self.attempted} attempted items failed (last: {last[0]} {last[2]})"
            )

    def write(self, root: Path) -> Path | None:
        """Write ``MISSING.tsv`` into the staged tree; nothing when no item was skipped,
        so a clean source's tree (and root digest) is byte-identical to before D-AF."""
        data = self.render()
        if data is None:
            return None
        path = root / MISSING_FILE
        path.write_bytes(data)
        return path

    def render(self) -> bytes | None:
        """The ``MISSING.tsv`` bytes (``None`` when nothing was skipped) — shared by disk
        mode (:meth:`write`) and WP-6h stream mode (PUT from memory)."""
        if not self.rows:
            return None
        lines = ["\t".join(COLUMNS)] + ["\t".join(r) for r in self.rows]
        return ("\n".join(lines) + "\n").encode()

    def restore(self, rows: list, attempted: int, consecutive: int) -> None:
        """WP-6h resume: re-apply a stream checkpoint's skipped rows and counters."""
        self.rows.extend(tuple(r) for r in rows)
        self.attempted, self.consecutive = int(attempted), int(consecutive)
