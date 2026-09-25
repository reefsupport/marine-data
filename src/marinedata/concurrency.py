"""Shared concurrency primitives for WP-6b: bounded per-host politeness and a retry-with-
backoff-and-jitter helper reused by the HTTP fetch path and the S3 upload path.

Nothing here starts unbounded work: callers own their own ``ThreadPoolExecutor``s, sized
by ``--jobs``/``--part-jobs`` (max 32, CLI-enforced). This module only supplies the two
bits that are identical in both places — "don't hammer one host/endpoint" and "a 429/5xx/
connection reset is not fatal, try again with jitter" — so behaviour (and its tests)
lives in one spot instead of being duplicated per caller.
"""

from __future__ import annotations

import random
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar
from urllib.parse import urlsplit

T = TypeVar("T")

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_BOTO_CODES = frozenset(
    {
        "SlowDown",
        "RequestTimeout",
        "ThrottlingException",
        "429",
        "InternalError",
        "ServiceUnavailable",
        "RequestTimeTooSkewed",
    }
)


def is_retryable_exc(exc: BaseException) -> bool:
    """429/5xx (HTTP or S3) and connection resets — never a 4xx auth/validation error."""
    code = getattr(exc, "code", None)  # urllib.error.HTTPError
    if isinstance(code, int) and code in _RETRYABLE_HTTP_STATUS:
        return True
    response = getattr(exc, "response", None)  # botocore.exceptions.ClientError
    if isinstance(response, dict):
        err = response.get("Error") or {}
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if err.get("Code") in _RETRYABLE_BOTO_CODES:
            return True
        if isinstance(status, int) and status in _RETRYABLE_HTTP_STATUS:
            return True
    return isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError, OSError))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base: float = 0.2,
    cap: float = 10.0,
    jitter: float = 0.3,
    retryable: Callable[[BaseException], bool] = is_retryable_exc,
) -> T:
    """Call ``fn()``; retry on a retryable ``Exception`` with capped exponential backoff
    plus jitter. A non-retryable exception, or ``BaseException`` that is not an
    ``Exception`` (e.g. a simulated kill or ``KeyboardInterrupt``), always propagates
    immediately — retrying must never mask a real interruption."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if not retryable(exc):
                raise
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(min(cap, base * (2**attempt)) + random.uniform(0, jitter))
    raise RuntimeError(f"failed after {retries} attempts: {last}") from last


class HostLimiter:
    """Bound concurrent requests per host (``--max-per-host``, default 4) across every
    worker thread, regardless of ``--jobs``. Politeness, not correctness: a single host
    never sees more than ``max_per_host`` in-flight requests at once."""

    def __init__(self, max_per_host: int = 4) -> None:
        self.max_per_host = max(1, int(max_per_host))
        self._sems: dict[str, threading.Semaphore] = {}
        self._guard = threading.Lock()

    def _sem(self, host: str) -> threading.Semaphore:
        with self._guard:
            sem = self._sems.get(host)
            if sem is None:
                sem = threading.Semaphore(self.max_per_host)
                self._sems[host] = sem
            return sem

    @contextmanager
    def acquire(self, url: str) -> Iterator[None]:
        sem = self._sem(urlsplit(url).netloc)
        sem.acquire()
        try:
            yield
        finally:
            sem.release()
