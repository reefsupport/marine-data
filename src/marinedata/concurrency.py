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
from typing import Any, TypeVar
from urllib.parse import urlsplit

import botocore.exceptions as _bce

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
# A lone PutObject 400 ("BadRequest ... N/A", empty body) hit a 2 h and a 6 h deepseagrass
# run and brackishmot; an identical re-run succeeded, so it is transient. ONLY this code
# (botocore reports "400" for body-less replies) at status 400 is retried: InvalidArgument,
# MalformedXML, EntityTooLarge etc. are 400s that are real client bugs and stay fatal.
_TRANSIENT_400_CODES = frozenset({"BadRequest", "400"})
# botocore transport failures that are not an ``OSError`` (timeouts already are one).
_RETRYABLE_BOTOCORE = (
    _bce.ReadTimeoutError,
    _bce.ConnectTimeoutError,
    _bce.EndpointConnectionError,
    _bce.ConnectionClosedError,
)


class RetriesExhausted(RuntimeError):
    """Every retry of a retryable failure (429/5xx/connection) failed; ``__cause__`` is the
    last underlying exception. A ``RuntimeError`` so existing callers are unaffected;
    the ``ingest-source`` runner uses the type to count it toward D-AF's thresholds."""


def is_retryable_exc(exc: BaseException) -> bool:
    """429/5xx (HTTP or S3), connection resets/timeouts and a transient S3 ``BadRequest``
    (400) — never a 403/404 or any other 4xx auth/validation error."""
    code = getattr(exc, "code", None)  # urllib.error.HTTPError
    if isinstance(exc, urllib.error.HTTPError):
        # HTTPError subclasses URLError, so without this a 404/410 fell through to the
        # URLError branch below and was retried `retries` times (D-AF: a dead link).
        return isinstance(code, int) and code in _RETRYABLE_HTTP_STATUS
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
        if status == 400 and err.get("Code") in _TRANSIENT_400_CODES:
            return True
    if isinstance(exc, _RETRYABLE_BOTOCORE):
        return True
    return isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError, OSError))


def _s3_response(exc: BaseException | None) -> dict[str, Any] | None:
    """The botocore ``response`` dict of ``exc`` or, for a ``RetriesExhausted`` wrapper,
    of its cause."""
    for cand in (exc, getattr(exc, "__cause__", None)):
        response = getattr(cand, "response", None)
        if isinstance(response, dict):
            return response
    return None


S3_CONTEXT_MARK = " [s3 "


def describe_error(exc: BaseException) -> str:
    """``"Type: message"`` plus, when the failure is an S3 call, `` [s3 op=… key=… status=…
    code=… request_id=…]``. ``key`` is set by the put path via :func:`tag_s3_key`. Only the
    key, status, error code and RequestId are read — never headers, URLs or credentials."""
    text = f"{type(exc).__name__}: {exc}"
    response = _s3_response(exc)
    ctx: list[str] = []
    op = getattr(exc, "operation_name", None) or getattr(exc.__cause__, "operation_name", None)
    if op:
        ctx.append(f"op={op}")
    key = getattr(exc, "s3_key", None)
    if key:
        ctx.append(f"key={key}")
    if response is not None:
        meta = response.get("ResponseMetadata") or {}
        code = (response.get("Error") or {}).get("Code")
        for name, val in (
            ("status", meta.get("HTTPStatusCode")),
            ("code", code),
            ("request_id", meta.get("RequestId")),
        ):
            if val not in (None, ""):
                ctx.append(f"{name}={val}")
    return f"{text}{S3_CONTEXT_MARK}{' '.join(ctx)}]" if ctx else text


def tag_s3_key(key: str, fn: Callable[[], T]) -> T:
    """Run ``fn``; if it raises, remember the S3 ``key`` on the exception (first tag wins,
    exception type unchanged) so :func:`describe_error` can name the failing object."""
    try:
        return fn()
    except Exception as exc:
        if getattr(exc, "s3_key", None) is None:
            exc.s3_key = key  # type: ignore[attr-defined]
        raise


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
    raise RetriesExhausted(f"failed after {retries} attempts: {last}") from last


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
