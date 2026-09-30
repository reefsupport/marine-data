"""Per-host HTTP 429 policy for adapter fetches (INT-ingest5c).

A 429 is the server saying "slow down", not "this item is gone". Two regimes:

* **Hugging Face hosts** (``huggingface.co``/``hf.co`` and their subdomains) keep the
  D-AA behaviour unchanged: :func:`~marinedata.adapters._http.open_url` retries a 429 like
  any other retryable status and then raises ``RetriesExhausted``, which aborts the source
  so ``ingest-batch`` can apply its 15-minute HF cooldown.
* **Every other host** gets per-item pacing: honour ``Retry-After`` (seconds or HTTP-date),
  otherwise exponential backoff with jitter, each wait capped at :data:`THROTTLE_CAP_S`,
  at most :data:`THROTTLE_TRIES` tries per item. A wait is host-wide (``not_before``), so
  sibling fetch threads also back off. An item that is still throttled after its tries
  raises :class:`ThrottleExhausted` (D-AF skips it as ``throttled-429``); the source is
  aborted only after :data:`MAX_CONSECUTIVE_429` 429s in a row from one host
  (:class:`HostThrottled`), i.e. when the host is refusing us, not one item.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import random
import threading
import time
from collections.abc import Mapping
from typing import Any

from ..concurrency import RetriesExhausted

THROTTLE_TRIES = 6
THROTTLE_CAP_S = 120.0
THROTTLE_BASE_S = 2.0
MAX_CONSECUTIVE_429 = 20
_HF_SUFFIXES = ("huggingface.co", "hf.co")


class ThrottleExhausted(RetriesExhausted):
    """One item still answered HTTP 429 after :data:`THROTTLE_TRIES` tries (non-HF host)."""


class HostThrottled(RuntimeError):
    """:data:`MAX_CONSECUTIVE_429` 429s in a row from one non-HF host: abort the source."""


def is_hf_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == s or host.endswith("." + s) for s in _HF_SUFFIXES)


def retry_after_s(headers: Mapping[str, Any] | Any, *, now: float | None = None) -> float | None:
    """Seconds requested by a ``Retry-After`` header (delta-seconds or HTTP-date), or None."""
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


class HostThrottle:
    """Shared state for one host: consecutive-429 count and a host-wide ``not_before``."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.consecutive = 0
        self.not_before = 0.0
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            wait = self.not_before - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def on_success(self) -> None:
        with self._lock:
            self.consecutive = 0

    def on_429(self, retry_after: float | None, item_try: int) -> float:
        """Record a 429 (``item_try`` = this item's 429 count, 1-based); return the delay
        now imposed on the host. Raises :class:`HostThrottled` on the 20th in a row."""
        backoff = THROTTLE_BASE_S * (2 ** (item_try - 1)) + random.uniform(0, 1)
        delay = min(THROTTLE_CAP_S, max(retry_after or 0.0, backoff))
        with self._lock:
            self.consecutive += 1
            n = self.consecutive
            if n >= MAX_CONSECUTIVE_429:
                self.consecutive = 0  # the next source to this host starts its own count
            self.not_before = max(self.not_before, time.monotonic() + delay)
        if n >= MAX_CONSECUTIVE_429:
            raise HostThrottled(f"HTTP 429: {n} consecutive 429s from {self.host} — source aborted")
        return delay


_REGISTRY: dict[str, HostThrottle] = {}
_REGISTRY_LOCK = threading.Lock()


def throttle_for(host: str) -> HostThrottle:
    with _REGISTRY_LOCK:
        t = _REGISTRY.get(host)
        if t is None:
            t = _REGISTRY[host] = HostThrottle(host)
        return t


def reset_throttles() -> None:
    """Forget every host's state (tests)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
