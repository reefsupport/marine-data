"""`fetch._get` retry policy (S44): 5xx, 429 and timeouts retry with exponential
backoff; any other 4xx raises at once."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

import marinedata.fetch as fetch
from marinedata.fetch import FetchError


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x.invalid/f", code, "err", {}, None)  # type: ignore[arg-type]


def test_5xx_429_and_timeout_retry_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[object] = [_http_error(503), _http_error(429), TimeoutError("read"), b"OK"]
    sleeps: list[float] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)

    assert fetch._get("https://x.invalid/f", retries=5) == b"OK"
    assert sleeps == [1.5, 3.0, 6.0]


def test_other_4xx_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        calls.append(request.full_url)
        raise _http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)

    with pytest.raises(FetchError, match="HTTP 404"):
        fetch._get("https://x.invalid/f", retries=5)
    assert len(calls) == 1
