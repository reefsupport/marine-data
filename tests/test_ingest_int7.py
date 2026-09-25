"""INT-ingest7: Deflate64 members over HTTP Range, and the PANGAEA 503 backoff."""

from __future__ import annotations

import email.message
import io
import os
import urllib.error

import pytest
from test_ingest_wp6j import _deflate64_zip
from test_range_wp6i import RangeServer

from marinedata.adapters import _http
from marinedata.adapters._range import open_remote_zip, read_member
from marinedata.concurrency import RetriesExhausted
from marinedata.ingest_missing import fetch_status


@pytest.fixture
def srv():
    s = RangeServer()
    yield s
    s.close()


def test_range_reader_inflates_deflate64_members(srv) -> None:
    members = {f"imgs/{i:03d}.jpg": os.urandom(20_000) + b"\0" * 20_000 for i in range(30)}
    srv.files["/d64.zip"] = blob = _deflate64_zip(members)
    zf, rf = open_remote_zip(srv.base + "/d64.zip", len(blob))
    assert {i.compress_type for i in zf.infolist()} == {9}  # every member is Deflate64
    for name in ("imgs/000.jpg", "imgs/017.jpg", "imgs/029.jpg"):
        before = rf.requests
        assert read_member(zf, rf, zf.getinfo(name)) == members[name]
        assert rf.requests == before + 1  # one ranged GET per member, served from its window
    assert all(r is not None for _, r in srv.log)  # never a whole-file GET
    assert rf.fetched < len(blob) // 4


def test_range_reader_deflate64_crc_mismatch_raises(srv) -> None:
    import zipfile

    raw = bytearray(_deflate64_zip({"a.bin": b"abc" * 5000, "b.bin": b"xyz" * 5000}))
    raw[40] ^= 0xFF  # flip a compressed byte of a.bin (local header 30 B + name 5 B)
    srv.files["/bad.zip"] = bytes(raw)
    zf, rf = open_remote_zip(srv.base + "/bad.zip", len(raw))
    with pytest.raises((zipfile.BadZipFile, ValueError, OSError)):
        read_member(zf, rf, zf.getinfo("a.bin"))


def _http_error(url: str, code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "Service Unavailable", hdrs, io.BytesIO(b""))


def test_pangaea_hosts_are_slow_5xx_hosts() -> None:
    for host in ("hs.pangaea.de", "download.pangaea.de", "doi.pangaea.de", "PANGAEA.DE"):
        assert _http.is_slow_5xx_host(host)
    for host in ("zenodo.org", "notpangaea.de", "pangaea.de.evil.com", "huggingface.co"):
        assert not _http.is_slow_5xx_host(host)


def test_pangaea_backoff_schedule() -> None:
    url = "https://hs.pangaea.de/x.jpg"
    plain = [_http.backoff_s(a, _http_error(url, 503), slow=True) for a in range(7)]
    assert plain == [2, 4, 8, 16, 32, 64, 120]  # exponential, capped at 120 s
    assert _http.backoff_s(0, _http_error(url, 503, "45"), slow=True) == 45  # Retry-After floor
    assert _http.backoff_s(4, _http_error(url, 503, "7"), slow=True) == 32  # tape-recall hint
    assert _http.backoff_s(0, _http_error(url, 503, "900"), slow=True) == 120  # still capped
    assert _http.backoff_s(3, ConnectionResetError(), slow=True) == 16
    assert [_http.backoff_s(a, None, slow=False) for a in (0, 3, 6)] == [1, 8, 30]  # unchanged


def _patch(monkeypatch, codes: list[int | None], retry_after: str | None = None):
    calls, sleeps = [], []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        code = codes[min(len(calls), len(codes)) - 1]
        if code is None:
            return io.BytesIO(b"ok")
        raise _http_error(req.full_url, code, retry_after)

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", sleeps.append)
    monkeypatch.setattr(_http.random, "uniform", lambda a, b: 0.0)
    return calls, sleeps


def test_pangaea_503_recovers_within_8_tries(monkeypatch) -> None:
    calls, sleeps = _patch(monkeypatch, [503] * 6 + [None])
    with _http.open_url("https://hs.pangaea.de/Images/PS96/a.jpg") as resp:
        assert resp.read() == b"ok"
    assert len(calls) == 7 and sleeps == [2, 4, 8, 16, 32, 64]


def test_pangaea_503_honours_retry_after_then_counts_missing(monkeypatch) -> None:
    calls, sleeps = _patch(monkeypatch, [503], retry_after="30")
    with pytest.raises(RetriesExhausted) as info:
        _http.open_url("https://download.pangaea.de/dataset/1/files/a.jpg")
    assert len(calls) == 8 and sleeps == [30, 30, 30, 30, 32, 64, 120]  # 8 tries, none after
    assert fetch_status(info.value) == "retries-exhausted"  # D-AF: skipped, not an abort


def test_other_hosts_keep_4_tries(monkeypatch) -> None:
    calls, sleeps = _patch(monkeypatch, [503])
    with pytest.raises(RetriesExhausted):
        _http.open_url("https://zenodo.org/records/1/files/a.zip")
    assert len(calls) == 4 and sleeps == [1, 2, 4]
