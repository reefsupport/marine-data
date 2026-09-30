"""INT-ingest5c: disk floor before listing, listing cache, per-host 429 pacing."""

from __future__ import annotations

import email.message
import io
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from marinedata.adapters import RemoteItem, _http, _throttle
from marinedata.adapters._throttle import (
    MAX_CONSECUTIVE_429,
    THROTTLE_CAP_S,
    HostThrottled,
    ThrottleExhausted,
    reset_throttles,
    retry_after_s,
)
from marinedata.adapters.manifest import RowJoinMixin
from marinedata.adapters.wikimedia import CommonsAdapter
from marinedata.cli_ingest_batch import _is_hf_rate_limited, run_one
from marinedata.concurrency import RetriesExhausted
from marinedata.ingest_listing import LISTING_DIR, cache_path, list_source
from marinedata.ingest_missing import fetch_status
from marinedata.ingest_source import IngestSpec, run_ingest
from marinedata.s3_upload import DiskFloorError, DiskGuard


@pytest.fixture(autouse=True)
def _fresh_throttles():
    reset_throttles()
    yield
    reset_throttles()


@pytest.fixture
def http_calls(monkeypatch):
    calls: list[str] = []

    def boom(req, *a, **k):
        calls.append(getattr(req, "full_url", str(req)))
        raise AssertionError("network call below the floor")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    return calls


class _NoS3:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"S3 {name} below the floor")


def _low_disk(monkeypatch):
    monkeypatch.setattr(DiskGuard, "free_bytes", lambda self: 1 << 30)  # 1 GiB < 40 GiB floor


def test_batch_below_floor_makes_zero_http_calls(tmp_path, monkeypatch, http_calls):
    _low_disk(monkeypatch)
    spec = tmp_path / "wm.yaml"
    spec.write_text(
        "id: wm\nadapter: commons-api\nlicense: CC-BY-4.0\nattribution: X\n"
        "params:\n  categories: ['Underwater photographs']\n"
    )
    s3 = _NoS3()
    res = run_one(spec, tmp_path / "work", s3)
    assert res["status"] == "error" and res["error"].startswith("DiskFloorError")
    assert http_calls == [] and s3.calls == []


def test_run_ingest_below_floor_makes_zero_http_calls(tmp_path, http_calls):
    spec = IngestSpec("wm", "commons-api", {"categories": ["X"]}, "CC-BY-4.0", "X")
    guard = DiskGuard(tmp_path / "w", floor_bytes=1 << 60)
    with pytest.raises(DiskFloorError):
        run_ingest(spec, tmp_path / "w", client=_NoS3(), guard=guard)
    assert http_calls == []


class _FakeAdapter:
    listing_cacheable = True

    def __init__(self) -> None:
        self.listed = 0
        self.restored = None

    def resolve_version(self):
        self.listed += 1
        return "v-live"

    def enumerate(self):
        self.listed += 1
        yield RemoteItem("a.jpg", "https://x/a.jpg", 10, None, "ab" * 32)
        yield RemoteItem("b.tar", "https://x/b.tar", None, "cd" * 16, None, ("https://x/b1",))

    def listing_state(self, item):
        return {"k": item.key, "n": [1, 2]}

    def restore_listing(self, items, states):
        self.restored = (items, states)


def test_listing_cache_is_written_then_reused(tmp_path):
    spec = IngestSpec("src", "fake", {"q": 1}, "CC-BY-4.0", "X")
    first = _FakeAdapter()
    v1, items1 = list_source(spec, first, tmp_path / LISTING_DIR)
    assert first.listed == 2 and cache_path(tmp_path / LISTING_DIR, spec).exists()
    again = _FakeAdapter()
    v2, items2 = list_source(spec, again, tmp_path / LISTING_DIR)
    assert again.listed == 0  # no resolve_version, no enumerate: zero upstream calls
    assert (v2, items2) == (v1, items1) == ("v-live", items1)
    assert again.restored[1] == [{"k": "a.jpg", "n": [1, 2]}, {"k": "b.tar", "n": [1, 2]}]
    changed = IngestSpec("src", "fake", {"q": 2}, "CC-BY-4.0", "X")
    third = _FakeAdapter()
    list_source(changed, third, tmp_path / LISTING_DIR)
    assert third.listed == 2  # different params -> different spec hash -> fresh listing


def test_listing_not_cached_when_adapter_not_opted_in_or_state_not_json(tmp_path):
    spec = IngestSpec("src", "fake", {}, "CC-BY-4.0", "X")
    a = _FakeAdapter()
    a.listing_cacheable = False
    list_source(spec, a, tmp_path)
    b = _FakeAdapter()
    b.listing_state = lambda item: {"when": object()}
    list_source(spec, b, tmp_path)
    assert not cache_path(tmp_path, spec).exists()


def test_real_adapters_restore_listing_state():
    items = [RemoteItem("k1", "https://u/1"), RemoteItem("k2", "https://u/2")]
    wm = CommonsAdapter({})
    wm.restore_listing(items, [{"fields": {}, "labels": {"sha1": "x"}}, None])
    assert wm._files == items and set(wm._meta) == {"k1"}
    assert wm.listing_state(items[0]) == {"fields": {}, "labels": {"sha1": "x"}}
    row = RowJoinMixin()
    row.restore_listing(items, [{"a": 1}, {"a": 2}])
    assert row.listing_state(items[1]) == {"a": 2}


def _err(url: str, code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "Too Many Requests", hdrs, None)


@pytest.fixture
def net(monkeypatch):
    """Scripted urlopen: pop one outcome per call (an exception to raise, else a body)."""
    state = {"script": [], "calls": 0, "sleeps": []}

    def fake(req, *a, **k):
        state["calls"] += 1
        out = state["script"].pop(0)
        if isinstance(out, BaseException):
            raise out
        return io.BytesIO(out)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(_throttle.time, "sleep", lambda s: state["sleeps"].append(s))
    monkeypatch.setattr(_http.time, "sleep", lambda s: state["sleeps"].append(s))
    return state


WM = "https://upload.wikimedia.org/a.jpg"


def test_non_hf_429_honours_retry_after_then_succeeds(net):
    net["script"] = [_err(WM, 429, "7"), _err(WM, 429, "7"), b"ok"]
    assert _http.open_url(WM).read() == b"ok"
    assert net["calls"] == 3
    assert sum(1 for s in net["sleeps"] if s >= 6.9) == 2  # host-wide wait ~ Retry-After


def test_retry_after_is_capped_and_http_date_parsed():
    hdrs = email.message.Message()
    hdrs["Retry-After"] = "Wed, 21 Oct 2015 07:28:10 GMT"
    assert retry_after_s(hdrs, now=1445412480.0) == pytest.approx(10.0)
    t = _throttle.throttle_for("h")
    assert t.on_429(10_000.0, 1) == THROTTLE_CAP_S


def test_non_hf_item_skipped_after_six_tries_not_aborted(net):
    net["script"] = [_err(WM, 429)] * 6
    with pytest.raises(ThrottleExhausted) as ei:
        _http.open_url(WM)
    assert net["calls"] == 6 and fetch_status(ei.value) == "throttled-429"
    assert max(net["sleeps"]) <= THROTTLE_CAP_S


def test_non_hf_source_aborts_on_20_consecutive_429s(net):
    net["script"] = [_err(WM, 429)] * MAX_CONSECUTIVE_429
    for _ in range(3):  # 3 items x 6 tries = 18 consecutive 429s: skipped, not aborted
        with pytest.raises(ThrottleExhausted):
            _http.open_url(WM)
    with pytest.raises(HostThrottled) as ei:
        _http.open_url(WM)
    assert net["calls"] == MAX_CONSECUTIVE_429 and fetch_status(ei.value) is None


def test_success_resets_the_consecutive_count(net):
    net["script"] = ([_err(WM, 429)] * 5 + [b"ok"]) * 5  # 25 x 429, never 20 in a row
    for _ in range(5):
        assert _http.open_url(WM).read() == b"ok"


def test_hf_429_keeps_d_aa_abort(net):
    url = "https://huggingface.co/datasets/x/y/resolve/main/a.parquet"
    net["script"] = [_err(url, 429)] * 4
    with pytest.raises(RetriesExhausted) as ei:
        _http.open_url(url)
    assert not isinstance(ei.value, ThrottleExhausted)
    assert net["calls"] == 4 and fetch_status(ei.value) is None
    res = {"status": "error", "error": f"RetriesExhausted: {ei.value}"}
    assert _is_hf_rate_limited(res)


def test_wikimedia_pace_is_thread_safe(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("marinedata.adapters.wikimedia.time.sleep", slept.append)
    wm = CommonsAdapter({"min_interval_s": 1.0})
    threads = [threading.Thread(target=wm._pace) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(slept) >= 7 and max(slept) >= 6.5  # 8 distinct slots, ~1 s apart


def test_listing_dir_under_queue_workdir():
    assert Path(LISTING_DIR).name == "_listings"
