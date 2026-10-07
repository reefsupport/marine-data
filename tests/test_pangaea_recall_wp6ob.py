"""WP-6o-b: the recall trigger loops until ready or max-hours, rate-limited (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recall_mod():
    spec = importlib.util.spec_from_file_location(
        "pangaea_recall_wp6ob", ROOT / "scripts/pangaea_recall.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeSession:
    """First loop: 503 for urls[0], urls[1]; 200 for urls[2], urls[3]. Second loop: 206."""

    def __init__(self, urls: list[str]) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._flaky = set(urls[:2])
        self._seen: set[str] = set()

    def get(self, url: str, headers: dict, timeout: float) -> _Resp:
        self.calls.append((url, dict(headers)))
        first_time = url not in self._seen
        self._seen.add(url)
        if url in self._flaky and first_time:
            return _Resp(503)
        return _Resp(200 if url not in self._flaky else 206)


class FakeClock:
    """Advances only on sleep(); recall() must stop once max_hours has passed."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_recall_two_loops_six_rows_rate_respected(recall_mod, tmp_path):
    urls = [f"https://download.pangaea.de/dataset/1/files/f{i}.jpg" for i in range(4)]
    session = FakeSession(urls)
    clock = FakeClock()
    tsv_path = tmp_path / "series.tsv"

    loops = recall_mod.recall(
        urls,
        tsv_path,
        max_hours=12.0,
        rate=2.0,
        session=session,
        clock=clock.clock,
        wall_clock=lambda: 1000.0,
        sleep=clock.sleep,
    )

    assert loops == 2
    rows = tsv_path.read_text().splitlines()
    assert len(rows) == 6
    statuses = [r.split("\t")[1] for r in rows]
    assert statuses == ["503", "503", "200", "200", "206", "206"]

    # rate respected: every in-loop gap is 1/rate = 0.5s (3 gaps in loop 1, 1 in loop 2);
    # the single 300s gap is the inter-loop wait before the retry pass.
    interval_sleeps = [s for s in clock.sleeps if s != recall_mod.SLEEP_BETWEEN_LOOPS_S]
    assert interval_sleeps == [pytest.approx(0.5)] * 4
    assert clock.sleeps.count(recall_mod.SLEEP_BETWEEN_LOOPS_S) == 1


def test_restart_skips_already_ready_urls(recall_mod, tmp_path):
    tsv_path = tmp_path / "series.tsv"
    tsv_path.write_text("https://x/a.jpg\t200\t1\nhttps://x/b.jpg\t503\t1\n")
    session = FakeSession([])
    clock = FakeClock()

    loops = recall_mod.recall(
        ["https://x/a.jpg", "https://x/b.jpg"],
        tsv_path,
        max_hours=12.0,
        rate=10.0,
        session=session,
        clock=clock.clock,
        wall_clock=lambda: 2000.0,
        sleep=clock.sleep,
    )

    assert loops == 1
    urls_requested = [c[0] for c in session.calls]
    assert urls_requested == ["https://x/b.jpg"]
