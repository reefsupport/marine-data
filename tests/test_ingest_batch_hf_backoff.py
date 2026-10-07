"""D-AA (2026-09-25 charter): ``ingest-batch`` cools down (>= 15 min) and retries an
HF source on a 429, keeping HF concurrency at 1, and gives up as ``needs-yohan:
hf-rate-limit`` after 3 cool-downs. No moto/boto3 needed — ``run_one`` is mocked and
``client`` is never touched, so this runs even where the S3-mocking extra is absent."""

from __future__ import annotations

import threading
from pathlib import Path

import marinedata.cli_ingest_batch as cib

_HF_429_ERROR = (
    "RuntimeError: GET https://huggingface.co/datasets/x/resolve/y/z.jpg "
    "failed after 4 attempts: HTTP Error 429: Too Many Requests"
)


def _spec(tmp_path: Path, source_id: str, adapter: str) -> Path:
    path = tmp_path / f"{source_id}.yaml"
    path.write_text(
        f"id: {source_id}\nadapter: {adapter}\nlicense: CC-BY-4.0\n"
        f"attribution: Synthetic Lab\nparams: {{}}\n"
    )
    return path


def _run_aware(spec_path, tmp_path, fake_run_one, monkeypatch, sleeps, **kw):
    monkeypatch.setattr(cib, "run_one", fake_run_one)
    return cib._run_one_hf_aware(
        spec_path,
        tmp_path / "work",
        client=None,
        hf_semaphore=threading.Semaphore(1),
        cooldown_s=kw.pop("cooldown_s", 900.0),
        max_cooldowns=kw.pop("max_cooldowns", 3),
        sleep=sleeps.append,
    )


def test_hf_429_cools_down_thrice_then_needs_yohan(tmp_path, monkeypatch):
    spec_path = _spec(tmp_path, "hf-source", "hf")
    calls = {"n": 0}

    def fake_run_one(path, work_root, client):
        calls["n"] += 1
        return {
            "source_id": "hf-source",
            "status": "error",
            "error": _HF_429_ERROR,
            "elapsed_s": 0.1,
        }

    sleeps: list[float] = []
    result = _run_aware(spec_path, tmp_path, fake_run_one, monkeypatch, sleeps)

    assert calls["n"] == 3
    assert sleeps == [900.0, 900.0]  # cools down before retries 2 and 3, not after the 3rd
    assert result["status"] == "needs-yohan"
    assert result["needs"] == "hf-rate-limit"
    assert result["cooldowns"] == 3


def test_hf_429_recovers_after_one_cooldown(tmp_path, monkeypatch):
    spec_path = _spec(tmp_path, "hf-source", "hf")
    calls = {"n": 0}

    def fake_run_one(path, work_root, client):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "source_id": "hf-source",
                "status": "error",
                "error": _HF_429_ERROR,
                "elapsed_s": 0.1,
            }
        return {"source_id": "hf-source", "status": "ok", "elapsed_s": 1.0}

    sleeps: list[float] = []
    result = _run_aware(spec_path, tmp_path, fake_run_one, monkeypatch, sleeps)

    assert calls["n"] == 2
    assert sleeps == [900.0]
    assert result["status"] == "ok"


def test_non_hf_429_like_error_is_not_cooled_down(tmp_path, monkeypatch):
    """The D-AA cool-down/requeue policy is HF-specific; another adapter's error (even
    one that happens to mention 429) is reported as a plain ``error`` immediately."""
    spec_path = _spec(tmp_path, "http-source", "http")
    calls = {"n": 0}

    def fake_run_one(path, work_root, client):
        calls["n"] += 1
        return {
            "source_id": "http-source",
            "status": "error",
            "error": _HF_429_ERROR.replace("huggingface.co", "example.com"),
            "elapsed_s": 0.1,
        }

    sleeps: list[float] = []
    result = _run_aware(spec_path, tmp_path, fake_run_one, monkeypatch, sleeps)

    assert calls["n"] == 1
    assert sleeps == []
    assert result["status"] == "error"


def test_hf_semaphore_caps_concurrency_at_one(tmp_path, monkeypatch):
    """Two HF sources submitted together never run their ``run_one`` body concurrently,
    independent of ``--jobs``."""
    import concurrent.futures as cf
    import time

    spec_a = _spec(tmp_path, "hf-a", "hf")
    spec_b = _spec(tmp_path, "hf-b", "hf")
    in_flight = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_run_one(path, work_root, client):
        with lock:
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
        time.sleep(0.05)
        with lock:
            in_flight["n"] -= 1
        return {"source_id": path.stem, "status": "ok", "elapsed_s": 0.05}

    monkeypatch.setattr(cib, "run_one", fake_run_one)
    sem = threading.Semaphore(1)
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(
                cib._run_one_hf_aware,
                p,
                tmp_path / "work",
                None,
                hf_semaphore=sem,
                cooldown_s=900.0,
                max_cooldowns=3,
                sleep=lambda _s: None,
            )
            for p in (spec_a, spec_b)
        ]
        results = [f.result() for f in futs]

    assert in_flight["max"] == 1
    assert {r["status"] for r in results} == {"ok"}
