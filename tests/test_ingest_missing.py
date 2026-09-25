"""D-AF (2026-09-25 charter): the runner skips a dead item into ``MISSING.tsv`` instead of
aborting the source, and aborts only past its thresholds (> 5% after >= 200 attempts,
50 consecutive, or every item failed). No network, no S3; the runner
tests are in ``test_ingest_missing_runner``."""

from __future__ import annotations

import io
import tarfile
import urllib.error

import pytest

from marinedata.adapters import AccessRefused, DigestMismatch
from marinedata.concurrency import RetriesExhausted, is_retryable_exc, retry_with_backoff
from marinedata.ingest_missing import MissingLedger, TooManyMissing, decode_status, fetch_status
from marinedata.s3_upload import DiskFloorError


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://h/x", code, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]


def _exhausted(cause: BaseException) -> RetriesExhausted:
    try:
        raise RetriesExhausted(f"GET http://h/x failed after 4 attempts: {cause}") from cause
    except RetriesExhausted as exc:
        return exc


# -- classification ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "want"),
    [
        (_http(404), "http-404"),
        (_http(410), "http-410"),
        (_exhausted(_http(404)), "http-404"),  # a 404 retried by an older retry path
        (_exhausted(urllib.error.URLError("refused")), "retries-exhausted"),
        (_exhausted(_http(503)), "retries-exhausted"),
        (_http(400), None),  # other 4xx: a request bug, not a dead item
        (_exhausted(_http(429)), None),  # a throttle: D-AA cools HF down on it
        (AccessRefused("http://h/x", "HTTP 403"), None),  # D-E -> needs-yohan
        (DigestMismatch("x: sha256 a != declared b"), None),  # pin violation
        (DiskFloorError("floor"), None),
        (ValueError("anything else"), None),
        (KeyboardInterrupt(), None),
    ],
)
def test_fetch_status(exc, want):
    assert fetch_status(exc) == want


def test_decode_status_skips_only_before_anything_was_written():
    err = tarfile.ReadError("bad tar")
    assert decode_status(err, wrote=False) == "decode-error:ReadError"
    assert decode_status(err, wrote=True) is None  # half-staged container: abort
    assert decode_status(DigestMismatch("x"), wrote=False) is None
    assert decode_status(KeyboardInterrupt(), wrote=False) is None


def test_http_404_is_not_retried_but_5xx_is():
    assert not is_retryable_exc(_http(404))
    assert not is_retryable_exc(_http(410))
    assert is_retryable_exc(_http(503)) and is_retryable_exc(_http(429))
    assert is_retryable_exc(urllib.error.URLError("reset"))
    calls = []

    def dead():
        calls.append(1)
        raise _http(404)

    with pytest.raises(urllib.error.HTTPError):
        retry_with_backoff(dead, base=0, jitter=0)
    assert len(calls) == 1


def test_retry_with_backoff_raises_retries_exhausted_with_cause():
    def down():
        raise _http(503)

    with pytest.raises(RetriesExhausted) as info:
        retry_with_backoff(down, retries=2, base=0, jitter=0)
    assert isinstance(info.value, RuntimeError) and info.value.__cause__.code == 503
    assert fetch_status(info.value) == "retries-exhausted"


# -- ledger thresholds ---------------------------------------------------------------------


def _ledger(tmp_path, now="2026-09-25T10:00:00Z"):
    return MissingLedger(tmp_path / "j.tsv", now=lambda: now)


def test_consecutive_failures_abort_at_50_and_a_success_resets(tmp_path):
    led = _ledger(tmp_path)
    for i in range(49):
        led.skip(f"k{i}", "u", "http-404")
    led.ok()  # resets the run
    for i in range(49):
        led.skip(f"j{i}", "u", "http-404")
    with pytest.raises(TooManyMissing, match="50 consecutive"):
        led.skip("last", "u", "retries-exhausted")


def test_fraction_aborts_only_after_200_attempts_and_above_5_percent(tmp_path):
    led = _ledger(tmp_path)
    skips = set(range(0, 199, 18)[:11])  # 11 skips in the first 199 (5.5%), < 200 attempts
    for i in range(199):
        led.skip(f"k{i}", "u", "http-404") if i in skips else led.ok()
    assert (len(led.rows), led.attempted) == (11, 199)
    with pytest.raises(TooManyMissing, match="11/200"):
        led.ok()  # the 200th attempt crosses MIN_ATTEMPTS with 11/200 = 5.5% skipped


def test_exactly_5_percent_does_not_abort(tmp_path):
    led = _ledger(tmp_path)
    for i in range(400):
        led.skip(f"k{i}", "u", "http-410") if i % 20 == 19 else led.ok()
    assert (len(led.rows), led.attempted) == (20, 400)
    led.finish()


def test_finish_aborts_when_every_item_failed(tmp_path):
    led = _ledger(tmp_path)
    for i in range(3):
        led.skip(f"k{i}", "u", "http-404")
    with pytest.raises(TooManyMissing, match="all 3"):
        led.finish()
    _ledger(tmp_path / "empty").finish()  # nothing attempted: not an abort


def test_write_and_first_seen_survives_a_resume(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert _ledger(tmp_path).write(root) is None  # clean source: no MISSING.tsv at all
    first = _ledger(tmp_path, now="2026-09-25T10:00:00Z")
    first.ok()
    first.skip("a.jpg", "http://h/a.jpg", "http-404")
    again = _ledger(tmp_path, now="2026-09-26T11:00:00Z")  # same journal: a resumed run
    again.skip("a.jpg", "http://h/a.jpg", "http-404")
    again.skip("b\tc.jpg", "http://h/b.jpg", "decode-error:ReadError")
    lines = again.write(root).read_text().splitlines()
    assert lines[0] == "key\turl\tstatus\tfirst_seen"
    assert lines[1] == "a.jpg\thttp://h/a.jpg\thttp-404\t2026-09-25T10:00:00Z"
    assert lines[2] == "b c.jpg\thttp://h/b.jpg\tdecode-error:ReadError\t2026-09-26T11:00:00Z"
