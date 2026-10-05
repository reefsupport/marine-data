"""RETRY400: a transient S3 PutObject 400 and botocore transport timeouts are retried with
the shared backoff; 403/404/other 4xx are not; a failing put names its key, status and
RequestId. No network: the S3 client is a stub."""

from __future__ import annotations

import botocore.exceptions as bce
import pytest

from marinedata import ingest_stream
from marinedata.cli_ingest_batch import _is_hf_rate_limited
from marinedata.concurrency import (
    RetriesExhausted,
    describe_error,
    is_retryable_exc,
    retry_with_backoff,
    tag_s3_key,
)


def _client_error(status: int, code: str, op: str = "PutObject", rid: str = "REQ123"):
    return bce.ClientError(
        {
            "Error": {"Code": code, "Message": "N/A"},
            "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": rid},
        },
        op,
    )


def test_put_400_badrequest_is_retryable():
    assert is_retryable_exc(_client_error(400, "BadRequest"))
    assert is_retryable_exc(_client_error(400, "400"))  # body-less reply: botocore code "400"


@pytest.mark.parametrize(
    ("status", "code"),
    [(403, "AccessDenied"), (404, "NoSuchKey"), (409, "BucketAlreadyOwnedByYou")],
)
def test_other_4xx_are_not_retryable(status, code):
    assert not is_retryable_exc(_client_error(status, code))


def test_other_400_codes_stay_fatal():
    assert not is_retryable_exc(_client_error(400, "InvalidArgument"))
    assert not is_retryable_exc(_client_error(400, "EntityTooLarge"))


def test_botocore_timeouts_are_retryable():
    url = "https://s3.example"
    assert is_retryable_exc(bce.ReadTimeoutError(endpoint_url=url))
    assert is_retryable_exc(bce.ConnectTimeoutError(endpoint_url=url))
    assert is_retryable_exc(bce.EndpointConnectionError(endpoint_url=url))
    assert is_retryable_exc(bce.ConnectionClosedError(endpoint_url=url))
    assert is_retryable_exc(TimeoutError("The read operation timed out"))


def test_400_is_retried_then_succeeds_and_403_is_not_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _client_error(400, "BadRequest")
        return "ok"

    assert retry_with_backoff(flaky, base=0, jitter=0) == "ok" and calls["n"] == 3

    denied = {"n": 0}

    def forbidden():
        denied["n"] += 1
        raise _client_error(403, "AccessDenied")

    with pytest.raises(bce.ClientError):
        retry_with_backoff(forbidden, base=0, jitter=0)
    assert denied["n"] == 1


def test_persistent_400_is_bounded_by_the_existing_attempts():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise _client_error(400, "BadRequest")

    with pytest.raises(RetriesExhausted):
        retry_with_backoff(always, retries=5, base=0, jitter=0)
    assert calls["n"] == 5


def test_describe_error_names_key_status_and_request_id():
    with pytest.raises(bce.ClientError) as ei:
        tag_s3_key(
            "open/deepseagrass/v1/img/a.jpg",
            lambda: (_ for _ in ()).throw(_client_error(400, "BadRequest")),
        )
    text = describe_error(ei.value)
    assert text.startswith("ClientError: An error occurred (BadRequest)")
    for part in (
        "op=PutObject",
        "key=open/deepseagrass/v1/img/a.jpg",
        "status=400",
        "code=BadRequest",
        "request_id=REQ123",
    ):
        assert part in text


def test_describe_error_follows_retries_exhausted_to_its_cause():
    def always():
        raise _client_error(400, "BadRequest", rid="RID9")

    with pytest.raises(RetriesExhausted) as ei:
        tag_s3_key("k/x", lambda: retry_with_backoff(always, retries=2, base=0, jitter=0))
    text = describe_error(ei.value)
    assert text.startswith("RetriesExhausted:")
    assert "key=k/x" in text and "status=400" in text and "request_id=RID9" in text


def test_describe_error_plain_exception_is_unchanged():
    assert describe_error(ValueError("boom")) == "ValueError: boom"


def test_s3_context_cannot_trigger_the_hf_429_cooldown():
    exc = _client_error(400, "BadRequest", rid="X4291")
    exc.s3_key = "img_429.jpg"
    assert not _is_hf_rate_limited({"status": "error", "error": describe_error(exc)})
    assert _is_hf_rate_limited({"status": "error", "error": "HTTPError: 429 Too Many Requests"})


def test_putter_send_tags_the_failing_key_and_retries_a_400(monkeypatch):
    monkeypatch.setattr("marinedata.concurrency.time.sleep", lambda _s: None)

    class Client:
        def __init__(self, fail: int) -> None:
            self.fail, self.calls = fail, 0

        def put_object(self, **kw):
            self.calls += 1
            if self.calls <= self.fail:
                raise _client_error(400, "BadRequest")
            return {"ETag": '"x"'}

    class Budget:
        def release(self, n):
            pass

        def sending(self, d):
            pass

    def putter(client):
        p = ingest_stream._Putter.__new__(ingest_stream._Putter)
        p.client, p.bucket, p.key_prefix, p.budget = client, "bkt", "pfx", Budget()
        import threading

        p.uploaded, p._lock = 0, threading.Lock()
        return p

    ok = Client(fail=2)
    p = putter(ok)
    p._send("a/b.jpg", b"data", "sha")
    assert ok.calls == 3 and p.uploaded == 1

    dead = Client(fail=99)
    with pytest.raises(RetriesExhausted) as ei:
        putter(dead)._send("a/b.jpg", b"data", "sha")
    assert dead.calls == 5 and ei.value.s3_key == "pfx/a/b.jpg"
