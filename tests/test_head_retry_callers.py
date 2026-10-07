"""WP-R15: every ``head_object`` caller retries a transient 405 (RGW) the way stream verify
does, and ``ingest-batch --verify-only`` verifies a staged version and writes the stub
without uploading."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import botocore.exceptions as bce
import pytest
import yaml

from marinedata.concurrency import RetriesExhausted

BUCKET = "bkt"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("marinedata.concurrency.time.sleep", lambda _s: None)


def _err(status: int, code: str = "405", op: str = "HeadObject"):
    return bce.ClientError(
        {"Error": {"Code": code, "Message": "x"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


class FlakyHead:
    """head_object fails ``fail`` times with a 405, then answers; 404 for ``absent``."""

    def __init__(self, fail: int = 0, absent: bool = False, size: int = 4) -> None:
        self.fail, self.absent, self.size, self.calls = fail, absent, size, 0

    def head_object(self, **kw):
        self.calls += 1
        if self.absent:
            raise _err(404, "404")
        if self.calls <= self.fail:
            raise _err(405)
        return {"ETag": '"e"', "ContentLength": self.size, "Metadata": {"sha256": "s"}}


def test_s3_upload_head_retries_405_and_keeps_404_as_absent():
    from marinedata.s3_upload import _head

    ok = FlakyHead(fail=2)
    assert _head(ok, BUCKET, "k")["ContentLength"] == 4 and ok.calls == 3
    absent = FlakyHead(absent=True)
    assert _head(absent, BUCKET, "k") is None and absent.calls == 1
    dead = FlakyHead(fail=99)
    with pytest.raises(RetriesExhausted) as ei:
        _head(dead, BUCKET, "pfx/k")
    assert dead.calls == 6 and ei.value.s3_key == "pfx/k"


def test_disk_mode_uploader_verify_retries_405_and_names_the_key():
    from marinedata.ingest_source import _Uploader

    def uploader(client):
        u = _Uploader.__new__(_Uploader)
        u.client, u.spec, u.key_prefix = client, SimpleNamespace(bucket=BUCKET), "pfx"
        u.ledger = {"a.jpg": {"size": 4, "etag": "e", "sha256": "s"}}
        return u

    ok = FlakyHead(fail=2)
    assert uploader(ok).verify(["a.jpg"]) == 1 and ok.calls == 3
    dead = FlakyHead(fail=99)
    with pytest.raises(RetriesExhausted) as ei:
        uploader(dead).verify(["a.jpg"])
    assert dead.calls == 6 and ei.value.s3_key == "pfx/a.jpg"


def test_coralseg_verify_heads_retries_405_before_calling_a_key_bad():
    from marinedata.task_layers.sources.coralseg_stream import verify_heads

    ok = FlakyHead(fail=2, size=4)
    assert verify_heads(ok, BUCKET, "pfx", {"a.jpg": 4}, workers=1) == [] and ok.calls == 3
    assert verify_heads(FlakyHead(size=9), BUCKET, "pfx", {"a.jpg": 4}, workers=1) == ["a.jpg"]
    assert verify_heads(FlakyHead(absent=True), BUCKET, "pfx", {"a.jpg": 4}, workers=1) == ["a.jpg"]


def test_metadata_release_fallback_retries_405_and_only_404_means_not_staged(monkeypatch, tmp_path):
    from marinedata import metadata_release as mr

    class Client(FlakyHead):
        def download_file(self, bucket, key, dest):
            self.got = key

    source = SimpleNamespace(id="s1", version="v1")
    dest = tmp_path / "out" / "metadata.parquet"

    def run(client):
        monkeypatch.setattr("marinedata.s3_upload.client_from_rclone", lambda *a, **k: client)
        return mr._fetch_staged_metadata_from_s3(source, dest)

    ok = Client(fail=2)
    assert run(ok) is True and ok.got == "sources/s1/v1/metadata.parquet" and ok.calls == 3
    assert run(Client(absent=True)) is False
    with pytest.raises(RetriesExhausted):  # a persistent 405 is not "not staged yet"
        run(Client(fail=99))


class FakeBucket:
    """In-memory S3: reads only. No put_object, so any upload attempt raises."""

    def __init__(self, objs: dict[str, bytes], flaky_key: str | None = None) -> None:
        self.objs, self.flaky_key, self.heads = objs, flaky_key, 0

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objs[Key])}

    def head_object(self, Bucket, Key):
        self.heads += 1
        if Key == self.flaky_key:
            self.flaky_key = None
            raise _err(405)
        data = self.objs[Key]
        return {
            "ETag": '"x"',
            "ContentLength": len(data),
            "Metadata": {"sha256": hashlib.sha256(data).hexdigest()},
        }

    def get_paginator(self, name):
        objs = self.objs

        class P:
            def paginate(self, Bucket, Prefix):
                yield {
                    "Contents": [
                        {"Key": k, "Size": len(v), "ETag": '"x"'}
                        for k, v in sorted(objs.items())
                        if k.startswith(Prefix)
                    ]
                }

        return P()


def _staged(version: str = "v1") -> dict[str, bytes]:
    pfx = f"sources/src1/{version}"
    body = {"images/a.jpg": b"aaaa", "images/b.jpg": b"bbbbbb"}
    body["INGEST.json"] = json.dumps({"images": 2, "layout": "objects"}).encode()
    manifest = "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in sorted(body.items()))
    return {f"{pfx}/{k}": v for k, v in {**body, "CHECKSUMS.sha256": manifest.encode()}.items()}


def _spec_file(tmp_path):
    p = tmp_path / "src1.yaml"
    p.write_text(
        f"id: src1\nadapter: http\nlicense: CC-BY-4.0\nattribution: Lab\nbucket: {BUCKET}\n"
        "params:\n  version: v1\n  urls:\n    - url: http://127.0.0.1:1/x\n"
        f"      sha256: {'0' * 64}"
    )
    return p


def test_verify_only_writes_the_stub_through_a_405_and_uploads_nothing(tmp_path, ample_disk):
    from marinedata.cli_ingest_batch import run_one

    objs = _staged()
    client = FakeBucket(objs, flaky_key="sources/src1/v1/images/b.jpg")
    res = run_one(_spec_file(tmp_path), tmp_path / "work", client, verify_only=True)
    assert res["status"] == "ok" and res["verified"] == 4 == res["files"]
    assert res["images"] == 2 and res["uploaded"] == 0 and client.heads >= 5  # 4 files + retry
    stub = yaml.safe_load((tmp_path / "work/src1/registry-stub-src1.yaml").read_text())
    manifest = objs["sources/src1/v1/CHECKSUMS.sha256"]
    assert stub["staged"]["root_digest"] == hashlib.sha256(manifest).hexdigest()
    assert stub["staged"]["files"] == 4 and stub["items"] == 2
    assert stub["staged"]["bytes"] == sum(len(v) for v in objs.values())
    assert stub["staged"]["prefix"] == "sources/src1/v1"


def test_verify_only_fails_on_a_missing_or_unexpected_object(tmp_path, ample_disk):
    from marinedata.cli_ingest_batch import run_one

    objs = _staged()
    objs["sources/src1/v1/images/stray.jpg"] = b"zz"
    res = run_one(_spec_file(tmp_path), tmp_path / "work", FakeBucket(objs), verify_only=True)
    assert res["status"] == "error" and "stray.jpg" in res["error"]
    assert not (tmp_path / "work/src1/registry-stub-src1.yaml").exists()
