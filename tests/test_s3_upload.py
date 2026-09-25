"""Resumable uploader + disk guard (moto-backed, network-free)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from marinedata.s3_upload import (  # noqa: E402
    DiskFloorError,
    DiskGuard,
    MiB,
    TempCapError,
    local_digest,
    upload_file,
)

PART = 5 * MiB  # moto (like S3) rejects non-final parts below 5 MiB


class SimulatedKill(BaseException):
    """Models SIGKILL: not an Exception, so nothing in the uploader can swallow it."""


class KillAfter:
    """Proxy a client; die right AFTER the n-th part reaches S3 (before the checkpoint)."""

    def __init__(self, client, n: int) -> None:
        self.client, self.n, self.count = client, n, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def upload_part(self, **kw):
        resp = self.client.upload_part(**kw)
        self.count += 1
        if self.count == self.n:
            raise SimulatedKill
        return resp


@pytest.fixture
def s3():
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="bkt")
        yield client


def _blob(path: Path, size: int, seed: int = 1) -> Path:
    path.write_bytes(random.Random(seed).randbytes(size))
    return path


def test_single_part_then_skip(s3, tmp_path):
    f = _blob(tmp_path / "small.bin", 1000)
    first = upload_file(s3, "bkt", "k/small.bin", f, checkpoint_dir=tmp_path / "ck")
    again = upload_file(s3, "bkt", "k/small.bin", f, checkpoint_dir=tmp_path / "ck")
    assert not first.skipped and again.skipped and again.parts_sent == 0
    head = s3.head_object(Bucket="bkt", Key="k/small.bin")
    assert head["ETag"].strip('"') == hashlib.md5(f.read_bytes()).hexdigest()
    assert head["Metadata"]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_kill_mid_upload_and_resume_gives_identical_etag_and_sha(s3, tmp_path):
    f = _blob(tmp_path / "big.bin", 3 * PART + 12345)  # 4 parts
    ck = tmp_path / "ck"
    with pytest.raises(SimulatedKill):
        upload_file(KillAfter(s3, 2), "bkt", "k/big.bin", f, checkpoint_dir=ck, part_size=PART)
    saved = json.loads(next(ck.glob("*.json")).read_text())
    assert list(saved["parts"]) == ["1"], "part 2 reached S3 but was never checkpointed"

    resumed = upload_file(s3, "bkt", "k/big.bin", f, checkpoint_dir=ck, part_size=PART)
    assert resumed.parts_sent == 2, "list_parts must reuse parts 1 AND 2"
    clean = upload_file(s3, "bkt", "k/clean.bin", f, checkpoint_dir=ck, part_size=PART)
    want = local_digest(f, PART)
    for key in ("k/big.bin", "k/clean.bin"):
        head = s3.head_object(Bucket="bkt", Key=key)
        assert head["ETag"].strip('"') == want.etag == clean.etag
        assert head["Metadata"]["sha256"] == want.sha256
    body = s3.get_object(Bucket="bkt", Key="k/big.bin")["Body"].read()
    assert hashlib.sha256(body).hexdigest() == want.sha256
    assert not list(ck.glob("*.json")), "checkpoint removed after verified completion"


def test_changed_file_never_splices_old_parts(s3, tmp_path):
    f = _blob(tmp_path / "big.bin", 2 * PART + 7)
    ck = tmp_path / "ck"
    with pytest.raises(SimulatedKill):
        upload_file(KillAfter(s3, 1), "bkt", "k/x.bin", f, checkpoint_dir=ck, part_size=PART)
    _blob(f, 2 * PART + 7, seed=2)  # same size, different bytes
    res = upload_file(s3, "bkt", "k/x.bin", f, checkpoint_dir=ck, part_size=PART)
    assert res.parts_sent == 3
    body = s3.get_object(Bucket="bkt", Key="k/x.bin")["Body"].read()
    assert body == f.read_bytes()


def test_disk_guard_refuses_below_floor(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(100, 90, 10))
    with pytest.raises(DiskFloorError):
        DiskGuard(tmp_path, temp_cap_bytes=10**9, floor_bytes=11).check()


def test_disk_guard_temp_cap_and_peak(tmp_path):
    g = DiskGuard(tmp_path, temp_cap_bytes=1000, floor_bytes=0)
    (tmp_path / "a").write_bytes(b"x" * 600)
    g.reserve(300)
    with pytest.raises(TempCapError):
        g.reserve(500)
    assert g.peak_bytes == 600
