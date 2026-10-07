"""WP-6h stream (diskless) staging: same staged tree as disk mode, resumable from S3."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

import test_ingest_source as tis  # noqa: E402
from _wp6_fixtures import LocalServer, png, sha256, tar_bytes  # noqa: E402

from marinedata import ingest_stream  # noqa: E402
from marinedata.cli import build_parser  # noqa: E402
from marinedata.ingest_source import IngestSpec, run_ingest  # noqa: E402
from marinedata.s3_upload import DiskFloorError, DiskGuard, MiB  # noqa: E402

M = 60
FETCH_DATE = __import__("datetime").date(2026, 9, 25)


def _guard(root: Path) -> DiskGuard:
    return DiskGuard(root, temp_cap_bytes=10**8, floor_bytes=0)


def _tree(s3, bucket: str, prefix: str) -> dict[str, bytes]:
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
    return out


@pytest.fixture
def loose(tmp_path):
    """M loose images with declared sizes, one dead URL (404 -> MISSING.tsv)."""
    srv = LocalServer()
    urls = []
    for i in range(M):
        blob = png(i, 24)
        srv.add(f"/img{i:03d}.png", blob)
        urls.append(
            f"    - url: {srv.base}/img{i:03d}.png\n      sha256: {sha256(blob)}\n"
            f"      size: {len(blob)}\n"
        )
    urls.insert(7, f"    - url: {srv.base}/gone.png\n")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-loose\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Reef Lab\n"
        "params:\n  version: '1'\n  urls:\n" + "".join(urls) + "\nbucket: disk\n"
    )
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        for b in ("disk", "strm"):
            s3.create_bucket(Bucket=b)
        yield s3, spec_path, tmp_path
    srv.close()


def _run(spec_path: Path, work: Path, s3, *, stream: bool, bucket: str, **kw):
    spec = IngestSpec.load(spec_path)
    spec.bucket = bucket
    for k in ("shard_threshold", "shard_bytes", "expected_images"):
        if k in kw:
            setattr(spec, k, kw.pop(k))
    return run_ingest(
        spec, work, client=s3, guard=_guard(work), fetch_date=FETCH_DATE, stream=stream, **kw
    )


def test_stream_tree_is_byte_identical_to_disk_objects_with_missing(loose):
    s3, spec_path, tmp = loose
    work = tmp / "w"  # shared, as a runner's per-source work dir is: MISSING first_seen
    strm = _run(spec_path, work, s3, stream=True, bucket="strm", jobs=4, checkpoint_items=7)
    assert not (work / "stage").exists(), "stream mode staged nothing on local disk"
    disk = _run(spec_path, work, s3, stream=False, bucket="disk", jobs=4)
    prefix = "sources/syn-loose/1/"
    a, b = _tree(s3, "disk", prefix), _tree(s3, "strm", prefix)
    assert a == b and len(a) == strm.files
    assert strm.root_digest == disk.root_digest and strm.missing == disk.missing == 1
    assert b"gone.png" in b[prefix + "MISSING.tsv"]
    assert strm.plan["mode"] == "stream" and strm.verified == strm.files
    parts = _tree(s3, "strm", "sources/syn-loose/_stream/1/")
    assert parts, "checkpoint parts live outside the version prefix"


def test_stream_shards_multipart_byte_identical(tmp_path):
    srv = LocalServer()
    members = {f"big{i:02d}.png": png(i, 300) for i in range(40)}  # ~270 KB each
    blob = tar_bytes(members)
    srv.add("/rec/data.tar", blob)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-big\nadapter: http\nlicense: CC-BY-4.0\nattribution: Lab\n"
        f"params:\n  version: '2'\n  urls:\n    - url: {srv.base}/rec/data.tar\n"
        f"      sha256: {sha256(blob)}\nbucket: disk\n"
    )
    shard = dict(shard_threshold=10, shard_bytes=6 * MiB, expected_images=40)
    try:
        with moto.mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            for b in ("disk", "strm"):
                s3.create_bucket(Bucket=b)
            kw = dict(part_size=5 * MiB, jobs=2)
            disk = _run(spec_path, tmp_path / "d", s3, stream=False, bucket="disk", **shard, **kw)
            strm = _run(spec_path, tmp_path / "s", s3, stream=True, bucket="strm", **shard, **kw)
            prefix = "sources/syn-big/2/"
            a, b = _tree(s3, "disk", prefix), _tree(s3, "strm", prefix)
            assert strm.layout == "shards" and a == b
            assert strm.root_digest == disk.root_digest
            head = s3.head_object(Bucket="strm", Key=prefix + "images/shard-00000.tar")
            assert "-" in head["ETag"], "a >1-part shard went up as a multipart upload"
            assert strm.plan["peak_memory_bytes"] <= 512 * MiB
    finally:
        srv.close()


def test_stream_resume_after_kill_refetches_only_uncheckpointed(loose, monkeypatch):
    s3, spec_path, tmp = loose
    ref = _run(spec_path, tmp / "ref", s3, stream=False, bucket="disk", jobs=1)
    fetched: list[str] = []
    lock = threading.Lock()
    real = ingest_stream._fetch_one

    def counting(adapter, item, *a, **kw):
        with lock:
            fetched.append(item.key)
        return real(adapter, item, *a, **kw)

    monkeypatch.setattr(ingest_stream, "_fetch_one", counting)
    work = tmp / "ref"  # same work dir: same MISSING first_seen as the reference
    killer = tis._KillAfterPuts(s3, n=35)
    with pytest.raises(tis.SimulatedKillMidRun):
        _run(spec_path, work, killer, stream=True, bucket="strm", jobs=4, checkpoint_items=10)
    parts = [k for k in _tree(s3, "strm", "sources/syn-loose/_stream/") if k.endswith(".json")]
    assert parts, "at least one checkpoint committed before the kill"
    fetched.clear()
    res = _run(spec_path, work, s3, stream=True, bucket="strm", jobs=4, checkpoint_items=10)
    done = res.plan["resumed_items"]
    assert done >= 10 and len(fetched) == M + 1 - done  # no re-fetch of checkpointed items
    assert res.skipped > 0, "objects PUT after the last checkpoint were not re-sent"
    prefix = "sources/syn-loose/1/"
    assert _tree(s3, "strm", prefix) == _tree(s3, "disk", prefix)
    assert res.root_digest == ref.root_digest and res.images == M


def test_memory_cap_respected(loose):
    s3, spec_path, tmp = loose
    item = len(png(0, 24))
    cap = 4 * item  # a quarter of the jobs=8 window: the window must shrink to fit
    rep = _run(spec_path, tmp / "m", s3, stream=True, bucket="strm", jobs=8, memory_cap=cap)
    assert rep.images == M
    biggest = max(len(v) for v in _tree(s3, "strm", "sources/syn-loose/1/").values())
    # Invariant: in-flight bytes <= cap + ONE object (an object bigger than the cap still
    # has to go whole — here metadata.parquet / INGEST.json at the end).
    assert 0 < rep.plan["peak_memory_bytes"] <= cap + biggest
    assert 0 < rep.plan["peak_memory_items_bytes"] <= cap + item  # item phase: one over
    loose_run = _run(spec_path, tmp / "n", s3, stream=True, bucket="strm", jobs=8)
    assert loose_run.plan["peak_memory_bytes"] >= rep.plan["peak_memory_bytes"]


def test_memory_budget_blocks_only_while_a_put_can_free_bytes():
    budget = ingest_stream.MemoryBudget(100)
    budget.acquire(80)
    budget.acquire(80)  # nothing in flight to free bytes: proceeds instead of deadlocking
    assert budget.used == 160
    budget.release(160)
    budget.acquire(90)
    budget.sending(+1)
    done = threading.Event()
    t = threading.Thread(target=lambda: (budget.acquire(50), done.set()))
    t.start()
    assert not done.wait(0.2), "blocked while a PUT holds the budget"
    budget.release(90)
    budget.sending(-1)
    assert done.wait(2.0)
    t.join()
    assert budget.peak <= 160


def test_stream_uses_the_low_floor_not_disk_floor(loose):
    s3, spec_path, tmp = loose
    spec = IngestSpec.load(spec_path)
    spec.bucket, spec.disk_floor_gib, spec.stream_disk_floor_gib = "strm", 10**6, 0.0
    with pytest.raises(DiskFloorError):
        run_ingest(spec, tmp / "x", client=s3, fetch_date=FETCH_DATE)
    rep = run_ingest(spec, tmp / "x", client=s3, fetch_date=FETCH_DATE, stream=True)
    assert rep.images == M


def test_stream_guard_admits_an_archive_only_if_it_fits_above_the_floor(tmp_path):
    g = ingest_stream.StreamGuard(tmp_path, temp_cap_bytes=1 << 62, floor_bytes=0)
    free = g.free_bytes()
    g.reserve(1024)
    with pytest.raises(DiskFloorError):
        g.reserve(free + (1 << 30))


def test_spec_and_cli_accept_stream():
    args = build_parser().parse_args(["ingest-batch", "specs", "--stream"])
    assert args.stream is True
    args = build_parser().parse_args(["ingest-source", "http", "s.yaml", "--stream"])
    assert args.stream is True
    spec = IngestSpec(id="x", adapter="http", params={}, license="CC0", attribution="a")
    assert spec.stream is False and spec.stream_disk_floor_gib == 3.0


def test_stream_shards_resume_after_kill_matches_disk(loose):
    """Shards layout checkpoints at each shard close on an item boundary."""
    s3, spec_path, tmp = loose
    shard = dict(shard_threshold=10, shard_bytes=10_000, expected_images=M)
    work = tmp / "sh"
    ref = _run(spec_path, work, s3, stream=False, bucket="disk", jobs=1, **shard)
    with pytest.raises(tis.SimulatedKillMidRun):
        _run(spec_path, work, tis._KillAfterPuts(s3, n=12), stream=True, bucket="strm", **shard)
    res = _run(spec_path, work, s3, stream=True, bucket="strm", jobs=4, **shard)
    assert res.layout == "shards" and 0 < res.plan["resumed_items"] < M
    prefix = "sources/syn-loose/1/"
    assert _tree(s3, "strm", prefix) == _tree(s3, "disk", prefix)
    assert res.root_digest == ref.root_digest and res.images == M
