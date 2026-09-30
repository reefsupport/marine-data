"""End-to-end ``ingest-source``: local HTTP upstream -> moto S3, bounded temp, resume."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

import pyarrow.parquet as pq  # noqa: E402
from _wp6_fixtures import LocalServer, png, sha256, tar_bytes  # noqa: E402

from marinedata import sample_schema  # noqa: E402
from marinedata.cli import build_parser  # noqa: E402
from marinedata.ingest_source import IngestSpec, run_ingest  # noqa: E402
from marinedata.s3_upload import DiskGuard  # noqa: E402

N = 80


@pytest.fixture
def env(tmp_path):
    srv = LocalServer()
    members = {}
    for i in range(N):
        members[f"img{i:03d}.png"] = png(i, 40)  # ~4.9 KB each, incompressible
        members[f"img{i:03d}.json"] = json.dumps(
            {"depth_m": 8.0 + i % 5, "depth_source": "metadata"}
        ).encode()
    blob = tar_bytes(members)
    srv.add("/rec/data.tar", blob)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-reef\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Reef Lab\n"
        f"params:\n  version: '2026.1'\n  urls:\n    - url: {srv.base}/rec/data.tar\n"
        f"      sha256: {sha256(blob)}\n"
        "defaults: {platform: diver, habitat: coral-reef}\nbucket: bkt\n"
    )
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        yield srv, s3, spec_path, tmp_path
    srv.close()


def _keys(s3) -> dict[str, int]:
    resp = s3.list_objects_v2(Bucket="bkt", Prefix="sources/")
    return {o["Key"]: o["Size"] for o in resp.get("Contents", [])}


def test_end_to_end_bounded_temp_then_resume_skips_everything(env):
    _, s3, spec_path, tmp = env
    spec = IngestSpec.load(spec_path, "http")
    cap = 96 * 1024
    guard = DiskGuard(tmp / "work", temp_cap_bytes=cap, floor_bytes=0)
    rep = run_ingest(spec, tmp / "work", client=s3, guard=guard, fetch_date=dt.date(2026, 9, 25))
    assert rep.images == N and rep.verified == rep.files
    assert rep.bytes > 3 * cap, "the source is several times the temp cap"
    assert guard.peak_bytes <= cap, f"peak temp {guard.peak_bytes} > cap {cap}"
    keys = _keys(s3)
    prefix = "sources/syn-reef/2026.1/"
    assert f"{prefix}CHECKSUMS.sha256" in keys and f"{prefix}metadata.parquet" in keys
    assert sum(k.startswith(f"{prefix}images/") for k in keys) == N
    manifest = s3.get_object(Bucket="bkt", Key=f"{prefix}CHECKSUMS.sha256")["Body"].read()
    assert hashlib.sha256(manifest).hexdigest() == rep.root_digest
    for line in manifest.decode().splitlines():
        sha, rel = line.split("  ", 1)
        body = s3.get_object(Bucket="bkt", Key=prefix + rel)["Body"].read()
        assert hashlib.sha256(body).hexdigest() == sha
    local_meta = tmp / "work/stage/syn-reef/2026.1/metadata.parquet"
    table = pq.read_table(local_meta)
    sample_schema.validate_table(table)
    row = table.to_pylist()[0]
    assert row["license"] == "CC-BY-4.0" and row["habitat"] == "coral-reef"
    assert row["depth_zone"] == "shallow" and row["upstream_digest"]
    assert not list((tmp / "work/stage/syn-reef/2026.1/images").glob("*.png")), (
        "own temp deleted once verified"
    )
    stub = (tmp / "work/registry-stub-syn-reef.yaml").read_text()
    assert rep.root_digest in stub

    again = run_ingest(
        spec,
        tmp / "work",
        client=s3,
        guard=DiskGuard(tmp / "work", temp_cap_bytes=cap, floor_bytes=0),
        fetch_date=dt.date(2026, 9, 25),
    )
    assert again.uploaded == 0 and again.skipped == again.files
    assert again.root_digest == rep.root_digest


def test_shard_layout_round_trips_through_s3(env):
    _, s3, spec_path, tmp = env
    spec = IngestSpec.load(spec_path)
    spec.shard_threshold, spec.shard_bytes, spec.expected_images = 50, 100_000, N
    rep = run_ingest(
        spec,
        tmp / "w2",
        client=s3,
        guard=DiskGuard(tmp / "w2", temp_cap_bytes=10**7, floor_bytes=0),
    )
    assert rep.layout == "shards"
    prefix = "sources/syn-reef/2026.1/"
    index = pq.read_table(
        __import__("io").BytesIO(
            s3.get_object(Bucket="bkt", Key=f"{prefix}images/index.parquet")["Body"].read()
        )
    ).to_pylist()
    assert len(index) == N
    for rec in index[:5]:
        rng = f"bytes={rec['offset']}-{rec['offset'] + rec['size'] - 1}"
        data = s3.get_object(Bucket="bkt", Key=prefix + rec["shard"], Range=rng)["Body"].read()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]


def test_dry_run_touches_nothing(env):
    _, s3, spec_path, tmp = env
    rep = run_ingest(IngestSpec.load(spec_path), tmp / "w3", client=s3, dry_run=True)
    assert rep.plan["items"] == 1 and rep.plan["target"].endswith("syn-reef/2026.1/")
    assert _keys(s3) == {} and not (tmp / "w3").exists()


def test_spec_requires_licence_and_rejects_unknown_keys(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("id: a\nadapter: http\nparams: {}\nlicense: ''\nattribution: x\n")
    with pytest.raises(ValueError, match="license"):
        IngestSpec.load(p)
    p.write_text("id: a\nadapter: http\nparams: {}\nlicense: MIT\nattribution: x\nfoo: 1\n")
    with pytest.raises(ValueError, match="foo"):
        IngestSpec.load(p)


def test_cli_parses_ingest_source():
    args = build_parser().parse_args(["ingest-source", "hf", "spec.yaml", "--dry-run"])
    assert args.adapter == "hf" and args.dry_run and callable(args.func)


# -- WP-6b: bounded concurrency ---------------------------------------------------------

M = 40  # independent loose items (not one container) so `--jobs` has something to overlap


@pytest.fixture
def many_urls_env(tmp_path):
    """``M`` independent small images as SEPARATE URLs — the shape `--jobs` speeds up."""
    srv = LocalServer()
    urls = []
    for i in range(M):
        blob = png(i, 24)
        srv.add(f"/img{i:03d}.png", blob)
        urls.append(f"    - url: {srv.base}/img{i:03d}.png\n      sha256: {sha256(blob)}\n")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-loose\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Reef Lab\n"
        "params:\n  version: '1'\n  urls:\n" + "".join(urls) + "\nbucket: bkt\n"
    )
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        yield srv, s3, spec_path, tmp_path
    srv.close()


def _no_floor_guard(root: Path) -> DiskGuard:
    # Hermetic: real free disk on the runner is irrelevant to whether N=1 and N=8
    # agree on layout/digest, and floor_bytes=0 keeps that true regardless of what
    # else is running on the machine.
    return DiskGuard(root, temp_cap_bytes=10**7, floor_bytes=0)


def test_jobs_n_matches_jobs_1_root_digest(many_urls_env):
    _, s3, spec_path, tmp = many_urls_env
    spec = IngestSpec.load(spec_path, "http")
    seq = run_ingest(
        spec,
        tmp / "w1",
        client=s3,
        jobs=1,
        guard=_no_floor_guard(tmp / "w1"),
        fetch_date=dt.date(2026, 9, 25),
    )
    with moto.mock_aws():
        s3b = boto3.client("s3", region_name="us-east-1")
        s3b.create_bucket(Bucket="bkt")
        par = run_ingest(
            IngestSpec.load(spec_path, "http"),
            tmp / "w8",
            client=s3b,
            jobs=8,
            max_per_host=4,
            guard=_no_floor_guard(tmp / "w8"),
            fetch_date=dt.date(2026, 9, 25),
        )
    assert seq.images == par.images == M
    assert seq.root_digest == par.root_digest
    assert seq.files == par.files


def test_temp_cap_never_exceeded_with_jobs(many_urls_env):
    _, s3, spec_path, tmp = many_urls_env
    spec = IngestSpec.load(spec_path, "http")
    cap = 64 * 1024  # well below the ~100 KB declared total, above per-flush granularity
    guard = DiskGuard(tmp / "work", temp_cap_bytes=cap, floor_bytes=0)
    rep = run_ingest(spec, tmp / "work", client=s3, guard=guard, jobs=8, max_per_host=4)
    assert rep.images == M
    assert guard.peak_bytes <= cap, f"peak temp {guard.peak_bytes} > cap {cap}"


class _KillAfterPuts:
    """Proxy an S3 client; die (not an ``Exception``) right after the n-th ``put_object``
    reaches S3 — models a hard kill of a `--jobs N` run partway through."""

    def __init__(self, client, n: int) -> None:
        self.client, self.n, self.count = client, n, 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def put_object(self, **kw):
        resp = self.client.put_object(**kw)
        with self._lock:
            self.count += 1
            local = self.count
        if local == self.n:
            raise SimulatedKillMidRun
        return resp


class SimulatedKillMidRun(BaseException):
    pass


def test_kill_mid_run_then_resume_uploads_only_missing(many_urls_env):
    _, s3, spec_path, tmp = many_urls_env
    spec = IngestSpec.load(spec_path, "http")
    # A tight cap forces several small `flush()` batches instead of one at the end, so the
    # kill (after the 5th successful put) lands during an EARLY batch — most of the M
    # images are never even fetched, a real "mid-run" kill rather than "everything raced
    # to S3 before the exception was noticed".
    guard = DiskGuard(tmp / "work", temp_cap_bytes=4096, floor_bytes=0)
    killer = _KillAfterPuts(s3, n=5)
    with pytest.raises(SimulatedKillMidRun):
        run_ingest(spec, tmp / "work", client=killer, guard=guard, jobs=8, max_per_host=4)
    before = len(_keys(s3))
    assert 0 < before < M, "some, not all, objects reached S3 before the kill"

    resumed = run_ingest(
        IngestSpec.load(spec_path, "http"),
        tmp / "work",
        client=s3,
        guard=DiskGuard(tmp / "work", temp_cap_bytes=4096, floor_bytes=0),
        jobs=8,
        max_per_host=4,
    )
    assert resumed.images == M and resumed.verified == resumed.files
    assert resumed.skipped == before, "already-uploaded objects were not re-sent"
    assert resumed.uploaded == resumed.files - before

    clean_rep = run_ingest(
        IngestSpec.load(spec_path, "http"),
        tmp / "clean",
        client=s3,  # same bucket: everything already present -> a pure skip/verify run
        jobs=1,
        guard=_no_floor_guard(tmp / "clean"),
    )
    assert clean_rep.root_digest == resumed.root_digest


def test_jobs_1_is_the_pre_wp6b_default_behaviour(many_urls_env):
    """``--jobs 1`` takes the sequential, no-pool code path (no prefetcher/put_pool)."""
    _, s3, spec_path, tmp = many_urls_env
    rep = run_ingest(
        IngestSpec.load(spec_path, "http"),
        tmp / "w",
        client=s3,
        jobs=1,
        guard=_no_floor_guard(tmp / "w"),
    )
    assert rep.images == M and rep.verified == rep.files
