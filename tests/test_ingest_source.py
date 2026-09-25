"""End-to-end ``ingest-source``: local HTTP upstream -> moto S3, bounded temp, resume."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

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
