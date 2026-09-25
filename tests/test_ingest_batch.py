"""``marinedata ingest-batch``: run every spec in a dir, resume via the CHECKSUMS marker,
collect NEEDS-YOHAN, and only fail the batch on a real error (WP-6c)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from _wp6_fixtures import LocalServer, png, sha256, tar_bytes  # noqa: E402

from marinedata.cli import build_parser  # noqa: E402
from marinedata.cli_ingest_batch import _cmd_ingest_batch, run_batch  # noqa: E402

BUCKET = "bkt"


def _spec(tmp_path: Path, source_id: str, url: str, sha: str, version: str = "2026.1") -> Path:
    path = tmp_path / f"{source_id}.yaml"
    path.write_text(
        f"id: {source_id}\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Lab\n"
        f"bucket: {BUCKET}\n"
        f"params:\n  version: '{version}'\n  urls:\n    - url: {url}\n      sha256: {sha}\n"
    )
    return path


def _blob() -> bytes:
    members = {f"img{i:02d}.png": png(i, 24) for i in range(6)}
    return tar_bytes(members)


@pytest.fixture
def env(tmp_path, ample_disk):
    srv = LocalServer()
    blob = _blob()
    srv.add("/a/data.tar", blob)
    srv.add("/b/data.tar", blob)
    srv.routes["/gated/data.tar"] = (403, "text/plain", b"nope", {})
    specs = tmp_path / "specs"
    specs.mkdir()
    _spec(specs, "source-a", f"{srv.base}/a/data.tar", sha256(blob))
    _spec(specs, "source-b", f"{srv.base}/b/data.tar", sha256(blob))
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield srv, s3, specs, tmp_path
    srv.close()


def _run(specs, work, **kw):
    import io

    buf = io.StringIO()
    results = run_batch(specs, work, out=buf, **kw)
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    return results, lines


def test_batch_runs_all_and_resumes_via_checksum_marker(env):
    _, _, specs, tmp = env
    work = tmp / "work"
    results, lines = _run(specs, work)
    assert {r["source_id"] for r in results} == {"source-a", "source-b"}
    assert all(r["status"] == "ok" for r in results)
    assert len(lines) == 2  # one JSONL line per finished source

    # Re-run: both sources are already complete on S3 -> skipped, no new work.
    results2, _ = _run(specs, tmp / "work2")
    assert all(r["status"] == "skip-done" for r in results2)
    assert {r["version"] for r in results2} == {"2026.1"}


def test_only_filter_selects_subset(env):
    _, _, specs, tmp = env
    results, _ = _run(specs, tmp / "work", only={"source-a"})
    assert [r["source_id"] for r in results] == ["source-a"]


def test_only_filter_rejects_unknown_id(env):
    _, _, specs, tmp = env
    with pytest.raises(ValueError, match="nope-id"):
        _run(specs, tmp / "work", only={"nope-id"})


def test_needs_yohan_collected_without_failing_the_batch(env):
    srv, _, specs, tmp = env
    _spec(specs, "source-gated", f"{srv.base}/gated/data.tar", "0" * 64, version="1")
    results, _ = _run(specs, tmp / "work")
    gated = next(r for r in results if r["source_id"] == "source-gated")
    assert gated["status"] == "needs-yohan"
    assert "gated" not in gated  # field name check omitted; just confirm url/needs present
    assert gated["url"].endswith("/gated/data.tar")
    assert "login" in gated["needs"] or "terms" in gated["needs"] or "gate" in gated["needs"]


def test_real_failure_is_reported_and_exits_nonzero(env, capsys):
    srv, _, specs, tmp = env
    _spec(specs, "source-missing", f"{srv.base}/missing/data.tar", "0" * 64, version="1")
    args = build_parser().parse_args(
        ["ingest-batch", str(specs), "--work", str(tmp / "work"), "--only", "source-missing"]
    )
    code = _cmd_ingest_batch(args)
    assert code == 1
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines()]
    assert any(line.get("status") == "error" for line in lines[:-1])
    summary = lines[-1]
    assert summary["errors"] and summary["errors"][0]["source_id"] == "source-missing"


def test_cli_parses_ingest_batch():
    args = build_parser().parse_args(
        ["ingest-batch", "specs/", "--only", "a,b", "--jobs", "2", "--work", "w"]
    )
    assert args.specs_dir == "specs/"
    assert args.only == "a,b"
    assert args.jobs == 2
    assert args.func is _cmd_ingest_batch
