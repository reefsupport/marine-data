"""``client_from_env``: the cluster Job creds path (WP-6c). Never printed or logged."""

from __future__ import annotations

import io
import json

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from _wp6_fixtures import LocalServer, png, sha256, tar_bytes  # noqa: E402

from marinedata.cli_ingest_batch import run_batch  # noqa: E402
from marinedata.s3_upload import client_from_env  # noqa: E402

# Well-known AWS docs example credentials (not real secrets) sized/shaped like the real
# thing so moto's format validation accepts them.
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SECRET_ACCESS_ID = "AKIAIOSFODNN7EXAMPLE"


def test_client_from_env_requires_both_vars(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(KeyError):
        client_from_env()


def test_client_from_env_prefixes_bare_host_with_https(monkeypatch):
    # Inspects the constructed client's metadata only — never opens a socket, so this
    # can't leak a real request at a real S3-compatible host (see the module docstring:
    # moto does not recognise a custom endpoint_url, so anything hitting the network
    # here would go out for real).
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", SECRET_ACCESS_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_ACCESS_KEY)
    monkeypatch.setenv("S3_ENDPOINT", "hel1.your-objectstorage.com")
    client = client_from_env()
    assert client.meta.endpoint_url == "https://hel1.your-objectstorage.com"


def test_client_from_env_builds_a_working_client(monkeypatch):
    # No S3_ENDPOINT here: a custom endpoint_url is NOT recognised by moto's request
    # matching, so setting one under mock_aws would escape the mock and hit the real
    # network (confirmed while writing this test — see git history/report). Leaving
    # S3_ENDPOINT unset exercises the default AWS endpoint, which moto does intercept.
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", SECRET_ACCESS_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_ACCESS_KEY)
    with moto.mock_aws():
        client = client_from_env()
        client.create_bucket(Bucket="bkt")  # proves the credentials actually work
        assert client.list_buckets()["Buckets"][0]["Name"] == "bkt"


def test_credentials_never_appear_in_batch_output(monkeypatch, tmp_path, capsys, ample_disk):
    """Run a real batch with env-var creds; grep every byte the job would emit to stdout
    for the literal secret. This is the WP-6c guarantee, not just an absence-of-a-log-call."""
    monkeypatch.delenv("S3_ENDPOINT", raising=False)  # see test above: real network otherwise
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", SECRET_ACCESS_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_ACCESS_KEY)

    srv = LocalServer()
    blob = tar_bytes({f"img{i:02d}.png": png(i, 24) for i in range(4)})
    srv.add("/a/data.tar", blob)
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "source-a.yaml").write_text(
        "id: source-a\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Lab\n"
        "bucket: bkt\nparams:\n  version: '1'\n"
        f"  urls:\n    - url: {srv.base}/a/data.tar\n      sha256: {sha256(blob)}\n"
    )
    try:
        with moto.mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="bkt")
            buf = io.StringIO()
            results = run_batch(specs, tmp_path / "work", out=buf)
    finally:
        srv.close()

    assert results[0]["status"] == "ok"
    emitted = buf.getvalue() + json.dumps(results)
    assert SECRET_ACCESS_KEY not in emitted
    assert SECRET_ACCESS_ID not in emitted
    captured = capsys.readouterr()
    assert SECRET_ACCESS_KEY not in captured.out and SECRET_ACCESS_KEY not in captured.err
