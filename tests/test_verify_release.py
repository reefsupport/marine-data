"""WP-3 AC: verify exits 1 on a mutated byte, 1 on a missing file, 0 on a clean build; an
s3:// check runs under moto with no real network."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from marinedata.manifest import build_manifest
from marinedata.manifest import read_manifest as _read_manifest
from marinedata.verify_release import verify, verify_local, verify_s3


def _make_build(tmp_path):
    root = tmp_path / "build"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "images.parquet").write_bytes(b"fake-parquet-bytes" * 100)
    (root / "README.md").write_text("hello")
    build_manifest(root)
    return root


def test_verify_local_clean_build_is_ok(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    report = verify_local(root, manifest)
    assert report.ok(strict=True)
    assert report.missing == report.mismatched == report.unlisted == []


def test_verify_mutated_byte_fails(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    target = root / "README.md"
    target.write_text("HELLO")  # mutate after manifest was built
    report = verify_local(root, manifest)
    assert not report.ok(strict=False)
    assert "README.md" in report.mismatched


def test_verify_missing_file_fails(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    (root / "README.md").unlink()
    report = verify_local(root, manifest)
    assert not report.ok(strict=False)
    assert "README.md" in report.missing


def test_verify_unlisted_only_fails_strict(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    (root / "extra.txt").write_text("surprise")
    report = verify_local(root, manifest)
    assert report.ok(strict=False)  # warn only
    assert not report.ok(strict=True)  # fail with --strict
    assert "extra.txt" in report.unlisted


def test_verify_entrypoint_local_dir(tmp_path):
    root = _make_build(tmp_path)
    report = verify(str(root))
    assert report.ok(strict=False)


@mock_aws
def test_verify_s3_clean_and_mutated(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    client = boto3.client("s3", region_name="us-east-1")
    bucket = "rs-storage-open-test"
    client.create_bucket(Bucket=bucket)
    prefix = "releases/v1/hf/"
    for rel in manifest:
        client.upload_file(str(root / rel), bucket, prefix + rel)

    report = verify_s3(f"s3://{bucket}/{prefix}", manifest, client=client)
    assert report.ok(strict=False)

    # Mutate the remote object's bytes -> size/etag mismatch -> fail.
    client.put_object(Bucket=bucket, Key=prefix + "README.md", Body=b"mutated")
    report2 = verify_s3(f"s3://{bucket}/{prefix}", manifest, client=client)
    assert not report2.ok(strict=False)
    assert "README.md" in report2.mismatched


@mock_aws
def test_verify_s3_deep_check_downloads_and_deletes(tmp_path):
    root = _make_build(tmp_path)
    manifest = _read_manifest(root / "MANIFEST.tsv")
    client = boto3.client("s3", region_name="us-east-1")
    bucket = "rs-storage-open-test"
    client.create_bucket(Bucket=bucket)
    prefix = "releases/v1/hf/"
    for rel in manifest:
        client.upload_file(str(root / rel), bucket, prefix + rel)

    scratch = tmp_path / "scratch"
    report = verify_s3(f"s3://{bucket}/{prefix}", manifest, client=client, deep=2, scratch=scratch)
    assert report.ok(strict=False)
    assert report.deep_checked == 2
    assert report.deep_failed == []
    # nothing left behind in scratch after the deep check
    assert list(scratch.iterdir()) == []


def test_verify_missing_manifest_argument_for_s3_raises(tmp_path):
    from marinedata.verify_release import VerifyError

    with pytest.raises(VerifyError):
        verify("s3://bucket/prefix")
