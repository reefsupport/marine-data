"""Manifest-driven (no-list) S3 fetch for pinned staged trees (7j).

`rs-storage-open` allows anonymous `GetObject` but denies anonymous `ListBucket` (7i),
so `fetch_s3`'s ordinary listing path 403s for every source on it. A source pinned by
`checksums.root_digest` with `loader.layout: staged-tree` does not need to list: the
root digest names the exact `CHECKSUMS.sha256` key, and that manifest names every other
file by key too. These tests stub the GET layer (`fetchers_remote._get`) — no network —
and cover: digest match, a stale/tampered root digest, a corrupted individual file, a
deterministic `--limit` subset, and that an unpinned source is untouched (still lists).
"""

from __future__ import annotations

import hashlib
import urllib.parse
from pathlib import Path

import pytest
from conftest import make_source

import marinedata.fetchers_remote as fetchers_remote
from marinedata.enums import AccessMethod
from marinedata.fetch import FetchError
from marinedata.models import Checksums

BUCKET = "rs-storage-open"
ENDPOINT = "hel1.example.invalid"
HOST = f"{BUCKET}.{ENDPOINT}"
PREFIX = "sources/fixture/2026-09-23-abc/"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_manifest(files: dict[str, bytes]) -> tuple[bytes, str]:
    """Render a `CHECKSUMS.sha256`-format manifest and its own sha256 (root digest)."""
    digests = {rel: _digest(data) for rel, data in files.items()}
    text = "".join(f"{digests[rel]}  {rel}\n" for rel in sorted(digests))
    manifest_bytes = text.encode("utf-8")
    return manifest_bytes, _digest(manifest_bytes)


def _pinned_source(root_digest: str, *, credentials_env: str | None = None):
    base = make_source("staged-tree")
    params: dict[str, str] = {"bucket": BUCKET, "endpoint": ENDPOINT, "prefix": PREFIX}
    if credentials_env:
        params["credentials_env"] = credentials_env
    return base.model_copy(
        update={
            "access": base.access.model_copy(
                update={"method": AccessMethod.S3, "params": params}
            ),
            "checksums": Checksums(
                version="2026-09-23-abc", root_digest=root_digest, files=5, size_bytes=100
            ),
        }
    )


def _unpinned_source():
    base = make_source("staged-tree")
    return base.model_copy(
        update={
            "access": base.access.model_copy(
                update={
                    "method": AccessMethod.S3,
                    "params": {"bucket": BUCKET, "endpoint": ENDPOINT, "prefix": PREFIX},
                }
            )
        }
    )


def _url_for(relative: str) -> str:
    return f"https://{HOST}/{urllib.parse.quote(PREFIX + relative)}"


def _fake_get(manifest_bytes: bytes, files: dict[str, bytes], calls: list[str]):
    manifest_url = _url_for("CHECKSUMS.sha256")

    def fake_get(url: str, *, headers: dict[str, str] | None = None, **kw: object) -> bytes:
        calls.append(url)
        if url == manifest_url:
            return manifest_bytes
        for rel, data in files.items():
            if url == _url_for(rel):
                return data
        raise AssertionError(f"unexpected GET {url}")

    return fake_get


_FILES = {
    "metadata.parquet": b"METADATA",
    "labels/points.parquet": b"POINTS",
    "images/a.jpg": b"AAA",
    "images/b.jpg": b"BBB",
    "images/c.jpg": b"CCC",
}


def test_digest_match_fetches_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_bytes, root_digest = _build_manifest(_FILES)
    source = _pinned_source(root_digest)
    calls: list[str] = []
    monkeypatch.setattr(fetchers_remote, "_get", _fake_get(manifest_bytes, _FILES, calls))

    result = fetchers_remote.fetch_s3(source, tmp_path, limit=2)

    assert result.method == "s3-manifest"
    assert result.items == 4  # metadata + labels/points.parquet + 2 of 3 images
    assert (tmp_path / "metadata.parquet").read_bytes() == _FILES["metadata.parquet"]
    assert (tmp_path / "labels" / "points.parquet").read_bytes() == _FILES["labels/points.parquet"]
    assert (tmp_path / "images" / "a.jpg").read_bytes() == _FILES["images/a.jpg"]
    assert (tmp_path / "images" / "b.jpg").read_bytes() == _FILES["images/b.jpg"]
    assert not (tmp_path / "images" / "c.jpg").exists()


def test_root_digest_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_bytes, _real_digest = _build_manifest(_FILES)
    source = _pinned_source("0" * 64)  # deliberately wrong pin
    calls: list[str] = []
    monkeypatch.setattr(fetchers_remote, "_get", _fake_get(manifest_bytes, _FILES, calls))

    with pytest.raises(FetchError, match="does not match"):
        fetchers_remote.fetch_s3(source, tmp_path, limit=2)


def test_one_file_digest_mismatch_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_bytes, root_digest = _build_manifest(_FILES)
    source = _pinned_source(root_digest)
    corrupted = dict(_FILES)
    corrupted["images/a.jpg"] = b"TAMPERED"  # manifest still expects the original hash
    calls: list[str] = []
    monkeypatch.setattr(fetchers_remote, "_get", _fake_get(manifest_bytes, corrupted, calls))

    with pytest.raises(FetchError, match=r"images/a\.jpg"):
        fetchers_remote.fetch_s3(source, tmp_path, limit=2)


def test_limit_subset_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_bytes, root_digest = _build_manifest(_FILES)
    source = _pinned_source(root_digest)

    for attempt in range(2):
        root = tmp_path / f"run-{attempt}"
        calls: list[str] = []
        monkeypatch.setattr(fetchers_remote, "_get", _fake_get(manifest_bytes, _FILES, calls))
        result = fetchers_remote.fetch_s3(source, root, limit=2)
        assert result.items == 4
        assert (root / "images" / "a.jpg").exists()
        assert (root / "images" / "b.jpg").exists()
        assert not (root / "images" / "c.jpg").exists()


def test_unpinned_source_still_takes_the_list_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _unpinned_source()
    listing_xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<Name>rs-storage-open</Name><Prefix>sources/fixture/2026-09-23-abc/</Prefix>"
        b"</ListBucketResult>"
    )
    calls: list[str] = []

    def fake_get(url: str, *, headers: dict[str, str] | None = None, **kw: object) -> bytes:
        calls.append(url)
        return listing_xml

    monkeypatch.setattr(fetchers_remote, "_get", fake_get)

    with pytest.raises(FetchError, match="no fetchable objects"):
        fetchers_remote.fetch_s3(source, tmp_path, limit=5)

    assert calls, "the list path must still call _get"
    assert "list-type=2" in calls[0]


def test_credentialed_pinned_source_still_takes_the_list_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Credentials mean the source can list — the manifest shortcut is only for the
    anonymous case where listing itself is the thing that 403s."""
    _manifest_bytes, root_digest = _build_manifest(_FILES)
    source = _pinned_source(root_digest, credentials_env="TESTS3")
    monkeypatch.setenv("TESTS3_ACCESS_KEY_ID", "AKIDTEST")
    monkeypatch.setenv("TESTS3_SECRET_ACCESS_KEY", "secret")
    calls: list[str] = []

    def fake_get(url: str, *, headers: dict[str, str] | None = None, **kw: object) -> bytes:
        calls.append(url)
        return (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            b"<Name>rs-storage-open</Name></ListBucketResult>"
        )

    monkeypatch.setattr(fetchers_remote, "_get", fake_get)

    with pytest.raises(FetchError, match="no fetchable objects"):
        fetchers_remote.fetch_s3(source, tmp_path, limit=5)

    assert calls and "list-type=2" in calls[0]
