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
import json
import threading
import time
import urllib.parse
from pathlib import Path

import pytest
from conftest import make_source

import marinedata.fetchers_remote as fetchers_remote
from marinedata.enums import AccessMethod
from marinedata.fetch import FetchError, fetch_sample
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
            "access": base.access.model_copy(update={"method": AccessMethod.S3, "params": params}),
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


_DENSE_MASK_FILES = {
    "metadata.parquet": b"METADATA",
    **{f"images/default/img{i:02d}.jpg": f"IMG{i}".encode() for i in range(10)},
    **{f"labels/masks/img{i:02d}.png": f"MASK{i}".encode() for i in range(10)},
}


def test_limit_fetches_only_the_sampled_images_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dense-mask source (10 images, 10 per-image labels): `--limit 2` must fetch
    exactly the 2 sampled images' labels, not all 10 (the coralscop-masks-rs bug)."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    calls: list[str] = []
    monkeypatch.setattr(
        fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, calls)
    )

    result = fetchers_remote.fetch_s3(source, tmp_path, limit=2)

    assert result.items == 5  # metadata + 2 sampled images + their 2 matching labels
    for i in range(2):
        assert (tmp_path / "images" / "default" / f"img{i:02d}.jpg").exists()
        assert (tmp_path / "labels" / "masks" / f"img{i:02d}.png").exists()
    for i in range(2, 10):
        assert not (tmp_path / "images" / "default" / f"img{i:02d}.jpg").exists()
        assert not (tmp_path / "labels" / "masks" / f"img{i:02d}.png").exists()


def test_no_limit_fetches_all_images_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A limit covering every image (i.e. no effective truncation) fetches all labels
    too, since every image's stem is sampled — unchanged end-to-end behaviour."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    calls: list[str] = []
    monkeypatch.setattr(
        fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, calls)
    )

    result = fetchers_remote.fetch_s3(source, tmp_path, limit=10)

    assert result.items == 21  # metadata + 10 images + 10 labels
    for i in range(10):
        assert (tmp_path / "images" / "default" / f"img{i:02d}.jpg").exists()
        assert (tmp_path / "labels" / "masks" / f"img{i:02d}.png").exists()


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


# ── cache reuse: a truncated `_fetch.json` must not satisfy a bigger request ─────────


def test_truncated_cache_is_not_reused_for_a_larger_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coralscop-masks-rs release bug: a bounded 2-image verification sample left
    `truncated: true` in `_fetch.json`; the release build's unbounded fetch then reused
    that cache as-is instead of re-fetching the remaining 8 images, and the loader
    later failed on images `metadata.parquet` named but the cache never held."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    calls: list[str] = []
    monkeypatch.setattr(
        fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, calls)
    )

    bounded = fetch_sample(source, root=tmp_path, limit=2)
    assert bounded.items == 5  # metadata + 2 sampled images + their 2 labels
    assert bounded.truncated is True
    for i in range(2, 10):
        assert not (tmp_path / "images" / "default" / f"img{i:02d}.jpg").exists()

    full = fetch_sample(source, root=tmp_path, limit=10)

    assert full.items == 21  # metadata + all 10 images + all 10 labels
    assert full.truncated is False
    for i in range(10):
        assert (tmp_path / "images" / "default" / f"img{i:02d}.jpg").exists()
        assert (tmp_path / "labels" / "masks" / f"img{i:02d}.png").exists()


def test_complete_cache_is_still_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache that already holds everything (not truncated) is reused for a bigger or
    unbounded request too — no network call is made the second time."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    calls: list[str] = []
    monkeypatch.setattr(
        fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, calls)
    )

    complete = fetch_sample(source, root=tmp_path, limit=10)
    assert complete.truncated is False
    calls.clear()

    def _forbidden_get(url: str, *, headers: dict[str, str] | None = None, **kw: object) -> bytes:
        raise AssertionError(f"unexpected GET {url} — cache should have been reused")

    monkeypatch.setattr(fetchers_remote, "_get", _forbidden_get)

    reused = fetch_sample(source, root=tmp_path, limit=10_000_000)

    assert reused.items == 21
    assert reused.truncated is False
    assert calls == []


# --- S44: complete + concurrent full fetch -------------------------------------------


def test_full_fetch_skips_files_already_present_with_the_right_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted release fetch resumes: a file already on disk with the manifest's
    sha256 is not fetched again, and one with the wrong bytes is re-fetched."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    for i in range(5):
        rel = f"labels/masks/img{i:02d}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(_DENSE_MASK_FILES[rel])
    stale = tmp_path / "images" / "default" / "img00.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"HALF-WRITTEN")
    calls: list[str] = []
    monkeypatch.setattr(
        fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, calls)
    )

    result = fetchers_remote.fetch_s3(source, tmp_path, limit=10)

    assert result.items == 21 and result.truncated is False
    fetched = set(calls) - {_url_for("CHECKSUMS.sha256")}
    assert len(fetched) == 21 - 5  # the 5 verified labels were skipped
    assert _url_for("images/default/img00.jpg") in fetched  # wrong bytes -> re-fetched
    assert stale.read_bytes() == _DENSE_MASK_FILES["images/default/img00.jpg"]
    for rel, data in _DENSE_MASK_FILES.items():
        assert (tmp_path / rel).read_bytes() == data


def test_full_fetch_with_failed_files_raises_with_the_missing_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full fetch never returns a partial root: two files that cannot be fetched make
    it raise naming how many are missing — after the rest have still been staged."""
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    broken = {_url_for("images/default/img03.jpg"), _url_for("labels/masks/img07.png")}
    inner = _fake_get(manifest_bytes, _DENSE_MASK_FILES, [])

    def flaky_get(url: str, **kw: object) -> bytes:
        if url in broken:
            raise FetchError(f"failed after 5 attempt(s) for {url}: HTTP 503")
        return inner(url, **kw)

    monkeypatch.setattr(fetchers_remote, "_get", flaky_get)

    with pytest.raises(FetchError, match=r"full fetch incomplete — 2 of 21 file\(s\) missing"):
        fetchers_remote.fetch_s3(source, tmp_path, limit=10_000_000)
    assert (tmp_path / "images" / "default" / "img04.jpg").is_file()
    assert not (tmp_path / "images" / "default" / "img03.jpg").exists()


def test_full_fetch_leaves_a_complete_marker_over_a_stale_truncated_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    monkeypatch.setattr(fetchers_remote, "_get", _fake_get(manifest_bytes, _DENSE_MASK_FILES, []))
    fetch_sample(source, root=tmp_path, limit=2)

    fetch_sample(source, root=tmp_path, limit=10_000_000)

    marker = json.loads((tmp_path / "_fetch.json").read_text(encoding="utf-8"))
    assert marker["truncated"] is False
    assert marker["items"] == 21
    assert "NOT the full dataset" not in marker["note"]


def test_manifest_files_are_fetched_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_bytes, root_digest = _build_manifest(_DENSE_MASK_FILES)
    source = _pinned_source(root_digest)
    inner = _fake_get(manifest_bytes, _DENSE_MASK_FILES, [])
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def slow_get(url: str, **kw: object) -> bytes:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return inner(url, **kw)

    monkeypatch.setattr(fetchers_remote, "_get", slow_get)
    monkeypatch.setenv("MARINEDATA_FETCH_WORKERS", "4")

    fetchers_remote.fetch_s3(source, tmp_path, limit=10)

    assert 1 < peak <= 4  # concurrent, and bounded by the configured pool
