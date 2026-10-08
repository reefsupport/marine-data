"""Integrity on fetched bytes: sha256 verification, the plaintext-http digest
requirement, byte/ratio-bounded extraction, and path-traversal coverage for every
extractor (`_extract_archive`'s zip and tar branches, plus the remote `_extract_remote_zip`
and `_extract_remote_tar` Range-based readers).

No network: `range_server` binds `127.0.0.1:0` (loopback only, kernel-assigned port,
started/torn down per test) and `file://` URLs read from `tmp_path` — nothing here
reaches a real host. Reuses fixtures/helpers from `test_fetch` rather than
re-implementing them, so the two files describe the same fetch machinery from two
angles: general HTTP-fetch behaviour there, integrity guarantees here.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from test_fetch import (
    _DUMMY_SHA256,
    _file_url,
    _http_source,
    _make_tar,
    _make_zip,
    _serve,
    _serve_tar,
    range_server,
)

from marinedata.fetch import FetchError, FetchNotSupported, fetch_sample

__all__ = ["range_server"]  # re-exported fixture; referenced only by pytest's own lookup


# ── sha256 verification ───────────────────────────────────────────────────────────


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    """⭐ Acceptance: a declared digest that does not match the fetched bytes must
    raise — D2 treats an absent-or-wrong checksum as the same class of quiet-skip
    failure as a licence denial, never silently accepted."""
    sample = tmp_path / "one.jpg"
    sample.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(FetchError, match="declared sha256 does not match"):
        fetch_sample(
            _http_source(_file_url(sample), sha256="0" * 64),
            root=tmp_path / "dest",
            force=True,
        )


def test_sha256_match_is_accepted(tmp_path: Path) -> None:
    """The mirror of the mismatch test: the real digest of the real bytes passes."""
    content = b"\xff\xd8\xff"
    sample = tmp_path / "one.jpg"
    sample.write_bytes(content)
    result = fetch_sample(
        _http_source(_file_url(sample), sha256=hashlib.sha256(content).hexdigest()),
        root=tmp_path / "dest",
        force=True,
    )
    assert result.items == 1


def test_sha256_prefixed_form_is_accepted(tmp_path: Path) -> None:
    """The `sha256:<hex>` form (as commonly published) must work, not just bare hex."""
    content = b"\xff\xd8\xff"
    sample = tmp_path / "one.jpg"
    sample.write_bytes(content)
    result = fetch_sample(
        _http_source(_file_url(sample), sha256=f"sha256:{hashlib.sha256(content).hexdigest()}"),
        root=tmp_path / "dest",
        force=True,
    )
    assert result.items == 1


def test_http_without_digest_is_refused(tmp_path: Path) -> None:
    """⭐ Acceptance: a plaintext `http://` source with no declared digest must fail
    before anything is fetched — nothing else vouches for what actually arrives. No
    server is started here: the guard fires ahead of any network I/O, so this test
    makes zero network calls, same as every other test in this file."""
    with pytest.raises(FetchNotSupported, match="sha256"):
        fetch_sample(
            _http_source("http://example.invalid/sample.zip"),
            root=tmp_path / "dest",
            force=True,
        )


# ── byte/ratio-bounded extraction, all four extraction paths ─────────────────────


def test_local_zip_member_over_cap_is_refused(tmp_path: Path) -> None:
    """`_extract_archive`'s zip branch: a single oversized member must be refused
    before being written, not merely capped by `limit` (member *count*)."""
    import marinedata.fetch as fetch_module

    archive = tmp_path / "sample.zip"
    _make_zip(archive, {"images/big.jpg": b"X" * 1000})
    original = fetch_module.MAX_MEMBER_BYTES
    fetch_module.MAX_MEMBER_BYTES = 100
    try:
        with pytest.raises(FetchError, match="per-member cap"):
            fetch_sample(_http_source(_file_url(archive)), root=tmp_path / "dest", force=True)
    finally:
        fetch_module.MAX_MEMBER_BYTES = original


def test_local_tar_member_over_cap_is_refused(tmp_path: Path) -> None:
    """`_extract_archive`'s tar branch, same guarantee as the zip case above."""
    import marinedata.fetch as fetch_module

    archive = tmp_path / "sample.tar"
    _make_tar(archive, {"images/big.jpg": b"X" * 1000})
    original = fetch_module.MAX_MEMBER_BYTES
    fetch_module.MAX_MEMBER_BYTES = 100
    try:
        with pytest.raises(FetchError, match="per-member cap"):
            fetch_sample(_http_source(_file_url(archive)), root=tmp_path / "dest", force=True)
    finally:
        fetch_module.MAX_MEMBER_BYTES = original


def test_remote_zip_member_over_cap_is_refused(range_server, tmp_path: Path) -> None:
    """`_extract_remote_zip`: the per-member cap must apply to a Range-read archive
    exactly as it does to a fully-downloaded one — checked from central-directory
    metadata alone, before the member's bytes are pulled."""
    import marinedata.fetch as fetch_module

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("images/big.jpg", b"X" * 1000)
    url = _serve(range_server, buf.getvalue())
    original_cap = fetch_module.MAX_MEMBER_BYTES
    original_threshold = fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE
    fetch_module.MAX_MEMBER_BYTES = 100
    fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = 1  # force the ranged path, not a full GET
    try:
        with pytest.raises(FetchError, match="per-member cap"):
            fetch_sample(
                _http_source(url, sha256=_DUMMY_SHA256), root=tmp_path / "dest", force=True
            )
    finally:
        fetch_module.MAX_MEMBER_BYTES = original_cap
        fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = original_threshold


def test_remote_tar_member_over_cap_is_refused(range_server, tmp_path: Path) -> None:
    """`_extract_remote_tar`: same guarantee, for the header-by-header remote reader."""
    import marinedata.fetch as fetch_module

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"X" * 1000
        info = tarfile.TarInfo(name="images/big.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    url = _serve_tar(range_server, buf.getvalue())
    original_cap = fetch_module.MAX_MEMBER_BYTES
    original_threshold = fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE
    fetch_module.MAX_MEMBER_BYTES = 100
    fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = 1
    try:
        with pytest.raises(FetchError, match="per-member cap"):
            fetch_sample(
                _http_source(url, sha256=_DUMMY_SHA256), root=tmp_path / "dest", force=True
            )
    finally:
        fetch_module.MAX_MEMBER_BYTES = original_cap
        fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = original_threshold


# ── path traversal, every extractor `test_zip_slip_is_refused` (test_fetch.py)
# does not already cover ─────────────────────────────────────────────────────────


def test_local_tar_slip_is_refused(tmp_path: Path) -> None:
    """`_extract_archive`'s tar branch: a `../` member must not write outside the
    destination directory — the tar counterpart of `test_zip_slip_is_refused`."""
    archive = tmp_path / "evil.tar"
    outside = tmp_path / "outside.txt"
    _make_tar(archive, {"../outside.txt": b"escaped", "images/ok.jpg": b"fine"})
    dest = tmp_path / "dest"
    dest.mkdir()
    result = fetch_sample(_http_source(_file_url(archive)), root=dest, force=True)
    assert not outside.exists()
    assert result.items == 1  # only the safe member


def test_remote_zip_slip_is_refused(range_server, tmp_path: Path) -> None:
    """`_extract_remote_zip`: the same traversal member, reached via a Range-read
    zip rather than an in-memory one — `_safe_extract_path` is shared code, but this
    is the only test that actually calls it through this extractor."""
    import marinedata.fetch as fetch_module

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("../outside.txt", b"escaped")
        zf.writestr("images/ok.jpg", b"fine")
    outside = tmp_path / "outside.txt"
    url = _serve(range_server, buf.getvalue())
    dest = tmp_path / "dest"
    dest.mkdir()
    original_threshold = fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE
    fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = 1
    try:
        result = fetch_sample(_http_source(url, sha256=_DUMMY_SHA256), root=dest, force=True)
    finally:
        fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = original_threshold
    assert not outside.exists()
    assert result.items == 1


def test_remote_tar_slip_is_refused(range_server, tmp_path: Path) -> None:
    """`_extract_remote_tar`: same traversal member, via the header-by-header remote
    tar reader — no prior test referenced `_extract_remote_tar` at all."""
    import marinedata.fetch as fetch_module

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (("../outside.txt", b"escaped"), ("images/ok.jpg", b"fine")):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    outside = tmp_path / "outside.txt"
    url = _serve_tar(range_server, buf.getvalue())
    dest = tmp_path / "dest"
    dest.mkdir()
    original_threshold = fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE
    fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = 1
    try:
        result = fetch_sample(_http_source(url, sha256=_DUMMY_SHA256), root=dest, force=True)
    finally:
        fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = original_threshold
    assert not outside.exists()
    assert result.items == 1
