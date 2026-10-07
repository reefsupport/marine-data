"""CHECKSUMS.sha256 — the manifest that makes a stored version checkable without us.

The format is not ours, so the load-bearing test is not an assertion about strings: it
is `shasum -a 256 -c` checking a real tree. The rest pin the properties that assertion
cannot see — determinism, self-exclusion, bounded reads, and the immutability contract
that makes re-running over a mutated version fatal rather than quietly corrective.
"""

from __future__ import annotations

import hashlib
import http.server
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import make_source

from marinedata.checksums import (
    CHECKSUM_FILE,
    ChecksumError,
    download_digest,
    file_digest,
    iter_lines,
    parse_checksums,
    scan_digests,
    write_checksums,
)
from marinedata.models import Source

# sha256 of b"" and of b"a" — fixed by the algorithm, so they pin the format end to end
# rather than merely agreeing with whatever this code happens to produce.
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
A_SHA256 = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"


def _tree(root: Path) -> Path:
    """A version dir with nesting, an empty file and a non-ASCII name."""
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images" / "train" / "a.jpg").write_bytes(b"a")
    (root / "images" / "train" / "b.jpg").write_bytes(b"bb")
    (root / "labels" / "points.parquet").write_bytes(b"ppp")
    (root / "labels" / "été.json").write_bytes(b"")
    (root / "SOURCE.json").write_bytes(b"{}")
    return root


# ── format ──────────────────────────────────────────────────────────────────


def test_line_format_is_sha256sum_compatible(tmp_path: Path) -> None:
    """`<64 hex><two spaces><posix relative path>\\n`, and the digest is real sha256."""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"a")

    write_checksums(tmp_path)
    text = (tmp_path / CHECKSUM_FILE).read_text(encoding="utf-8")

    assert text == f"{A_SHA256}  images/a.jpg\n"


def test_empty_file_is_listed_not_skipped(tmp_path: Path) -> None:
    """A zero-byte file is part of the version; omitting it would let one appear or
    vanish without the manifest noticing."""
    (tmp_path / "EMPTY").write_bytes(b"")

    write_checksums(tmp_path)

    assert parse_checksums((tmp_path / CHECKSUM_FILE).read_text()) == {"EMPTY": EMPTY_SHA256}


def test_a_path_sha256sum_cannot_represent_is_refused(tmp_path: Path) -> None:
    """A backslash starts an escape in sha256sum's own format, so a manifest containing
    one unescaped would check a different file than the one on disk."""
    (tmp_path / "we\\ird.json").write_bytes(b"x")

    with pytest.raises(ChecksumError, match="sha256sum cannot represent"):
        write_checksums(tmp_path)


# ── sort order and self-exclusion ───────────────────────────────────────────


def test_paths_are_sorted_bytewise_including_non_ascii(tmp_path: Path) -> None:
    """Sorting is by UTF-8 bytes. Python sorts strings by code point and UTF-8 preserves
    code-point order bytewise, so the two coincide — proven here rather than asserted,
    because the format's requirement is stated in bytes."""
    _tree(tmp_path)

    write_checksums(tmp_path)
    lines = (tmp_path / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]

    assert paths == sorted(paths, key=lambda p: p.encode("utf-8"))
    assert paths[0] == "SOURCE.json"  # uppercase sorts before lowercase, bytewise
    assert paths[-1] == "labels/été.json"  # non-ASCII sorts last


def test_manifest_never_lists_itself(tmp_path: Path) -> None:
    """A file cannot contain its own digest, so listing itself would make every check
    fail — and re-running would keep changing the answer."""
    (tmp_path / "a.txt").write_bytes(b"a")

    write_checksums(tmp_path)
    digests = parse_checksums((tmp_path / CHECKSUM_FILE).read_text())

    assert CHECKSUM_FILE not in digests
    assert list(digests) == ["a.txt"]


# ── determinism ─────────────────────────────────────────────────────────────


def test_same_tree_gives_byte_identical_file(tmp_path: Path) -> None:
    """Two independent copies of the same content produce the same bytes — nothing
    timestamped, path-absolute or filesystem-ordered leaks into the manifest."""
    first = _tree(tmp_path / "first")
    second = _tree(tmp_path / "second")

    write_checksums(first)
    write_checksums(second)

    assert (first / CHECKSUM_FILE).read_bytes() == (second / CHECKSUM_FILE).read_bytes()


def test_rewriting_an_unchanged_tree_is_a_no_op(tmp_path: Path) -> None:
    """Idempotence: a second run over untouched bytes must succeed and change nothing,
    or every pipeline that retries would be a false alarm."""
    _tree(tmp_path)
    first = write_checksums(tmp_path)

    second = write_checksums(tmp_path)

    assert second.root_digest == first.root_digest
    assert (second.files, second.size_bytes) == (first.files, first.size_bytes)


def test_root_digest_is_the_manifest_file_hashed(tmp_path: Path) -> None:
    """The registry stores this one value, so it must be exactly the digest of the file
    on disk — not of an in-memory rendering that might differ in newlines."""
    _tree(tmp_path)

    manifest = write_checksums(tmp_path)

    assert manifest.root_digest == file_digest(tmp_path / CHECKSUM_FILE)
    assert manifest.files == 5
    assert manifest.size_bytes == 1 + 2 + 3 + 0 + 2


# ── immutability: a changed version raises ──────────────────────────────────


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: (r / "labels" / "points.parquet").write_bytes(b"qqq"), "changed"),
        (lambda r: (r / "labels" / "extra.json").write_bytes(b"x"), "added"),
        (lambda r: (r / "labels" / "points.parquet").unlink(), "removed"),
    ],
    ids=["content-changed", "file-added", "file-removed"],
)
def test_mutating_a_written_version_raises(
    tmp_path: Path, mutate: Callable[[Path], object], expected: str
) -> None:
    """A published version is immutable, so all three ways a tree can drift from its
    manifest are equally violations. The gate raises; it never rewrites."""
    _tree(tmp_path)
    write_checksums(tmp_path)
    before = (tmp_path / CHECKSUM_FILE).read_bytes()

    mutate(tmp_path)
    with pytest.raises(ChecksumError, match=expected):
        write_checksums(tmp_path)

    assert (tmp_path / CHECKSUM_FILE).read_bytes() == before  # refused, not half-written


def test_malformed_manifest_raises_rather_than_skipping_the_line() -> None:
    """A dropped line is a file that quietly stops being verified."""
    with pytest.raises(ChecksumError, match="line 2"):
        parse_checksums(f"{A_SHA256}  a.jpg\nnot-a-digest  b.jpg\n")


# ── reuse of already-recorded digests ───────────────────────────────────────


def test_recorded_digests_are_used_without_reading_the_file(tmp_path: Path) -> None:
    """The corpus is terabytes. A caller that already streamed a file's bytes supplies
    the digest and the file is not opened again — proven by making the file unreadable,
    which is the only way to show a read did not happen."""
    (tmp_path / "big.bin").write_bytes(b"pretend this is 200GB")
    (tmp_path / "small.txt").write_bytes(b"a")
    (tmp_path / "big.bin").chmod(0o000)

    try:
        manifest = write_checksums(tmp_path, recorded={"big.bin": EMPTY_SHA256})
    finally:
        (tmp_path / "big.bin").chmod(0o644)

    digests = parse_checksums((tmp_path / CHECKSUM_FILE).read_text())
    assert digests == {"big.bin": EMPTY_SHA256, "small.txt": A_SHA256}
    assert manifest.files == 2


def test_recorded_digests_are_advisory_not_authoritative(tmp_path: Path) -> None:
    """The tree decides which paths exist. A stale map naming a file that is not there
    is ignored, and a path it does not cover is hashed — a partial map can make the run
    slower, never wrong."""
    (tmp_path / "present.txt").write_bytes(b"a")

    scanned = scan_digests(tmp_path, recorded={"deleted.txt": EMPTY_SHA256})

    assert scanned == {"present.txt": A_SHA256}


def test_a_malformed_recorded_digest_raises(tmp_path: Path) -> None:
    """Garbage accepted here becomes a manifest that fails forever against good bytes."""
    (tmp_path / "a.txt").write_bytes(b"a")

    with pytest.raises(ChecksumError, match="expected 64 lowercase hex"):
        scan_digests(tmp_path, recorded={"a.txt": "sha256:" + A_SHA256[:16]})


# ── the real thing: an external checker agrees ──────────────────────────────


@pytest.mark.skipif(shutil.which("shasum") is None, reason="shasum not installed")
def test_shasum_c_verifies_the_written_manifest(tmp_path: Path) -> None:
    """The point of the format: verification needs no code of ours. Run from inside the
    version dir, `shasum -a 256 -c` must pass, and must fail once a byte moves."""
    _tree(tmp_path)
    write_checksums(tmp_path)

    ok = subprocess.run(
        ["shasum", "-a", "256", "-c", CHECKSUM_FILE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr

    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"z")
    bad = subprocess.run(
        ["shasum", "-a", "256", "-c", CHECKSUM_FILE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "images/train/a.jpg" in bad.stdout


def test_iter_lines_is_the_only_renderer(tmp_path: Path) -> None:
    """write_checksums streams these lines straight to disk, so joining them must
    reproduce the file exactly — otherwise the format has two definitions."""
    _tree(tmp_path)

    write_checksums(tmp_path)
    digests = scan_digests(tmp_path)

    assert "".join(iter_lines(digests)) == (tmp_path / CHECKSUM_FILE).read_text(encoding="utf-8")


# ── the registry field ──────────────────────────────────────────────────────


def _with_checksums(declared: str, covering: str | None = None) -> Source:
    """A source declaring version ``declared`` whose checksums cover ``covering``."""
    base = make_source("image-folder").model_dump()
    base["version"] = declared
    base["checksums"] = {
        "version": covering or declared,
        "root_digest": A_SHA256,
        "files": 5,
        "size_bytes": 8,
    }
    return Source(**base)


def test_checksums_default_to_none_because_nothing_is_ingested() -> None:
    """Absent means not ingested. No source carries a digest today and none should."""
    assert make_source("image-folder").checksums is None


def test_checksums_covering_another_version_raise() -> None:
    """A digest naming a different version would pass every automated check while
    describing bytes nobody serves — worse than having no digest at all."""
    with pytest.raises(ValueError, match="checksums cover version"):
        _with_checksums("v1", covering="v2")


def test_a_valid_checksums_block_is_accepted() -> None:
    source = _with_checksums("v2")

    assert source.checksums is not None
    assert source.checksums.root_digest == A_SHA256


# ── download_digest: a real HTTP server, no mocks ───────────────────────────


class _BodyHandler(http.server.BaseHTTPRequestHandler):
    body: bytes = b""
    status: int = 200

    def log_message(self, *args: object) -> None:  # quiet the test output
        pass

    def do_GET(self) -> None:
        if self.status != 200:
            self.send_response(self.status)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@pytest.fixture
def body_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _BodyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_download_digest_matches_hashlib(body_server, tmp_path: Path) -> None:
    """The digest returned is exactly ``hashlib.sha256`` of the bytes landed on disk —
    one streamed pass, not a second read to verify."""
    payload = b"reef" * 5000
    body_server.RequestHandlerClass.body = payload
    body_server.RequestHandlerClass.status = 200
    url = f"http://127.0.0.1:{body_server.server_port}/blob.bin"
    dest = tmp_path / "blob.bin"

    digest = download_digest(url, dest)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_digest_leaves_no_part_file_on_error(body_server, tmp_path: Path) -> None:
    """A 5xx mid-download must not leave a ``.part`` file, or a partial ``dest``, behind."""
    body_server.RequestHandlerClass.body = b""
    body_server.RequestHandlerClass.status = 500
    url = f"http://127.0.0.1:{body_server.server_port}/blob.bin"
    dest = tmp_path / "blob.bin"

    import urllib.error

    with pytest.raises(urllib.error.HTTPError):
        download_digest(url, dest)

    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()
