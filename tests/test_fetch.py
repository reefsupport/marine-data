"""HTTP sample fetching — no network. Uses `file://` URLs, which `_get` handles
identically to `http(s)://` (urllib dispatches on scheme), so these exercise the real
code path without needing a server.

The gap this closes: `_fetch_http`'s own docstring said it retrieves "a single declared
sample archive or file", but the implementation only ever wrote the raw bytes of
whatever `sample_url` pointed at — no `.zip` or `.tar` was ever extracted. That silently
blocked every HTTP-access source needing a multi-file layout (image-mask-pairs,
coco-json, yolo-txt) from ever being verified, however correct the URL was: a bare-file
fetch cannot satisfy a layout that needs an `images/` + `masks/` pair. Found while
working through the 28-source verification backlog — every HTTP source failed with "no
sample_url declared" even for sources whose real download IS a small zip.
"""

from __future__ import annotations

import http.server
import io
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest
from conftest import make_source

from marinedata.enums import AccessMethod
from marinedata.fetch import MAX_DOWNLOAD_WITHOUT_RANGE, FetchError, FetchNotSupported, fetch_sample


def _http_source(sample_url: str):
    base = make_source("image-mask-pairs")
    return base.model_copy(
        update={
            "access": base.access.model_copy(
                update={"method": AccessMethod.HTTP, "params": {"sample_url": sample_url}}
            )
        }
    )


def _file_url(path: Path) -> str:
    return f"file://{path}"


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _make_tar(path: Path, files: dict[str, bytes], *, compression: str = "") -> None:
    mode = f"w:{compression}" if compression else "w"
    with tarfile.open(path, mode) as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def test_bare_file_still_works(tmp_path: Path) -> None:
    """The original behaviour — a single non-archive file — must be unaffected."""
    sample = tmp_path / "one.jpg"
    sample.write_bytes(b"\xff\xd8\xff")
    result = fetch_sample(_http_source(_file_url(sample)), root=tmp_path / "dest", force=True)
    assert result.items == 1
    assert (tmp_path / "dest" / "one.jpg").read_bytes() == b"\xff\xd8\xff"


def test_zip_archive_is_extracted_with_directory_structure(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    _make_zip(
        archive,
        {
            "images/f0.jpg": b"img0",
            "images/f1.jpg": b"img1",
            "masks/f0.png": b"mask0",
            "masks/f1.png": b"mask1",
        },
    )
    dest = tmp_path / "dest"
    result = fetch_sample(_http_source(_file_url(archive)), root=dest, force=True)
    assert result.items == 4
    assert (dest / "images" / "f0.jpg").read_bytes() == b"img0"
    assert (dest / "masks" / "f1.png").read_bytes() == b"mask1"


@pytest.mark.parametrize(
    "compression,suffix", [("", ".tar"), ("gz", ".tar.gz"), ("bz2", ".tar.bz2")]
)
def test_tar_archives_are_extracted(tmp_path: Path, compression: str, suffix: str) -> None:
    archive = tmp_path / f"sample{suffix}"
    _make_tar(archive, {"images/a.jpg": b"A", "masks/a.png": b"M"}, compression=compression)
    dest = tmp_path / "dest"
    result = fetch_sample(_http_source(_file_url(archive)), root=dest, force=True)
    assert result.items == 2
    assert (dest / "images" / "a.jpg").read_bytes() == b"A"
    assert (dest / "masks" / "a.png").read_bytes() == b"M"


def test_archive_extraction_is_bounded_by_limit(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    _make_zip(archive, {f"images/f{i}.jpg": f"img{i}".encode() for i in range(20)})
    dest = tmp_path / "dest"
    result = fetch_sample(_http_source(_file_url(archive)), root=dest, limit=5, force=True)
    assert result.items == 5
    assert len(list((dest / "images").iterdir())) == 5


def test_zip_slip_is_refused(tmp_path: Path) -> None:
    """A path-traversal member must not write outside the destination directory."""
    archive = tmp_path / "evil.zip"
    outside = tmp_path / "outside.txt"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", b"escaped")
        zf.writestr("images/ok.jpg", b"fine")
    dest = tmp_path / "dest"
    dest.mkdir()
    result = fetch_sample(_http_source(_file_url(archive)), root=dest, force=True)
    assert not outside.exists()
    assert result.items == 1  # only the safe member


def test_extraction_works_when_destination_is_reached_via_a_symlink(tmp_path: Path) -> None:
    """⭐ Real bug, caught by live verification, invisible to every prior test here.

    `_safe_extract_path` compared an unresolved `root` against a `.resolve()`d target —
    fine when `root` has no symlinks in its ancestry (true of every `tmp_path` pytest
    hands out), but on macOS `/tmp` itself is a symlink to `/private/tmp`, so comparing
    `/tmp/x` against `/private/tmp/x/...` rejected every single member as "escaping",
    silently turning a perfectly good archive into "contained no extractable files".
    Reproduced with a real 487 MB archive over the real network before being traced to
    this. This fixture recreates the shape (destination reached through a symlink)
    without needing an actual symlinked filesystem to already exist.
    """
    real_dest = tmp_path / "real"
    real_dest.mkdir()
    symlinked_dest = tmp_path / "via_symlink"
    symlinked_dest.symlink_to(real_dest)

    archive = tmp_path / "sample.zip"
    _make_zip(archive, {"images/a.jpg": b"A", "masks/a.png": b"M"})

    result = fetch_sample(
        _http_source(_file_url(archive)), root=symlinked_dest / "dest", force=True
    )
    assert result.items == 2
    assert (real_dest / "dest" / "images" / "a.jpg").read_bytes() == b"A"


def test_empty_archive_raises(tmp_path: Path) -> None:
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    with pytest.raises(FetchError, match="no extractable files"):
        fetch_sample(_http_source(_file_url(archive)), root=tmp_path / "dest", force=True)


def test_missing_sample_url_is_unfetchable() -> None:
    base = make_source("image-mask-pairs")
    source = base.model_copy(
        update={"access": base.access.model_copy(update={"method": AccessMethod.HTTP})}
    )
    with pytest.raises(FetchNotSupported, match="no `sample_url` declared"):
        fetch_sample(source)


# ── remote zip: a real Range-capable server, no mocks ───────────────────────
#
# Python's own http.server.SimpleHTTPRequestHandler does not support Range requests
# (verified: it always returns 200 with the whole body, no Accept-Ranges header), so a
# minimal handler that does is needed here — this is exactly the class of server the
# real fix targets (HF's CDN, S3-backed hosts), and a synthetic fixture that could not
# express Range support would not actually exercise the code path it is meant to prove.


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    body: bytes = b""

    def log_message(self, *args: object) -> None:  # quiet the test output
        pass

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)
            return
        start, end = range_header.removeprefix("bytes=").split("-")
        start, end = int(start), int(end)
        chunk = self.body[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.body)}")
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def range_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _serve(server, zip_bytes: bytes) -> str:
    server.RequestHandlerClass.body = zip_bytes
    return f"http://127.0.0.1:{server.server_port}/sample.zip"


def test_remote_zip_reads_only_what_it_needs(range_server, tmp_path: Path) -> None:
    """⭐ The whole point: a zip served with Range support must never be downloaded in
    full — only its central directory and the requested members travel over the wire."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(200):
            zf.writestr(f"images/f{i:04d}.jpg", bytes([i % 256]) * 2000)
    payload = buf.getvalue()

    import marinedata.fetch as fetch_module

    ranges_requested: list[tuple[int, int]] = []
    original = fetch_module._get_range

    def spying_get_range(url, start, end, **kwargs):
        ranges_requested.append((start, end))
        return original(url, start, end, **kwargs)

    fetch_module._get_range = spying_get_range
    # A full download is faster and just as safe for a small archive, so the dispatch
    # in _fetch_http only takes the ranged path once size exceeds this threshold — lower
    # it here so a small test archive still exercises the ranged path being tested.
    original_threshold = fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE
    fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = 1
    try:
        url = _serve(range_server, payload)
        result = fetch_sample(_http_source(url), root=tmp_path / "dest", limit=3, force=True)
    finally:
        fetch_module._get_range = original
        fetch_module.MAX_DOWNLOAD_WITHOUT_RANGE = original_threshold

    assert result.items == 3
    total_ranged = sum(end - start + 1 for start, end in ranges_requested)
    assert total_ranged < len(payload) / 4, (
        f"ranged reads pulled {total_ranged} bytes of a {len(payload)}-byte, 200-member "
        f"archive to extract only 3 members — should be a small fraction of the whole"
    )


def test_remote_zip_extraction_matches_content(range_server, tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/a.jpg", b"AAAA")
        zf.writestr("masks/a.png", b"MMMM")
    url = _serve(range_server, buf.getvalue())
    dest = tmp_path / "dest"
    result = fetch_sample(_http_source(url), root=dest, force=True)
    assert result.items == 2
    assert (dest / "images" / "a.jpg").read_bytes() == b"AAAA"
    assert (dest / "masks" / "a.png").read_bytes() == b"MMMM"


def test_large_zip_without_range_support_is_refused(tmp_path: Path) -> None:
    """A host that ignores Range and reports a huge Content-Length must be refused
    outright rather than silently downloaded in full."""

    class _NoRangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_HEAD(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(MAX_DOWNLOAD_WITHOUT_RANGE * 2))
            self.end_headers()

        def do_GET(self) -> None:  # pragma: no cover - must never be reached
            raise AssertionError("a full GET must not be attempted for a huge, non-ranged zip")

    server = http.server.HTTPServer(("127.0.0.1", 0), _NoRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/huge.zip"
        with pytest.raises(FetchNotSupported, match="does not support Range"):
            fetch_sample(_http_source(url), root=tmp_path / "dest", force=True)
    finally:
        server.shutdown()
        thread.join(timeout=5)
