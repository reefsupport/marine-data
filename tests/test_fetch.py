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

import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from conftest import make_source

from marinedata.enums import AccessMethod
from marinedata.fetch import FetchError, FetchNotSupported, fetch_sample


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
