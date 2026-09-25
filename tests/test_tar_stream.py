"""WP-6n: a tar container has no central directory (unlike zip, WP-6k), so its members
are streamed sequentially, once, straight off a fake HTTP response object -- never
written to disk. See :func:`marinedata.adapters._range.iter_tar_stream`."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from marinedata.adapters import SPOOLED, STREAMABLE, suffix_of
from marinedata.adapters._range import iter_tar_stream

MAX_BYTES = 100


def _member_ok(name: str) -> bool:
    return suffix_of(name) in STREAMABLE + SPOOLED


def _make_tar_gz() -> bytes:
    """5 images (4 small + 1 oversize), 1 non-image -- 4 members should pass the filter."""
    members = {
        "img1.jpg": b"a" * 10,
        "img2.jpg": b"b" * 10,
        "img3.jpg": b"c" * 10,
        "img4.jpg": b"d" * 10,
        "big.jpg": b"e" * (MAX_BYTES + 1),  # oversize: filtered by the size cap, not read
        "readme.txt": b"not an image",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_iter_tar_stream_filters_oversize_and_non_image_members() -> None:
    blob = _make_tar_gz()
    out = list(iter_tar_stream(io.BytesIO(blob), member_ok=_member_ok, max_bytes=MAX_BYTES))
    assert [name for name, _ in out] == ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    assert out[0][1] == b"a" * 10


def test_iter_tar_stream_writes_nothing_under_tmp_path_except_the_checkpoint(
    tmp_path: Path,
) -> None:
    blob = _make_tar_gz()
    last = None
    for name, _data in iter_tar_stream(io.BytesIO(blob), member_ok=_member_ok, max_bytes=MAX_BYTES):
        last = name  # the pipeline's existing checkpoint records the last completed key
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(f'{{"last_member": "{last}"}}')
    assert last == "img4.jpg"
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_iter_tar_stream_resume_after_member_2_yields_members_3_and_4() -> None:
    blob = _make_tar_gz()
    out = list(
        iter_tar_stream(
            io.BytesIO(blob), member_ok=_member_ok, max_bytes=MAX_BYTES, resume_after="img2.jpg"
        )
    )
    assert [name for name, _ in out] == ["img3.jpg", "img4.jpg"]
