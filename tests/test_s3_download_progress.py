"""Progress reporting in :func:`marinedata.s3_download.download_images` (D3f).

One line to stderr every 100 completed images, flushed, carrying the running byte
total — never written into the staged tree itself.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marinedata import s3_download


def _stub_stage_one(stem, image_groups, plan, version_root):
    key, _size, _etag = image_groups[stem]
    return stem, key, f"digest-{stem}", 10, 10


@pytest.fixture(autouse=True)
def _no_real_downloads(monkeypatch):
    monkeypatch.setattr(s3_download, "_stage_one", _stub_stage_one)


def _plan() -> SimpleNamespace:
    return SimpleNamespace(partition="default", image_suffix=".jpg")


def test_progress_line_every_100_images(capsys, tmp_path: Path) -> None:
    selected = [f"img{i:04d}" for i in range(250)]
    image_groups = {stem: (f"{stem}.jpg", 1000, "etag") for stem in selected}

    s3_download.download_images(selected, image_groups, _plan(), tmp_path, workers=4)

    lines = [line for line in capsys.readouterr().err.splitlines() if line]
    assert len(lines) == 2
    assert lines[0] == "100/250 images, 100000 B"
    assert lines[1] == "200/250 images, 200000 B"


def test_no_progress_line_under_100_images(capsys, tmp_path: Path) -> None:
    selected = [f"img{i:04d}" for i in range(5)]
    image_groups = {stem: (f"{stem}.jpg", 1000, "etag") for stem in selected}

    s3_download.download_images(selected, image_groups, _plan(), tmp_path, workers=2)

    assert capsys.readouterr().err == ""


def test_progress_never_written_to_a_staged_file(capsys, tmp_path: Path) -> None:
    selected = [f"img{i:04d}" for i in range(100)]
    image_groups = {stem: (f"{stem}.jpg", 500, "etag") for stem in selected}

    s3_download.download_images(selected, image_groups, _plan(), tmp_path, workers=1)

    capsys.readouterr()  # drain — asserted above
    staged_files = list(tmp_path.rglob("*"))
    assert staged_files == []
