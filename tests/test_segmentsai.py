"""``segmentsai-instance`` loader tests (WSD S7e / S7f).

Fixtures under ``tests/fixtures/segmentsai/case{1,2}/`` pair a real Segments.ai
instance-id PNG (``labelled_data/segments/*_label_ground-truth.png``, byte-for-byte
from the Hetzner bucket) with a synthetic same-size ATL placeholder jpg — the loader
only reads the ATL image's dimensions, never its pixels, so a placeholder keeps the
fixture tiny without weakening the assertions. The v0.4 JSON is the real
``coral_reef_annotated_images-v0.4.json`` categories table, trimmed to the two samples
under test.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest
from conftest import make_source

from marinedata.loaders import LoaderError, build_loader

FIXTURES = Path(__file__).parent / "fixtures" / "segmentsai"


def _pixel_counts(path: Path) -> Counter:
    from PIL import Image

    with Image.open(path) as im:
        return Counter(im.tobytes())


def _fixture_copy(name: str, tmp_path: Path) -> Path:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    return root


def test_sample_17001738002_hard_and_soft_coral(tmp_path: Path) -> None:
    """ids 1,2 -> Hard Coral; ids 3,4 -> Soft Coral (S7e test 1)."""
    source = make_source("segmentsai-instance")
    samples = list(build_loader(source, _fixture_copy("case1", tmp_path)))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.mask is not None
    counts = _pixel_counts(Path(sample.mask))
    assert counts == {0: 717_456, 1: 341_367, 4: 4_138}
    assert sample.meta["objects_without_pixels"] == 0


def test_sample_17002369001_larger_image(tmp_path: Path) -> None:
    """1398x1398 sample; ids 1-4 -> Hard Coral, id 5 -> Soft Coral (S7e test 2)."""
    source = make_source("segmentsai-instance")
    samples = list(build_loader(source, _fixture_copy("case2", tmp_path)))
    assert len(samples) == 1
    counts = _pixel_counts(Path(samples[0].mask))
    assert counts == {0: 966_443, 1: 747_047, 4: 240_914}


def test_unmapped_instance_id_raises(tmp_path: Path) -> None:
    """A pixel id with no v0.4 annotation entry (injected id 6) raises."""
    from PIL import Image

    root = _fixture_copy("case1", tmp_path)
    mask_path = next((root / "labelled_data" / "segments").glob("*.png"))
    with Image.open(mask_path) as im:
        im.putpixel((0, 0), 6)
        im.save(mask_path)

    source = make_source("segmentsai-instance")
    with pytest.raises(LoaderError, match="no annotation entry"):
        list(build_loader(source, root))
