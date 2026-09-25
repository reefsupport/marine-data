"""``coralseg-r-channel`` loader tests (WSD S6x §2f / S7f).

Fixture under ``tests/fixtures/coralseg/fr3/`` is a single-tile crop of the real
Hetzner bucket (``benthic_datasets/mask_labels/Coralseg/test/``), copied byte-for-byte
so the pixel-count assertion is the real R-channel counts, not a synthetic one.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata.loaders import LoaderError, build_loader

FIXTURES = Path(__file__).parent / "fixtures" / "coralseg"


def _pixel_counts(path: Path) -> Counter:
    from PIL import Image

    with Image.open(path) as im:
        return Counter(im.tobytes())


def _fixture_copy(name: str, tmp_path: Path) -> Path:
    """A writable copy of ``FIXTURES / name`` — the loader caches converted PNGs next
    to the mask it read, and the checked-in fixture tree must never gain untracked
    output from running the test suite."""
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    return root


def test_fr3_tile_r_channel_pixel_counts(tmp_path: Path) -> None:
    """R=0/R=1 decode into the two declared classes with the real tile's counts."""
    source = make_source("coralseg-r-channel")
    samples = list(build_loader(source, _fixture_copy("fr3", tmp_path)))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.mask is not None
    counts = _pixel_counts(Path(sample.mask))
    assert counts == {0: 221_708, 1: 40_436}
    assert sample.meta["mask_classes"] == {"0": "Other", "1": "Hard Coral", "2": "Soft Coral"}


def test_unknown_r_value_raises(tmp_path: Path) -> None:
    """A red value outside {0, 1, 2} raises rather than being clipped/guessed."""
    from PIL import Image

    root = _fixture_copy("fr3", tmp_path)
    mask_path = next((root / "Mask").glob("*.png"))
    with Image.open(mask_path) as im:
        im = im.convert("RGB")
        im.putpixel((0, 0), (3, 0, 0))
        im.save(mask_path)

    source = make_source("coralseg-r-channel")
    with pytest.raises(LoaderError, match="not in \\{0, 1, 2\\}"):
        list(build_loader(source, root))


def test_registry_source_resolves_to_staged_tree_with_all_three_mask_values() -> None:
    """coralseg-flip-d (D-AI3): the registry entry now reads its own staged copy via
    ``staged-tree``, declaring every red-channel value the live masks contain."""
    registry = Registry.load()
    source = registry.source("coralseg-ucsd-mosaics")
    assert source.loader is not None
    assert source.loader.layout == "staged-tree"
    assert source.loader.params["mask_channel"] == "r"
    assert source.loader.params["mask_values"] == "0=Other,1=Hard Coral,2=Soft Coral"
