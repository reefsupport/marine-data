"""``labelbox-rgb`` loader tests (WSD S6x §2a / S7a).

Fixtures under ``tests/fixtures/labelbox_rgb/`` are single-image crops of the real
Hetzner bucket: one exact NDJSON line (or two, for SEAVIEW_ATL's dedup case) plus the
matching ``masks_stitched/*.png``, copied byte-for-byte so the pixel-count assertions
below are the real prototype numbers from the spec, not synthetic ones.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata.loaders import LoaderError, build_loader, loader_for
from marinedata.loaders.labelbox import LabelboxRgbMaskLoader

FIXTURES = Path(__file__).parent / "fixtures" / "labelbox_rgb"


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


def test_seaflower_courtown_hc_and_sc(tmp_path: Path) -> None:
    """Both classes present; matches rs_labelled's stitched copy of the same image."""
    source = make_source("labelbox-rgb")
    samples = list(build_loader(source, _fixture_copy("seaflower_courtown", tmp_path)))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.mask is not None
    counts = _pixel_counts(Path(sample.mask))
    assert counts == {0: 3_391_583, 1: 839, 2: 16_454}
    assert sample.meta["objects_without_pixels"] == 0


def test_tayrona_milleporid_has_no_pixels(tmp_path: Path) -> None:
    """NDJSON has 1 Milleporid object; it is painted black, not an error."""
    source = make_source("labelbox-rgb")
    samples = list(build_loader(source, _fixture_copy("tayrona", tmp_path)))
    assert len(samples) == 1
    sample = samples[0]
    counts = _pixel_counts(Path(sample.mask))
    assert counts == {0: 12_049_977, 1: 692_943}
    assert sample.meta["objects_without_pixels"] == 1


def test_seaview_atl_old_export_cross_checks(tmp_path: Path) -> None:
    """The -old export (line with the most objects, per external_id) matches the mask."""
    source = make_source("labelbox-rgb", {"annotations": "export-result-old.ndjson"})
    samples = list(build_loader(source, _fixture_copy("seaview_atl", tmp_path)))
    assert len(samples) == 1
    counts = _pixel_counts(Path(samples[0].mask))
    assert counts == {0: 653_830, 1: 382_139, 2: 26_992}


def test_seaview_atl_new_export_raises_on_mismatch(tmp_path: Path) -> None:
    """The newer export has 0 objects for this image — the cross-check must catch it."""
    source = make_source("labelbox-rgb", {"annotations": "export-result.ndjson"})
    with pytest.raises(LoaderError, match="do not match"):
        list(build_loader(source, _fixture_copy("seaview_atl", tmp_path)))


def test_unknown_colour_raises(tmp_path: Path) -> None:
    """A colour outside the 5-entry LUT raises — the guard for unsampled regions."""
    from PIL import Image

    root = _fixture_copy("seaflower_courtown", tmp_path)
    mask_path = next((root / "masks_stitched").glob("*.png"))
    with Image.open(mask_path) as im:
        im = im.convert("RGB")
        im.putpixel((0, 0), (1, 2, 3))
        im.save(mask_path)

    source = make_source("labelbox-rgb")
    with pytest.raises(LoaderError, match="not in the labelbox-rgb LUT"):
        list(build_loader(source, root))


def test_seaview_labels_staged_tree_keeps_the_labelbox_rgb_converter() -> None:
    """RB-3 (2026-10-06): `reef-support-seaview-labels` moved from the legacy `labelbox-rgb`
    mask_labels tree to a staged tree; the stitched masks under `labels/masks/` still use the
    fixed Labelbox colour LUT, so the converter stays registered and the entry says so."""
    source = Registry.load().source("reef-support-seaview-labels")
    assert source.loader is not None
    assert source.loader.layout == "staged-tree"
    assert "labelbox" in str(source.loader.params["masks"]).lower()
    assert loader_for("labelbox-rgb") is LabelboxRgbMaskLoader
