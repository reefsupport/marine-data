"""``coco-rle-binary`` loader tests (WSD S6x §2e).

The real-sample fixture under ``tests/fixtures/coralscop/sample/`` is the smaller of
two live ``coralscop-masks-rs`` bucket JSONs fetched anonymously during development
(4.4 KB, well under the 50 KB cap) — its expected union pixel count (71,670 of a
1020x1797 image, 11 annotations, all compressed RLE) was cross-checked against a real
``pycocotools.mask.decode()`` in a throwaway venv, since pycocotools is not a project
dependency here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata.loaders import LoaderError, build_loader, loader_for
from marinedata.loaders.coralscop import CoralscopRleMaskLoader

FIXTURES = Path(__file__).parent / "fixtures" / "coralscop"


def _pixel_counts(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        from collections import Counter

        return Counter(im.tobytes())


def _write_doc(root: Path, stem: str, *, h: int, w: int, annotations: list[dict]) -> None:
    images = root / "images"
    jsons = root / "jsons"
    images.mkdir(parents=True, exist_ok=True)
    jsons.mkdir(parents=True, exist_ok=True)
    (images / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    doc = {
        "image": {"file_name": f"{stem}.jpg", "height": h, "width": w, "id": 0},
        "annotations": annotations,
    }
    (jsons / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")


def _encode_compressed(runs: list[int]) -> str:
    """Inverse of the loader's decoder — ``maskApi.c``'s ``rleToString``, used only to
    synthesise compressed-RLE fixtures for the round-trip tests."""
    out: list[str] = []
    for i, value in enumerate(runs):
        x = value - (runs[i - 2] if i > 2 else 0)
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def _runs_from_mask(mask) -> list[int]:
    """Column-major run-lengths of a 2D {0,1} array, background-first."""
    flat = mask.reshape(-1, order="F")
    runs: list[int] = []
    current = 0
    run_len = 0
    for v in flat.tolist():
        if v == current:
            run_len += 1
        else:
            runs.append(run_len)
            current = v
            run_len = 1
    runs.append(run_len)
    return runs


def _checker_mask(h: int, w: int):
    import numpy as np

    arr = np.zeros((h, w), dtype=np.uint8)
    arr[: h // 2, w // 2 :] = 1
    arr[h // 2 :, : w // 2] = 1
    return arr


def test_compressed_and_uncompressed_rle_agree(tmp_path: Path) -> None:
    """The same mask encoded as compressed-string vs uncompressed-list RLE decodes to
    the identical binary raster."""
    mask = _checker_mask(6, 8)
    runs = _runs_from_mask(mask)
    root_compressed = tmp_path / "compressed"
    root_uncompressed = tmp_path / "uncompressed"
    ann_compressed = {
        "id": 1,
        "image_id": 0,
        "area": int(mask.sum()),
        "segmentation": {"size": [6, 8], "counts": _encode_compressed(runs)},
        "bbox": [0, 0, 8, 6],
    }
    ann_uncompressed = {**ann_compressed, "segmentation": {"size": [6, 8], "counts": runs}}
    _write_doc(root_compressed, "img", h=6, w=8, annotations=[ann_compressed])
    _write_doc(root_uncompressed, "img", h=6, w=8, annotations=[ann_uncompressed])

    source = make_source("coco-rle-binary")
    s_compressed = next(iter(build_loader(source, root_compressed)))
    s_uncompressed = next(iter(build_loader(source, root_uncompressed)))

    counts_c = _pixel_counts(Path(s_compressed.mask))
    counts_u = _pixel_counts(Path(s_uncompressed.mask))
    assert counts_c == counts_u
    assert counts_c[1] == int(mask.sum())
    assert s_compressed.meta["raster_ignore_value"] == 0
    assert s_compressed.meta["mask_classes"] == {"0": "background", "1": "foreground"}


def test_overlapping_annotations_union(tmp_path: Path) -> None:
    """Two overlapping annotations union to their combined footprint, not double-counted."""
    import numpy as np

    h, w = 4, 4
    a = np.zeros((h, w), dtype=np.uint8)
    a[:, :3] = 1  # 12 px
    b = np.zeros((h, w), dtype=np.uint8)
    b[:, 1:] = 1  # 12 px, overlapping columns 1-2 with `a`
    expected_union = int((a | b).sum())  # 16, the full 4x4 grid

    root = tmp_path / "root"
    anns = [
        {
            "id": 1,
            "image_id": 0,
            "area": int(a.sum()),
            "segmentation": {"size": [h, w], "counts": _runs_from_mask(a)},
            "bbox": [0, 0, 3, 4],
        },
        {
            "id": 2,
            "image_id": 0,
            "area": int(b.sum()),
            "segmentation": {"size": [h, w], "counts": _runs_from_mask(b)},
            "bbox": [1, 0, 3, 4],
        },
    ]
    _write_doc(root, "img", h=h, w=w, annotations=anns)

    source = make_source("coco-rle-binary")
    sample = next(iter(build_loader(source, root)))
    counts = _pixel_counts(Path(sample.mask))
    assert counts[1] == expected_union
    assert sample.meta["num_annotations"] == 2


def test_segmentation_size_mismatch_raises(tmp_path: Path) -> None:
    """An annotation's declared RLE size disagreeing with the image size is a real
    format defect, not something to guess through."""
    root = tmp_path / "root"
    ann = {
        "id": 1,
        "image_id": 0,
        "area": 4,
        "segmentation": {"size": [3, 3], "counts": [9]},
        "bbox": [0, 0, 3, 3],
    }
    _write_doc(root, "img", h=4, w=4, annotations=[ann])

    source = make_source("coco-rle-binary")
    with pytest.raises(LoaderError, match="segmentation size"):
        list(build_loader(source, root))


def test_polygon_segmentation_raises(tmp_path: Path) -> None:
    """A polygon (list-of-points) segmentation is reported, not silently dropped."""
    root = tmp_path / "root"
    ann = {
        "id": 1,
        "image_id": 0,
        "area": 4,
        "segmentation": [[0, 0, 0, 2, 2, 2, 2, 0]],
        "bbox": [0, 0, 2, 2],
    }
    _write_doc(root, "img", h=4, w=4, annotations=[ann])

    source = make_source("coco-rle-binary")
    with pytest.raises(LoaderError, match="polygon"):
        list(build_loader(source, root))


def test_real_sample_json(tmp_path: Path) -> None:
    """Live bucket sample (00HKNAAI2KC52W4QK9NT, 11 objects, all compressed RLE):
    union pixel count matches a real ``pycocotools.mask.decode()`` cross-check."""
    root = tmp_path / "sample"
    shutil.copytree(FIXTURES / "sample", root)

    source = make_source("coco-rle-binary")
    samples = list(build_loader(source, root))
    assert len(samples) == 1
    sample = samples[0]
    counts = _pixel_counts(Path(sample.mask))
    assert counts[1] == 71_670
    assert sample.meta["num_annotations"] == 11


def test_coco_rle_binary_loader_still_registered() -> None:
    """The converter stays registered under `coco-rle-binary` even though no
    registry entry declares that layout anymore (`coralscop-masks-rs` moved to
    `staged-tree` once its staged tree was pinned, WSD S15b)."""
    assert loader_for("coco-rle-binary") is CoralscopRleMaskLoader


def test_registry_source_resolves_to_staged_tree() -> None:
    """`coralscop-masks-rs` moved to `staged-tree` once its staged tree was pinned
    (WSD S15b), same pattern as `reef-support-benthic-own` (WSD S8e)."""
    registry = Registry.load()
    source = registry.source("coralscop-masks-rs")
    assert source.loader is not None
    assert source.loader.layout == "staged-tree"
