"""Loader tests.

Two layers:

1. **Functional** — each layout is exercised against a synthetic fixture. Every source
   declaring that layout inherits the coverage, which is the only way ~40 loaders get
   tested without terabytes of downloads.
2. **Contract, per source** — every registered source is checked for a resolvable
   layout, coherent params, a constructible loader, and (where declared) a crosswalk
   that actually builds. These run over the real registry, so a broken entry fails CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _touch_image, make_source

from marinedata import Registry
from marinedata.loaders import (
    DataNotAvailable,
    LoaderError,
    build_loader,
    registered_layouts,
)
from marinedata.schema import Axis

# ── functional: one test per layout ───────────────────────────────────────


def test_image_folder(image_folder_root: Path) -> None:
    loader = build_loader(make_source("image-folder"), image_folder_root)
    samples = list(loader)
    assert len(samples) == 5
    assert {s.meta["native_label"] for s in samples} == {"Hard Coral", "Soft Coral"}
    assert all(Axis.TAXON in s.supervised for s in samples)
    assert all(s.image is not None for s in samples)


def test_image_folder_with_nested_images_dir(tmp_path: Path) -> None:
    """Fish4Knowledge's tar extracts to fish_image/<class>/*.png, not <class>/*.png."""
    root = tmp_path / "f4k"
    for name, count in (("fish_01", 2), ("fish_02", 1)):
        for i in range(count):
            _touch = root / "fish_image" / name / f"img_{i}.jpg"
            _touch.parent.mkdir(parents=True, exist_ok=True)
            _touch.write_bytes(b"\x89PNG")
    loader = build_loader(make_source("image-folder", {"images_dir": "fish_image"}), root)
    samples = list(loader)
    assert len(samples) == 3
    assert {s.meta["native_label"] for s in samples} == {"fish_01", "fish_02"}


def test_image_folder_rejects_flat_directory(tmp_path: Path) -> None:
    (tmp_path / "loose.jpg").write_bytes(b"\x89PNG")
    loader = build_loader(make_source("image-folder"), tmp_path)
    with pytest.raises(LoaderError, match="one directory per class"):
        list(loader)


def test_image_mask_pairs(image_mask_root: Path) -> None:
    loader = build_loader(make_source("image-mask-pairs"), image_mask_root)
    samples = list(loader)
    assert len(samples) == 3
    assert all(s.mask is not None for s in samples)
    assert all(Path(s.mask).stem == Path(s.image).stem for s in samples)


def test_image_mask_pairs_missing_mask_is_an_error(image_mask_root: Path) -> None:
    """A missing mask in a segmentation set is a defect, not a sample to skip."""
    next((image_mask_root / "masks").iterdir()).unlink()
    loader = build_loader(make_source("image-mask-pairs"), image_mask_root)
    with pytest.raises(LoaderError, match="no mask for image"):
        list(loader)


def test_image_mask_pairs_with_no_crosswalk_is_genuinely_unsupervised(
    image_mask_root: Path,
) -> None:
    """A dense mask with `supervises: []` (deepfish, uieb, ...) must not claim TAXON.

    Found while verifying the "unlabelled data already flows through correctly" claim:
    this loader used to hardcode `supervised={TAXON}` regardless of what the source
    actually declares, so a source with a segmentation mask but no crosswalk (real
    examples: deepfish, uieb, squid, suim, atlantis-synthetic-depth) silently reported
    100% TAXON supervision in `supervision_coverage()` and every shard sidecar, with an
    empty `labels` dict underneath. Encoding happened to still yield IGNORE_INDEX (since
    `index_of(None)` does that too), so no loss curve ever caught it — exactly the class
    of silent defect this codebase's design principles warn about.
    """
    from marinedata.models import Annotation

    source = make_source(
        "image-mask-pairs",
        annotations=(Annotation(kind="dense-mask", supervises=()),),
    )
    samples = list(build_loader(source, image_mask_root))
    assert all(s.supervised == frozenset() for s in samples)
    assert all(s.labels == {} for s in samples)


def test_image_mask_pairs_respects_declared_supervised_axes(image_mask_root: Path) -> None:
    """The complement: a source that DOES declare supervision keeps it."""
    from marinedata.models import Annotation

    source = make_source(
        "image-mask-pairs",
        annotations=(Annotation(kind="dense-mask", supervises=("taxon", "condition")),),
    )
    samples = list(build_loader(source, image_mask_root))
    assert all(s.supervised == frozenset({Axis.TAXON, Axis.CONDITION}) for s in samples)


@pytest.fixture
def dual_condition_root(tmp_path: Path) -> Path:
    np = pytest.importorskip("numpy")
    PIL_Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "dual-condition"
    for i in range(2):
        _touch_image(root / "images" / f"f{i}.jpg")
        positive = np.zeros((4, 4), dtype="uint8")
        positive[0, 0] = 255
        negative = np.zeros((4, 4), dtype="uint8")
        negative[1, 1] = 255
        (root / "masks_bleached").mkdir(parents=True, exist_ok=True)
        (root / "masks_non_bleached").mkdir(parents=True, exist_ok=True)
        PIL_Image.fromarray(positive, mode="L").save(root / "masks_bleached" / f"f{i}_bleached.png")
        PIL_Image.fromarray(negative, mode="L").save(
            root / "masks_non_bleached" / f"f{i}_non_bleached.png"
        )
    return root


def test_dual_condition_masks_combines_into_one_raster(dual_condition_root: Path) -> None:
    np = pytest.importorskip("numpy")
    PIL_Image = pytest.importorskip("PIL.Image")
    samples = list(build_loader(make_source("dual-condition-masks"), dual_condition_root))
    assert len(samples) == 2
    for sample in samples:
        combined = np.asarray(PIL_Image.open(sample.mask))
        assert combined[0, 0] == 1  # positive
        assert combined[1, 1] == 2  # negative
        assert combined[2, 2] == 0  # unlabelled
    assert (dual_condition_root / "masks_combined" / "f0.png").is_file()


def test_dual_condition_masks_meta_uses_configured_labels(dual_condition_root: Path) -> None:
    """A crosswalk needs real words to key on, not the generic positive/negative."""
    source = make_source(
        "dual-condition-masks",
        {"positive_label": "bleached", "negative_label": "non_bleached"},
    )
    samples = list(build_loader(source, dual_condition_root))
    assert samples[0].meta["mask_values"] == {
        "0": "unlabelled",
        "1": "bleached",
        "2": "non_bleached",
    }


def test_dual_condition_masks_respects_declared_supervised_axes(dual_condition_root: Path) -> None:
    from marinedata.models import Annotation

    source = make_source(
        "dual-condition-masks",
        annotations=(Annotation(kind="dense-mask", supervises=("condition",)),),
    )
    samples = list(build_loader(source, dual_condition_root))
    assert all(s.supervised == frozenset({Axis.CONDITION}) for s in samples)


def test_dual_condition_masks_rejects_overlap(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    PIL_Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "overlap"
    _touch_image(root / "images" / "f0.jpg")
    overlap = np.zeros((4, 4), dtype="uint8")
    overlap[0, 0] = 255
    (root / "masks_bleached").mkdir(parents=True)
    (root / "masks_non_bleached").mkdir(parents=True)
    PIL_Image.fromarray(overlap, mode="L").save(root / "masks_bleached" / "f0_bleached.png")
    PIL_Image.fromarray(overlap, mode="L").save(root / "masks_non_bleached" / "f0_non_bleached.png")

    loader = build_loader(make_source("dual-condition-masks"), root)
    with pytest.raises(LoaderError, match="overlap"):
        list(loader)


def test_dual_condition_masks_missing_pair_is_an_error(dual_condition_root: Path) -> None:
    next((dual_condition_root / "masks_bleached").iterdir()).unlink()
    loader = build_loader(make_source("dual-condition-masks"), dual_condition_root)
    with pytest.raises(LoaderError, match="no positive mask"):
        list(loader)


def test_coco_json(coco_root: Path) -> None:
    loader = build_loader(make_source("coco-json"), coco_root)
    samples = list(loader)
    assert len(samples) == 2
    first = next(s for s in samples if s.key == "im0.jpg")
    assert len(first.boxes) == 2
    assert first.meta["native_labels"] == ["Hard Coral", "fish"]


def test_coco_json_malformed(coco_root: Path) -> None:
    (coco_root / "annotations.json").write_text("{not json", encoding="utf-8")
    loader = build_loader(make_source("coco-json"), coco_root)
    with pytest.raises(LoaderError, match="malformed COCO JSON"):
        list(loader)


@pytest.fixture
def labelme_root(tmp_path: Path) -> Path:
    import json as _json

    root = tmp_path / "labelme"
    _touch_image(root / "a.jpg")
    (root / "a.json").write_text(
        _json.dumps(
            {
                "shapes": [
                    {
                        "label": "Mussismilia hispida",
                        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    },
                    {"label": "Siderastrea stellata", "points": [[20, 20], [30, 20], [30, 30]]},
                ]
            }
        )
    )
    _touch_image(root / "b.jpg")
    (root / "b.json").write_text(_json.dumps({"shapes": []}))
    return root


def test_labelme_json(labelme_root: Path) -> None:
    samples = list(build_loader(make_source("labelme-json"), labelme_root))
    assert len(samples) == 2
    a = next(s for s in samples if s.key.endswith("a.jpg"))
    assert len(a.boxes) == 2
    assert a.boxes[0] == (0, 0, 10, 10)
    assert a.meta["native_labels"] == ["Mussismilia hispida", "Siderastrea stellata"]
    assert len(a.meta["polygons"]) == 2
    assert a.meta["polygons"][0]["points"] == [[0, 0], [10, 0], [10, 10], [0, 10]]

    b = next(s for s in samples if s.key.endswith("b.jpg"))
    assert b.boxes == ()
    assert b.supervised == frozenset()


def test_labelme_json_missing_sidecar_is_an_error(labelme_root: Path) -> None:
    (labelme_root / "a.json").unlink()
    loader = build_loader(make_source("labelme-json"), labelme_root)
    with pytest.raises(LoaderError, match=r"no a\.json sidecar"):
        list(loader)


def test_labelme_json_malformed_sidecar(labelme_root: Path) -> None:
    (labelme_root / "a.json").write_text("{not json")
    loader = build_loader(make_source("labelme-json"), labelme_root)
    with pytest.raises(LoaderError, match="malformed LabelMe JSON"):
        list(loader)


def test_yolo_txt(yolo_root: Path) -> None:
    source = make_source("yolo-txt", {"names": "coral,fish"})
    samples = list(build_loader(source, yolo_root))
    assert len(samples) == 2
    f0 = next(s for s in samples if s.key.endswith("f0.jpg"))
    assert len(f0.boxes) == 2
    assert f0.meta["native_labels"] == ["coral", "fish"]
    assert f0.meta["normalised_boxes"] is True


def test_yolo_txt_malformed_line(yolo_root: Path) -> None:
    (yolo_root / "labels" / "f0.txt").write_text("0 0.5 0.5\n")
    loader = build_loader(make_source("yolo-txt"), yolo_root)
    with pytest.raises(LoaderError, match="malformed YOLO label"):
        list(loader)


def test_csv_points(csv_points_root: Path) -> None:
    samples = list(build_loader(make_source("csv-points"), csv_points_root))
    assert len(samples) == 2
    a = next(s for s in samples if s.key == "a.jpg")
    assert a.meta["n_points"] == 2
    assert a.points == ((10.0, 20.0), (30.0, 40.0))


def test_csv_points_missing_column(csv_points_root: Path) -> None:
    (csv_points_root / "annotations.csv").write_text("Name,Row\na.jpg,1\n", encoding="utf-8")
    loader = build_loader(make_source("csv-points"), csv_points_root)
    with pytest.raises(LoaderError, match="missing columns"):
        list(loader)


def test_labelbox_ndjson(labelbox_root: Path) -> None:
    source = make_source("labelbox-ndjson", {"geometry_labels": "SCALE"})
    samples = list(build_loader(source, labelbox_root))
    assert len(samples) == 2

    s1 = next(s for s in samples if s.key == "s1.jpg")
    assert s1.meta["native_labels"] == ["Hard Coral", "Hard Coral", "Soft Coral"]
    assert s1.meta["geometry_labels"] == []

    s2 = next(s for s in samples if s.key == "s2.jpg")
    assert s2.meta["geometry_labels"] == ["SCALE"]
    assert s2.meta["native_labels"] == []
    # A scale bar is geometry, not biota — it must not become a taxon label.
    assert s2.labels == {}


def test_audio_clips(audio_root: Path) -> None:
    samples = list(build_loader(make_source("audio-clips"), audio_root))
    assert len(samples) == 2
    assert all(s.audio is not None for s in samples)
    assert {s.meta["native_label"] for s in samples} == {"fish_chorus", "snapping_shrimp"}


def test_audio_clips_with_nested_images_dir_and_extra_files(tmp_path: Path) -> None:
    """The Watkins IA mirror extracts to <top>/<species>/sound/*.wav plus a sibling
    metadata/*.csv per species — rglob must find clips and ignore the metadata."""
    root = tmp_path / "watkins"
    for species in ("orca", "beluga"):
        (root / "top" / species / "sound").mkdir(parents=True)
        (root / "top" / species / "sound" / "clip1.wav").write_bytes(b"RIFF")
        (root / "top" / species / "metadata").mkdir(parents=True)
        (root / "top" / species / "metadata" / "info.csv").write_text("x")
    samples = list(build_loader(make_source("audio-clips", {"images_dir": "top"}), root))
    assert len(samples) == 2
    assert {s.meta["native_label"] for s in samples} == {"orca", "beluga"}


def test_metadata_only_refuses_clearly(tmp_path: Path) -> None:
    loader = build_loader(make_source("metadata-only"), tmp_path)
    with pytest.raises(LoaderError, match="metadata-only"):
        list(loader)


# ── absence vs malformation ───────────────────────────────────────────────


def test_missing_root_says_fetch_it(tmp_path: Path) -> None:
    loader = build_loader(make_source("image-folder"), tmp_path / "nope")
    with pytest.raises(DataNotAvailable, match="Fetch it from"):
        list(loader)


def test_empty_root_is_data_not_available(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = build_loader(make_source("image-folder"), empty)
    with pytest.raises(DataNotAvailable, match="empty"):
        list(loader)


def test_unknown_layout_lists_known_ones() -> None:
    with pytest.raises(LoaderError, match="No loader registered"):
        build_loader(make_source("does-not-exist"), "/tmp")


# ── contract, per registered source ───────────────────────────────────────


def _sources_with_loaders(reg: Registry):
    return [s for s in reg if s.loader is not None]


def test_every_source_declares_a_loader(registry: Registry) -> None:
    """A source with no loader is an unfinished entry. metadata-only is the honest
    way to say 'this is a reference, not a training set'."""
    missing = [s.id for s in registry if s.loader is None]
    assert not missing, f"sources with no loader declared: {missing}"


@pytest.mark.parametrize("source_id", [s.id for s in Registry.load()])
def test_source_layout_is_registered(registry: Registry, source_id: str) -> None:
    source = registry.source(source_id)
    assert source.loader is not None
    assert source.loader.layout in registered_layouts(), (
        f"{source_id} declares unknown layout '{source.loader.layout}'"
    )


@pytest.mark.parametrize("source_id", [s.id for s in Registry.load()])
def test_source_loader_constructs(registry: Registry, source_id: str, tmp_path: Path) -> None:
    """Construction must succeed without the data present."""
    loader = build_loader(registry.source(source_id), tmp_path / source_id)
    assert loader.source.id == source_id


@pytest.mark.parametrize(
    "source_id",
    [s.id for s in Registry.load() if s.loader and s.loader.crosswalk_id],
)
def test_declared_crosswalk_builds(registry: Registry, source_id: str) -> None:
    """A crosswalk that targets a missing node must fail here, not at epoch 1."""
    harmonizer = registry.harmonizer_for(source_id)
    assert harmonizer is not None
    assert harmonizer.coverage_report()


def test_layouts_have_at_least_one_source(registry: Registry) -> None:
    """Flags a layout implemented but never used — usually a rename that half-landed."""
    declared = {s.loader.layout for s in _sources_with_loaders(registry)}
    orphans = set(registered_layouts()) - declared
    assert not orphans - {"metadata-only"}, f"layouts with no sources: {sorted(orphans)}"
