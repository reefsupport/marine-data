"""Generic layout readers.

Most marine datasets use one of a handful of on-disk conventions. Implementing each
convention once, and testing it against a synthetic fixture, gives every source that
declares it real coverage — which is the only way ~40 loaders can be honestly tested.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from ..harmonize import Harmonizer
from ..sample import Sample
from ..schema import Axis
from .base import LoaderError, SourceLoader, register_loader

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"})
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg"})


def _images_under(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


class _HarmonizingLoader(SourceLoader):
    """Shared plumbing for loaders that map native labels through a crosswalk."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._harmonizer: Harmonizer | None = None

    def bind_harmonizer(self, harmonizer: Harmonizer) -> None:
        """Attach a harmonizer. Without one, native labels pass through unmapped."""
        self._harmonizer = harmonizer

    def _resolve(self, native_label: str, axis: Axis = Axis.TAXON) -> tuple[dict, frozenset]:
        if self._harmonizer is None:
            from ..sample import LabelValue

            return {axis: LabelValue(node_id=native_label)}, frozenset({axis})
        result = self._harmonizer.map_label(native_label)
        return result.labels, result.supervised


@register_loader
class ImageFolderLoader(_HarmonizingLoader):
    """``root/<class_name>/<image>`` — one directory per class.

    The most common layout for classification sets (bleaching, species, coral health).

    Params:
        images_dir: subdirectory holding the per-class folders. Default: the root
            itself. Needed when an archive nests them under a top-level folder — e.g.
            Fish4Knowledge's tar extracts to ``fish_image/<class>/*.png``, not
            ``<class>/*.png`` directly.
    """

    layout = "image-folder"

    def _classes_dir(self) -> Path:
        sub = self._param("images_dir")
        return self.root / str(sub) if sub else self.root

    def validate(self) -> None:
        super().validate()
        classes_dir = self._classes_dir()
        if not classes_dir.is_dir():
            raise LoaderError(
                f"{self.source.id}: layout 'image-folder' expects images_dir at {classes_dir}"
            )
        subdirs = [d for d in classes_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise LoaderError(
                f"{self.source.id}: layout 'image-folder' expects one directory per class "
                f"under {classes_dir}, found none."
            )

    def _iter_samples(self) -> Iterator[Sample]:
        for class_dir in sorted(d for d in self._classes_dir().iterdir() if d.is_dir()):
            for image in _images_under(class_dir):
                labels, supervised = self._resolve(class_dir.name)
                yield Sample(
                    source_id=self.source.id,
                    key=self._relative(image),
                    image=image,
                    labels=labels,
                    supervised=supervised,
                    licence_tier=self.source.licence.tier,
                    split=self.split,
                    meta={"native_label": class_dir.name},
                )


@register_loader
class FlatImagesLoader(_HarmonizingLoader):
    """Unlabelled images, flat or nested, with no class structure.

    The right layout for pretraining corpora and for distributions whose annotations
    live elsewhere. Declaring this is more honest than declaring a labelled layout the
    files do not actually satisfy — the HuggingFace mirror of MOUSS, for instance, ships
    images only while the bounding boxes live in a separate release.

    Params:
        images_dir: subdirectory to read. Default: the root itself.
    """

    layout = "flat-images"

    def _images_dir(self) -> Path:
        sub = self._param("images_dir")
        return self.root / str(sub) if sub else self.root

    def _iter_samples(self) -> Iterator[Sample]:
        for image in _images_under(self._images_dir()):
            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                labels={},
                supervised=frozenset(),  # genuinely unsupervised, not "unknown"
                licence_tier=self.source.licence.tier,
                split=self.split,
            )


@register_loader
class ImageMaskPairLoader(_HarmonizingLoader):
    """Parallel ``images/`` and ``masks/`` directories with matching stems.

    Params:
        images_dir: default ``"images"``
        masks_dir: default ``"masks"``
        mask_suffix: appended to the stem, e.g. ``"_mask"``
    """

    layout = "image-mask-pairs"

    def _dirs(self) -> tuple[Path, Path]:
        images = self.root / str(self._param("images_dir", "images"))
        masks = self.root / str(self._param("masks_dir", "masks"))
        return images, masks

    def validate(self) -> None:
        super().validate()
        images, masks = self._dirs()
        for label, path in (("images_dir", images), ("masks_dir", masks)):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'image-mask-pairs' expects {label} at {path}"
                )

    def _iter_samples(self) -> Iterator[Sample]:
        images, masks = self._dirs()
        suffix = str(self._param("mask_suffix", ""))
        by_stem = {p.stem: p for p in masks.rglob("*") if p.is_file()}
        supervised = self.source.declared_supervision()

        for image in _images_under(images):
            mask = by_stem.get(f"{image.stem}{suffix}") or by_stem.get(image.stem)
            if mask is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                # In a full dataset a missing mask is a real defect, not a sample to
                # quietly skip.
                raise LoaderError(f"{self.source.id}: no mask for image '{image.name}' in {masks}")
            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                mask=mask,
                labels={},
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={"mask_is_dense": True},
            )


@register_loader
class DualConditionMaskLoader(_HarmonizingLoader):
    """Two mutually-exclusive binary masks per image, combined into one dense mask.

    Built for ``reef-support-bleaching``: the annotation tool exports "bleached" and
    "non-bleached" as two separate 0/255 rasters rather than one indexed mask, so
    ``ImageMaskPairLoader`` — which reads exactly one ``masks_dir`` — cannot represent
    it without discarding one side of the pair. Guessing which side to keep, or
    treating them as independent axes, would either drop half the labels or invent an
    overlap policy nobody asked for.

    Verified against real data (2026-08-28) that the two masks never mark the same
    pixel: ``positive`` and ``negative`` are a true partition, not two axes that could
    disagree. That is what makes a single combined raster the honest representation
    rather than a simplification — there is no ambiguity being thrown away. A source
    where the two masks *can* overlap needs a different, deliberate policy, not this
    loader; :meth:`_iter_samples` raises if it ever finds one, rather than silently
    picking a side.

    The combined raster (0 = unlabelled, 1 = positive, 2 = negative) is written once
    per pair into ``combined_dir`` and reused on subsequent loads; the two source mask
    directories are only ever read, never modified.

    Params:
        images_dir: default ``"images"``
        positive_dir: default ``"masks_bleached"``
        negative_dir: default ``"masks_non_bleached"``
        positive_suffix: appended to the image stem to find the positive mask,
            default ``"_bleached"``
        negative_suffix: default ``"_non_bleached"``
        combined_dir: default ``"masks_combined"``
    """

    layout = "dual-condition-masks"

    def _dirs(self) -> tuple[Path, Path, Path, Path]:
        images = self.root / str(self._param("images_dir", "images"))
        positive = self.root / str(self._param("positive_dir", "masks_bleached"))
        negative = self.root / str(self._param("negative_dir", "masks_non_bleached"))
        combined = self.root / str(self._param("combined_dir", "masks_combined"))
        return images, positive, negative, combined

    def validate(self) -> None:
        super().validate()
        images, positive, negative, _combined = self._dirs()
        for label, path in (
            ("images_dir", images),
            ("positive_dir", positive),
            ("negative_dir", negative),
        ):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'dual-condition-masks' expects {label} at {path}"
                )

    def _iter_samples(self) -> Iterator[Sample]:
        images, positive_dir, negative_dir, combined_dir = self._dirs()
        positive_suffix = str(self._param("positive_suffix", "_bleached"))
        negative_suffix = str(self._param("negative_suffix", "_non_bleached"))
        by_stem_positive = {p.stem: p for p in positive_dir.rglob("*") if p.is_file()}
        by_stem_negative = {p.stem: p for p in negative_dir.rglob("*") if p.is_file()}
        supervised = self.source.declared_supervision()

        for image in _images_under(images):
            positive = by_stem_positive.get(f"{image.stem}{positive_suffix}")
            negative = by_stem_negative.get(f"{image.stem}{negative_suffix}")
            missing = [
                name
                for name, mask in (("positive", positive), ("negative", negative))
                if mask is None
            ]
            if missing:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(
                    f"{self.source.id}: no {'/'.join(missing)} mask for image "
                    f"'{image.name}' in {positive_dir if 'positive' in missing else negative_dir}"
                )

            combined = combined_dir / f"{image.stem}.png"
            if not combined.is_file():
                combined.parent.mkdir(parents=True, exist_ok=True)
                _write_combined_condition_mask(
                    positive, negative, combined, source_id=self.source.id, key=image.name
                )

            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                mask=combined,
                labels={},
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={
                    "mask_is_dense": True,
                    "mask_values": {"0": "unlabelled", "1": "positive", "2": "negative"},
                },
            )


def _require_pillow_and_numpy():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Combining dual-condition masks needs Pillow and numpy: pip install 'marinedata[torch]'"
        ) from exc
    return Image, np


def _write_combined_condition_mask(
    positive: Path, negative: Path, out: Path, *, source_id: str, key: str
) -> None:
    """0 = unlabelled, 1 = positive, 2 = negative. Raises on any pixel marked in both —
    see :class:`DualConditionMaskLoader` for why that must not be silently resolved."""
    Image, _np = _require_pillow_and_numpy()

    positive_arr = _binary_mask_array(positive)
    negative_arr = _binary_mask_array(negative)
    if positive_arr.shape != negative_arr.shape:
        raise LoaderError(
            f"{source_id}/{key}: positive mask {positive_arr.shape} and negative mask "
            f"{negative_arr.shape} have different dimensions"
        )
    if (positive_arr & negative_arr).any():
        raise LoaderError(
            f"{source_id}/{key}: positive and negative masks overlap — this loader "
            f"assumes they partition the image; an overlapping pair needs a different, "
            f"deliberate policy, not a silent pick-one"
        )

    combined = positive_arr.astype("uint8") * 1 + negative_arr.astype("uint8") * 2
    Image.fromarray(combined, mode="L").save(out)


def _binary_mask_array(path: Path):
    Image, np = _require_pillow_and_numpy()
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) > 127


@register_loader
class CocoJsonLoader(_HarmonizingLoader):
    """COCO-style detection/segmentation JSON.

    Params:
        annotations: path to the JSON, relative to root. Default ``"annotations.json"``
        images_dir: image directory, relative to root. Default ``"images"``
    """

    layout = "coco-json"

    def _annotation_path(self) -> Path:
        return self.root / str(self._param("annotations", "annotations.json"))

    def validate(self) -> None:
        super().validate()
        path = self._annotation_path()
        if not path.is_file():
            raise LoaderError(f"{self.source.id}: COCO annotations not found at {path}")

    def _iter_samples(self) -> Iterator[Sample]:
        path = self._annotation_path()
        images_dir = self.root / str(self._param("images_dir", "images"))
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LoaderError(f"{self.source.id}: malformed COCO JSON at {path} — {exc}") from exc

        categories = {c["id"]: c["name"] for c in doc.get("categories", [])}
        by_image: dict[int, list[dict]] = {}
        for ann in doc.get("annotations", []):
            by_image.setdefault(ann["image_id"], []).append(ann)

        for image in doc.get("images", []):
            anns = by_image.get(image["id"], [])
            boxes = tuple(
                (float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][2]), float(a["bbox"][3]))
                for a in anns
                if "bbox" in a
            )
            native = [categories.get(a.get("category_id"), "unknown") for a in anns]
            labels, supervised = self._resolve(native[0]) if native else ({}, frozenset())
            yield Sample(
                source_id=self.source.id,
                key=image["file_name"],
                image=images_dir / image["file_name"],
                boxes=boxes,
                labels=labels,
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={"native_labels": native, "n_annotations": len(anns)},
            )


@register_loader
class LabelMeJsonLoader(_HarmonizingLoader):
    """LabelMe-style polygon annotations: one ``<stem>.json`` sidecar per image, in
    the same directory, holding ``shapes: [{label, points: [[x, y], ...]}]``.

    Built for #DeOlhoNosCorais: its real annotation format turned out to be this, not
    the ``image-mask-pairs`` PNG masks originally declared — found by actually opening
    a sidecar after reaching the data (see the source's ``verified_note``), the same
    class of surprise that ``docs/`` warns synthetic fixtures cannot catch.

    Polygons are summarised as their bounding box (``sample.boxes``) so this source is
    usable by the same box-based tooling as ``coco-json``/``yolo-txt``, without adding
    a first-class polygon field to :class:`Sample` for what only one source needs so
    far. Nothing is discarded: the full vertex list survives in
    ``meta["polygons"]`` for anyone who does want the exact shape.

    Params:
        images_dir: default the root itself (this layout is normally used per-
            partition, e.g. ``partition_glob: "*"`` over ``train``/``test``/``val``).
    """

    layout = "labelme-json"

    def _images_dir(self) -> Path:
        sub = self._param("images_dir")
        return self.root / str(sub) if sub else self.root

    def _iter_samples(self) -> Iterator[Sample]:
        for image in _images_under(self._images_dir()):
            sidecar = image.with_suffix(".json")
            if not sidecar.is_file():
                if self.partial:
                    continue
                raise LoaderError(f"{self.source.id}: no {sidecar.name} sidecar for {image.name}")
            try:
                doc = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise LoaderError(f"{self.source.id}: malformed LabelMe JSON at {sidecar}") from exc

            shapes = doc.get("shapes", [])
            boxes: list[tuple[float, float, float, float]] = []
            native: list[str] = []
            polygons: list[dict] = []
            for shape in shapes:
                points = shape.get("points") or []
                label = shape.get("label")
                if not points or not label:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
                native.append(label)
                polygons.append({"label": label, "points": points})

            resolved, supervised = self._resolve(native[0]) if native else ({}, frozenset())
            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                boxes=tuple(boxes),
                labels=resolved,
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={"native_labels": native, "polygons": polygons},
            )


@register_loader
class YoloTxtLoader(_HarmonizingLoader):
    """YOLO layout: ``images/`` and ``labels/`` with one ``.txt`` per image.

    Each line is ``class_id cx cy w h`` in normalised coordinates. Boxes are converted
    to absolute-free ``(cx, cy, w, h)`` and left normalised — callers know their own
    image dimensions, and decoding them here would force eager image reads.

    Params:
        images_dir: default ``"images"``
        labels_dir: default ``"labels"``
        names: comma-separated class names, indexed by class_id
    """

    layout = "yolo-txt"

    def _dirs(self) -> tuple[Path, Path]:
        return (
            self.root / str(self._param("images_dir", "images")),
            self.root / str(self._param("labels_dir", "labels")),
        )

    def _names(self) -> list[str]:
        raw = str(self._param("names", ""))
        return [part.strip() for part in raw.split(",") if part.strip()]

    def validate(self) -> None:
        super().validate()
        images, labels = self._dirs()
        for label, path in (("images_dir", images), ("labels_dir", labels)):
            if not path.is_dir():
                raise LoaderError(f"{self.source.id}: layout 'yolo-txt' expects {label} at {path}")

    def _iter_samples(self) -> Iterator[Sample]:
        images, labels_dir = self._dirs()
        names = self._names()

        for image in _images_under(images):
            label_file = labels_dir / f"{image.stem}.txt"
            boxes: list[tuple[float, float, float, float]] = []
            native: list[str] = []

            if label_file.is_file():
                for lineno, line in enumerate(
                    label_file.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) < 5:
                        raise LoaderError(
                            f"{self.source.id}: malformed YOLO label at "
                            f"{label_file}:{lineno} — expected 5 fields, got {len(parts)}"
                        )
                    class_id = int(parts[0])
                    boxes.append(tuple(float(v) for v in parts[1:5]))  # type: ignore[arg-type]
                    native.append(names[class_id] if class_id < len(names) else str(class_id))

            resolved, supervised = self._resolve(native[0]) if native else ({}, frozenset())
            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                boxes=tuple(boxes),
                labels=resolved,
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={"native_labels": native, "normalised_boxes": True},
            )


@register_loader
class CsvPointsLoader(_HarmonizingLoader):
    """CoralNet-style sparse point annotations in CSV.

    Params:
        annotations: CSV path relative to root. Default ``"annotations.csv"``
        images_dir: default ``"images"``
        image_column / row_column / column_column / label_column
    """

    layout = "csv-points"

    def _annotation_path(self) -> Path:
        return self.root / str(self._param("annotations", "annotations.csv"))

    def validate(self) -> None:
        super().validate()
        if not self._annotation_path().is_file():
            raise LoaderError(f"{self.source.id}: point CSV not found at {self._annotation_path()}")

    def _iter_samples(self) -> Iterator[Sample]:
        path = self._annotation_path()
        images_dir = self.root / str(self._param("images_dir", "images"))
        img_col = str(self._param("image_column", "Name"))
        row_col = str(self._param("row_column", "Row"))
        col_col = str(self._param("column_column", "Column"))
        lbl_col = str(self._param("label_column", "Label"))

        grouped: dict[str, list[tuple[float, float, str]]] = {}
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {img_col, row_col, col_col, lbl_col} - set(reader.fieldnames or [])
            if missing:
                raise LoaderError(
                    f"{self.source.id}: point CSV {path} is missing columns "
                    f"{sorted(missing)}; found {reader.fieldnames}"
                )
            for record in reader:
                grouped.setdefault(record[img_col], []).append(
                    (float(record[row_col]), float(record[col_col]), record[lbl_col])
                )

        for name, entries in sorted(grouped.items()):
            points = tuple((r, c) for r, c, _ in entries)
            native = [lbl for _, _, lbl in entries]
            labels, supervised = self._resolve(native[0]) if native else ({}, frozenset())
            yield Sample(
                source_id=self.source.id,
                key=name,
                image=images_dir / name,
                points=points,
                labels=labels,
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={"native_labels": native, "n_points": len(points)},
            )


@register_loader
class AudioClipsLoader(_HarmonizingLoader):
    """``root/<class_name>/<clip>`` for bioacoustic sets.

    Params:
        images_dir: subdirectory holding the per-class folders. Default: the root
            itself. Same purpose as the identically-named param on ``image-folder`` and
            ``image-mask-pairs`` — an archive's real content is often nested under a
            top-level folder (the Watkins "best of" mirror extracts to
            ``watkins_best_of_whales/<species>/sound/*.wav``, so clips are found via
            ``rglob`` under each species directory rather than requiring them flat).
    """

    layout = "audio-clips"

    def _classes_dir(self) -> Path:
        sub = self._param("images_dir")
        return self.root / str(sub) if sub else self.root

    def _iter_samples(self) -> Iterator[Sample]:
        for class_dir in sorted(d for d in self._classes_dir().iterdir() if d.is_dir()):
            for clip in sorted(class_dir.rglob("*")):
                if not clip.is_file() or clip.suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                labels, supervised = self._resolve(class_dir.name)
                yield Sample(
                    source_id=self.source.id,
                    key=self._relative(clip),
                    audio=clip,
                    labels=labels,
                    supervised=supervised,
                    licence_tier=self.source.licence.tier,
                    split=self.split,
                    meta={"native_label": class_dir.name},
                )


@register_loader
class MetadataOnlyLoader(SourceLoader):
    """For reference material that is not a training set.

    Taxonomic authorities, occurrence databases, satellite products consumed as context
    layers. Declaring this explicitly is better than leaving `loader` unset, because it
    distinguishes "deliberately not loadable" from "nobody has written the adapter yet".
    """

    layout = "metadata-only"

    def validate(self) -> None:
        return

    def __iter__(self) -> Iterator[Sample]:
        # Bypass the base root checks: "this is not a training set" is the useful
        # message, not "the directory you did not need is missing".
        return self._iter_samples()

    def _iter_samples(self) -> Iterator[Sample]:
        raise LoaderError(
            f"{self.source.id} is registered as metadata-only — a reference or context "
            f"layer, not a training set. It has no sample-level loader by design."
        )
