"""Segments.ai instance-id mask converter (WSD S7e / S7f).

``labelled_data/segments/*_label_ground-truth.png`` is a Segments.ai SDK cache dump:
each PNG is mode ``I;16``/``I``, pixel value = annotation *instance id* (0 = none), not
a class index. ``coral_reef_annotated_images-v0.4.json`` is the matching Segments.ai
release: it maps each sample's instance ids to a ``category_id``, and
``task_attributes.categories`` names those categories. This loader remaps id -> class
per image and caches the result exactly like
:class:`marinedata.loaders.labelbox.LabelboxRgbMaskLoader` and
:class:`marinedata.loaders.coralseg.CoralsegRMaskLoader` do — decode once, write an
indexed PNG via :func:`marinedata.normalise._encode_indexed_png`, reuse it on repeat
loads.

Wiring only: which source these rows should live under is an open provenance question
(S7e §Open), so this loader is declared but not yet attached to a registry entry.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..normalise import _encode_indexed_png
from ..sample import Sample
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader, _images_under, _require_pillow_and_numpy

_UNLABELLED_CLASS_NAME = "unlabelled"


class _Categories:
    """id -> name table plus the derived class count, read once per loader run."""

    def __init__(self, raw: list[dict]) -> None:
        self.name_by_id: dict[int, str] = {int(c["id"]): str(c["name"]) for c in raw}
        self.classes = (max(self.name_by_id, default=-1)) + 2  # +1 unlabelled, +1 count->size

    def mask_classes(self) -> dict[str, str]:
        out = {"0": _UNLABELLED_CLASS_NAME}
        out.update({str(cat_id + 1): name for cat_id, name in self.name_by_id.items()})
        return out


def _load_dataset(path: Path) -> tuple[_Categories, dict[str, dict]]:
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    dataset = doc.get("dataset") or {}
    categories = _Categories((dataset.get("task_attributes") or {}).get("categories", []))
    by_stem = {Path(s["name"]).stem: s for s in dataset.get("samples", []) if s.get("name")}
    return categories, by_stem


def _id_to_class_lut(annotations: list[dict], categories: _Categories, *, label: str):
    """Build a ``numpy`` LUT: pixel instance id -> class index (``category_id + 1``)."""
    _Image, np = _require_pillow_and_numpy()
    max_id = max((int(a["id"]) for a in annotations), default=0)
    lut = np.zeros(max_id + 1, dtype=np.int64)
    for ann in annotations:
        ann_id = int(ann["id"])
        category_id = int(ann["category_id"])
        if category_id not in categories.name_by_id:
            raise LoaderError(
                f"{label}: annotation id {ann_id} has category_id {category_id}, which "
                "is not in the v0.4 categories table"
            )
        lut[ann_id] = category_id + 1
    return lut


@register_loader
class SegmentsAiInstanceMaskLoader(_HarmonizingLoader):
    """``ATL/`` images + ``labelled_data/segments/`` instance-id masks, remapped via
    ``labelled_data/coral_reef_annotated_images-v0.4.json``.

    Params:
        images_dir: default ``"ATL"``
        masks_dir: default ``"labelled_data/segments"``
        mask_suffix: appended to the image stem to find its mask, default
            ``"_label_ground-truth"``
        annotations: v0.4 JSON path relative to root, default
            ``"labelled_data/coral_reef_annotated_images-v0.4.json"``
        converted_dir: where the derived indexed PNGs are cached, default
            ``"masks_converted"``
    """

    layout = "segmentsai-instance"

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        images = self.root / str(self._param("images_dir", "ATL"))
        masks = self.root / str(self._param("masks_dir", "labelled_data/segments"))
        converted = self.root / str(self._param("converted_dir", "masks_converted"))
        annotations = self.root / str(
            self._param("annotations", "labelled_data/coral_reef_annotated_images-v0.4.json")
        )
        return images, masks, converted, annotations

    def validate(self) -> None:
        super().validate()
        images, masks, _converted, annotations = self._paths()
        for name, path in (("images_dir", images), ("masks_dir", masks)):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'segmentsai-instance' expects {name} at {path}"
                )
        if not annotations.is_file():
            raise LoaderError(f"{self.source.id}: v0.4 JSON not found at {annotations}")

    def _iter_samples(self) -> Iterator[Sample]:
        images, masks, converted, annotations_path = self._paths()
        suffix = str(self._param("mask_suffix", "_label_ground-truth"))
        by_stem = {p.stem: p for p in masks.rglob("*") if p.is_file()}
        categories, samples_by_stem = _load_dataset(annotations_path)
        supervised = self.source.declared_supervision()

        for image in _images_under(images):
            stem = image.stem
            mask = by_stem.get(f"{stem}{suffix}") or by_stem.get(stem)
            if mask is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(f"{self.source.id}: no mask for image '{image.name}' in {masks}")

            record = samples_by_stem.get(stem)
            label_entry = (record or {}).get("labels", {}).get("ground-truth")
            if record is None or label_entry is None:
                if self.partial:
                    continue
                raise LoaderError(
                    f"{self.source.id}: no v0.4 JSON sample for '{stem}' in {annotations_path}"
                )

            label = f"{self.source.id}/{image.name}"
            ann_list = (label_entry.get("attributes") or {}).get("annotations", [])
            lut = _id_to_class_lut(ann_list, categories, label=label)

            # The raw instance-id array is re-read and re-validated on every load (unlike
            # LabelboxRgbMaskLoader's decode cache): `objects_without_pixels` needs the
            # per-instance id set, which only the raw mask carries — the cached indexed
            # PNG is class-level and collapses several ids into one value. Only the
            # (comparatively expensive) indexed-PNG *write* is skipped when cached.
            Image, np = _require_pillow_and_numpy()
            with Image.open(mask) as im:
                if im.mode not in {"I;16", "I"}:
                    raise LoaderError(
                        f"{label}: mode {im.mode!r} is not a supported segments.ai "
                        "instance mask mode (need 'I;16' or 'I')"
                    )
                with Image.open(image) as img:
                    if im.size != img.size:
                        raise LoaderError(
                            f"{label}: mask size {im.size} does not match ATL image size {img.size}"
                        )
                arr = np.asarray(im)
                raw_ids = frozenset(int(i) for i in np.unique(arr))
                unknown = sorted(i for i in raw_ids if i >= len(lut))
                if unknown:
                    raise LoaderError(
                        f"{label}: pixel id(s) {unknown} have no annotation entry in the v0.4 JSON"
                    )
                dest = converted / f"{stem}.png"
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    out = lut[arr].astype(np.uint8)
                    indexed = Image.frombytes("L", im.size, out.tobytes())
                    _encode_indexed_png(indexed, dest, classes=categories.classes, label=label)

            annotated_ids = {int(a["id"]) for a in ann_list}
            objects_without_pixels = sum(1 for i in annotated_ids if i not in raw_ids)

            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                mask=dest,
                labels={},
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={
                    "mask_is_dense": True,
                    "raster_ignore_value": 0,
                    "mask_classes": categories.mask_classes(),
                    "objects_without_pixels": objects_without_pixels,
                },
            )
