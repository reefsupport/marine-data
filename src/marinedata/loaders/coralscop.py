"""CoralSCOP class-agnostic COCO-RLE mask converter (WSD S6x §2e).

CoralSCOP ships one COCO-style JSON per image (``{"image": {...}, "annotations": [...]}
``), each annotation carrying a ``segmentation`` RLE rather than a category — the masks
are SAM pseudo-labels over an unlabelled image pool, not ground-truth classes. There is
no ``pycocotools`` in this environment, so RLE is decoded with a small pure-numpy port
of ``maskApi.c``'s ``rleFrString`` (compressed, LEB128-like varint deltas) plus the
uncompressed list form; both were checked against a live ``pycocotools.mask.decode()``
on two real bucket samples during development (byte-identical union masks) but that
check is not repeatable in this repo since the dependency is absent.

All annotations for an image are unioned into a single binary raster (index 1 = any
CoralSCOP object, 0 = none) and cached as an indexed PNG via
:func:`marinedata.normalise._encode_indexed_png`, mirroring
:mod:`marinedata.loaders.coralseg`. Polygon segmentations (a list of point arrays
rather than an RLE dict) are not silently dropped: they raise, since this source has
none in the samples inspected and a change would be a real format shift worth seeing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..normalise import _encode_indexed_png
from ..sample import Sample
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader, _images_under, _require_pillow_and_numpy

_CLASSES = 2
_IGNORE_INDEX = 0
_MASK_CLASSES = {"0": "background", "1": "foreground"}


def _decode_compressed_counts(s: str) -> list[int]:
    """Pure-Python port of ``maskApi.c``'s ``rleFrString``: LEB128-like varints, each
    a delta from the count two positions back once the third value is reached."""
    cnts: list[int] = []
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def _runs_to_mask(runs: Sequence[int], h: int, w: int, *, label: str):
    """Column-major (Fortran-order) run-length expansion, alternating background/
    foreground starting with background — the COCO RLE convention."""
    _Image, np = _require_pillow_and_numpy()
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in runs:
        if run < 0:
            raise LoaderError(f"{label}: negative run length {run} in RLE counts")
        if value:
            flat[idx : idx + run] = 1
        idx += run
        value ^= 1
    if idx != total:
        raise LoaderError(
            f"{label}: RLE counts sum to {idx} pixels, expected {total} ({h}x{w})"
        )
    return flat.reshape((h, w), order="F")


def _decode_segmentation(segmentation: object, *, label: str):
    """Decode one annotation's ``segmentation`` into an ``{0,1}`` (h, w) array.

    Raises on a polygon (a list of point arrays) rather than dropping it — this
    source's samples are all RLE, so a polygon would mean the upstream export
    changed shape and needs a human, not a silent skip.
    """
    if isinstance(segmentation, list):
        raise LoaderError(
            f"{label}: segmentation is a polygon (list of point arrays), not an RLE "
            "dict — coco-rle-binary does not decode polygons"
        )
    if not isinstance(segmentation, dict):
        raise LoaderError(
            f"{label}: segmentation is {type(segmentation).__name__}, expected an RLE dict"
        )
    size = segmentation.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise LoaderError(f"{label}: segmentation has no valid 'size' [h, w]")
    h, w = int(size[0]), int(size[1])
    counts = segmentation.get("counts")
    if isinstance(counts, str):
        runs = _decode_compressed_counts(counts)
    elif isinstance(counts, list):
        runs = [int(c) for c in counts]
    else:
        raise LoaderError(
            f"{label}: segmentation 'counts' is {type(counts).__name__}, expected "
            "str (compressed) or list (uncompressed)"
        )
    return _runs_to_mask(runs, h, w, label=label)


@register_loader
class CoralscopRleMaskLoader(_HarmonizingLoader):
    """One COCO-RLE JSON per image, class-agnostic SAM pseudo-masks unioned to binary.

    Params:
        images_dir: default ``"images"``
        jsons_dir: default ``"jsons"``
        converted_dir: where the derived indexed PNGs are cached, default
            ``"masks_converted"``
    """

    layout = "coco-rle-binary"

    def _paths(self) -> tuple[Path, Path, Path]:
        images = self.root / str(self._param("images_dir", "images"))
        jsons = self.root / str(self._param("jsons_dir", "jsons"))
        converted = self.root / str(self._param("converted_dir", "masks_converted"))
        return images, jsons, converted

    def validate(self) -> None:
        super().validate()
        images, jsons, _converted = self._paths()
        for name, path in (("images_dir", images), ("jsons_dir", jsons)):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'coco-rle-binary' expects {name} at {path}"
                )

    def _iter_samples(self) -> Iterator[Sample]:
        images, jsons, converted = self._paths()
        by_stem = {p.stem: p for p in jsons.glob("*.json") if p.is_file()}
        supervised = self.source.declared_supervision()

        for image in _images_under(images):
            doc_path = by_stem.get(image.stem)
            if doc_path is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(
                    f"{self.source.id}: no COCO-RLE json for image '{image.name}' in {jsons}"
                )

            label = f"{self.source.id}/{image.name}"
            try:
                doc = json.loads(doc_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise LoaderError(f"{label}: malformed JSON at {doc_path} — {exc}") from exc

            image_meta = doc.get("image") or {}
            try:
                h, w = int(image_meta["height"]), int(image_meta["width"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LoaderError(
                    f"{label}: missing/invalid image height/width in {doc_path}"
                ) from exc

            annotations = doc.get("annotations") or []
            dest = converted / f"{image.stem}.png"
            if not dest.is_file():
                Image, np = _require_pillow_and_numpy()
                union = np.zeros((h, w), dtype=np.uint8)
                for ann in annotations:
                    segmentation = ann.get("segmentation")
                    size = segmentation.get("size") if isinstance(segmentation, dict) else None
                    if isinstance(size, (list, tuple)) and list(size) != [h, w]:
                        raise LoaderError(
                            f"{label}: annotation {ann.get('id')} segmentation size "
                            f"{list(size)} != image size [{h}, {w}]"
                        )
                    mask = _decode_segmentation(segmentation, label=label)
                    union |= mask
                dest.parent.mkdir(parents=True, exist_ok=True)
                indexed = Image.frombytes("L", (w, h), union.tobytes())
                _encode_indexed_png(indexed, dest, classes=_CLASSES, label=label)

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
                    "raster_ignore_value": _IGNORE_INDEX,
                    "mask_classes": _MASK_CLASSES,
                    "num_annotations": len(annotations),
                },
            )
