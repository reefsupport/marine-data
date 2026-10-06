"""Coralseg (UCSD) R-channel mask converter (WSD S6x §2f / S7f).

The Coralseg mosaics ship one RGB PNG per image where the class lives entirely in the
red channel (0 Other, 1 Hard Coral, 2 Soft Coral — the ``Mask conversion`` line of
``s3://rs-storage-open/benthic_datasets/README.md``, D-AI3) and green/blue are always (legacy:)
zero — a convention distinct from :mod:`.labelbox`'s fill+outline palette, so it gets
its own tiny decode rather than overloading the labelbox LUT. Modelled on
:class:`marinedata.loaders.labelbox.LabelboxRgbMaskLoader`: decode once, cache the
derived indexed PNG next to the mask via
:func:`marinedata.normalise._encode_indexed_png`, and raise rather than clip on any
value the convention does not define.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..normalise import _encode_indexed_png
from ..sample import Sample
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader, _images_under, _require_pillow_and_numpy

_CLASSES = 3
_MASK_CLASSES = {"0": "Other", "1": "Hard Coral", "2": "Soft Coral"}


def _r_channel_indices(im, *, label: str) -> bytes:
    """Decode the red channel of a Coralseg mask into raw per-pixel indices.

    Raises if any observed red value is outside ``{0, 1, 2}`` — an unrecognised value here
    means the fixed 3-class convention this dataset was verified against (S6x §2f) does
    not hold for this file, and guessing which class it belongs to would silently
    corrupt the mask.
    """
    _Image, np = _require_pillow_and_numpy()
    arr = np.asarray(im.convert("RGB"))
    red = arr[:, :, 0]
    bad = red > 2
    if bad.any():
        y, x = (int(v) for v in np.argwhere(bad)[0])
        raise LoaderError(
            f"{label}: pixel at (x={x}, y={y}) has R={int(red[y, x])}, which is not in "
            "{0, 1, 2} for the coralseg R-channel convention — refusing to guess"
        )
    return red.astype(np.uint8).tobytes()


@register_loader
class CoralsegRMaskLoader(_HarmonizingLoader):
    """``Image/`` + ``Mask/`` pairs where the mask's red channel is the class index.

    Params:
        images_dir: default ``"Image"``
        masks_dir: default ``"Mask"``
        mask_suffix: appended to the image stem to find its mask, default ``""``
        converted_dir: where the derived indexed PNGs are cached, default
            ``"masks_converted"``
    """

    layout = "coralseg-r-channel"

    def _paths(self) -> tuple[Path, Path, Path]:
        images = self.root / str(self._param("images_dir", "Image"))
        masks = self.root / str(self._param("masks_dir", "Mask"))
        converted = self.root / str(self._param("converted_dir", "masks_converted"))
        return images, masks, converted

    def validate(self) -> None:
        super().validate()
        images, masks, _converted = self._paths()
        for name, path in (("images_dir", images), ("masks_dir", masks)):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'coralseg-r-channel' expects {name} at {path}"
                )

    def _iter_samples(self) -> Iterator[Sample]:
        images, masks, converted = self._paths()
        suffix = str(self._param("mask_suffix", ""))
        by_stem = {p.stem: p for p in masks.rglob("*") if p.is_file()}
        supervised = self.source.declared_supervision()

        for image in _images_under(images):
            mask = by_stem.get(f"{image.stem}{suffix}") or by_stem.get(image.stem)
            if mask is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(f"{self.source.id}: no mask for image '{image.name}' in {masks}")

            label = f"{self.source.id}/{image.name}"
            dest = converted / f"{image.stem}.png"
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image, _np = _require_pillow_and_numpy()
                with Image.open(mask) as im:
                    raw = _r_channel_indices(im, label=label)
                    indexed = Image.frombytes("L", im.size, raw)
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
                meta={"mask_is_dense": True, "mask_classes": _MASK_CLASSES},
            )
