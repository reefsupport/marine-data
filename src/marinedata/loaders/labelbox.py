"""Labelbox NDJSON export reader.

The format Reef Support's own annotations ship in: one JSON object per line, with
projects → labels → annotations → objects. Objects carry either a mask reference or a
geometry (line, polygon, point).

Worth noting for anyone reusing this: the export nests three levels deep before reaching
anything useful, and the same file mixes annotation *kinds* — instance masks and scale
lines sit side by side under `objects`. Splitting them by `annotation_kind` rather than
assuming homogeneity is the difference between a working loader and one that silently
treats a scale bar as a coral colony.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..normalise import _encode_indexed_png
from ..sample import Sample
from ..schema import Axis
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader, _images_under, _require_pillow_and_numpy


@register_loader
class LabelboxNdjsonLoader(_HarmonizingLoader):
    """Labelbox v2 NDJSON export.

    Params:
        annotations: NDJSON path relative to root. Default ``"export-result.ndjson"``
        images_dir: default ``"images"``
        geometry_labels: comma-separated label names that are geometry rather than
            biota — e.g. ``"SCALE"``. These are emitted with their geometry but
            contribute no taxon supervision.
    """

    layout = "labelbox-ndjson"

    def _annotation_path(self) -> Path:
        return self.root / str(self._param("annotations", "export-result.ndjson"))

    def validate(self) -> None:
        super().validate()
        path = self._annotation_path()
        if not path.is_file():
            raise LoaderError(f"{self.source.id}: Labelbox export not found at {path}")

    def _geometry_labels(self) -> frozenset[str]:
        raw = str(self._param("geometry_labels", ""))
        return frozenset(part.strip() for part in raw.split(",") if part.strip())

    def _iter_samples(self) -> Iterator[Sample]:
        path = self._annotation_path()
        images_dir = self.root / str(self._param("images_dir", "images"))
        geometry_labels = self._geometry_labels()

        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LoaderError(
                        f"{self.source.id}: malformed NDJSON at {path}:{lineno} — {exc}"
                    ) from exc

                name = (doc.get("data_row") or {}).get("external_id") or doc.get("external_id")
                if not name:
                    continue

                biota: list[str] = []
                geometry: list[str] = []
                for project in (doc.get("projects") or {}).values():
                    for label in project.get("labels", []):
                        for obj in (label.get("annotations") or {}).get("objects", []):
                            obj_name = obj.get("name")
                            if not obj_name:
                                continue
                            (geometry if obj_name in geometry_labels else biota).append(obj_name)

                if not biota and not geometry:
                    continue

                labels, supervised = ({}, frozenset())
                if biota:
                    labels, supervised = self._resolve(biota[0], Axis.TAXON)
                elif geometry:
                    # Geometry-only frames are still supervised for taxon — the
                    # annotator looked and found no biota — but carry no label.
                    supervised = frozenset({Axis.TAXON})

                yield Sample(
                    source_id=self.source.id,
                    key=name,
                    image=images_dir / name,
                    labels=labels,
                    supervised=supervised,
                    licence_tier=self.source.licence.tier,
                    split=self.split,
                    meta={
                        "native_labels": biota,
                        "geometry_labels": geometry,
                        "n_objects": len(biota) + len(geometry),
                    },
                )


# ── labelbox-rgb: stitched RGB mask → indexed PNG (WSD S6x §2a) ────────────

_RGB_LUT: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,  # background / unlabelled
    (255, 0, 0): 1,  # Hard Coral — fill
    (255, 255, 0): 1,  # Hard Coral — outline
    (0, 0, 255): 2,  # Soft Coral — fill
    (255, 165, 0): 2,  # Soft Coral — outline
}
"""Fill+outline colour per class, verified against the NDJSON on 540/540 stitched
masks (2026-09-23 spec §2a). The outline lies inside the object footprint, so it
carries the same class index as the fill — there is no separate 'outline' class."""

_CLASS_NAMES: dict[int, str] = {1: "Hard Coral", 2: "Soft Coral"}
"""Only the two classes ever rendered into a stitched mask. Milleporid, Other
Sessile Invertebrates and SCALE occupy indices 3-5 in native.yaml's node order but
are never painted (§2a) — they reach 0 (background) like everything else unlabelled."""

_CLASSES = 6
"""native.yaml node order: 0 unlabelled, 1 HC, 2 SC, 3 Milleporid, 4 OSI, 5 SCALE."""

_IGNORE_INDEX = 0


def _rgb_lut_indices(im, *, label: str):
    """Decode a stitched RGB mask into raw per-pixel indices via ``_RGB_LUT``, exact
    match only (no tolerance — the source has no anti-aliasing, per spec). Any pixel
    colour absent from the LUT raises: it is the guard for a rendering the pipeline
    has never seen (the 3 unsampled SEAVIEW regions), not a value to approximate.
    """
    _Image, np = _require_pillow_and_numpy()
    arr = np.asarray(im.convert("RGB"))
    out = np.full(arr.shape[:2], 255, dtype=np.uint8)
    for (r, g, b), index in _RGB_LUT.items():
        out[(arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)] = index
    unknown = out == 255
    if unknown.any():
        y, x = (int(v) for v in np.argwhere(unknown)[0])
        colour = tuple(int(c) for c in arr[y, x])
        raise LoaderError(
            f"{label}: pixel at (x={x}, y={y}) has colour {colour}, which is not in "
            f"the labelbox-rgb LUT {sorted(_RGB_LUT)} — refusing to guess"
        )
    return out.tobytes()


def _ndjson_objects_by_external_id(path: Path) -> dict[str, list[dict]]:
    """Per ``external_id``, the mask-relevant objects of the NDJSON line with the
    most objects (SEAVIEW_ATL's ``export-result-old.ndjson`` has 1,410 lines for 705
    images, each twice with one empty line — spec §2a). A no-op dedup on a file with
    no duplicate lines, so this is safe to run unconditionally.
    """
    best_count: dict[str, int] = {}
    best_objects: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoaderError(f"malformed NDJSON at {path}:{lineno} — {exc}") from exc

            name = (doc.get("data_row") or {}).get("external_id") or doc.get("external_id")
            if not name:
                continue

            objects: list[dict] = []
            for project in (doc.get("projects") or {}).values():
                for entry in project.get("labels", []):
                    for obj in (entry.get("annotations") or {}).get("objects", []):
                        obj_name = obj.get("name")
                        if not obj_name:
                            continue
                        objects.append({"name": obj_name, "has_mask": obj.get("mask") is not None})

            if name not in best_count or len(objects) > best_count[name]:
                best_count[name] = len(objects)
                best_objects[name] = objects
    return best_objects


def _cross_check_classes(observed: frozenset[int], objects: list[dict], *, label: str) -> int:
    """{classes present in the pixels} == {HC, SC} ∩ {NDJSON mask-object names}
    (spec §2a), raising on mismatch. Returns ``objects_without_pixels``: mask-type
    NDJSON objects whose name is neither Hard Coral nor Soft Coral (Milleporid/OSI —
    declared with a mask but painted black, never a pixel error).
    """
    mask_objects = [o for o in objects if o["has_mask"]]
    pixel_names = {_CLASS_NAMES[i] for i in observed if i in _CLASS_NAMES}
    ndjson_names = {o["name"] for o in mask_objects if o["name"] in _CLASS_NAMES.values()}
    if pixel_names != ndjson_names:
        raise LoaderError(
            f"{label}: pixel classes {sorted(pixel_names)} do not match the NDJSON "
            f"mask-object classes {sorted(ndjson_names)}"
        )
    return sum(1 for o in mask_objects if o["name"] not in _CLASS_NAMES.values())


@register_loader
class LabelboxRgbMaskLoader(_HarmonizingLoader):
    """Stitched RGB mask → indexed PNG, for Reef Support's own Labelbox exports
    (``reef-support-benthic-own``, ``reef-support-seaview-labels``).

    The masks were rendered locally from the Labelbox export with a fixed 5-colour
    palette (fill + outline per class, see ``_RGB_LUT``) rather than shipped as an
    indexed raster, so this loader — unlike :class:`ImageMaskPairLoader` — decodes
    colour to class index itself, cross-checks the result against the NDJSON, and
    caches the converted PNG (:func:`marinedata.normalise._encode_indexed_png`, the
    same shared palette/range-check every other mask conversion in this package
    uses) so repeat loads never re-decode.

    Params:
        images_dir: default ``"images"``
        stitched_dir: default ``"masks_stitched"``
        mask_suffix: appended to the image stem to find its stitched mask, default
            ``"_mask"``
        annotations: NDJSON path relative to root, default ``"export-result.ndjson"``.
            SEAVIEW_ATL sets this to ``"export-result-old.ndjson"`` — the newer export
            is lossy (0 objects for 320/705 images, spec §2a).
        converted_dir: where the derived indexed PNGs are cached, default
            ``"masks_converted"``
    """

    layout = "labelbox-rgb"

    def _paths(self) -> tuple[Path, Path, Path, Path]:
        images = self.root / str(self._param("images_dir", "images"))
        stitched = self.root / str(self._param("stitched_dir", "masks_stitched"))
        converted = self.root / str(self._param("converted_dir", "masks_converted"))
        annotations = self.root / str(self._param("annotations", "export-result.ndjson"))
        return images, stitched, converted, annotations

    def validate(self) -> None:
        super().validate()
        images, stitched, _converted, annotations = self._paths()
        for name, path in (("images_dir", images), ("stitched_dir", stitched)):
            if not path.is_dir():
                raise LoaderError(
                    f"{self.source.id}: layout 'labelbox-rgb' expects {name} at {path}"
                )
        if not annotations.is_file():
            raise LoaderError(f"{self.source.id}: Labelbox export not found at {annotations}")

    def _iter_samples(self) -> Iterator[Sample]:
        images, stitched, converted, annotations = self._paths()
        suffix = str(self._param("mask_suffix", "_mask"))
        by_stem = {p.stem: p for p in stitched.rglob("*") if p.is_file()}
        records = _ndjson_objects_by_external_id(annotations)

        for image in _images_under(images):
            mask = by_stem.get(f"{image.stem}{suffix}")
            if mask is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(
                    f"{self.source.id}: no stitched mask for image '{image.name}' in {stitched}"
                )

            objects = records.get(image.name)
            if objects is None:
                if self.partial:
                    continue
                raise LoaderError(
                    f"{self.source.id}: no NDJSON record for '{image.name}' in {annotations}"
                )

            label = f"{self.source.id}/{image.name}"
            dest = converted / f"{image.stem}.png"
            if dest.is_file():
                Image, _np = _require_pillow_and_numpy()
                with Image.open(dest) as written:
                    observed = frozenset(written.tobytes())
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image, _np = _require_pillow_and_numpy()
                with Image.open(mask) as im:
                    raw = _rgb_lut_indices(im, label=label)
                    indexed = Image.frombytes("L", im.size, raw)
                _digest, observed = _encode_indexed_png(
                    indexed, dest, classes=_CLASSES, label=label
                )

            objects_without_pixels = _cross_check_classes(observed, objects, label=label)

            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                mask=dest,
                labels={},
                supervised=frozenset({Axis.TAXON}),
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta={
                    "mask_is_dense": True,
                    "raster_ignore_value": _IGNORE_INDEX,
                    "mask_classes": {
                        "0": "unlabelled",
                        "1": "Hard Coral",
                        "2": "Soft Coral",
                        "3": "Milleporid",
                        "4": "Other Sessile Invertebrates",
                        "5": "SCALE",
                    },
                    "objects_without_pixels": objects_without_pixels,
                },
            )
