"""Training label mapping for every published source (HF ``reefsupport/marine-data`` and ``-nc``).

The published masks carry each source's *native* ids, which collide across sources. This module
turns them into fixed, comparable class ids per **scheme**:

========================  =====================================================================
``benthic-coarse``        HC, MIL, SC, ALGAE, ABIOTIC, OTHER_FAUNA (registry task order)
``coral-binary``          NOT_CORAL (0), CORAL (1)
``scene``                 SUIM's 8 classes (BW HD PF WR RO RI FV SR; ids = SUIM pixel values)
========================  =====================================================================

255 is the ignore value in every scheme: unannotated pixels, labels with no class in the scheme,
and anything excluded by the options. Which labels land where is data
(``registry/label-schemes/*.yaml`` + ``registry/crosswalks``), not code. numpy only.

>>> from datasets import load_dataset
>>> ds = load_dataset("reefsupport/marine-data", "coral-masks", split="train")
>>> ds = ds.map(lambda row: remap_row(row, "benthic-coarse"))   # adds ``label`` (HxW uint8)

``is_dense(source)`` is False for own / seaview / CoralSCOP: there 255 means *unannotated*, not
background, so train those with :func:`supervised_classes` (loss restricted to those classes).

Options (both validated, typos raise): ``exclude_conditions=("dead", "bleached")`` sends labels
whose crosswalk condition is (under) those conditions to 255; ``ignore=("sand", "RK")`` sends extra
native labels or taxon nodes (with their subtree) to 255.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from . import label_schemes as _ls

IGNORE_INDEX = _ls.IGNORE


def schemes() -> tuple[str, ...]:
    """Names of the available schemes."""
    return tuple(_ls.schemes())


def class_names(scheme: str) -> tuple[str, ...]:
    """Class names of ``scheme``; the id of a class is its position."""
    return _ls.scheme_spec(scheme).classes


def is_dense(source: str) -> bool:
    """True when every pixel of the source's masks is annotated (255 = ignore only).

    False for own / seaview / CoralSCOP and for instance / box sources: there 255 (or an absent
    object) means *unannotated*, not background.
    """
    return _ls.source_spec(source).dense


def supervised_classes(source: str, scheme: str) -> tuple[str, ...]:
    """Classes of ``scheme`` that ``source`` can ever label, in id order.

    For a non-dense source, restrict the loss to these (e.g. mask the other logits): the source
    shows no negatives for the rest.
    """
    src = _ls.source_spec(source, scheme)
    reached = {_ls.resolve_class(source, label, scheme) for label in src.native_labels}
    return tuple(c for c in class_names(scheme) if c in reached)


def map_label(source: str, label_native: str, scheme: str, **opts: Iterable[str]) -> int | None:
    """Scheme class id of one native label (instances, boxes, class_map entries); ``None`` = 255."""
    name = _ls.resolve_class(source, label_native, scheme, _ls.make_options(**opts))
    return None if name is None else _ls.class_id(scheme, name)


@lru_cache(maxsize=256)
def _lut(source: str, scheme: str, options: _ls.Options) -> np.ndarray:
    src = _ls.source_spec(source, scheme)
    if src.kind != "semantic":
        raise ValueError(
            f"source {source!r} has no pixel-id LUT (kind {src.kind!r}); "
            "use map_label for its labels"
        )
    table = np.full(256, _ls.IGNORE, dtype=np.uint8)
    for pixel, label in src.ids.items():
        name = _ls.resolve_class(source, label, scheme, options)
        table[pixel] = _ls.class_id(scheme, name)
    table.setflags(write=False)
    return table


def lut(source: str, scheme: str, **opts: Iterable[str]) -> np.ndarray:
    """``(256,) uint8`` table: native pixel value -> scheme id (255 for unmapped or ignored)."""
    return _lut(source, scheme, _ls.make_options(**opts)).copy()


def remap_mask(mask: Any, source: str, scheme: str, **opts: Iterable[str]) -> np.ndarray:
    """Remap a native-id mask (any integer array-like, values 0..255) to scheme ids, uint8."""
    array = np.asarray(mask)
    if array.ndim == 3:  # an RGB-decoded index PNG: ids live in the first channel
        array = array[..., 0]
    if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
        raise ValueError("mask values outside 0..255 cannot be native ids")
    return _lut(source, scheme, _ls.make_options(**opts))[array.astype(np.intp, copy=False)]


def _decode_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, Mapping):  # raw HF struct {"bytes", "path"}
        from PIL import Image

        with Image.open(io.BytesIO(mask["bytes"])) as handle:
            return np.asarray(handle)
    return np.asarray(mask)  # PIL image (decoded by `datasets`), ndarray or nested list


def _class_map(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("class_map")
    if raw is None:
        return []
    return json.loads(raw) if isinstance(raw, str) else list(raw)


def _validate_class_map(
    entries: list[Mapping[str, Any]], pixels: np.ndarray, source: str, scheme: str
) -> None:
    """Raise unless the row's ``class_map`` agrees with the registry tables for this source."""
    src = _ls.source_spec(source, scheme)
    for entry in entries:
        expected = src.ids.get(int(entry["id"]))
        if expected != entry["label_native"]:
            raise ValueError(
                f"{source}: class_map says id {entry['id']} = {entry['label_native']!r}, "
                f"the registry says {expected!r}"
            )
        if scheme == "benthic-coarse":
            published = entry.get("coarse")
            derived = _ls.resolve_class(source, entry["label_native"], scheme)
            if published != derived:
                raise ValueError(
                    f"{source}: class_map coarse {published!r} for {entry['label_native']!r} "
                    f"disagrees with the {scheme} LUT ({derived!r})"
                )
    present = {int(v) for v in np.unique(pixels)} - {_ls.IGNORE}
    unknown = present - {int(e["id"]) for e in entries}
    if entries and unknown:
        raise ValueError(f"{source}: mask has ids {sorted(unknown)} missing from its class_map")


def remap_row(row: Mapping[str, Any], scheme: str, **opts: Iterable[str]) -> dict[str, Any]:
    """Copy of an HF mask row with ``label`` added: the mask remapped to ``scheme`` (HxW uint8).

    Usable in ``Dataset.map`` / ``with_transform``. The row's ``class_map`` is validated against the
    registry tables (and, for benthic-coarse, its published ``coarse``): a mismatch raises.
    """
    source = row["source"]
    pixels = _decode_mask(row["mask"])
    if pixels.ndim == 3:
        pixels = pixels[..., 0]
    _validate_class_map(_class_map(row), pixels, source, scheme)
    return {**row, "label": remap_mask(pixels, source, scheme, **opts)}


__all__ = [
    "IGNORE_INDEX",
    "class_names",
    "is_dense",
    "lut",
    "map_label",
    "remap_mask",
    "remap_row",
    "schemes",
    "supervised_classes",
]
