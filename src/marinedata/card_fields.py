# ruff: noqa: E501
"""Per-config and per-column text for the generated dataset card (PA-P1.9).

``FIELD_DOCS`` is the single source of the card's ``Data fields`` descriptions: one line per column
name, whatever config carries it. The tables themselves are generated from the build's parquet
schema (the export layout), so a column that exists but is missing here shows up as undocumented
and fails ``tests/test_release_card.py::test_every_layout_column_is_described``.
"""

from __future__ import annotations

from dataclasses import dataclass

FIELD_DOCS: dict[str, str] = {
    # identity, pixels, labels
    "image_sha256": "SHA-256 of the image bytes. Stable id and join key across configs.",
    "image": "RGB photo (embedded bytes).",
    "mask": (
        "Label mask (PNG). Semantic: `uint8` class id, `255` = ignore. "
        "Instance: `uint16` instance id, `0` = background."
    ),
    "class_map": (
        "List of structs, one per class present: `id` (value in the mask), `label_native`, "
        "`taxon_node`, `coarse` (may be null)."
    ),
    "instances": (
        "List of structs, one per instance id: `id`, `label_native`, `taxon_node`, `coarse`, "
        "`bbox_xyxy_norm` [x_min, y_min, x_max, y_max] in 0-1."
    ),
    "boxes": (
        "List of structs: `x_min`, `y_min`, `x_max`, `y_max` (0-1 of width and height) and `label`."
    ),
    "points": "List of point annotations (`x`, `y`, `label`, `t1_class`); empty in this release.",
    "label": "Task class of the sample (null = the source abstains or does not supervise this axis).",
    "native_label": "The source's own label for the sample.",
    "label_reason": "How `label` was reached from `native_label`.",
    "mask_class_map": "JSON map from mask pixel value to task class (null = unlabelled or abstain).",
    "mask_values": "JSON description of the native pixel values of the mask, as the source published.",
    "sample_key": "Key of one (image, source) sample.",
    "annotator_type": "Who made the annotation: `human` or `model`.",
    "ann_license": "Licence of the annotation itself, where it differs from the image's.",
    "ann_attribution": "Attribution text for the annotation itself.",
    # source, split, licence
    "source": "Source id; see the licensing table.",
    "source_id": "Source id; see the licensing table.",
    "source_ids": "Comma-separated ids of every source that published these image bytes.",
    "source_version": "Version of the source copy the row was built from.",
    "licence": "Licence of the row's data under its source.",
    "annotation_licence": "Licence of the annotations.",
    "license": "Licence of the row's image under its source.",
    "licence_class": "Licence class of the row (`open` for every row of the open flavour).",
    "license_url": "URL of the licence text named in `license`.",
    "license_verified_by": "Who verified the licence of the row.",
    "license_verified_on": "Date the licence of the row was verified.",
    "attribution": "Text to reproduce when you use or share the row.",
    "rights_holder": "Rights holder to credit, where the source names one.",
    "split": "Split of the row: `train`, `validation` or `test`.",
    "split_group": "Group that is never divided across splits (site, station, transect or upstream group).",
    # provenance
    "upstream_id": "Identifier of the image in the upstream source.",
    "upstream_url": "URL of the image or its record upstream.",
    "provenance_url": "URL of the record the row's provenance was read from.",
    "fetch_date": "ISO-8601 date the source copy was fetched.",
    "lineage_root_digest": "Digest of the lineage root the image bytes trace back to.",
    # geometry, position, time
    "width": "Image width in pixels.",
    "height": "Image height in pixels.",
    "min_side": "Shorter image side in pixels.",
    "lat": "Latitude, WGS84 degrees (null unless the source provides it).",
    "lon": "Longitude, WGS84 degrees (null unless the source provides it).",
    "geo_precision": "Precision class of `lat`/`lon`.",
    "geo_source": "Where `lat`/`lon` came from.",
    "gps_precision_m": "Stated precision of `lat`/`lon` in metres.",
    "coord_precision_m": "Stated precision of `lat`/`lon` in metres.",
    "location_generalized": "True when the position was deliberately coarsened for a sensitive site.",
    "depth_m": "Depth in metres (null unless the source provides it).",
    "depth_source": "Where `depth_m` came from.",
    "depth_zone": "Depth zone class derived from `depth_m`.",
    "capture_datetime": "ISO-8601 capture time (null unless the source provides it).",
    "date": "Capture date, ISO-8601; its precision is in `date_precision`.",
    "date_precision": "Precision of `date`: year, month or day.",
    "year": "Capture year.",
    "date_source": "Where `date` came from.",
    "platform": "Capture platform (diver, ROV, drone, towed camera, ...), null when unknown.",
    "camera": "Camera or instrument, where known.",
    "site_id": "Site the image was taken at, as the source names it.",
    "quadrat_id": "Quadrat the image was taken at, as the source names it.",
    "survey_id": "Survey the image belongs to, as the source names it.",
    "region": "Region name, as the source gives it.",
    "meow_realm": "Marine Ecoregions of the World realm of the position.",
    "meow_province": "Marine Ecoregions of the World province of the position.",
    "meow_ecoregion": "Marine Ecoregions of the World ecoregion of the position.",
    "habitat": "Habitat, where the source states one (for example `lab` for aquarium photos).",
    # quality and privacy
    "q_blur": "Blur score of the image (variance of the Laplacian).",
    "q_clip_lo": "Fraction of pixels clipped at the dark end.",
    "q_clip_hi": "Fraction of pixels clipped at the bright end.",
    "q_uiqm": "Underwater image quality measure (UIQM).",
    "q_entropy": "Entropy of the image.",
    "q_blank": "True when the image is blank or near-uniform.",
    "quality_flags": "Comma-separated quality flags raised for the image.",
    "privacy_flag": "Outcome of the privacy screening for the image.",
    "face_score": "Highest face-detector score found in the image (0-1).",
}


@dataclass(frozen=True)
class ConfigDoc:
    """How the card describes one config. ``blurb`` is a ``str.format`` template."""

    task: str
    blurb: str
    order: int = 100
    categories: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    table: str = "splits"
    """Per-config source table: ``splits`` (rows + per-split counts), ``plain`` or ``none``."""


CONFIG_DOCS: dict[str, ConfigDoc] = {
    "coral-masks": ConfigDoc(
        "Benthic semantic segmentation",
        "Benthic semantic segmentation of coral-reef imagery: `image`, a `uint8` index `mask` "
        "(`255` = ignore) and a per-image `class_map`. The {rows} images come from "
        "{sources_phrase}{vocab_note}. Native labels are kept and mapped to a shared taxon node "
        "(`taxon_node`) and a coarse group (`coarse`) where a mapping exists.",
        10,
        ("image-segmentation",),
        ("semantic-segmentation",),
        ("coral", "benthic"),
    ),
    "scene-masks": ConfigDoc(
        "Underwater scene semantic segmentation",
        "Underwater scene segmentation: {rows} images with `uint8` index masks (`255` = ignore) "
        "and a per-image `class_map`.",
        20,
        ("image-segmentation",),
        ("semantic-segmentation",),
        table="none",
    ),
    "instance-masks": ConfigDoc(
        "Underwater instance segmentation",
        "Underwater instance segmentation: `image`, a `uint16` instance-id `mask` "
        "(`0` = background) and `instances`, a list with the label and a normalised box for "
        "every instance id. {rows} images.",
        30,
        ("image-segmentation",),
        ("instance-segmentation",),
    ),
    "fish-boxes": ConfigDoc(
        "Object detection (fish and fauna boxes)",
        "Fish and fauna photos with bounding boxes (`x_min`, `y_min`, `x_max`, `y_max` "
        "normalised to 0-1, native `label`). {rows} images.",
        40,
        ("object-detection",),
        (),
        ("fish",),
        "plain",
    ),
    "points": ConfigDoc(
        "Point annotations with geo, date and licence",
        "Point annotations, each row carrying the image-level geo, date and licence columns "
        "(`lat`, `lon`, `date`, `license`, ...) joined by `image_sha256`. {rows} rows.",
        50,
        table="plain",
    ),
    "metadata": ConfigDoc(
        "Per-image metadata (no pixels)",
        "No pixels: licence, provenance, position, depth, ecology and quality columns, one row "
        "per `image_sha256`. {rows} rows.",
        60,
        table="plain",
    ),
    "images": ConfigDoc(
        "Image pixels",
        "Every image once (pixels embedded), keyed by `image_sha256`. {rows} images.",
        70,
        table="plain",
    ),
}


def config_doc(name: str) -> ConfigDoc:
    """The catalogue entry of ``name``; an unknown config gets a neutral one that sorts last."""
    return CONFIG_DOCS.get(name) or ConfigDoc(name, "`" + name + "`: {rows} rows.", 1000)


def field_doc(column: str) -> str | None:
    return FIELD_DOCS.get(column)
