"""Static v1.1.0 statements (PR-E, E3/E4): one set of fields rendered into the card sections and the
Croissant RAI properties (design section 4, DATASHEET <-> card <-> Croissant).

Every sentence is a fact of the pipeline or a declared defect (design 3.5, 8); nothing here names a
private repo or bucket (``hf_export_v11.assert_public_safe`` re-checks the rendered texts)."""

from __future__ import annotations

VERSION = "1.1.0"
V1_DATE = "2026-10-07"
V1_NOTES = {
    "open": "coral-masks, scene-masks, instance-masks and fish-boxes: 8 sources, 58,803 rows. "
    "Frozen: files, rows and paths are unchanged in v1.1.0.",
    "nc": "coral-masks and coral-masks-machine. Frozen: files, rows and paths are unchanged in "
    "v1.1.0.",
}
V11_ORDER = (
    "images", "metadata", "points", "bleaching", "boxes", "semseg", "instances", "benthic-cover",
    "sites", "taxonomy", "label-map", "sources",
)  # fmt: skip
"""The design 2.1 order of the v1.1.0 configs."""
CONFIG_TASKS: dict[str, tuple[str, str]] = {
    "images": (
        "Pixels for retrieval, pretraining and image classification",
        "Bytes once per `image_sha256`; images a v1.0 config already embeds are not repeated "
        "(`metadata.bytes_config` names the config that holds each image).",
    ),
    "metadata": (
        "Survey and site analytics, joins, retrieval",
        "One row per image, v1.0 images included; no pixels. Default config.",
    ),
    "points": ("Point-based benthic cover", "One row per labelled point; label-only, no image."),
    "bleaching": ("Image classification (bleaching condition)", "One row per (image, label)."),
    "boxes": ("Object detection", "One row per box; label-only."),
    "semseg": (
        "Semantic segmentation",
        "One row per (image, source) mask. New masks are embedded as palette PNG; v1.0 masks are "
        "referenced by `mask_ref` / `mask_config`.",
    ),
    "instances": (
        "Instance segmentation",
        "One row per instance. No mask pixels: each row holds the instance geometry (`polygon`, "
        "`rle` or a `mask_ref` pointer), the label and its taxonomy nodes; the masks stay in the "
        "v1.0 `instance-masks` config.",
    ),
    "benthic-cover": ("Benthic cover", "Derived per image or survey from points and masks."),
    "sites": ("Site analytics", "One row per (source, site), derived from `metadata`."),
    "taxonomy": ("Label space", "Taxonomy nodes and the edges of the crosswalks in this repo."),
    "label-map": (
        "Label harmonisation",
        "One row per (source, native label or pixel value) with its taxonomy nodes and match type.",
    ),
    "sources": (
        "Provenance, licence and attribution",
        "One row per source: licence, attribution, rights holder, verification, counts.",
    ),
}
LABEL_SPACE = (
    "Labels are harmonised into the pinned taxonomy release (`taxonomy_version` is stamped on "
    "every row of every new table). Three axes per row: `taxon_node_id`, `form_node_id` and "
    "`condition_node_id`, with nodes keyed `<schema>:<id>`. `rs-benthic-v1` is the functional "
    "schema (L0 biotic/abiotic, L1 phyla, L2 the 14 MariMap codes, L3 genera), `rs-taxa-v1` the "
    "identity schema (`A<AphiaID>` nodes, AphiaID also shipped as `worms_aphia_id`) and "
    "`rs-fauna-v1` the fish and fauna schema. Node ids are namespaced in taxonomy 3.0.0: an "
    "`rs-taxa-v1` id is `A` followed by the WoRMS AphiaID, and the non-taxon nodes of "
    "`rs-benthic-v1` and `rs-fauna-v1` (debris, equipment, divers, wrecks, seafloor, unknown) "
    "carry `NT_*` ids. `label-map` maps each source's raw labels to these nodes, one row per "
    "source and native label or pixel value. `match_type` is exact, broader, narrower, related "
    "or unmapped; `narrower` is stored for audit and never used as a positive. A native label "
    "with no mapping is a row with `match_type = unmapped` and null node columns, never dropped, "
    "so the unmapped labels of each source are listed in `label-map` (the mapped share per "
    "source is tabulated below); the training views send them to `255` (ignore)."
)
SPLITS_TEXT = (
    "The card publishes the **measured** split shares of every config, never a nominal ratio: "
    "upstream splits dominate several sources, so the v1.0 shares are not 80/10/10 or "
    "70/15/15. New rows target 70/15/15 with the split names `train`, `validation`, `test` "
    "and `test_ood`. Precedence, first match wins: (1) an image that matches a v1.0 image by "
    "sha256, or within Hamming distance 4 of its dhash, keeps that v1.0 split unless an "
    "upstream benchmark manifest pins it (see below), and an image within 5-8 of a v1.0 image "
    "is never used across the train/eval boundary; (2) the out-of-distribution holdout rules "
    "(deepfish, temperate-australasia, mediterranean, a data-driven tropical province, depth "
    "of 800 m or more, platform AUV, year 2025 or later) go to `test_ood`, which is never "
    "mixed into `test`; (3) the upstream split where defined; (4) a group hash on survey, "
    "site or quadrat, else the near-duplicate group. Images of one group share a split.\n"
    "\n"
    "Version 1.1.0 inherits the v1.0 splits by `image_sha256` from the published v1.0 tree, "
    "with one exception: six benchmark sources (`coralscapes`, `coralvqa`, `suim`, `uiis`, "
    "`uiis10k` and `usis10k`) are pinned to their upstream manifests. A pin routes the "
    "manifest's evaluation split to `test` and a second held-out split, where the source has "
    "one, to `validation`. A pin wins over an inherited v1.0 split; the out-of-distribution "
    "holdout and any rule that keeps an image out of evaluation win over a pin. The pins "
    "moved images that were in v1.0 `train` into the v1.1.0 `validation` and `test` splits "
    "and resolved near-duplicate groups that straddle v1.0 splits, while the frozen v1.0 "
    "configs keep their v1.0 splits. A group that cannot be assigned consistently is withheld "
    "from every table and listed under `withheld` in `SPLIT_MAP.json`.\n"
    "\n"
    "Because several of these sources carry large upstream test sets, the overall split is "
    "not the 70/15/15 target: the test share is well above 15 %, and `boxes` and `instances` "
    "are the most affected tables.\n"
    "\n"
    "**Evaluation caveat.** A model trained on the v1.0 configs must not be evaluated on the "
    "v1.1.0 `validation` or `test` splits of these six sources, because some of their images "
    "were in `train` in v1.0."
)
PERSONAL = (
    "The images are underwater and reef photographs. Divers, and rarely faces, can appear. The "
    "release gates include a privacy pass (face detections at a score of 0.85 or more are "
    "excluded, 0.60 to 0.85 are flagged) and a check for EXIF GPS; their results are published "
    "in `provenance/GATES.json`. Images the detector flags are reviewed on contact sheets: an "
    "image with an identifiable person is withheld from every table, and a detection that "
    "proves to be a false positive is recorded in `registry/privacy/review.yaml` in the code "
    "repository. No image with an identifiable person is published knowingly; one Roboflow fork "
    "was excluded from this release for that reason (see Excluded sources). A gate and a review "
    "lower the risk; they do not remove it. The EXIF GPS of `seaview-survey-imagery` images is "
    "retained by design, because that source publishes per-image coordinates. `metadata` "
    "carries position at the precision the source provides (`coord_precision_m`) and no source "
    "is currently flagged location-sensitive in the registry."
)
LIMITATIONS = (
    "Coralscapes has no soft-coral or octocoral class. `label-map` records the absent classes, "
    "so a model trained on Coralscapes alone cannot predict them and cover for them is "
    "undercounted.",
    "Absence is not a negative. For the reef-support and Seaview label sources `255` means "
    "unannotated, not background; Coralscapes uses its own ignore id. `label-map` carries the "
    "per-source ignore semantics.",
    "Cross-source benthic training should use the coarse training views, or mask the classes a "
    "source cannot contain: the label overlap table shows how few classes more than one source "
    "reaches.",
    "Labels without a taxonomy mapping are kept (`mapped = false`) and ignored in the training "
    "views. A source whose mapped share is below the 95 % floor is listed in the sources table "
    "with its share.",
    "Until the declared bridges from `rs-taxa-v1` to `rs-benthic-v1` and `rs-fauna-v1` exist, "
    "rows labelled by an identity node carry a null `rs_benthic_code`.",
    "A small number of identical images carry conflicting labels; every party has "
    "`label_status = conflict` and all rows are kept. The sampled label-quality audit has not "
    "been run.",
    "Confidence is present only where the upstream source supplies it; forks and crowd sources "
    "have none. It is never invented.",
    "v1.0 files are frozen as published, with their known defects: Seaview rows say "
    "`CC-BY-3.0-AU` where the University of Queensland licence is CC BY 3.0 unported (new rows "
    "and `metadata.license` say `CC-BY-3.0`); v1.0 `fish-boxes` uses `license`, `source_id` and "
    "int32 where the other configs use `licence`, `source` and int64; v1.0 masks render black "
    "in the viewer because they hold class ids, not colours (new semantic masks carry a palette).",
    "Coordinates, dates and depths are as complete as each source made them; analytics resolvers "
    "exist for a subset of the sources.",
    "The point labels of `seaview-survey-imagery` are in `points` only and are not part of "
    "`benthic-cover`: the cover rollup reads the original point sources, and Seaview points "
    "stay out of it until a cover decision is made for that source.",
    "**Licensing and provenance caveats.** Four sources are forks of Roboflow Universe projects "
    "(`roboflow-coral-bleaching-final-v6i`, `roboflow-coral-bleaching-general-v1-yolov8s`, "
    "`roboflow-coral-classification-copy-changed-v13i`, `roboflow-coral-reef-classification-v3i`). "
    "They are web-scraped photo sets re-shared on Roboflow. Their CC-BY-4.0 tag is the "
    "uploader's claim, which Reef Support could not verify with the original photographers, and "
    "third-party watermarks appear sporadically. Near-duplicate images across these four "
    "sources are reported by the dedup gate and are not removed. A fifth fork, "
    "`roboflow-coral-reef-bleach-detection-v2i`, is excluded from this release: its images are "
    "augmented broadcast-video frames that show identifiable people, and its licence cannot be "
    "verified. Rights holders can use the takedown route below.",
)
MASKS = (
    "**Masks.** A `semseg` row names a v1.0 mask config in `mask_config` when that frozen config "
    "holds the image's mask, and the mask stays there; otherwise the mask is embedded in the row "
    "as a palette PNG. The `instances` table carries no masks: each row holds the instance's "
    "`polygon` or `rle` (or a `mask_ref` pointer), `instance_id`, the native label with its "
    "taxonomy nodes and `match_type`, `annotator_type` and `ann_license`, and the instance "
    "masks stay in the v1.0 `instance-masks` config."
)
EXCLUDED_INTRO = (
    "Every registered source that is not in this release falls under one of the reasons below. "
    "The sources that are in it are listed in the `sources` config."
)
NOT_INCLUDED = (
    (
        "`roboflow-coral-reef-bleach-detection-v2i`",
        "tagged release-excluded. Its images are augmented broadcast-video frames that show "
        "identifiable people, and its licence tag comes from the uploader's sidecar and cannot "
        "be verified",
    ),
    (
        "`large-scale-fish`",
        "tagged release-excluded and quarantined: out-of-water market photographs of fish",
    ),
    (
        "sonar sources",
        "side-scan and forward-looking sonar imagery is a different sensing modality from the "
        "camera images of this release; none is included",
    ),
    (
        "open-licence sources not yet staged",
        "registered sources with an open licence that were not in the v1.1.0 fetch list (for "
        "example `noaa-pifsc-esa-coral-icra`, `marineevt` and `salmon-cage`) are deferred to a "
        "later release",
    ),
    ("non-image sources", "text-only sources such as `oceaninstruct` hold no images"),
)
"""Named exclusions of the v1.1.0 card (the licence classes stay in ``EXCLUDED_CLASSES``). No
count is written here: the sources that are in the release are the ``sources`` config."""
NOT_INCLUDED_OPEN = (
    (
        "`synthetic-seabed-debris`",
        "licensed for non-commercial use only; it ships in the separate non-commercial flavour "
        "of the release, not in this repository. No other non-commercial source is part of "
        "v1.1.0",
    ),
)
"""Open flavour only: the one source of the non-commercial flavour (the open card never names
the other repository)."""
RAI = {
    "rai:dataCollection": (
        "Compiled by Reef Support from the open image datasets listed in the `sources` config. "
        "Each source is fetched from its primary location, pinned by its root digest in "
        "`provenance/release.lock.yaml` and verified against its checksums before use."
    ),
    "rai:dataBiases": (
        "Coverage is uneven by region, platform and source; some classes (octocorals) are "
        "absent from some sources; crowd and fork-derived labels carry no confidence. See the "
        "limitations section of the dataset card."
    ),
    "rai:personalSensitiveInformation": PERSONAL,
    "rai:dataLimitations": " ".join(LIMITATIONS[:5]),
    "rai:dataUseCases": (
        "Benthic and scene segmentation, object detection, bleaching classification, "
        "point-based cover estimation, survey analytics and retrieval research. Not a "
        "substitute for in-water survey data."
    ),
    "rai:dataReleaseMaintenancePlan": (
        "Released versions are not edited. Corrections and accepted takedown requests are "
        "published as a new version and listed in the changelog."
    ),
}
CHANGELOG = (
    "**Added (additive; no v1.0 file, row, path or `configs:` entry changed):** the configs "
    "`images`, `metadata`, `points`, `bleaching`, `boxes`, `semseg`, `instances`, "
    "`benthic-cover`, `sites`, `label-map` and `sources`, and `taxonomy` in the open repo; "
    "`LICENSE`, `NOTICE`, `CITATION.cff`, `croissant.json` and `provenance/`.",
    "**Unchanged:** every v1.0 config, with its files, rows and paths; the release is additive.",
    "**Task layers:** the v2 task layers are new tables on the `image_sha256` key: `metadata`, "
    "`points`, `bleaching`, `boxes`, `semseg` and `instances` (`benthic-cover` and `sites` are "
    "derived from them).",
    "**Changed:** this card (text only; the v1.0 `configs:` entries are carried verbatim). "
    "`dataset_info` is written for the new configs only.",
    "**Label space:** taxonomy 3.0.0. Every new table row carries `taxonomy_version`; the "
    'Coralscapes class "unknown hard substrate" is unmapped in the new tables; three wrong '
    "AphiaIDs were corrected (205902, 206485, 758259). v1.0 `class_map` columns are untouched.",
    "**Splits:** a split map that inherits the v1.0 splits by `image_sha256` (v1.0 continuity); "
    "six benchmark sources are pinned to their upstream manifests instead, which moved some "
    "images out of v1.0 `train` into `validation` and `test`; the frozen v1.0 configs keep "
    "their v1.0 splits (see Splits).",
    "**Excluded:** `roboflow-coral-reef-bleach-detection-v2i` (see Excluded sources).",
    "**Licences:** new Seaview rows and `metadata.license` use `CC-BY-3.0` (v1.0 rows keep "
    "`CC-BY-3.0-AU`, declared above).",
)
MEOW = (
    "Marine Ecoregions of the World (MEOW): Spalding, M. D. et al. (2007), BioScience 57(7), "
    "573-583. Only the derived realm, province and ecoregion names ship; the MEOW geometry does "
    "not."
)
WORMS = (
    "World Register of Marine Species (WoRMS Editorial Board), https://www.marinespecies.org, "
    "CC BY 4.0. AphiaIDs identify taxa in the taxonomy, `label-map` and the label tables."
)
