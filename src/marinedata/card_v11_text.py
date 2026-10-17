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
        "One row per instance; the masks stay in the v1.0 `instance-masks` config.",
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
    "`rs-fauna-v1` the fish and fauna schema. `match_type` is exact, broader, narrower, related "
    "or unmapped; `narrower` is stored for audit and never used as a positive. A native label "
    "with no mapping is a row with `match_type = unmapped` and null node columns, never dropped; "
    "the training views send it to `255` (ignore)."
)
SPLITS_TEXT = (
    "The card publishes the **measured** split shares of every config, never a nominal ratio: "
    "upstream splits dominate several sources, so the v1.0 shares are not 80/10/10 or 70/15/15. "
    "New rows target 70/15/15 with the split names `train`, `validation`, `test` and `test_ood`. "
    "Precedence, first match wins: (1) an image that matches a v1.0 image by sha256, or within "
    "Hamming distance 4 of its dhash, keeps that v1.0 split, and an image within 5-8 of a v1.0 "
    "image is never used across the train/eval boundary; (2) the out-of-distribution holdout "
    "rules (deepfish, temperate-australasia, mediterranean, a data-driven tropical province, "
    "depth of 800 m or more, platform AUV, year 2025 or later) go to `test_ood`, which is never "
    "mixed into `test`; (3) the upstream split where defined; (4) a group hash on survey, site "
    "or quadrat, else the near-duplicate group. Images of one group share a split."
)
PERSONAL = (
    "The images are underwater and reef photographs. Divers, and rarely faces, can appear. The "
    "release gates include a privacy pass (face detections at a score of 0.85 or more are "
    "excluded, 0.60 to 0.85 are flagged) and a check for EXIF GPS; their results are published "
    "in `provenance/GATES.json`. A gate lowers the risk; it does not remove it. `metadata` "
    "carries position at the precision the source provides (`coord_precision_m`) and no source "
    "is currently flagged location-sensitive in the registry."
)
DEDUP_FAMILY = (
    "Near-duplicate images inside one split that come from different Roboflow Universe forks of "
    "one image set, or from the Seaview survey imagery and our labelled copies of its "
    "photographs, are disclosed, not removed: the dedup gate counts them as "
    "`near_pairs_across_sources_in_family` in `provenance/GATES.json`, and a near-duplicate "
    "across splits still fails it."
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
)
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
    "**Changed:** this card (text only; the v1.0 `configs:` entries are carried verbatim). "
    "`dataset_info` is written for the new configs only.",
    "**Label space:** taxonomy 3.0.0. Every new table row carries `taxonomy_version`; the "
    'Coralscapes class "unknown hard substrate" is unmapped in the new tables; three wrong '
    "AphiaIDs were corrected (205902, 206485, 758259). v1.0 `class_map` columns are untouched.",
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
