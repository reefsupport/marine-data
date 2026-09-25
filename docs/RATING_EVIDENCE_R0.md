# Rating evidence, quoted verbatim (r0, 2026-09-25)

This file exists so `docs/DATASHEET.md`'s Limitations table can be checked by
`tests/test_datasheet.py` without depending on files outside this git repository. The
5-star rating audit, the release manifest and the charter are **operational task state**
that live under `~/.claude-state/projects/reefsupport/tasks/` and
`~/dev/reefsupport/data/_release/…/v1/RELEASE.json` — outside this repo, and not
something this project commits wholesale. The quotes below are copied verbatim from
those files on 2026-09-25 so every number in the datasheet's Limitations table traces to
a file a test can actually open. If a later rating re-run changes these numbers, refresh
this file in the same commit as the datasheet update.

## From `2026-09-25-5star-rating-r0.md` (S58 rating audit)

> D1 Scale: There are 69,600 unique sha images, before near-dup collapse (14,265
> near-dup pairs; 7,085 at d=0 with a different sha). 29,984 unique human-labeled
> images: 28,734 in `coral-health-binary` plus 1,250 with masks. Only one task clears
> 10k.

> D2 Diversity: There are about 3 independent providers: NOAA PIFSC, the Roboflow
> community (5 ids, including NOAA copies) and Reef Support's own. Habitat is coral
> reef only and depth is shallow only. Region, depth, platform and year are not
> recorded per sample; the registry says `regions: [unknown]` for Roboflow. coralscop
> makes up 54% of images.

> D3 Task coverage: Human ground truth exists for two types: image classification
> (health/bleaching) and semantic segmentation (1,250 images). There is no detection,
> points (the card says "not part of v1"), captions, VQA, tracking or enhancement.
> CoralSCOP masks are pseudo-labels.

> D4 Ontology: `rs-benthic-v1` has 58 taxon-axis nodes, 32 of them with an AphiaID
> (55%); `rs-fauna-v1` has 21/21. 15 of 54 image sources have a `crosswalk_id` (28%).
> In the release, `benthic-coarse`/`benthic-l2` have 0 image labels (only 1,908
> `mask_class_map` rows) and `coral-genus-caribbean` has 0. The labels that ship are
> condition classes, not taxa.

> D5 Label quality: Abstain and conflict handling runs per release:
> `partial_abstain_excluded` holds 4,556 rows, and 399 multi-label rows in v2i abstain.
> `labelcheck.py` audits crosswalks, and pseudo-masks are segregated. Noise and
> agreement have never been measured; 14 identical-sha images carry conflicting
> labels.

> D6 Dedup/decontam: sha256 plus dHash-64 (union ≤ 4, never-eval exclusion ≤ 8, chain
> guard), within the 9 release sources only. There is no confirm hash (the card says
> "planned for v2") and no whole-corpus pass. Upstream splits are ignored, and no
> benchmark decontamination was done.

> D7 Splits/eval: The group split is frozen, one map serves every task, and there is a
> byte-identical regeneration test. The realized split is 85.7/7.2/7.1 against a
> 70/15/15 target, stratified by source only. There are no OOD holdouts, no metrics, no
> harness and no baselines.

> D8 Metadata: HF `images` columns are `image_sha256, image, source_id, source_ids,
> split_group`. Staged `metadata.parquet` has `stem, partition, upstream_path,
> upstream_split, width, height, split_group`. None of GPS, depth, date, camera or
> licence is present.

> D9 Quality filter: `src/` has no blur, exposure or blank scoring. NOAA PIFSC is
> 10,419 images at 224×224, and Roboflow is about 23k images resized to 640×640.
> Neither carries a resolution flag.

## From `RELEASE.json` (`_release/2026-09-24-neardup/releases/v1/`)

```json
"never_eval_near_dup_excluded": { "count": 557 },
"partial_abstain_excluded": [
  { "abstaining_labels": 1, "rows_dropped": 4556,
    "source": "roboflow-coral-reef-classification-v3i", "task": "bleaching-condition" }
]
```

## From the charter (`2026-09-25-5star-charter.md`)

> D-A … `coral-genus-caribbean` (0 labels in v1) is dropped from the HF configs in the
> release.

> D-B Reef Support's own imagery (benthic-own, rs-*) is licensed CC-BY-4.0 in the
> registry/card, with the note "set 2026-09-25 by delegation; confirm before publish".

## From `_hf/v1/README.md` (the generated Hub card, current build)

Per-config row counts (train / validation / test): `images` 59732 / 4947 / 4921;
`masks` 1412 / 313 / 183; `coralscop-pseudo-masks` 37273 / — / —; `benthic-coarse` and
`benthic-l2` 63330 / 5357 / 5228; `coral-health-binary` 63330 / 5357 / 5228;
`bleaching-condition` 60130 / 4678 / 4551; `general-pretraining` 63330 / 5357 / —.

Sources table (id, licence, images): `coralscop-masks-rs` CC-BY-NC-SA-4.0 37273;
`noaa-pifsc-bleaching` US-GOV-PD 10419; `reef-support-benthic-own` CC-BY-4.0 1250;
`reef-support-bleaching` CC-BY-4.0 658; `roboflow-coral-bleaching-final-v6i` CC-BY-4.0
2550; `roboflow-coral-bleaching-general-v1-yolov8s` CC-BY-4.0 2543;
`roboflow-coral-classification-copy-changed-v13i` CC-BY-4.0 2785;
`roboflow-coral-reef-bleach-detection-v2i` CC-BY-4.0 10544;
`roboflow-coral-reef-classification-v3i` CC-BY-4.0 4541.
