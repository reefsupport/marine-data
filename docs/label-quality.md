# Label quality — v1 release (WP-9, 2026-09-25)

This is a read-only audit of the v1 HF build (`data/_hf/v1`). No label in the release was changed.

- Outputs (outside git): `data/_labelquality/2026-09-25/` — `agreement.json`, `conflicts-<task>.tsv`, `label_issues.parquet`, `noise.json`.
- Code: `src/marinedata/labelquality/`; run it with `marinedata labelquality {features,agreement,confident}`.
- Origins: `registry/label-origin.yaml`.
- Audit sheets: `docs/label-quality/`.

## 1. Cross-source agreement

**Population.** The S47 "7,085 d=0 pairs" are dHash-0 sha pairs over the eval-capable images (never-eval `coralscop-masks-rs` is excluded): 7,085 sha pairs in 4,087 clusters.

They are **not** all cross-source. Only 4,421 sha pairs have differing source sets.

**Method.**
- Raters are `(sha, source, label)` units, and every unit pair inside a cluster is scored.
- Pairs are oriented by `(source, sha, label)`, which makes Cohen's κ order-stable. Scott's π is the symmetric check.
- CIs come from a cluster bootstrap (n=1000, seed 20260925, percentile).
- "identical_sha" means the same file shipped by more than one source.

| task | scope | pairs | clusters | agreement [95% CI] | Cohen κ [95% CI] | Scott π | confusion (a\|b:n) |
|---|---|---|---|---|---|---|---|
| bleaching-condition | dhash0/cross_source | 4,110 | 1,156 | 0.995 [0.991, 0.998] | 0.988 [0.980, 0.995] | 0.988 | BLEACHED|BLEACHED:2754 BLEACHED|HEALTHY:21 HEALTHY|HEALTHY:1335 |
| bleaching-condition | dhash0/within_source | 2,641 | 2,010 | 0.997 [0.993, 1.000] | 0.994 [0.986, 0.999] | 0.994 | BLEACHED|BLEACHED:1530 BLEACHED|HEALTHY:8 HEALTHY|HEALTHY:1103 |
| bleaching-condition | identical_sha/cross_source | 1,143 | 876 | 0.993 [0.986, 0.998] | 0.981 [0.963, 0.995] | 0.981 | BLEACHED|BLEACHED:862 BLEACHED|HEALTHY:8 HEALTHY|HEALTHY:273 |
| bleaching-condition | identical_sha/within_source | 230 | 230 | 0.996 [0.987, 1.000] | 0.991 [0.974, 1.000] | 0.991 | BLEACHED|BLEACHED:116 BLEACHED|HEALTHY:1 HEALTHY|HEALTHY:113 |
| coral-health-binary | dhash0/cross_source | 4,731 | 1,372 | 0.965 [0.956, 0.975] | 0.922 [0.899, 0.945] | 0.922 | HEALTHY|HEALTHY:1514 HEALTHY|UNHEALTHY:166 UNHEALTHY|UNHEALTHY:3051 |
| coral-health-binary | dhash0/within_source | 2,872 | 2,234 | 0.991 [0.986, 0.995] | 0.981 [0.972, 0.989] | 0.981 | HEALTHY|HEALTHY:1183 HEALTHY|UNHEALTHY:27 UNHEALTHY|UNHEALTHY:1662 |
| coral-health-binary | identical_sha/cross_source | 1,196 | 881 | 0.982 [0.972, 0.993] | 0.952 [0.921, 0.979] | 0.952 | HEALTHY|HEALTHY:274 HEALTHY|UNHEALTHY:21 UNHEALTHY|UNHEALTHY:901 |
| coral-health-binary | identical_sha/within_source | 245 | 245 | 0.988 [0.971, 1.000] | 0.975 [0.943, 1.000] | 0.975 | HEALTHY|HEALTHY:115 HEALTHY|UNHEALTHY:3 UNHEALTHY|UNHEALTHY:127 |

**Reading.**
- Cross-source κ is high: health 0.92, bleaching 0.99. The bleaching figure is inflated, because v6i, v1-yolov8s and v13i are **one annotation lineage** (the same uploader and footage, with 99.4% agreement between v6i and v1). They are not independent raters.
- All real disagreement sits between `v3i` ("Healthy/Unhealthy", a folder layout) and that family, at 73–79% agreement. v3i's "Unhealthy" is a broader concept than "Bleached" (it includes disease, algae and dead coral), so the health roll-up mixes two definitions.

## 2. The 14 conflicting identical images

These are files whose identical bytes carry different labels. The category was assigned by rule and checked against the contact sheet (`$SP/wp9/sheets/conflicts.jpg`).

| # | sha | task | split | source=label | category |
|---|---|---|---|---|---|
| 1 | `08b95db91f46` | bleaching-condition | train | bleaching-final-v6i=HEALTHY; bleaching-final-v6i=BLEACHED; bleaching-general-v1-yolov8s=HEALTHY | a within-source duplicate, two labels |
| 2 | `74376de31249` | coral-health-binary | train | reef-classification-v3i=HEALTHY; reef-classification-v3i=UNHEALTHY | a within-source duplicate, two labels |
| 3 | `f3d127721863` | coral-health-binary | train | reef-classification-v3i=HEALTHY; reef-classification-v3i=UNHEALTHY | a within-source duplicate, two labels |
| 4 | `0ba39a23bf1b` | coral-health-binary | train | bleaching-final-v6i=UNHEALTHY; bleaching-general-v1-yolov8s=UNHEALTHY; classification-copy-changed-v13i=UNHEALTHY; reef-classification-v3i=HEALTHY | b v3i Unhealthy≠Bleached concept |
| 5 | `383948f66254` | coral-health-binary | train | classification-copy-changed-v13i=UNHEALTHY; reef-classification-v3i=HEALTHY | b v3i Unhealthy≠Bleached concept |
| 6 | `aef0fd35836e` | coral-health-binary | validation | bleaching-final-v6i=HEALTHY; bleaching-general-v1-yolov8s=HEALTHY; reef-classification-v3i=UNHEALTHY | b v3i Unhealthy≠Bleached concept |
| 7 | `c386fc9bbc5c` | coral-health-binary | train | bleaching-general-v1-yolov8s=HEALTHY; reef-classification-v3i=UNHEALTHY | b v3i Unhealthy≠Bleached concept |
| 8 | `e81fb7fcc0d7` | coral-health-binary | train | bleaching-final-v6i=UNHEALTHY; bleaching-general-v1-yolov8s=UNHEALTHY; classification-copy-changed-v13i=UNHEALTHY; reef-classification-v3i=HEALTHY | b v3i Unhealthy≠Bleached concept |
| 9 | `f1833d1136db` | coral-health-binary | validation | bleaching-final-v6i=UNHEALTHY; bleaching-general-v1-yolov8s=UNHEALTHY; classification-copy-changed-v13i=UNHEALTHY; reef-classification-v3i=HEALTHY | b v3i Unhealthy≠Bleached concept |
| 10 | `3e57e6182a51` | bleaching-condition | train | bleaching-general-v1-yolov8s=HEALTHY; classification-copy-changed-v13i=BLEACHED | c v13i relabel vs v1 |
| 11 | `7800a3ba43dc` | bleaching-condition | train | bleaching-general-v1-yolov8s=HEALTHY; classification-copy-changed-v13i=BLEACHED | c v13i relabel vs v1 |
| 12 | `f2f71758ce1e` | bleaching-condition | train | bleaching-general-v1-yolov8s=HEALTHY; classification-copy-changed-v13i=BLEACHED | c v13i relabel vs v1 |
| 13 | `32407fa9107e` | bleaching-condition | test | bleaching-final-v6i=BLEACHED; bleaching-general-v1-yolov8s=HEALTHY; classification-copy-changed-v13i=BLEACHED | d v1 vs v6i+v13i |
| 14 | `d033200f82c1` | bleaching-condition | test | bleaching-final-v6i=BLEACHED; bleaching-general-v1-yolov8s=HEALTHY; classification-copy-changed-v13i=BLEACHED | d v1 vs v6i+v13i |

- **a (3): within-source duplicates.** The uploader labelled the same frame twice, differently. These are pure noise, and dedup-by-sha resolves them.
- **b (6): v3i against the bleaching family.** This is a concept mismatch, not a mistake. v3i "Unhealthy" includes pale or diseased coral that the family calls Healthy.
- **c (3): v13i relabels visibly pigmented brain coral as Bleached, where v1 has Healthy.** The v13i label looks wrong. It is the "Copy (Changed)" fork editing labels.
- **d (2): v1 against v6i+v13i on pale leather soft coral.** This is genuinely ambiguous, because soft coral paling is not scleractinian bleaching. Both are in the test split, so they are a candidate for the expert sheet.

## 3. Feature cache (design §4.4)

- **Location and format.** `data/_features/dinov2-small/v1/part-*.parquet` plus `FEATURES.json`, which holds the per-part sha256. Each row is keyed by `(image_sha256, model, revision)`.
- **Model.** `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- **Preprocessing.** RGB. The short side is resized to 256 (bicubic, HF `shortest_edge` rule), centre-cropped to 224 and normalised with the ImageNet mean/std.
- **Feature.** CLS ⊕ the mean patch token of the layer-normed `last_hidden_state`, giving 768-d float16.
- **Run.** 69,600 images (every v1 image), 101 MB, 493 s on MPS. Versions: torch 2.14.0, transformers 5.17.0, pillow 11.0.0.

## 4. Confident learning

- **Probe.** An L2 logistic probe on standardised features (numpy IRLS). C is chosen from {0.001, 0.01, 0.1, 1} by out-of-fold log-loss, and 0.01 was an interior pick.
- **Folds.** Out-of-fold predictions use 5 folds **grouped by WP-10 `split_group_id`**, so no near-duplicate group straddles train and held-out.
- **Cleanlab (Northcutt et al. 2021).**
  - The per-class threshold is the mean p_j over samples labelled j.
  - The confident joint is calibrated to the given counts.
  - Noise is the off-diagonal mass.
  - Issues use `filter_by=confident_learning`.
- **CIs.** A group bootstrap (n=1000).

| task | n | C | OOF acc | OOF logloss | thresholds | flagged | noise [95% CI] |
|---|---|---|---|---|---|---|---|
| bleaching-condition | 25,455 | 0.01 | 0.861 | 0.319 | BLEACHED 0.794 / HEALTHY 0.827 | 1,129 | 6.08% [5.12, 6.96] |
| coral-health-binary | 30,011 | 0.01 | 0.848 | 0.349 | HEALTHY 0.809 / UNHEALTHY 0.778 | 1,603 | 7.36% [6.55, 8.21] |

| task | source | n | flagged | noise | 95% CI |
|---|---|---|---|---|---|
| bleaching-condition | noaa-pifsc-bleaching | 10,419 | 645 | 8.74% | [7.30, 10.64] |
| bleaching-condition | roboflow-coral-bleaching-final-v6i | 1,453 | 46 | 4.16% | [2.37, 6.69] |
| bleaching-condition | roboflow-coral-bleaching-general-v1-yolov8s | 1,471 | 48 | 4.41% | [2.55, 7.01] |
| bleaching-condition | roboflow-coral-classification-copy-changed-v13i | 1,730 | 77 | 6.25% | [4.02, 8.87] |
| bleaching-condition | roboflow-coral-reef-bleach-detection-v2i | 10,382 | 313 | 4.17% | [1.51, 5.51] |
| coral-health-binary | noaa-pifsc-bleaching | 10,419 | 673 | 8.96% | [7.51, 10.96] |
| coral-health-binary | roboflow-coral-bleaching-final-v6i | 1,453 | 45 | 4.13% | [2.41, 6.15] |
| coral-health-binary | roboflow-coral-bleaching-general-v1-yolov8s | 1,471 | 49 | 4.51% | [2.77, 6.93] |
| coral-health-binary | roboflow-coral-classification-copy-changed-v13i | 1,730 | 82 | 6.57% | [4.60, 9.00] |
| coral-health-binary | roboflow-coral-reef-bleach-detection-v2i | 10,382 | 344 | 4.47% | [1.57, 5.99] |
| coral-health-binary | roboflow-coral-reef-classification-v3i | 4,556 | 410 | 14.84% | [12.11, 17.70] |

`label_issues.parquet` has 2,732 flagged rows over 1,797 distinct images. Its columns are `image_sha256, task, source_id, sample_key, split, given, suggested, self_confidence, suggested_prob, fold`.

## 5. Model audit of the flags (D-I: a model rater, not an expert)

- **Sample.** 235 flagged images were drawn stratified by task × source (seeded) and shuffled into `M001–M235`.
- **Blinding.** Each was judged from a 256-px thumbnail **without** seeing the given or suggested label. The verdicts are H (healthy), U (bleached/unhealthy), N (not coral) and ? (cannot tell).
- **What was scored.** 115 verdicts (M121–M235) were recorded in `docs/label-quality/model-audit.tsv`. The M001–M120 verdicts were lost in a session compaction before they were written to disk. Their contact sheets are kept for a re-read, and the 115 audited images span all six sources and both tasks.

| | value |
|---|---|
| audited / decided / unsure | 115 / 65 / 50 |
| **flag precision** (given label wrong), decided | **53.8%** (35/65), Wilson 95% [41.8, 65.4] |
| conservative (unsure counted as not-an-error) | 30.4% (35/115) |
| suggestion correct, decided | 46.2% |
| flags pointing to HEALTHY (given BLEACHED/UNHEALTHY) | 19/23 = 83% correct |
| flags pointing to BLEACHED/UNHEALTHY (given HEALTHY) | 16/42 = 38% correct |
| per source (decided n) | noaa 75% (12), v2i 64% (14), v3i 52% (21), v6i 43% (7), v1 33% (6), v13i 20% (5) |

**Mechanism.**
- The probe's "should be bleached" flags fire on pale, blurred and colour-cast frames: a cyan cast, blue strobe light and 224-px NOAA tiles. The DINOv2 CLS+mean feature confounds whiteness with bleaching. Those flags are mostly wrong, so treat them as **"hard or ambiguous"**, not as "mislabelled".
- The "should be healthy" flags are mostly right: given-Bleached images show pigmented coral. They are real label errors, concentrated in the crowd sources.
- Five images labelled for coral health show no coral (anemones, a sponge, a nudibranch).
- Rough corrected noise = CL noise × flag precision ≈ 7.4% × 0.54 ≈ **4% (health)** and 6.1% × 0.53 ≈ **3% (bleaching)**. Both are wide; the expert audit (§6) replaces this estimate.

## 6. Expert-audit protocol (500 samples; to be run by Yohan's team)

- **Files.**
  - `expert-audit-sheet.tsv` is **blind**: audit_id, sha, sample_key, split and blank expert columns.
  - `expert-audit-key.tsv` holds source, given label, CL flag, stratum and weight. **Do not give it to the annotators.**
- **Strata.** Source × given health label × CL flag, 24 strata over 29,766 (sha, source) units. Half the budget is spread equally across strata and half proportionally, and small strata are taken whole. 131 of the 500 rows are CL-flagged.
- **Weight.** `N_h/n_h`. Estimate population noise with the weighted (Horvitz-Thompson) mean, never the raw mean.
- **Annotators.** Two independent annotators with coral-reef survey experience (e.g. CoralWatch or Reef Check trained), plus an adjudicator for disagreements.
- **Conditions.** `HEALTHY, PALE, BLEACHED, OTHER_UNHEALTHY, DEAD, NOT_CORAL, UNSURE`. Confidence 1–3.
- **Instructions.**
  1. Judge the coral that dominates the frame. If there is no scleractinian or soft coral, use `NOT_CORAL`.
  2. `PALE` is visible pigment loss with tissue still coloured. `BLEACHED` is white tissue over an intact skeleton (with polyps visible). `DEAD` is skeleton overgrown by turf or algae. `OTHER_UNHEALTHY` covers disease bands, lesions and predation scars (e.g. crown-of-thorns).
  3. Do not correct for colour cast by guessing. If a cyan or blue cast, blur or 224-px resolution prevents a call, use `UNSURE`.
  4. Work alone and do not look at the other annotator's column. The adjudicator fills `adjudicated_condition` only where they differ.
- **Analysis.**
  - Inter-expert κ gives the human ceiling.
  - Per-source weighted error comes with a bootstrap CI, using the mappings HEALTHY→HEALTHY and PALE, BLEACHED, OTHER_UNHEALTHY and DEAD→UNHEALTHY. The bleaching task uses BLEACHED only.
  - The flag stratum gives the CL flag's precision and recall.
- **Effort.** At about 20 s per image per annotator, the job is roughly 6 h of annotation in total.

## 7. Residuals (human-only, D-I)

1. The expert sheet (§6) is unfilled. Until it is filled, all noise figures are model-estimated.
2. NOAA annotator training and the `reef-support-bleaching` annotators need confirming (see `label-origin.yaml`).
3. Whether v3i "Unhealthy" should roll up into coral-health-binary at all is a curation decision, not a measurement.
4. The 2 category-d soft-coral conflicts in the test split need an expert call.

## Reproduce

```
marinedata labelquality features  --hf data/_hf/v1 --out data/_features/dinov2-small/v1 --revision ed25f3a31f01632728cabb09d1542f84ab7b0056
marinedata labelquality agreement --hf data/_hf/v1 --dhash-db ~/.cache/marinedata/_dhash/dhash-pillow-12.3.0.sqlite --out data/_labelquality/<date>
marinedata labelquality confident --hf data/_hf/v1 --groups <WP-10 groups.parquet> --features data/_features/dinov2-small/v1 --out data/_labelquality/<date>
```
