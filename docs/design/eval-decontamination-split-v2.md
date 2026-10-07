# Design: benchmark decontamination, split v2 with OOD holdouts, eval harness

Status: design, 2026-09-25. Owner of the decisions below: this document. Implementation: the five packages in §5.
Inputs: rating r0 (decontamination and split specifications), the rating rubric, the source catalog (168 sources), the dedup stage specification,
the metadata specification (MEOW/depth/habitat fields), `sample_schema.py` (field names used verbatim).
Companion file: `registry/benchmarks.yaml` (28 benchmarks, schema in its header).

## 0. What is actually wrong with v1's split (the 85.7/7.2/7.1 drift)

The drift is **not** a group-packing failure. Measured on the v1 HF `metadata` config (69,600 rows):

| pool | rows | train | val | test |
|---|---|---|---|---|
| `coralscop-masks-rs` (tag `never-eval`, model pseudo-masks) | 37,273 | 100% | 0 | 0 |
| every other source (eval-eligible) | 32,327 | 22,459 (69.5%) | 4,947 (15.3%) | 4,921 (15.2%) |
| all rows | 69,600 | 85.8% | 7.1% | 7.1% |

`release._never_eval_source_ids` forces the never-eval source to train, and `check_ratios` then measures the union against
70/15/15. The eligible pool is on target. Consequences for v2:
1. Ratios are defined and gated over the **eval-eligible in-distribution (ID) pool only**. Never-eval / pseudo-label /
   unlabelled sources form a separate `train-only` pool that the card reports on its own line.
2. Group-size-awareness is still needed, but for a different, smaller defect: coarse strata (v1's
   `roboflow-coral-classification-copy-changed-v13i`, 1,172 images, came out 50/25/25) and upstream-pinned strata (§3.3).
3. Rejected: raising eligible sources' val/test to ~32% each so the all-rows number reads 70/15/15. That starves
   labelled training data to fix a reporting artefact.

## 1. Benchmark registry (`registry/benchmarks.yaml`)

One entry per public benchmark whose test images could leak into our train. Fields (all required unless marked):
`id`, `name`, `task` (cls|det|inst-seg|sem-seg|points|vqa|enhancement|salient|depth), `catalog_id` (S57 TSV id),
`registry_id` (our `registry/sources` id or null), `upstream_split` {`rule`, `eval_split` (the split papers report on),
`counts` {split: n}, `definition_url`}, `split_verified` (true only if the split files/counts were seen in upstream metadata),
`verified_by` (what was seen), `obtain` {`status`: staged|w1|w2|w3|needs-yohan|unfetchable, `via`}, `policy`
(route-to-our-test|exclude), `policy_reason`, `chain` (S57 duplicate-chain ids it shares images with).

Policy rules (the yaml applies them; a new benchmark follows the same rules):
- **route-to-our-test** (default): the upstream `eval_split` images are pinned to our `test` (or to an OOD split if an OOD rule
  claims them first, §3.2). A second held-out upstream split (val when test also exists) is pinned to our `val`.
  Upstream train images are free for the allocator. Any `split_group_id` containing a pinned image takes the pin.
- **exclude**: the benchmark's eval images and every image in their `split_group_id` are removed from **all** splits of our
  release, and stay in the decontamination manifest. Used when (a) the benchmark is eval-only and its value depends on staying
  external (MarineEval, U45, SQUID), or (b) the set is a test set we do not carry as a task layer. Licence is never a reason.
- A benchmark we cannot fetch (`needs-yohan`/`unfetchable`) is listed in the card as "not checked"; it counts toward neither
  the ≥ 5 nor the ≥ 15 target.

Coverage (details and URLs in the yaml): 28 benchmarks, 14 with the split verified from upstream metadata this session
(Coralscapes 1517/166/392, SUIM train_val 1525 / TEST 110, UIEB-mirror 800/90, DeepFish `{split}.csv` rule, FathomNet FGVC23
depth rule, FathomNet VME 23,636/5,846/685, RUOD `instances_{train,val}.json`, USIS10K `*_{train,val,test}_annotations.json`,
UIIS 3,937/691, UIIS10K `multiclass_{train,test}.json`, MarineEval eval-only tree, PLC no-split README, CoralVQA 226,726/27,984
QA rows, URPC-mirror 4,602/1,973). The other 14 carry the paper's split as `split_verified: false`; package P1 must fetch and
confirm each before its manifest is marked complete.

## 2. Contamination thresholds and the CI gate

### 2.1 Stages (reuse the `marinedata.dedup` functions; no second implementation)
Evaluated for every (our image, benchmark eval image) pair the index returns, cheapest first; the first stage that fires wins.

| stage | signal | contaminated when | review band (logged) |
|---|---|---|---|
| S0 id | `upstream_id` / FathomNet uuid / CoralNet image id / URL | exact equality | — |
| S1 bytes | sha256 of file bytes | equal | — |
| S2 pixels | sha256 of decoded RGB8 (EXIF-transposed, alpha dropped) + shape | equal | — |
| S3 perceptual | candidates: dHash-64 Hamming ≤ 12 via the dedup band index; decide on pHash-256 | pHash-256 ≤ 32 **and** grey entropy ≥ 4.0 bits | pHash 33–48, or ≤ 32 with entropy < 4.0 (goes to S4) |
| S4 embedding | cosine of the dedup-pinned embedding (default `facebook/dinov2-small` CLS, L2-norm) on S3 candidates **plus** ANN top-10 per benchmark image | cosine ≥ τ_decon | τ_decon − 0.05 ≤ cos < τ_decon |
| S5 patch→parent | our patch's parent (key (parent id,row,col) or dedup matcher NCC ≥ 0.90) is a benchmark eval image, or a benchmark patch's parent is ours | parent contaminated | matcher 0.80–0.90 |

τ_decon = τ_dedup − 0.03, clamped to [0.85, 0.95], where τ_dedup is the dedup precision-95 operating point from its audited
pairs. Until the dedup report publishes it, τ_dedup = 0.93, so τ_decon = 0.90. Decontamination deliberately sits on the recall side:
a false positive only costs one dropped training image. The entropy guard exists because of the S47 low-texture
(blue-water) false merges: for those, perceptual alone never decides.
The thresholds, the embedding model id and revision, and τ_dedup's calibration-report sha256 are pinned in the yaml's
`thresholds:` block. Changing any of them changes `benchmarks_sha256` (§3.5).

### 2.2 Gate semantics — `marinedata decon check <release_dir>` (exit 1 on any failure)
1. Any confirmed hit (S0–S5) in `train`, `val`, the `train-only` pool or `general-pretraining` for any benchmark: **count ≥ 1 fails**.
2. Any hit, in any split, for an `exclude` benchmark: count ≥ 1 fails.
3. A `route-to-our-test` eval image found anywhere except `test`/`ood-*`: fails. This is the same as rule 1, stated per pin.
4. Coverage: a benchmark with `obtain.status` in {staged, w1, w2} and no manifest, or with manifest rows < 99% of
   `upstream_split.counts[eval_split]`, fails. That turns a silent skip into a failure.
5. Review-band hits in train/val/train-only > max(5, 1% of the benchmark's eval n): fails (it means the thresholds are
   miscalibrated). Otherwise each review hit is listed, not failed.
The gate runs inside `release.py` behind `--decon` (default off until P2 lands, then on for v2) and as a `make ci` step on the
fixture release.

### 2.3 Overlap table (card section "Benchmark contamination"; also `decon/overlap.parquet`)
```
| benchmark | task | eval split (n) | hashed | policy | train S0-2/S3/S4/S5 | val | train-only | test | ood-* | excluded | status |
| coralscapes | sem-seg | test (392) | 392 (100%) | route | 0/0/0/0 | 0 | 0 | 392 | 0 | 0 | clean |
```
The table is followed by a thresholds line (S3 ≤ 32 / S4 ≥ 0.90 dinov2-small@rev / S5 NCC ≥ 0.90), the manifest set's
sha256, and a "not checked" list with the reason for each benchmark. `status` ∈ clean | contaminated | not-checked.

## 3. Split v2

### 3.1 Pools, in the order assignment runs (a sample's `split_group_id` is the unit throughout)
1. **excluded**: groups containing an `exclude` benchmark eval image. Not shipped.
2. **ood-***: holdouts (§3.2), evaluated per sample; a group goes OOD if **any** member matches. First match in the precedence
   order below wins `split`; every matching holdout is recorded in a new `ood_tags` list column.
3. **pinned**: upstream eval → `test`, second upstream held-out split → `val` (§1); then **v1 continuity**: a group
   whose images were in v1 `val`/`test` is pinned to the same v2 split, so nobody who evaluated on v1 test finds it in v2 train.
4. **train-only**: groups whose sources are all tagged `never-eval` (pseudo labels, unlabelled pretrain sources).
5. **free**: everything else → the allocator (§3.3).

### 3.2 OOD holdouts (6). Rules live in `registry/splits/v2.yaml`; precedence = list order
| # | split name | exact rule (per sample) | fallback when the field is null |
|---|---|---|---|
| 1 | `ood-source-deepfish` | `source_id == "deepfish"` | none needed |
| 2 | `ood-geo-temperate-australasia` | `meow_realm == "Temperate Australasia"` | `geo_fallback[source_id].realm` if the source is single-realm |
| 3 | `ood-geo-mediterranean` | `meow_province == "Mediterranean Sea"` (realm Temperate Northern Atlantic) | `geo_fallback[source_id].province` |
| 4 | `ood-depth-deep` | `depth_m >= 800` (the FathomNet FGVC23 train ceiling; its eval spans 0–1300 m) | source nominal depth range only if its minimum is ≥ 800 |
| 5 | `ood-platform-auv` | `platform == "auv"` | `platform_fallback[source_id]` if single-platform |
| 6 | `ood-time-2025` | `capture_datetime >= 2025-01-01T00:00Z`, only for sources with groups on both sides of the cutoff | none; null stays ID |

Missing metadata **never** sends a sample OOD; it stays ID. Initial `geo_fallback` entries: deepseagrass → Temperate
Australasia; benthoz15 → Temperate Australasia; obsea-fish, seaclear-marine-debris, posidonia, med-fish → Mediterranean Sea;
mlc-moorea → Eastern Indo-Pacific; coralscapes → Western Indo-Pacific / Red Sea and Gulf of Aden; noaa-pifsc-bleaching →
Eastern Indo-Pacific / Hawaii; reef-support-benthic-own → Tropical Atlantic. Multi-realm sources without per-sample
geography (roboflow-*, mermaid-aws, seaview) get no fallback.
Size guards (CI fails, and a human narrows the rule in the yaml; nothing is silently subsampled):
- A holdout is **constructible** if it has ≥ 500 images and ≥ 5 groups. Otherwise it is reported as "not constructible in <release>".
  #5 widens once to `platform in {auv, towed}` before it gives up.
- Each holdout is ≤ 10% of the eligible corpus, and all holdouts together are ≤ 25%.
Choice rationale: two realms far from the tropical-reef core (#2, #3) satisfy the "≥ 2 realms" criterion without removing
Reef Support's Caribbean/Colombia data or Coralscapes (Red Sea) from training. DeepFish is the source holdout: fixed-camera
fish imagery with no S57 duplicate chain.

### 3.3 Allocator (group-size-aware, per stratum = `source_id`, shared groups as in `strata.py`)
Targets: `train/val/test = 0.70/0.15/0.15` of each stratum's **eligible ID** images (pinned + free).
1. Pinned groups fill their split first.
2. **Test-overfull stratum** (pinned test ≥ 0.15 × stratum, e.g. MLC, RUOD, FGVC23): the non-test pool P = free + pinned-val
   is split train:val = 70:15 (82.35/17.65), with pinned val counting toward val. The stratum is exempt from the ratio gate
   and listed in the card as `test-heavy (upstream)`.
3. Otherwise sort free groups by `(-size, sha256(f"{seed}:{split_group_id}"))`. A group goes to the split with the largest
   normalised deficit `(target − filled) / target` **among splits where `filled + size ≤ target × 1.10`**. If no split
   qualifies, it goes to `train`. The v1 packer let a big group overshoot val/test; this is the fix.
4. Repair pass for strata off by more than tolerance: pairwise swaps between `train` and the short split, trying the
   smallest groups first, at most 200 swaps, accepting only swaps that reduce the L1 error. Deterministic order.
5. Strata with < 3 groups → train-only (the existing `DEFAULT_MIN_GROUPS` rule, disclosed).
Tolerances (the gate is `marinedata splits check`): the eligible ID pool overall within ±1.5 pt of each target; every
unpinned stratum with ≥ 20 groups within ±2.0 pt; a coarse stratum (< 20 groups, or largest group > 10% of the stratum)
within ±max(2.0 pt, largest-group share); test-overfull strata exempt but reported.

### 3.4 Pretrain relation
`general-pretraining` = ID `train` ∪ `train-only`. Its `probe` split = ID `val`. ID `test`, `ood-*` and `excluded` never
enter pretraining. Decon gate rule 1 covers it.

### 3.5 Determinism
Seed `0` (continuity with v1). SPLIT_MAP v2 = `schema_version: 2`, keyed by `split_group_id`, append-only as in v1, and it adds
`rules_sha256` (of `registry/splits/v2.yaml`), `benchmarks_sha256` (of `registry/benchmarks.yaml`), and `map_sha256` =
sha256 of the canonical JSON (sorted keys, `separators=(",", ":")`) of {seed, ratios, rules_sha256, benchmarks_sha256,
assignments}. RELEASE.json's existing `split_map_sha256` records it. A regeneration test must reproduce it byte for byte.

### 3.6 Worked example: v1 sources + W1 sources (≈ = count from the paper, not yet verified)
| source | images | pool/rule | train | val | test | OOD / other |
|---|---|---|---|---|---|---|
| v1 eligible (7 sources) | 32,327 | pinned by v1 continuity | 22,459 | 4,947 | 4,921 | — |
| coralscop-masks-rs | 37,273 | train-only (never-eval) | — | — | — | 37,273 train-only |
| coralscapes (have) | 2,075 | route; test 392 overfull | 1,386 | 297 | 392 | — |
| mlc-moorea | 2,055 | route 2009 (695) + 2010 (689) ≈; overfull | 553 | 118 | 1,384 | — |
| plc-beijbom2015 | ≈5,090 | no upstream split; free | 3,562 | 764 | 764 | — |
| noaa-benthic-t1 (patches) | 790,352 | free, grouped patch→parent | 553,246 | 118,553 | 118,553 | — |
| noaa-benthic-t3 | 0 new | label table on t1 | — | — | — | — |
| usis10k | 10,632 | route test ≈1,596, val ≈1,594 | 7,441 | 1,595 | 1,596 | — |
| uiis ∪ uiis10k (UIIS ⊂ UIIS10K) | 10,048 | route UIIS10K test ≈2,010 ∪ UIIS val 691 | 6,050–6,620 | 1,297–1,418 | 2,010–2,701 | — |
| ruod | 14,000 | route test ≈4,200; overfull | 8,071 | 1,729 | 4,200 | — |
| trashcan | 7,212 | route val ≈1,147; overfull | 4,995 | 1,070 | 1,147 | — |
| fathomnet-fgvc23 | ≈16,694 | route eval ≈10,744; overfull | 4,900 | 1,050 | ≤10,744 | eval images ≥ 800 m → ood-depth-deep |
| fathomnet-vme | 30,167 | route test 685, val 5,846 | 23,636 | 5,846 | 685 | — |
| deepseagrass | ≈1,701 parents | fallback realm | — | — | — | 1,701 ood-geo-temperate-australasia |
| obsea-fish | ? | fallback province | — | — | — | all ood-geo-mediterranean |
| marineeval | 827 files | exclude | — | — | — | excluded |
| **ID total (UIIS lower bound)** | 920,652 | | 636,869 (69.2%) | 137,387 (14.9%) | 146,396 (15.9%) | passes ±1.5 |
With v1 + W1 only, holdouts #1, #5 and #6 are **not constructible** (DeepFish is not staged; platform and datetime
coverage is 0% until the metadata backfill lands), #4 depends on FathomNet depth metadata, and #2 (1,701) and #3 are
constructible. The four-realm criterion therefore needs metadata coverage plus the W2 sources (benthoz15, seaclear, deepfish) before v2 is cut.
FGVC23, VME and later FathomNet share images: the dedup unions merge their groups, so the true totals are lower.

## 4. Eval harness

### 4.1 CLI (new package `src/marinedata/eval/`, entry `marinedata eval`)
- `eval score --release <dir|tag> --task <task> --split test|val|ood-<name>|all --pred <preds.parquet> [--boot 1000] [--seed 20260925] --out <dir>`
  Predictions: parquet with `image_sha256` plus a task-typed column: `pred_label` + `pred_probs` (cls/points; points also carry
  `point_id`), `pred_mask_png` (sem-seg, same H×W as GT), COCO results JSON rows (det/inst-seg), `pred_answer` (vqa).
  Writes `metrics.json` (point estimate, CI, n_images, n_groups) and a per-sample scores parquet.
- `eval baseline --task <task> --method probe|finetune --backbone dinov2-small --seed <s> --device mps|cpu|cuda --out <dir>`
- `eval reproduce --release v2 --config configs/baselines/v2.yaml` runs every configured baseline and compares against
  `results/v2/baselines.parquet`; exit 1 if any cell differs by more than 0.5 pt.
- `eval table --release v2 --out results/v2/` writes `baselines.parquet` and `baselines.md` (the card table).

### 4.2 Metric registry (`eval/metrics.py`, `METRICS: dict[str, Metric]`, each with `task_types`, `fn`, `higher_is_better`, `unit="pt"`)
- `macro_f1`: unweighted mean F1 over the classes with ≥ 1 GT sample in the evaluated split. Predicted-but-absent classes
  only add false positives. Cross-checked against `sklearn.metrics.f1_score(average="macro", labels=present)`.
- `miou`: one confusion matrix accumulated over the split (dataset-level, not a per-image mean), `ignore_index=255`,
  classes with union 0 dropped.
- `map`: COCO mAP@[.5:.95] via `pycocotools` (bbox and segm), 101-point interpolation, maxDets 100; `map50` reported alongside.
- `point_acc`: micro top-1 accuracy over points; `point_macro_recall` alongside.
- `vqa_acc`: exact match after normalisation (lowercase, strip punctuation and articles, number words → digits); a
  multiple-choice item matches on the option letter.
- `ece`: 15 equal-width bins on max-softmax confidence, L1-weighted; reported for every cls/points task.
Golden-value tests: each metric against a hand-computed fixture and its reference library.

### 4.3 Bootstrap CIs
Cluster bootstrap over `split_group_id` (images in a group are correlated), n = 1000, seed 20260925, 95% **percentile**
interval. mAP resamples cached per-image match arrays rather than re-running COCOeval, so n = 1000 stays cheap.
Rejected: BCa. Its jackknife needs one metric recompute per group (thousands for mAP), and cluster-BCa is non-standard.
At test sizes ≥ 1k groups the percentile interval is adequate. Revisit if a task's bootstrap skewness exceeds |0.5|.

### 4.4 Baselines (`configs/baselines/v2.yaml`; weights pinned by HF revision sha, no login)
**A. Frozen linear probe**, backbone `facebook/dinov2-small` (ViT-S/14).
- Tasks: coral-health-binary, bleaching-condition, benthic-coarse, benthic-l2 (cls); points (224 crop centred on the point);
  sem-seg (Coralscapes-class tasks).
- Input: resize the short side to 256, centre-crop 224, ImageNet mean/std. Features = CLS ⊕ mean patch token (768-d), cached
  as float16 parquet keyed by (image_sha256, model, revision).
- cls/points: `LogisticRegression(lbfgs, multinomial, max_iter=1000)`, C ∈ {0.01, 0.1, 1, 10} chosen on val by macro-F1, CPU,
  train capped at 100k images per task (seeded group sample). Deterministic.
- sem-seg: a linear 1×1 head on the 16×16 patch grid, bilinear upsample, AdamW lr 1e-3, 20 epochs, batch 32, seed 0.
**B. Small fine-tune**, the same backbone: last 4 blocks + head trainable.
- Optimiser: AdamW, lr 5e-5 (backbone) / 1e-3 (head), wd 0.05, cosine schedule, 1 warm-up epoch, 10 epochs, batch 64.
- Precision: fp32 on MPS, bf16 on CUDA. Seeds {0, 1, 2}, reported as the mean.
- Tasks: the cls tasks, points, sem-seg (a linear head as in A, backbone unfrozen), and det via
  `torchvision fasterrcnn_mobilenet_v3_large_fpn` (COCO weights), 12 epochs, on ≤ 20k train images.
- inst-seg and VQA baselines are deferred: inst-seg needs a GPU, and VQA moves to the captions layer (zero-shot open VLM). Their metrics exist now.
**Compute** (planning numbers; P5 must replace them with measured ones):
- Feature extraction: ~150 img/s on M-series MPS (decode-bound), ~8 img/s on the server Job (ingest Job: 4 vCPU/8 GiB, no GPU).
  At ~1M images that is about 2 h on the Mac against about 35 h on the Job. Features are therefore extracted on the Mac;
  the Job runs the CPU probe fits and the bootstraps.
- Fine-tune: ~50 img/s on MPS. Capped at 50k images × 10 epochs ≈ 3 h per seed per task, so it is Mac-only (overnight).
  The Job is not viable for it.
**Reproducibility**:
- The probe is bit-stable given cached features. Re-extracting on MPS moves metrics by < 0.1 pt; checked in P5 by
  re-extracting a 2k subset.
- Fine-tune: `torch.use_deterministic_algorithms(True)` on CPU/CUDA. MPS is not fully deterministic, so the ±0.5 pt
  tolerance applies to the 3-seed mean per (task, split, metric) cell. If a cell's seed std is > 0.3 pt, that task moves to 5 seeds.

### 4.5 Card results table (`results/<release>/baselines.md`, committed per release)
```
| task | split | metric | method | backbone@rev | score | 95% CI | seeds | n img / groups | config sha | harness commit |
| bleaching-condition | test | macro_f1 | probe | dinov2-small@<sha> | 71.3 | [69.8, 72.7] | 0 | 4,921 / 1,012 | 3f2a… | abcd123 |
| bleaching-condition | ood-geo-temperate-australasia | macro_f1 | probe | … | … | … | … | … | … | … |
```
Each task gets one row per (split ∈ test + every constructible `ood-*`) × method, then an ID-to-OOD gap line. The numbers
in the example row are illustrative only.

## 5. Implementation plan (5 sonnet packages)
| pkg | scope | files | tests / acceptance | needs merged first |
|---|---|---|---|---|
| P1 bench-registry | load + validate `registry/benchmarks.yaml`; confirm the 14 unverified splits (metadata only); build per-benchmark eval-image manifests (S0–S5 signals) for every `obtain` in {staged, w1} | `src/marinedata/benchmarks.py`, `cli_bench.py`, `registry/benchmarks/manifests/<id>.parquet`, yaml edits | schema test over all entries; manifest row count ≥ 99% of eval n for ≥ 5 benchmarks; `split_verified` flips only with a URL | dedup (`dedup/` hashes, embed cache), W1-A |
| P2 decon-gate | `marinedata decon check` + overlap parquet/md; `--decon` flag in `release.py` (one line) | `src/marinedata/decon.py`, `cli_decon.py`, `tests/test_decon.py` | fixture: planted S1/S2/S3/S4/S5 hits each fail; clean fixture passes; exclude-policy hit in test fails; coverage < 99% fails; review-band limit | P1, dedup |
| P3 split-v2 | pools, OOD rules, allocator, repair pass, SPLIT_MAP v2 + hashes, `marinedata splits check` | `src/marinedata/splitv2/{rules,holdouts,allocate,mapfile}.py`, `registry/splits/v2.yaml`, `tests/test_splitv2*.py` | §3.6 recomputed from a synthetic fixture to the same counts; v13i-like coarse stratum within tolerance; any-member-OOD group rule; missing metadata stays ID; byte-identical regeneration; v1 map untouched | dedup (`split_group_id`), metadata (meow/depth fields; fallbacks cover its absence), P1 (pins) |
| P4 eval-core | `eval score`, metric registry, cluster bootstrap, prediction readers | `src/marinedata/eval/{cli,metrics,bootstrap,predictions}.py`, `tests/test_eval_metrics.py` | golden tests vs sklearn/pycocotools within 1e-6; bootstrap is seed-stable; group resampling proven by fixture | none (parallel-safe; split names from P3's yaml) |
| P5 baselines | feature cache, probe, fine-tune, `eval reproduce`/`eval table`, card table (one-line hook in `hf_card.py`, disclosed) | `src/marinedata/eval/baselines/{features,probe,finetune}.py`, `configs/baselines/v2.yaml`, `results/v2/` | `eval reproduce` on a 2k fixture matches within ±0.5 pt twice; measured MPS/Job throughput recorded | P3, P4 (`hf_card` owner) |
Order: P1 ∥ P4 → P2 ∥ P3 → P5. The integrator flips `--decon` and `--dedup-v2` together at the v2 build.
