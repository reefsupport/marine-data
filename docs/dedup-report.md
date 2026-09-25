# Dedup v2 report (WP-10, 2026-09-25)

Package `src/marinedata/dedup/`. CLI: `marinedata dedup {run,eval,bench,audit,gate}`.
Outputs are in `~/dev/reefsupport/data/_dedup/2026-09-25/`: `hashes`, `pairs`, `clusters` and
`groups` parquet files, `summary.json`, `eval.json`, and `audit/`.

## Corpus

| corpus | image rows | access |
|---|---|---|
| v1 HF export, `images` config (`data/_hf/v1/`) | 69,600 | read-only |
| mermaid-aws staged tree, 2026-09-19 | 18,029 | read-only |
| coralscapes 1.0 | 2,074 | read-only |

There are 90,236 records and 89,703 unique sha256. Decode errors: 0.

## Pipeline

1. **Features.** From one decode: sha256 of the bytes, and sha256 of the decoded pixels.
   - dHash-64, bit-identical to v1 `neardup.dhash_file`.
   - pHash-256 (64x64 DCT, 16x16 block) plus pHash-64.
   - Flip variants of each hash.
   - A texture margin: the mean adjacent difference in the 9x8 thumbnail. An image is lowtex when the margin is < 3.0.
2. **Candidates.** There are three channels:
   - equal pixel sha;
   - multi-index Hamming search (4x16-bit bands, radius 8) on dHash and pHash-64, with identity, h-flip and v-flip probes;
   - SSCD (`sscd_disc_mixup`, 512-d) cosine kNN, k=10, cos >= 0.5. Embeddings are cached by sha256.
3. **Confirmation.**
   - An exact or pixel match always confirms.
   - Otherwise the pair confirms on `cos >= 0.75`, or on `cos >= 0.5` when a hash agrees (dHash <= 8 or pHash-256 <= 56).
   - **Texture guard:** if either image is lowtex, a hash match proves nothing, so the pair needs `cos >= 0.85`.
   - **Crop matcher:** used when the area ratio is <= 0.95 and cos >= 0.5. It runs a multi-scale FFT NCC between the patch and its parent, and confirms at NCC >= 0.85.
4. **Grouping.** Union-find over confirmed pairs gives `dup_cluster_id`.
   - `split_group_id` also unions the declared keys: video/sequence/dive/campaign, stereo, and patch→parent.
   - Upstream split membership is carried per group as `group_upstream_splits`.

## Full run

| stage | total | breakdown |
|---|---|---|
| candidates | 221,899 | pixel 6 · hash 107,801 · embed 197,167 |
| confirmed | 71,476 | copy 65,979 · agree 5,306 · pixel 6 · crop 185 |

- The crop check covered 8,476 pairs (5,537 images) and took about 18 min.
- There are 4,359 multi-member dup clusters. The largest has 190 members.
- 100 images are lowtex, and none of them is in a confirmed lowtex pair.

## Cross-source overlap

6,686 unique images sit in 1,271 dup clusters that span more than one source. Nearly all of them are
inside v1, where the Roboflow datasets re-export the same footage: flips, rotations and recolours
of the same video frames.

Across corpora (v1 ↔ mermaid ↔ coralscapes) there are only 3 clusters (7 images). All 3 were
confirmed by the crop matcher at cos 0.50–0.53 and NCC 0.86–0.93. They were not audited, so treat
them as suspect (see Open).

`matrix[A][B]` counts the unique images of A whose dup cluster also holds an image of B. The
diagonal counts images of A that have another A image in their cluster. The table lists only the
sources with some off-diagonal overlap.

| source (unique imgs) | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|---|
| S0 coralscapes@1.0 (2074) | 8 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| S1 mermaid-aws@2026-09-19 (18029) | 0 | 622 | 2 | 0 | 0 | 0 | 0 | 0 |
| S2 v1:coralscop-masks-rs (37273) | 1 | 3 | 3062 | 67 | 64 | 33 | 5 | 20 |
| S3 v1:roboflow-coral-bleaching-final-v6i (2550) | 0 | 0 | 167 | 2154 | 2289 | 1064 | 16 | 988 |
| S4 v1:roboflow-coral-bleaching-general-v1-yolov8s (1925) | 0 | 0 | 145 | 1925 | 1482 | 730 | 13 | 602 |
| S5 v1:roboflow-coral-classification-copy-changed-v13i (1172) | 0 | 0 | 26 | 418 | 355 | 79 | 1 | 405 |
| S6 v1:roboflow-coral-reef-bleach-detection-v2i (10544) | 0 | 0 | 102 | 21 | 21 | 1 | 10380 | 25 |
| S7 v1:roboflow-coral-reef-classification-v3i (4467) | 0 | 0 | 91 | 793 | 640 | 967 | 10 | 4197 |

## Synthetic recall (brief: >= 90% on >= 2,000 pairs)

The smoke run (v1 test shard, 150 derivatives, every family re-encoded to JPEG) reached **97.3%** recall with hash + SSCD kNN (crop 90%) and 80.0% with hash candidates only. Positive-pair cosine: p1 0.66 · p5 0.72 · p10 0.77 · p50 0.96.

The acceptance run (`marinedata dedup eval --per-family 400`, n = 2,000) was still running at commit time. Its results land in `eval.json`.

## Precision audit (D-I model audit, not a human audit)

- 120 confirmed pairs were sampled, stratified by distance band and lowtex. 100 were judged from contact sheets (`audit/sheet_00..04.jpg`).
- Every judged pair is a duplicate (100/100):
  - 96 are the same image or a derivative (flip, rotate, crop, recolour, re-encode);
  - 4 are the same scene (adjacent video frames with a different subtitle or overlay). These are true positives for split leakage.
- Wilson 95% lower bound: **96.3%**.

| stratum | true / judged |
|---|---|
| d0-2 | 24/24 |
| d3-5 | 24/24 |
| d6-8 | 24/24 |
| embed-only | 24/24 |
| exact | 4/4 |

Strata with no coverage: lowtex (no confirmed lowtex pairs exist), the 185 `crop` confirmations,
and cross-corpus pairs. Verdicts are in `audit/audit.tsv`. P100–P119 are not reviewed.

## Low-texture false merges (S47)

This replays v1 rule A (dHash <= 8) over the 69,600 v1 images and asks what v2 confirms.

| | v1 pairs | kept by v2 |
|---|---|---|
| all | 26,855 | 22,135 |
| lowtex pairs | **10** | **0** |
| cross split-group | 14,526 | 9,823 |
| lowtex, cross split-group | 10 | 0 |

Rule B (dHash <= 4) is in `eval.json`, the same pending run.

v2 rejects all 10 lowtex pairs because their SSCD cosine is < 0.85. Nobody has visually checked
whether all 10 were false.

## Scaling: multi-index Hamming at 1M synthetic codes

The benchmark planted 1% near-duplicates (3 flipped bits).

| measure | result |
|---|---|
| build | 0.02 s |
| index size | 24.9 MB |
| r3 query | 0.6 s, planted recall 1.0 |
| r7 query | 12.6 s, planted recall 1.0 |
| peak RSS | 0.58 GB |

At 5M codes the index is about 125 MB. That figure is extrapolated, not measured.

## Leak gate

`marinedata dedup gate <release_dir> --groups groups.parquet [--allow-ungrouped] [--write]` fails
in three cases:

- a split_group spans splits;
- a group that holds an upstream-test image appears in train;
- a release sha is ungrouped, unless `--allow-ungrouped` is set.

`marinedata release build --dedup-v2 GROUPS` runs the gate after the manifests are written and
before RELEASE.json. On success it writes `DEDUP_GATE.json` and adds `dedup_v2` to RELEASE.json.
The option defaults to off, and the v1 code path is unchanged when it is off.

## Deviations from the brief

1. **SSCD kNN runs over all images, not "candidates only".** No 64-bit hash reaches a 60% crop.
   Hash-only recall is reported next to it.
2. **Upstream split membership is carried, not unioned as a key.** Unioning it would fold a whole
   upstream test split into one group. The gate checks it directly.
3. **mermaid-aws has no sequence or dive key in the staged tree.** Only dup clusters group it.
4. **The gate has not been run on a built release directory.** Its tests use fixtures.
   `audit render` assumes every item has a sha256, so staged-only pairs are not renderable yet.
