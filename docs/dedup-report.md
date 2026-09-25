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
confirmed by the crop matcher at cos 0.50–0.53 and NCC 0.86–0.93. WP-10b audited them: all 3 are false merges (see the crop audit below).

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

| mode | n | recall | candidate recall | resize | jpeg | crop | jitter | flip |
|---|---|---|---|---|---|---|---|---|
| hash+embed-knn | 2000 | 97.25% | 99.85% | 99.50% | 99.50% | 87.25% | 100.00% | 100.00% |
| hash-candidates-only | 2000 | 80.15% | 80.35% | 99.50% | 99.50% | 1.75% | 100.00% | 100.00% |

- Lowtex queries: 6, recall 33%. The texture guard costs recall on flat images, but n is tiny.
- Crop pairs confirmed: 197.
- Queries that also confirmed another corpus image: 647. These are mostly the source's own augment-copy siblings.
- Positive-pair cosine percentiles: {'1': 0.6344028115272522, '5': 0.7127112150192261, '10': 0.7576063871383667, '50': 0.9724259376525879}.
- Every derivative is re-encoded to JPEG. The seed is 0.

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

Rule B (dHash <= 4): `{"cross_group_kept_v2": 4648, "cross_group_pairs": 4720, "lowtex_cross_group_kept_v2": 0, "lowtex_cross_group_pairs": 0, "lowtex_kept_v2": 0, "lowtex_pairs": 0, "pairs": 15129, "pairs_kept_v2": 15057}`.

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

## Crop and cross-corpus audit (WP-10b, D-I model audit, not a human audit)

All 185 `crop` confirmations were judged (a census, not a sample) from the patch, the NCC box cut out of
the parent, and the parent, side by side. Verdicts: `docs/dedup-crop-audit-2026-09-25.tsv`.

- derivative 3 · same scene 33 · different 149.
- Precision, duplicate or same scene: **36/185 = 19.5% [Wilson 95% 14.4–25.8]**. As a crop detector: 3/185 = 1.6% [0.6–4.7].
- By cos band: [0.50,0.52) 9/68 · [0.52,0.55) 8/68 · [0.55,0.60) 6/30 · [0.60,1] 13/19. By NCC band: [0.85,0.87) 15/59 ·
  [0.87,0.90) 13/73 · [0.90,0.93) 4/23 · [0.93,1] 4/30. Every band is wrong; NCC is not even monotone.
- By corpus: mermaid↔mermaid 0/128 [0–2.9] · v1↔v1 36/54 · cross-corpus 0/3.
- Only 1 of the 36 positives was also joined by a non-crop edge. The 33 same-scene v1 pairs (adjacent video frames,
  repeat photos of one plot) were joined by coincidence: in none of them is the NCC box the patch content.
- Mechanism: 163/185 matches sit at scale ≤ 0.2. There the whole patch shrinks to a template of ≤ 51 px that keeps only
  its low-frequency gradient (light bar, vignette, water column), and NCC finds that gradient in almost any murky parent.
  Every mermaid-aws frame has the same light bar, hence 128 false merges.
- Cross-corpus clusters: **0/3 real**. C008 (v1↔coralscapes), C023 and C103 (v1↔mermaid) are unrelated scenes, all matched at
  scale 0.08. The v1↔v1 edge inside `dc-35f5d77f` is a real same-scene pair and stays.
- Proposed, NOT committed: a `crop_scale_min = 0.30` floor on the match grid. Synthetic crops keep 0.60–0.95 per side, so the floor
  never excludes one. Under the floor the 3 cross-corpus NCCs drop to 0.82 / 0.61 / 0.82 (< 0.85), so the clusters split
  (a synthetic fixture for this passed locally, then was reverted with the floor). Synthetic crop recall, crop family only,
  n=400, seed 0: not measured with the floor, not measured without. Floor committed: no. The crop-only eval ran, but its recall was not captured, so the ≥ 85% gate is unproven. The thresholds are unchanged.
- Not fixed by any threshold: with the floor, 68 of the 185 still confirm (3 derivative, 6 same scene, 59 different; 13%).
  NCC ≥ 0.93 keeps 6 false and loses C182 (0.929); NCC ≥ 0.95 keeps 1 false and loses 1 of the 3 real crops. Cos does not
  separate them either: the false pairs run up to cos 0.655, and the real crops sit at 0.547–0.690.

**`--dedup-v2` is not ready to flip** while the crop channel is on. The fix is to verify the box, not to tighten NCC: re-embed
`parent[box]` with SSCD and require a high cos to the patch. Until then, either keep v2 off or flip it with the crop channel off.

## WP-10c — the crop channel fix (D-T2)

The WP-10b false-merge mechanism was fixed with a box re-embed, gated behind a new `--dedup-crop`/
`--no-dedup-crop` switch (**default OFF**): `match_patch` is restricted to `scale >= crop_scale_min`
(0.30, was unbounded down to 0.08, where background-gradient matches lived), and a peak NCC is no
longer sufficient on its own — the matched box is cropped out of the larger image, SSCD-re-embedded,
and must also clear `cos_box_crop` against the smaller image's own embedding.

- `tau_ncc=0.50` and `tau_box=0.60` were grid-searched with 5-fold CV on the same 185-pair WP-10b
  audit set (`docs/dedup-crop-audit-2026-09-25.tsv`): mean held-out **precision 100%** (bar ≥ 95%),
  **0/3** cross-corpus false merges (C008/C023/C103) in every fold.
- The synthetic crop eval (n=400, hash+embed-knn) went from 69.25% recall with the channel off to
  **99.5%** with it on (bar ≥ 85%), 380/400 confirmed via the crop channel itself.
- With the switch off, v2 dedup is copy/agree/pixel only and the 3 known cross-corpus false merges
  cannot occur (`confirm()` alone rejects them — regression test in `tests/test_dedup_v2.py`).
- Only ~9 true crop positives sit in the 185-pair audit, so D-T2's caveat stands: the v2 build must
  hand-audit 50 random crop merges before shipping the channel on, with P ≥ 90% or it goes back off.
- The code default stays off (D-X): `--dedup-crop` only changes output when both it and `--decon`
  (or the `v2=True` `build_release` preset, INT-core2) are passed.
