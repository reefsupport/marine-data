# Ethics: diver / face / person presence audit (WP-5b/5d, 2026-09-25)

v1 (WP-5) used an OpenCV Haar-cascade proxy (12,433/69,600 = 17.9% flagged, dominated
by coral-texture false positives, no manual review). WP-5b replaced it with two real,
open-weight detectors and WP-5d ran them over the full corpus and manually audited a
stratified sample — this is now a **calibrated** privacy signal, not just a screening
count.

## Method

- **Faces**: OpenCV Zoo YuNet (`face_detection_yunet_2023mar.onnx`,
  `cv2.FaceDetectorYN`), returns boxes + 5 landmarks + score.
- **People/divers**: torchvision `ssdlite320_mobilenet_v3_large` (COCO `person` class,
  `COCO_V1` weights).
- Both run over the embedded bytes of every row in the `images` HF config (train +
  validation + test, 69,600 rows), resumable/cached by `image_sha256`
  (`marinedata privacy-scan`, `src/marinedata/privacy.py`).
- `face_identifiable` (per box): short side >= 32px AND score >= 0.8 AND not flagged by
  `_looks_occluded()` (a landmark-geometry mask/regulator proxy — see calibration below).
- Output: `data/_privacy/v1/privacy.parquet`, one row per image with `face_boxes`,
  `person_boxes`, `face_identifiable`, `people_present`.

## D-I manual audit (2026-09-25)

Stratified sample, fixed seed=42, drawn from the full 69,600-row scan; rendered as
boxed contact sheets and reviewed image-by-image (never a face crop committed —
`docs/privacy-audit-2026-09-25.tsv` records `image_sha256` + boxes only):

- 150 face-flagged images, stratified by detector score band (`<0.7`, `0.7-0.85`,
  `>=0.85`) x source
- 100 person-flagged images, stratified by source
- 100 negatives (neither detector fired), stratified by source

| Metric | Result | 95% Wilson CI |
|---|---|---|
| Face precision (image-level: >=1 real face in the flagged box(es)) | **18/150 = 12.0%** | 7.7–18.2% |
| Person precision | **33/100 = 33.0%** | 24.6–42.7% |
| Missed face/person in the 100 negatives | 0/100 | upper bound 3.7% |
| Identifiable faces in the face-flagged stratum | 15/150 = 10.0% | 6.2–15.9% |

Both detectors are still dominated by false positives — coral polyp clusters, brain
coral texture, dive-survey grid quadrats, and JPEG/scan-line chromatic-aberration
artifacts routinely pattern-match as faces or people. Almost every genuine hit came
from a small number of recurring sources: TV-news b-roll frames (a "Great Barrier Reef
bleaching" broadcast clip, several near-duplicate frames) and open-water diver shots.
One face-detector false negative was found in the person-flagged sample: idx 185, a
two-person interview frame with two clearly identifiable faces that YuNet missed
entirely (caught only because the person detector separately fired on their bodies) —
noted as a real recall gap, not fixed here.

### Identifiable-face prevalence, extrapolated to all 69,600

Stratified estimator (non-overlapping strata: face-flagged 3,387 / person-flagged-only
821 / negatives 65,392; the 296 face+person-overlap rows are counted once, under
face-flagged):

| Stratum | n | audit identifiable rate | extrapolated |
|---|---|---|---|
| face-flagged | 3,387 | 15/150 = 10.0% | ~339 |
| person-flagged-only | 821 | 1/86 = 1.2% | ~10 |
| negatives | 65,392 | 0/100 = 0.0% | ~0 |
| **Total** | **69,600** | | **~348 (0.50%)** |

95% CI (sum of per-stratum Wilson bounds — conservative, dominated by the zero-count
negative stratum's wide upper bound): **0.30%–4.32%** (~210–3,007 images). The point
estimate (~348) is the more useful planning number; the CI's upper end reflects that a
100-image sample cannot rule out a small residual rate among 65,392 unreviewed
negatives, not a specific finding of missed faces.

### WP-5f: second-stage verifier fit and dry run (2026-09-25)

All 5,148 first-stage face candidates were re-scored with a second-stage YuNet
pass on each box's 50%-expanded crop (`marinedata.privacy.verify`). The D-I
audit above was extended with a further blind, decile-stratified review of
130 more candidates (idx 350–479 in `docs/privacy-audit-2026-09-25.tsv`,
`auditor=agent`), bringing the audited-true face count to **26** (18 original
+ 8 new) — at the 25-positive floor needed to fit a threshold rather than
spot-check one.

`VERIFY_THRESHOLD = 0.3292` is the highest score that still recalls >=95% of
those 26 audited-true faces:

| Metric | Result | 95% Wilson CI |
|---|---|---|
| Recall (of 26 audited-true faces) | **25/26 = 96.2%** | 81.1–99.3% |
| Precision (of 280 audited face-kind rows) | **12.3%** | 8.4–17.5% |

Precision is below the 20% floor a fully-fit threshold should clear; kept
anyway per `docs/PRIVACY.md` because raising it would drop recall
below 95% on real faces. Applying this threshold to every scored candidate
(not an extrapolation — every one of the 3,387 face-flagged images was
scored) gives **2,274 images (3.27% of the full 69,600-image corpus)** that
would be blurred as containing an identifiable face — see
`data/_privacy/2026-09-25/privacy_v2.parquet`.

### `_looks_occluded()` calibration

Compared the heuristic's per-box `occluded` flag against the audit's
`occluded_by_mask_or_regulator` ground truth, over every true-positive face in the
sample (26 images: 18 from the face stratum + 8 also flagged in the person stratum):

- **Precision: 0/3 = 0%** — it fired 3 times in the full 350-image sample, always on a
  coral-texture false positive, never on a real face.
- **Recall: 0/5 = 0%** — none of the 5 real masked/regulator-obscured diver faces in
  the sample were flagged.
- **Not retuned.** 5 ground-truth positives is too small a signal to fit a new
  threshold without overfitting, and the sample gives no evidence a different
  threshold would help (it isn't a threshold problem — the landmark-ratio signal
  itself doesn't discriminate on this sample). In practice this doesn't change
  `face_identifiable` for the masked-diver cases either way: those boxes are already
  excluded by the score/size gate before `_looks_occluded()` is even consulted. Flagged
  for a future WP: the heuristic adds noise (false positives on coral texture) without
  demonstrated protective value; consider dropping it or replacing it with a real
  mask-classifier if mask detection specifically becomes a requirement.

### Per-source flag rates (all 69,600, no manual review — see the audit for calibrated precision)

| Source | n | face-flagged | rate | person-flagged | rate |
|---|---|---|---|---|---|
| `coralscop-masks-rs` | 37,273 | 2,467 | 6.6% | 421 | 1.1% |
| `roboflow-coral-reef-bleach-detection-v2i` | 10,544 | 515 | 4.9% | 614 | 5.8% |
| `noaa-pifsc-bleaching` | 10,419 | 26 | 0.2% | 25 | 0.2% |
| `roboflow-coral-reef-classification-v3i` | 4,467 | 97 | 2.2% | 27 | 0.6% |
| `roboflow-coral-bleaching-final-v6i` | 2,550 | 52 | 2.0% | 12 | 0.5% |
| `roboflow-coral-bleaching-general-v1-yolov8s` | 1,925 | 38 | 2.0% | 10 | 0.5% |
| `reef-support-benthic-own` | 1,250 | 149 | 11.9% | 2 | 0.2% |
| `roboflow-coral-classification-copy-changed-v13i` | 1,172 | 43 | 3.7% | 6 | 0.5% |
| **Total** | **69,600** | **3,387** | **4.9%** | **1,117** | **1.6%** |

`reef-support-bleaching` (658 registry images) does not appear as a distinct
`source_id` here — confirmed in the WP-5b report to be canonicalized under
`reef-support-benthic-own` via near-dup collapse, provenance preserved in `source_ids`.

## v2 policy

See `docs/PRIVACY.md`: the v2 build blurs every `face_identifiable` box (Gaussian,
padded 20%) in the released pixels via `blur_faces()`; originals are never rewritten;
`face_identifiable` and `people_present` ship as `metadata` columns (wired at the
WP-2 integration merge, not in this worktree — see the WP-5d report).

## Sensitive-species location policy

Unchanged from WP-5 (see `docs/DATASHEET.md`): no per-sample GPS exists in v1, so
there is nothing to redact today; the 0.1° rounding rule for CITES/IUCN-listed taxa is
recorded pre-emptively for the release that adds coordinates.
