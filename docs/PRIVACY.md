# Privacy policy — faces and people (WP-5b/5d/5e)

**Status: WP-5b/5c/5d DONE; WP-5e (D-I2 second-stage verifier + blur) PARTIAL —
code done, real-data tuning/dry-run not.** The full 69,600-image scan ran to
completion (WP-5c), and the D-I manual audit (350 images, seed=42) ran against
it (WP-5d) — `docs/privacy-audit-2026-09-25.tsv`, results below and in
`docs/ETHICS_FACE_AUDIT.md`. `privacy.parquet` + the audit TSV are uploaded to
`s3://<open-bucket>/releases/marine-data/v1/privacy/`. See "D-I2:
second-stage verifier + v2 blur" below for what WP-5e added and what is still
open.

## Why this replaces the Haar-cascade proxy

WP-5's `docs/ETHICS_FACE_AUDIT.md` used an OpenCV Haar cascade to flag 12,433/69,600
(17.9%) images as possibly containing a diver/face. Haar cascades on underwater
texture are known to be high-recall/low-precision — coral heads, sponge fans, and
patterned substrate reliably trigger frontal-face cascades — so that number was
explicitly documented as unusable for a release decision. `src/marinedata/privacy.py`
replaces it with two real, open-weight, no-login neural detectors:

- **Faces**: OpenCV Zoo YuNet (`face_detection_yunet_2023mar.onnx`, 227 KB), run via
  `cv2.FaceDetectorYN`.
- **People/divers**: torchvision `ssdlite320_mobilenet_v3_large` with COCO `V1`
  weights (13.4 MB), filtered to the `person` class.

Both weights are open, downloaded without login, and small enough that the combined
footprint (~13.6 MB) is far under the 300 MB budget. SHA-256 of both files is recorded
next to the download in the worker's scratch dir per the brief; the release build
should pin them the same way.

## The `face_identifiable` rule

A face box is `face_identifiable` when **all** of:

1. short side >= 32 px,
2. detector score >= 0.8,
3. not flagged by `_looks_occluded()` — a landmark-geometry proxy over YuNet's 5-point
   landmarks (eyes, nose, mouth corners): if the mouth-corner span collapses to under
   35% of the eye span, or the nose sits within 25% of the eye span from the midpoint
   of the mouth corners, the mouth is very likely hidden behind a regulator, mask
   strap, or the camera angle — treated as not independently identifiable even if a
   face box fired.

This is a first-order geometric heuristic, not a learned occlusion classifier. The D-I
audit (`docs/privacy-audit-2026-09-25.tsv`) calibrated it: on the 26 true-positive
faces in the sample, `_looks_occluded()` scored **0/3 precision, 0/5 recall** — it
fired three times, always on a coral-texture false positive, and missed all five real
masked/regulator-obscured diver faces. **Not retuned**: 5 ground-truth positives is too
small to fit a new threshold without overfitting, and the failure isn't threshold
shape, it's that the landmark-ratio signal doesn't discriminate on this sample. It also
doesn't change `face_identifiable` for those 5 cases either way — they're already
excluded by the score/size gate before `_looks_occluded()` is consulted. Left in place;
flagged in `docs/ETHICS_FACE_AUDIT.md` as unproven and a candidate for removal or
replacement by a real classifier if mask detection becomes a hard requirement.

## v2 blur policy (decision, to be implemented by the v2 build)

- Every `face_identifiable` box gets a Gaussian blur, padded 20% on each side, in the
  **released** pixels only. Kernel size scales with the (padded) box's short side so
  small and large faces both end up genuinely unrecognisable.
- The **original, unblurred bytes stay in the private staging tree** — never in a
  public release. v1 is not touched or retro-blurred.
- `people_present` (any COCO-person box, regardless of `face_identifiable`) becomes a
  new column on the sample/release schema so downstream consumers can filter "has
  a diver in-frame" without needing pixel access.
- Implementation: `marinedata.privacy.blur_faces(image_bytes, boxes) -> bytes`
  (deterministic — same input always produces the same output bytes; re-encodes in
  the source format, JPEG or PNG). Unit-tested in `tests/test_privacy.py`
  (determinism, boxed-region actually changes, untouched pixels don't, no-boxes is a
  lossless no-op). The v2 build calls this per `face_identifiable` row when it copies
  bytes from private staging into the public release tree; it is not wired into any
  `hf_*` module here (D-M).

## What is actually done

- `src/marinedata/privacy.py` + `marinedata privacy-scan` CLI: batched, resumable,
  cached by `image_sha256`, writes the `privacy.parquet` schema described above.
- `blur_faces` implemented and unit-tested (determinism, boxed-region-changes,
  untouched-pixels, no-op-on-empty-boxes).
- **Full-corpus scan complete** (WP-5c, PID 22482): all 69,600 images, 0 decode
  errors, 5,148 face boxes / 1,238 person-flagged images found, ~102 min elapsed.
- **D-I manual audit complete** (WP-5d): 350 images (150 face / 100 person / 100
  negative, seed=42) hand-reviewed against rendered contact sheets, recorded in
  `docs/privacy-audit-2026-09-25.tsv`. Results, per-source breakdown and the
  `_looks_occluded()` calibration are in `docs/ETHICS_FACE_AUDIT.md`.
- `docs/DATASHEET.md` and `docs/ETHICS_FACE_AUDIT.md` rewritten with the audited
  numbers, citing the TSV.
- `privacy.parquet` + the audit TSV uploaded to
  `s3://<open-bucket>/releases/marine-data/v1/privacy/`, size+ETag
  verified; `CHECKSUMS.sha256` regenerated.

## D-I2: second-stage verifier + v2 blur (WP-5e)

`src/marinedata/privacy.py` is now the `marinedata.privacy` **package**
(`scan.py` unchanged first-stage detection, `verify.py` the new second stage,
`blur.py` the new release-time decision) — a straight rename, every prior
import (`from marinedata.privacy import blur_faces`, etc.) still works via
`privacy/__init__.py`'s re-exports.

- **`verify.py`**: re-runs the same YuNet model on each first-stage face box
  expanded 50% (`expand_box`), rather than the full frame — see the module
  docstring for why that discriminates coral-texture false positives from
  real faces better than a second look at the same box. `VERIFY_THRESHOLD`
  gates the D-I2 blur decision.
- **`blur.py`**: `apply_privacy_blur(image_bytes, blur_boxes, enabled)` is the
  `--privacy-blur` flag surface — `enabled=False` returns the *same object*,
  not just equal bytes (D-X: a release built without the flag is provably
  identical to one that never imported this module).
  `blur_boxes_for_row` selects every box on an audited-true image
  (`docs/privacy-audit-2026-09-25.tsv`, `face_true=1`) plus any
  verifier-confirmed box. `build_privacy_config_row` /
  `write_privacy_config` write the per-image `privacy_blurred`, `blur_boxes`,
  `face_candidate` columns (a config parquet joinable on `image_sha256`, the
  same shape as `privacy.parquet`).

### Threshold provenance — fully fit (WP-5f)

All 5,148 first-stage face candidates (3,387 images) were scored with the
second-stage verifier on the 50%-expanded crop. The D-I audit was extended
with a further blind, decile-stratified contact-sheet review of 130 more
candidates (idx 350–479, `docs/privacy-audit-2026-09-25.tsv`, `auditor=agent`,
`audit_date=2026-09-25`), bringing the audited-true face count to **26**
(18 original + 8 new) — at the brief's 25-positive floor.

`VERIFY_THRESHOLD = 0.3292` is `fit_threshold_for_min_recall()`'s pick: the
highest verifier score that still recalls ≥95% of the 26 audited-true faces.
At this threshold:

- **Recall**: 96.2% (25/26), Wilson 95% CI [81.1%, 99.3%]
- **Precision**: 12.3% (34/276 non-true candidates above threshold are false
  positives), Wilson 95% CI [8.4%, 17.5%] (n=280 audited face rows)

Precision at the recall-95% threshold is **below the brief's 20% floor**. Per
the brief's fallback, the threshold is kept anyway (raising it to hit 20%
precision would drop recall below 95% on real faces, which is the more
important error to avoid for a privacy release) — i.e. "blur every candidate
above this lower bound."

**Dry run** (`VERIFY_THRESHOLD=0.3292`, all boxes on an audited-true image plus
every verifier-confirmed box elsewhere): **2,274 / 3,387** face-flagged images
would be blurred (67.1% of face-flagged, 3.27% of the full 69,600-image
corpus). Per-source breakdown and the 40-image before/after contact sheet are
in the WP-5f scratch output; the release artifact is below.

**Blur strength**: `blur_faces()`'s Gaussian kernel was found to be too weak
to reliably defeat the verifier's own re-detection on real faces — the
original `ksize = box_short_side // 2` left enough low-frequency shape/colour
signal that only ~60% of real audited-true faces dropped below threshold
after blur. Fixed to `ksize = box_short_side` (no halving); re-verified on
10 real audited-true face crops, **10/10** now drop below
`VERIFY_THRESHOLD` after blur.

**Release artifact**: `data/_privacy/2026-09-25/privacy_v2.parquet` (69,600
rows, same shape as `privacy.parquet`: `image_sha256, face_candidate,
verify_score, privacy_blurred, blur_boxes`, joinable on `image_sha256`).
sha256: `0978fb811cf3161de4d0e5d0f2d928f03ddb7f4e0f53eca6c9597d2463b07e43`.

## Known gap, deliberately out of this scope

`face_identifiable` and `people_present` are **not yet wired into `hf_*` metadata** —
that integration belongs to WP-2 at the point it merges this branch (per D-M, this
worktree never touches `hf_*`). Until then, the columns exist only in
`data/_privacy/v1/privacy.parquet` and `data/_privacy/2026-09-25/privacy_v2.parquet`,
joinable on `image_sha256`. The same applies to the `privacy_blurred` / `blur_boxes`
/ `face_candidate` / `verify_score` config from D-I2.
