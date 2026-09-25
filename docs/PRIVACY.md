# Privacy policy — faces and people (WP-5b)

**Status: pipeline built and validated; full-corpus scan and the D-I manual audit are
NOT complete** (see "What is actually done" below). This document records the policy
decision so the v2 build can implement it as soon as the full scan/audit lands; it does
not itself certify that v1 or the current `privacy.parquet` is audit-complete.

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

This is a first-order geometric heuristic, not a learned occlusion classifier. **It
still needs calibration against a real manual audit** (D-I: >= 150 face positives
across score bands, 100 person positives, 100 random negatives,
`docs/privacy-audit-2026-09-25.tsv`) before the thresholds above should be treated as
final — that audit did not run in this pass (see "What is actually done").

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

## What is actually done vs. not (read before relying on this doc)

Done and verified in this pass:
- `src/marinedata/privacy.py` + `marinedata privacy-scan` CLI: batched, resumable
  (skips `image_sha256` already in the output parquet), flushes incrementally, runs
  both detectors, writes the `privacy.parquet` schema described above.
- `blur_faces` implemented and unit-tested (8 tests, all passing).
- Smoke-tested end-to-end against real images from `data/_hf/v1/data/images/`: 260
  images scanned (60 then +200 with cache-skip verified), 10 face boxes, 2 person
  boxes found — proves the pipeline runs correctly on real data at real resolution,
  including detector-vs-architecture pitfalls (see code comment on
  `PersonDetector.__init__` about the `reduce_tail` backbone-width trap).

Not done — out of reach of this worker's budget, not faked:
- **The full-corpus scan.** Measured throughput on this pass is ~0.1–0.3 s/image on
  CPU/MPS, i.e. roughly 2–6 hours for all 69,600 images — longer than one worker
  session's tool-call budget allows to run and verify interactively. The CLI is
  resumable specifically so this can be restarted/continued rather than re-run from
  scratch.
- **The D-I manual audit** (`docs/privacy-audit-2026-09-25.tsv`, >= 150/100/100
  stratified samples, precision per detector, FN rate, 95% CI on the identifiable
  count). This requires the full scan's output first, plus visual review time this
  pass did not have budget for.
- **`docs/DATASHEET.md` / `docs/ETHICS_FACE_AUDIT.md` numbers** are therefore left
  as the Haar-cascade numbers for now rather than being overwritten with an
  incomplete pilot's counts — replacing 12,433/69,600 with a number from 260 sampled
  images would be less honest, not more.
- **S3 upload of `privacy.parquet`** — not done; the artifact from this pass only
  covers the 260-image smoke sample and is not the release deliverable.

## Next steps for whoever continues this

1. Run `marinedata privacy-scan --input data/_hf/v1/data/images --output
   data/_privacy/v1/privacy.parquet --face-weights <yunet.onnx> --person-weights
   <torch-cache-dir>` to completion (resumable — safe to kill and restart).
2. Pull a stratified sample from the resulting parquet (by score band / source) and
   run the D-I manual audit into `docs/privacy-audit-2026-09-25.tsv`.
3. Use the audit's precision/FN numbers to recheck (or retune) the `MIN_FACE_PX` /
   `MIN_FACE_SCORE` / `_looks_occluded` thresholds in `src/marinedata/privacy.py`.
4. Update `docs/DATASHEET.md` and `docs/ETHICS_FACE_AUDIT.md` with the real
   full-corpus numbers, keeping the Haar-cascade history to one sentence.
5. Upload the finished `privacy.parquet` to
   `s3://rs-storage-open/releases/open-marine-imagery/v1/privacy/` with size+ETag
   verification.
