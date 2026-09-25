# Ethics: diver / face presence audit (WP-5, 2026-09-25)

This is a **coarse, high-recall/low-precision proxy**, not a per-image ground-truth
label. It exists to answer one narrow ethics question honestly — "roughly how many
images in this release could show a person's face?" — using an open, reproducible
detector rather than skipping the question because no human review budget exists yet
(D-I: human-only items are listed as residuals, never faked).

## Method

- Detector: OpenCV Haar cascades, `haarcascade_frontalface_default.xml` +
  `haarcascade_profileface.xml` (frontal and profile face), applied to every image.
- Preprocessing: decode to grayscale, downscale so the longest side is <=480px.
- Parameters: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(24, 24)` px.
- An image is flagged "face-positive" if either cascade returns at least one
  detection. No manual review of detections was performed.
- Corpus scanned: every row of the `images` HF config (train + validation + test),
  read directly from the built Parquet shards.
- opencv-python-headless version: `4.10.0` (pinned; the 5.x line removed
  `cv2.CascadeClassifier` from the top-level namespace at audit time).

## Result

- **12,433 of 69,600 images (17.9%)** were flagged face-positive by at least one
  cascade.
- This rate is almost certainly dominated by **false positives**: Haar cascades are
  known to fire on textured, high-contrast patterns (coral rubble, bleached
  branching structures, dive-gear edges) that are not faces. No claim is made here
  about how many of the 12,433 flags are real divers or bystanders — that requires a
  human audit, which has not been run. Reporting the raw flag count without a
  confidence claim is the honest option; suppressing it because the detector is
  imprecise is not.

### Per-source breakdown

The `images` config's `source_id` column attributes each row to one canonical
source (after near-duplicate collapse). The table below covers all 69,600 scanned
rows across 8 of the 9 registry sources; `reef-support-bleaching` (658 images in the
registry) does not appear as a distinct canonical `source_id` in this table, i.e.
its rows were canonicalized under another source's id during near-dup collapse —
noted here rather than investigated further (a residual, per D-I).

| Source (registry id) | n scanned | Face-positive | Rate |
|---|---|---|---|
| `roboflow-coral-reef-bleach-detection-v2i` | 10,544 | 2,054 | 19.5% |
| `noaa-pifsc-bleaching` | 10,419 | 290 | 2.8% |
| `roboflow-coral-reef-classification-v3i` | 4,467 | 1,071 | 24.0% |
| `roboflow-coral-classification-copy-changed-v13i` | 1,172 | 247 | 21.1% |
| `roboflow-coral-bleaching-general-v1-yolov8s` | 1,925 | 495 | 25.7% |
| `reef-support-benthic-own` | 1,250 | 211 | 16.9% |
| `roboflow-coral-bleaching-final-v6i` | 2,550 | 672 | 26.4% |
| `coralscop-masks-rs` | 37,273 | 7,393 | 19.8% |
| **Total** | **69,600** | **12,433** | **17.9%** |

NOAA PIFSC's much lower rate (2.8%) is consistent with it being uncropped/consistent
224×224 seafloor-transect imagery with little dive-gear or diver presence, versus
the Roboflow community exports and CoralSCOP frames, which include more
close-in, gear-adjacent shots.

## Residual (human-only, listed per D-I, never faked)

No human reviewer has confirmed or refuted any of the 12,433 flagged images, and no
manual audit sample has been drawn from them. Until that review happens: treat the
12,433 figure as an upper-bound screening count for "images worth a privacy pass
before any redistribution that requires diver consent," not as a count of images
containing real, identifiable people.

## Sensitive-species location policy

No per-sample GPS is recorded in this release (see `docs/RATING_EVIDENCE_R0.md`,
D8), so there is currently no coordinate data to round or redact. This policy is
recorded pre-emptively for the moment location metadata is added in a future
release: any sample geotagged to a CITES Appendix I/II or IUCN Red List
Critically Endangered/Endangered taxon (e.g. certain reef shark, ray or giant clam
species) must have its published coordinates rounded to 0.1° (roughly 11 km) before
public release, to avoid pinpointing poaching-vulnerable populations. This mirrors
GBIF's and OBIS's standard practice for sensitive-species occurrence data.
