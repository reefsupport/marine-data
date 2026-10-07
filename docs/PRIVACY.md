# Privacy: faces and people

This document describes the face and person detection tooling in `marinedata.privacy` and the audit that was used to
calibrate it. For the position on the published release (v1.0, October 2026) see the Ethics section of
[`DATASHEET.md`](DATASHEET.md). Depicted people can request removal through the contacts listed there.

## Detection

`marinedata privacy-scan` runs two open-weight detectors over image bytes, in batches, resumable and cached by
`image_sha256`:

- **Faces**: OpenCV Zoo YuNet (`face_detection_yunet_2023mar.onnx`, via `cv2.FaceDetectorYN`). Returns a box, five
  landmarks and a score.
- **People and divers**: torchvision `ssdlite320_mobilenet_v3_large` with COCO `V1` weights, filtered to the `person`
  class.

Both are downloaded without login and total about 13.6 MB. The output has one row per image with `face_boxes`,
`person_boxes`, `face_identifiable` and `people_present`.

A face box is `face_identifiable` when all of the following hold:

1. the short side is at least 32 px,
2. the detector score is at least 0.8,
3. it is not flagged by `_looks_occluded()`, a landmark-geometry proxy that treats a face as covered when the mouth-corner
   span collapses to under 35% of the eye span, or the nose sits within 25% of the eye span from the midpoint of the mouth
   corners.

`_looks_occluded()` is a first-order heuristic, not a learned classifier. On the audit sample below it fired three times,
always on a coral-texture false positive, and flagged none of the five real masked or regulator-covered diver faces. It
is kept but is not relied on: those boxes are already excluded by the score and size gate.

## Audit

A stratified sample of 350 images from an earlier internal build of 69,600 images (fixed seed 42) was reviewed against
rendered contact sheets: 150 face-flagged, 100 person-flagged and 100 images neither detector flagged. The per-image
record, `docs/privacy-audit-2026-09-25.tsv`, holds the image hash and boxes only, never a face crop.

| Metric | Result | 95% Wilson interval |
|---|---|---|
| Face precision (at least one real face in the flagged boxes) | 18/150 = 12.0% | 7.7 to 18.2% |
| Person precision | 33/100 = 33.0% | 24.6 to 42.7% |
| Faces or people missed in the 100 negatives | 0/100 | upper bound 3.7% |

Both detectors over-trigger on underwater texture: coral polyp clusters, brain coral, survey quadrat grids and
compression artefacts match as faces or people. Almost all genuine hits came from a few recurring sources, mainly
television news footage and open-water diver photographs. The face detector also missed one clearly identifiable
two-person frame that the person detector caught. The audit was run on an earlier build, so its rates describe the
detectors, not the published release.

## Second-stage verifier and blur

- `marinedata.privacy.verify` re-runs YuNet on each first-stage face box expanded by 50%, which separates coral-texture
  false positives from real faces better than a second pass over the same box. `VERIFY_THRESHOLD` (0.3292) is the highest
  verifier score that still recalls at least 95% of the audited true faces (25 of 26, Wilson interval 81.1 to 99.3%).
  Precision at that threshold is 12.3% (interval 8.4 to 17.5%): the threshold deliberately favours recall, because for
  privacy a missed face is the more serious error.
- `marinedata.privacy.blur_faces(image_bytes, boxes)` applies a deterministic Gaussian blur to each box, padded by 20% on
  each side, with a kernel as wide as the padded box's short side. It re-encodes in the source format (JPEG or PNG) and
  returns the input unchanged when there are no boxes. Of ten audited true faces re-checked after blurring, all ten fell
  below `VERIFY_THRESHOLD`.
- `apply_privacy_blur(image_bytes, blur_boxes, enabled)` is the build-time switch. With `enabled=False` it returns the
  same object it was given, so a build without the flag is byte-identical to one that never imported the module.
  `blur_boxes_for_row` selects every box on an audited true image plus every verifier-confirmed box, and
  `write_privacy_config` writes the per-image `privacy_blurred`, `blur_boxes` and `face_candidate` columns, joinable on
  `image_sha256`.

Original, unblurred images are never rewritten.

## Sensitive-species locations

The release carries no coordinates of its own; location fields are null unless the source provides them. Where a source
does provide coordinates for a CITES- or IUCN-listed taxon, they are to be rounded to 0.1 degrees.
