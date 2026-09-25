"""Face / person privacy for the open-imagery release (WP-5b/5d/5e).

A small package rather than one file, split by concern:

- :mod:`.scan` — first-stage detection (YuNet faces, SSDLite person), the
  resumable ``privacy-scan`` pipeline, and ``blur_faces`` (the low-level pixel
  op). Unchanged from WP-5b/5c/5d; only its location moved.
- :mod:`.verify` — the WP-5e second-stage face verifier (D-I2): re-runs YuNet
  on an expanded crop around each first-stage candidate box to cut false
  positives before anything is blurred.
- :mod:`.blur` — the WP-5e release-time blur decision and the
  ``--privacy-blur`` flag (D-X: default OFF, byte-identical when off).

Everything previously imported as ``from marinedata.privacy import X`` keeps
working — this module re-exports the full public surface of all three.
"""

from __future__ import annotations

from .blur import (
    apply_privacy_blur,
    blur_boxes_for_row,
    build_privacy_config_row,
    write_privacy_config,
)
from .scan import (
    COCO_PERSON_LABEL,
    MIN_FACE_PX,
    MIN_FACE_SCORE,
    PERSON_MODEL_VERSION,
    YUNET_VERSION,
    Box,
    FaceBox,
    FaceDetector,
    PersonDetector,
    ScanRow,
    ScanStats,
    _looks_occluded,
    blur_faces,
    iter_image_rows,
    load_cached_shas,
    scan,
    sha256_bytes,
    write_output,
)
from .verify import (
    VERIFY_MODEL_VERSION,
    VERIFY_THRESHOLD,
    FaceVerifier,
    expand_box,
    recall_precision_at_threshold,
)

__all__ = [
    "COCO_PERSON_LABEL",
    "MIN_FACE_PX",
    "MIN_FACE_SCORE",
    "PERSON_MODEL_VERSION",
    "VERIFY_MODEL_VERSION",
    "VERIFY_THRESHOLD",
    "YUNET_VERSION",
    "Box",
    "FaceBox",
    "FaceDetector",
    "FaceVerifier",
    "PersonDetector",
    "ScanRow",
    "ScanStats",
    "_looks_occluded",
    "apply_privacy_blur",
    "blur_boxes_for_row",
    "blur_faces",
    "build_privacy_config_row",
    "expand_box",
    "iter_image_rows",
    "load_cached_shas",
    "recall_precision_at_threshold",
    "scan",
    "sha256_bytes",
    "write_output",
    "write_privacy_config",
]
