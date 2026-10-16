"""G8 privacy (v1.1.0 gate side): EXIF GPS and faces over the ``images`` config only (every new
image lives there once; the v1.0 configs were checked in v1.0 and are frozen).

The detector is the pinned YuNet model, loaded through :func:`.yunet_model.verified_model_path`
(sha256 checked, never downloaded); ``--detector MODULE:ATTR`` replaces it. A detector that cannot
load, or an image it cannot read, FAILS the gate: a gate that could not look has not passed. A face
at or above the threshold fails with the full ``image_sha256`` (also listed in
``counts.detected_image_sha256``, first 1000).

A person's look at a detection is recorded in ``registry/privacy/review.yaml``
(:mod:`marinedata.privacy.review`; ``--privacy-review FILE`` replaces the path). A detection on a
``cleared`` image is not a failure: it is counted under ``cleared_reviewed`` and listed in
``counts.cleared_reviewed_entries`` (sha256, reviewer, date). A ``withhold`` image that is still
in the build fails as ``withheld_image_present`` (the build drops it). A list that cannot be read
or validated fails the gate, and an unreviewed detection fails as before. A review clears the
face detection only: EXIF GPS still fails the image."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..privacy.review import ReviewEntry, load_review
from . import GateResult, finish, skip
from .config import exif_gps_allowed
from .privacy import (
    DEFAULT_THRESHOLD,
    GPS_INFO_TAG,
    IMAGE_COLUMNS,
    UNREADABLE,
    _people,
    exif_has_gps,
)
from .privacy_detectors import Detection, Detector
from .shards import config_shards, iter_rows, shard_columns
from .yunet_model import YUNET_COMMIT, YuNetModelError, verified_model_path

NAME = "privacy"
IMAGES = "images"
MAX_LISTED = 1000


class YuNetDetector:
    """OpenCV YuNet over the pinned ONNX file; faces only."""

    name = "yunet"
    version = YUNET_COMMIT[:12]

    def __init__(self, model_path: Path, score_threshold: float = 0.5) -> None:
        import cv2

        self._cv2 = cv2
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, image: bytes) -> list[Detection]:
        import numpy as np

        frame = self._cv2.imdecode(np.frombuffer(image, np.uint8), self._cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("image bytes do not decode")
        height, width = frame.shape[:2]
        self._net.setInputSize((width, height))
        _, faces = self._net.detect(frame)
        return [] if faces is None else [Detection("face", float(row[-1])) for row in faces]


def load_yunet_detector(path: str | Path | None = None) -> YuNetDetector:
    """The YuNet detector over ``verified_model_path(path)``: sha256 verified, never downloaded."""
    return YuNetDetector(verified_model_path(path))


def run_privacy_v11_gate(
    build_dir: Path,
    *,
    detector: Detector | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    allow_exif_gps=None,
    registry_root: Path | None = None,
    review_path: str | Path | None = None,
) -> GateResult:
    """``review_path`` replaces ``registry/privacy/review.yaml`` (under ``registry_root``)."""
    build = Path(build_dir)
    files = config_shards(build).get(IMAGES)
    if not files:
        return skip(NAME, f"no {IMAGES} config: nothing to scan (EXIF GPS, faces)", _people(build))
    try:
        review = load_review(review_path, registry_root=registry_root)
    except (OSError, ValueError) as exc:
        return finish(
            NAME,
            {"review_list": "unavailable", "scope": IMAGES},
            [f"privacy review list unavailable: {type(exc).__name__}: {str(exc)[:200]}"],
        )
    cleared, withheld = review.cleared, review.withheld
    if detector is None:
        try:
            detector = load_yunet_detector()
        except (YuNetModelError, ImportError, OSError, ValueError) as exc:
            return finish(
                NAME,
                {"detector": "unavailable", "scope": IMAGES},
                [f"detector unavailable: {type(exc).__name__}: {str(exc)[:200]}"],
            )
    allowed = frozenset(
        exif_gps_allowed(registry_root) if allow_exif_gps is None else allow_exif_gps
    )
    stats: Counter = Counter()
    detected: dict[str, float] = {}
    cleared_seen: dict[str, ReviewEntry] = {}
    seen_allowed: set[str] = set()
    failures: list[str] = []
    present = shard_columns(files[0])
    columns = [
        c
        for c in dict.fromkeys([*IMAGE_COLUMNS, "image", "image_sha256", "source_id"])
        if c in present
    ]
    for path, index, _split, row in iter_rows(files, columns):
        data = (row.get("image") or {}).get("bytes")
        sha, source = str(row.get("image_sha256")), str(row.get("source_id") or row.get("source"))
        who = f"{IMAGES}/{path.name}[{index}] image_sha256={sha} source={source}"
        if sha in withheld:  # present with or without bytes: it must not be in the build at all
            stats["withheld_image_present"] += 1
            entry = withheld[sha]
            failures.append(
                f"{who} withheld_image_present: privacy review {entry.reviewed_at} "
                f"({entry.reviewer}) says withhold"
            )
            continue
        if not data:
            continue
        stats["images_scanned"] += 1
        try:
            gps = exif_has_gps(data)
        except UNREADABLE as exc:
            stats["exif_unreadable"] += 1
            failures.append(f"{who} exif_unreadable: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if gps:
            stats["exif_gps_images"] += 1
            if source in allowed:
                stats["exif_gps_allowed"] += 1
                seen_allowed.add(source)
            else:
                failures.append(f"{who} exif_gps_present: GPSInfo (tag {GPS_INFO_TAG})")
        try:
            hits = [h for h in detector.detect(data) if h.score >= threshold]
        except Exception as exc:
            stats["detector_errors"] += 1
            failures.append(f"{who} detector_error: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if not hits:
            continue
        if sha in cleared:  # a person looked at this image: the detection is not a failure
            cleared_seen[sha] = cleared[sha]
            continue
        top = max(hits, key=lambda h: h.score)
        detected[sha] = top.score
        kinds = "+".join(sorted({h.kind for h in hits}))
        failures.append(
            f"{who} {kinds}_detected: score {top.score:.2f} >= {threshold:.2f} "
            f"({detector.name} {detector.version})"
        )
    counts = _people(build) | {
        "scope": IMAGES,
        "images_scanned": stats["images_scanned"],
        "exif_gps_images": stats["exif_gps_images"],
        "exif_gps_allowed": stats["exif_gps_allowed"],
        "exif_gps_allowed_sources": sorted(seen_allowed),
        "exif_unreadable": stats["exif_unreadable"],
        "detector_errors": stats["detector_errors"],
        "detector": f"{detector.name} {detector.version}",
        "threshold": threshold,
        "images_with_detections": len(detected),
        "detected_image_sha256": sorted(detected)[:MAX_LISTED],
        "cleared_reviewed": len(cleared_seen),
        "cleared_reviewed_entries": [
            {"image_sha256": sha, "reviewer": e.reviewer, "reviewed_at": e.reviewed_at}
            for sha, e in sorted(cleared_seen.items())
        ][:MAX_LISTED],
        "withheld_image_present": stats["withheld_image_present"],
        "review_entries_cleared": len(cleared),
        "review_entries_withhold": len(withheld),
    }
    if not stats["images_scanned"] and not failures:
        return skip(NAME, f"the {IMAGES} config holds no image bytes to scan", counts)
    return finish(NAME, counts, failures)
