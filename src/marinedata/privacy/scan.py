"""Face / person privacy scan for the open-imagery release (WP-5b).

Replaces the Haar-cascade proxy from :mod:`marinedata` WP-5 (documented in
``docs/ETHICS_FACE_AUDIT.md``) with two real, open-weight, no-login detectors:

- **Faces**: OpenCV Zoo YuNet (``cv2.FaceDetectorYN``), ONNX weights
  ``face_detection_yunet_2023mar.onnx``.
- **People/divers**: a COCO ``person``-class detector, torchvision's
  ``ssdlite320_mobilenet_v3_large`` with ``COCO_V1`` weights (13 MB — chosen over
  the larger Faster R-CNN variant to stay well under the 300 MB weight budget and
  keep CPU/MPS batch scanning fast).

Both run CPU by default and use MPS (Apple GPU) for the person detector when
available (``torch.backends.mps.is_available()``); YuNet's ONNX backend in this
OpenCV build only supports CPU.

**Identifiability heuristic** (calibrated against ``docs/privacy-audit-2026-09-25.tsv``,
see ``docs/PRIVACY.md``): a face is ``face_identifiable`` when its short side is
>= ``MIN_FACE_PX`` pixels, its detector score is >= ``MIN_FACE_SCORE``, and it is not
flagged as mask/regulator-occluded by :func:`_looks_occluded`, a landmark-geometry
proxy — YuNet returns 5 landmarks (eyes, nose, mouth corners); a regulator or mask
typically collapses the mouth-corner-to-nose distance relative to the eye span.

**Resumability**: :func:`scan` loads any existing output parquet's ``image_sha256``
set first and skips those rows, so a killed/restarted run only redoes new images.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_FACE_PX = 32
MIN_FACE_SCORE = 0.8
YUNET_VERSION = "yunet_2023mar"
PERSON_MODEL_VERSION = "ssdlite320_mobilenet_v3_large_coco_v1"
COCO_PERSON_LABEL = 1  # torchvision COCO_V1 category index for "person"


@dataclass(frozen=True)
class Box:
    """A pixel-space detection box, top-left origin."""

    x: float
    y: float
    w: float
    h: float
    score: float

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": self.score}


@dataclass(frozen=True)
class FaceBox(Box):
    identifiable: bool = False
    occluded: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        d: dict[str, float | bool] = dict(super().as_dict())
        d["identifiable"] = self.identifiable
        d["occluded"] = self.occluded
        return d


@dataclass(frozen=True)
class ScanRow:
    image_sha256: str
    face_boxes: tuple[FaceBox, ...] = ()
    person_boxes: tuple[Box, ...] = ()
    face_detector_version: str = YUNET_VERSION
    person_detector_version: str = PERSON_MODEL_VERSION

    @property
    def face_identifiable(self) -> bool:
        return any(b.identifiable for b in self.face_boxes)

    @property
    def people_present(self) -> bool:
        return bool(self.person_boxes)

    def to_record(self) -> dict[str, Any]:
        return {
            "image_sha256": self.image_sha256,
            "face_boxes": [b.as_dict() for b in self.face_boxes],
            "person_boxes": [b.as_dict() for b in self.person_boxes],
            "face_identifiable": self.face_identifiable,
            "people_present": self.people_present,
            "face_detector_version": self.face_detector_version,
            "person_detector_version": self.person_detector_version,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_occluded(landmarks: Sequence[tuple[float, float]]) -> bool:
    """Proxy for "face behind a mask/regulator" from YuNet's 5 landmarks.

    Order: right eye, left eye, nose, right mouth corner, left mouth corner.
    A regulator/mask strap tends to pull the (noisy, extrapolated) mouth-corner
    landmarks in toward the nose; a bare face keeps mouth corners well below and
    outside the eye span. Ratio and threshold calibrated in
    ``docs/PRIVACY.md`` against the manual audit.
    """
    if len(landmarks) < 5:
        return False
    (rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy) = landmarks[:5]
    eye_span = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    mouth_span = ((lmx - rmx) ** 2 + (lmy - rmy) ** 2) ** 0.5
    nose_to_mouth = ((nx - (rmx + lmx) / 2) ** 2 + (ny - (rmy + lmy) / 2) ** 2) ** 0.5
    if eye_span <= 1e-6:
        return False
    return (mouth_span / eye_span) < 0.35 or (nose_to_mouth / eye_span) < 0.25


class FaceDetector:
    """Wraps ``cv2.FaceDetectorYN`` (OpenCV Zoo YuNet, CPU only)."""

    def __init__(self, weights_path: str | Path, score_threshold: float = 0.6) -> None:
        import cv2

        self._cv2 = cv2
        self._model = cv2.FaceDetectorYN.create(
            str(weights_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, image_bgr: Any) -> list[FaceBox]:
        h, w = image_bgr.shape[:2]
        self._model.setInputSize((w, h))
        _, faces = self._model.detect(image_bgr)
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, bw, bh = row[0], row[1], row[2], row[3]
            score = float(row[14])
            landmarks = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
            short_side = min(bw, bh)
            identifiable = (
                short_side >= MIN_FACE_PX
                and score >= MIN_FACE_SCORE
                and not _looks_occluded(landmarks)
            )
            out.append(
                FaceBox(
                    x=float(x),
                    y=float(y),
                    w=float(bw),
                    h=float(bh),
                    score=score,
                    identifiable=identifiable,
                    occluded=_looks_occluded(landmarks),
                )
            )
        return out


class PersonDetector:
    """Wraps torchvision ``ssdlite320_mobilenet_v3_large`` (COCO_V1 weights).

    ``weights_cache_dir`` is a ``TORCH_HOME``-shaped directory (i.e. containing
    ``hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-*.pth``, exactly what
    ``torch.hub`` writes when it first downloads the checkpoint). Pointing this at a
    pre-populated cache makes the run fully offline; letting torchvision build the
    model from the ``COCO_V1`` enum (rather than hand-loading a bare state dict into
    a randomly-initialised model) is required — ``ssdlite320_mobilenet_v3_large()``
    with ``weights=None`` silently builds a *narrower* backbone (half the channels
    in the later blocks) than the one the released checkpoint was trained with, so a
    manual ``load_state_dict`` on that path always fails with a shape mismatch.
    """

    def __init__(self, weights_cache_dir: str | Path | None = None, score_threshold: float = 0.5):
        import os

        import torch
        import torchvision

        if weights_cache_dir is not None:
            os.environ["TORCH_HOME"] = str(weights_cache_dir)
        weights_enum = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights_enum)
        model.eval()
        self._model = model.to(self.device)
        self._threshold = score_threshold

    def detect_batch(self, images_rgb: Sequence[Any]) -> list[list[Box]]:
        import numpy as np

        tensors = [
            self._torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
            for img in images_rgb
        ]
        tensors = [t.to(self.device) for t in tensors]
        with self._torch.no_grad():
            predictions = self._model(tensors)
        results = []
        for pred in predictions:
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            rows = []
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels, strict=True):
                if label == COCO_PERSON_LABEL and score >= self._threshold:
                    box = Box(float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score))
                    rows.append(box)
            results.append(rows)
        return results


def blur_faces(
    image_bytes: bytes, boxes: Iterable[Mapping[str, float]], pad_frac: float = 0.2
) -> bytes:
    """Gaussian-blur every box (padded ``pad_frac`` on each side) in ``image_bytes``.

    Deterministic: same input bytes + boxes always produce the same output bytes.
    Re-encodes in the source format (falls back to PNG if the format can't be
    inferred) at full fidelity otherwise. Used by the v2 release build to blur
    ``face_identifiable`` boxes in the RELEASED pixels; never mutates v1.
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("blur_faces: could not decode image_bytes")
    h, w = img.shape[:2]
    for box in boxes:
        bx, by, bw, bh = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
        pad_x, pad_y = bw * pad_frac, bh * pad_frac
        x0 = max(0, round(bx - pad_x))
        y0 = max(0, round(by - pad_y))
        x1 = min(w, round(bx + bw + pad_x))
        y1 = min(h, round(by + bh + pad_y))
        if x1 <= x0 or y1 <= y0:
            continue
        region = img[y0:y1, x0:x1]
        ksize = max(3, (min(region.shape[:2]) // 2) | 1)  # odd, scales with box size
        img[y0:y1, x0:x1] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    ext = _guess_ext(image_bytes)
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("blur_faces: re-encode failed")
    return encoded.tobytes()


def _guess_ext(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".png"


def iter_image_rows(parquet_paths: Sequence[Path]) -> Iterator[tuple[str, bytes]]:
    """Yield ``(image_sha256, image_bytes)`` across a set of HF ``images`` shards."""
    import pyarrow.parquet as pq

    for path in parquet_paths:
        table = pq.read_table(path, columns=["image_sha256", "image"])
        shas = table.column("image_sha256").to_pylist()
        images = table.column("image").to_pylist()
        for sha, image_struct in zip(shas, images, strict=True):
            yield sha, image_struct["bytes"]


@dataclass
class ScanStats:
    scanned: int = 0
    skipped_cached: int = 0
    decode_errors: int = 0
    faces_found: int = 0
    people_found: int = 0


def load_cached_shas(output_path: Path) -> set[str]:
    """Existing ``image_sha256`` values in ``output_path``, or empty if absent."""
    if not output_path.exists():
        return set()
    import pyarrow.parquet as pq

    table = pq.read_table(output_path, columns=["image_sha256"])
    return set(table.column("image_sha256").to_pylist())


def write_output(output_path: Path, rows: Sequence[ScanRow], append: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = [r.to_record() for r in rows]
    box_type = pa.struct(
        [
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("w", pa.float64()),
            ("h", pa.float64()),
            ("score", pa.float64()),
        ]
    )
    face_type = pa.struct(
        [
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("w", pa.float64()),
            ("h", pa.float64()),
            ("score", pa.float64()),
            ("identifiable", pa.bool_()),
            ("occluded", pa.bool_()),
        ]
    )
    schema = pa.schema(
        [
            ("image_sha256", pa.string()),
            ("face_boxes", pa.list_(face_type)),
            ("person_boxes", pa.list_(box_type)),
            ("face_identifiable", pa.bool_()),
            ("people_present", pa.bool_()),
            ("face_detector_version", pa.string()),
            ("person_detector_version", pa.string()),
        ]
    )
    new_table = pa.Table.from_pylist(records, schema=schema)
    if append and output_path.exists():
        existing = pq.read_table(output_path)
        new_table = pa.concat_tables([existing, new_table])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, output_path)


def scan(
    parquet_paths: Sequence[Path],
    output_path: Path,
    face_weights: Path,
    person_weights: Path | None,
    batch_size: int = 16,
    limit: int | None = None,
    flush_every: int = 500,
    log: Any = None,
) -> ScanStats:
    """Batched, resumable, cached-by-sha scan over a set of ``images`` shards.

    Skips any ``image_sha256`` already present in ``output_path`` (resumability),
    processes in batches of ``batch_size`` for the person detector, and flushes to
    ``output_path`` every ``flush_every`` new rows so a kill mid-run loses at most
    one flush interval of work.
    """
    import cv2
    import numpy as np

    face_detector = FaceDetector(face_weights)
    person_detector = PersonDetector(person_weights) if person_weights else None
    cached = load_cached_shas(output_path)
    stats = ScanStats()
    pending: list[ScanRow] = []
    batch_imgs: list[Any] = []
    batch_shas: list[str] = []
    batch_faces: dict[str, list[FaceBox]] = {}

    def flush_person_batch() -> None:
        if not batch_imgs:
            return
        if person_detector:
            person_results = person_detector.detect_batch(batch_imgs)
        else:
            person_results = [[] for _ in batch_imgs]
        for sha, persons in zip(batch_shas, person_results, strict=True):
            row = ScanRow(
                image_sha256=sha,
                face_boxes=tuple(batch_faces[sha]),
                person_boxes=tuple(persons),
            )
            pending.append(row)
            stats.faces_found += len(row.face_boxes)
            stats.people_found += len(row.person_boxes)
        batch_imgs.clear()
        batch_shas.clear()
        batch_faces.clear()

    for sha, image_bytes in iter_image_rows(parquet_paths):
        if limit is not None and stats.scanned >= limit:
            break
        if sha in cached:
            stats.skipped_cached += 1
            continue
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            stats.decode_errors += 1
            continue
        faces = face_detector.detect(img_bgr)
        batch_faces[sha] = faces
        batch_shas.append(sha)
        batch_imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        stats.scanned += 1
        if len(batch_imgs) >= batch_size:
            flush_person_batch()
        if len(pending) >= flush_every:
            write_output(output_path, pending, append=True)
            if log:
                log(f"flushed {len(pending)} rows ({stats.scanned} scanned total)")
            pending.clear()
    flush_person_batch()
    if pending:
        write_output(output_path, pending, append=True)
    return stats
