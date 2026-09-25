"""WP-5b: privacy-scan module — deterministic blur, occlusion heuristic, schema.

Detector-model tests (``FaceDetector``/``PersonDetector``) need the ``privacy`` extra
(``uv sync --extra privacy``: torch, torchvision, opencv-python-headless), which is
NOT part of the standard ``[dev]`` CI install (see ``.github/workflows/ci.yml``), so
they ``importorskip`` — this keeps ``make ci`` green without the heavy ML deps while
still giving full coverage in an environment that has them.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from marinedata.privacy import (  # noqa: E402
    Box,
    FaceBox,
    ScanRow,
    _looks_occluded,
    blur_faces,
    sha256_bytes,
)


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 120  # flat-ish colour so blur has something to smooth
    img[10:30, 10:30] = (255, 0, 0)  # a distinct patch to blur
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def test_blur_faces_is_deterministic() -> None:
    image_bytes = _make_jpeg()
    boxes = [{"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.95}]
    out1 = blur_faces(image_bytes, boxes)
    out2 = blur_faces(image_bytes, boxes)
    assert out1 == out2


def test_blur_faces_changes_the_boxed_region() -> None:
    image_bytes = _make_jpeg()
    boxes = [{"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.95}]
    blurred = blur_faces(image_bytes, boxes)
    assert blurred != image_bytes

    original = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = cv2.imdecode(np.frombuffer(blurred, dtype=np.uint8), cv2.IMREAD_COLOR)
    # centre of the padded box should have moved away from pure red after blur
    assert not np.array_equal(original[15:25, 15:25], result[15:25, 15:25])


def test_blur_faces_leaves_untouched_pixels_alone() -> None:
    image_bytes = _make_jpeg()
    boxes = [{"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.95}]
    blurred = blur_faces(image_bytes, boxes)
    result = cv2.imdecode(np.frombuffer(blurred, dtype=np.uint8), cv2.IMREAD_COLOR)
    # far corner, well outside the padded box, should be exactly the flat fill colour
    assert tuple(int(c) for c in result[60, 60]) == (120, 0, 0)


def test_blur_faces_no_boxes_is_a_noop_pixelwise() -> None:
    # PNG round-trips losslessly (unlike JPEG), so a real pixel-equality check is valid.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    image_bytes = encoded.tobytes()

    blurred = blur_faces(image_bytes, [])
    result = cv2.imdecode(np.frombuffer(blurred, dtype=np.uint8), cv2.IMREAD_COLOR)
    original = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(original, result)


def test_sha256_bytes_matches_hashlib() -> None:
    import hashlib

    data = b"some image bytes"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_looks_occluded_bare_face_geometry() -> None:
    # eyes wide apart, mouth corners well below and spread out, nose between: bare face
    landmarks = [(20.0, 20.0), (40.0, 20.0), (30.0, 30.0), (22.0, 45.0), (38.0, 45.0)]
    assert _looks_occluded(landmarks) is False


def test_looks_occluded_collapsed_mouth_is_flagged() -> None:
    # mouth corners collapsed near the nose: proxy for a mask/regulator occluding the mouth
    landmarks = [(20.0, 20.0), (40.0, 20.0), (30.0, 30.0), (29.0, 31.0), (31.0, 31.0)]
    assert _looks_occluded(landmarks) is True


def test_scan_row_face_identifiable_is_any_of_boxes() -> None:
    row = ScanRow(
        image_sha256="abc",
        face_boxes=(
            FaceBox(x=0, y=0, w=10, h=10, score=0.9, identifiable=False),
            FaceBox(x=0, y=0, w=40, h=40, score=0.95, identifiable=True),
        ),
        person_boxes=(Box(x=0, y=0, w=50, h=100, score=0.8),),
    )
    assert row.face_identifiable is True
    assert row.people_present is True
    record = row.to_record()
    assert record["face_identifiable"] is True
    assert record["people_present"] is True
    assert len(record["face_boxes"]) == 2
