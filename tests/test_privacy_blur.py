"""WP-5e: the release-time blur decision and the ``--privacy-blur`` flag (D-I2/D-X).

Pure-function tests (no network, no ONNX weights) plus one ``integration``-marked
test that proves the actual acceptance criterion — blurring drops the verifier's
own score below its threshold — against a real face and the real YuNet model,
gated the same way as the rest of the repo's network tests
(``MARINEDATA_INTEGRATION=1``, see ``test_integration.py``).
"""

from __future__ import annotations

import os

import pytest

from marinedata.privacy.blur import (
    apply_privacy_blur,
    blur_boxes_for_row,
    build_privacy_config_row,
    write_privacy_config,
)
from marinedata.privacy.verify import VERIFY_THRESHOLD

_FACE_BOXES = (
    {"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.9},
    {"x": 50, "y": 50, "w": 15, "h": 15, "score": 0.55},
)


def test_blur_boxes_for_row_audited_true_takes_every_box() -> None:
    # A human already confirmed this image has a real face: blur ALL face
    # boxes on it, even ones the verifier itself scored low.
    boxes = blur_boxes_for_row(_FACE_BOXES, verify_scores=None, audited_true=True)
    assert boxes == list(_FACE_BOXES)


def test_blur_boxes_for_row_filters_by_verifier_score() -> None:
    scores = [0.9, 0.1]  # first box confirmed, second not
    boxes = blur_boxes_for_row(_FACE_BOXES, verify_scores=scores, audited_true=False)
    assert boxes == [_FACE_BOXES[0]]


def test_blur_boxes_for_row_no_scores_and_not_audited_is_empty() -> None:
    assert blur_boxes_for_row(_FACE_BOXES, verify_scores=None, audited_true=False) == []


def test_blur_boxes_for_row_mismatched_scores_raises() -> None:
    with pytest.raises(ValueError):
        blur_boxes_for_row(_FACE_BOXES, verify_scores=[0.9], audited_true=False)


def test_apply_privacy_blur_disabled_is_byte_identical() -> None:
    original = b"not actually an image but bytes are bytes for this check"
    out = apply_privacy_blur(original, blur_boxes=_FACE_BOXES, enabled=False)
    assert out == original
    assert out is original  # D-X: no-op, not just equal — never even copies


def test_apply_privacy_blur_enabled_with_no_boxes_is_also_a_noop() -> None:
    original = b"still not an image"
    assert apply_privacy_blur(original, blur_boxes=[], enabled=True) == original


def test_apply_privacy_blur_enabled_with_boxes_changes_real_image_bytes() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    img[10:30, 10:30] = (255, 0, 0)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    image_bytes = encoded.tobytes()

    out = apply_privacy_blur(image_bytes, blur_boxes=[_FACE_BOXES[0]], enabled=True)
    assert out != image_bytes


def test_build_privacy_config_row_flag_off_never_carries_boxes() -> None:
    row = build_privacy_config_row(
        "sha1", _FACE_BOXES, blur_boxes=[_FACE_BOXES[0]], privacy_blur_enabled=False
    )
    assert row["privacy_blurred"] is False
    assert row["blur_boxes"] == []
    assert row["face_candidate"] is True  # signal survives even with the flag off


def test_build_privacy_config_row_flag_on_carries_boxes() -> None:
    row = build_privacy_config_row(
        "sha2", _FACE_BOXES, blur_boxes=[_FACE_BOXES[0]], privacy_blur_enabled=True
    )
    assert row["privacy_blurred"] is True
    assert row["blur_boxes"] == [dict(_FACE_BOXES[0])]


def test_build_privacy_config_row_no_face_boxes_face_candidate_false() -> None:
    row = build_privacy_config_row("sha3", (), blur_boxes=[], privacy_blur_enabled=True)
    assert row["face_candidate"] is False
    assert row["privacy_blurred"] is False


def test_write_privacy_config_roundtrip(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")

    rows = [
        build_privacy_config_row("sha1", _FACE_BOXES, [_FACE_BOXES[0]], privacy_blur_enabled=True),
        build_privacy_config_row("sha2", (), [], privacy_blur_enabled=True),
    ]
    out = tmp_path / "privacy_config.parquet"
    write_privacy_config(out, rows)
    table = pq.read_table(out)
    assert table.column("image_sha256").to_pylist() == ["sha1", "sha2"]
    assert table.column("privacy_blurred").to_pylist() == [True, False]
    assert table.column("face_candidate").to_pylist() == [True, False]
    assert len(table.column("blur_boxes").to_pylist()[0]) == 1


def test_blur_faces_kernel_spans_the_full_box_not_half_of_it(monkeypatch) -> None:
    """Regression for WP-5f: a kernel scaled to half the box's short side left
    enough low-frequency shape/colour signal that the verifier still re-detected
    ~40% of real audited-true faces post-blur (see docs/PRIVACY.md D-I2). The
    kernel must span the box's own short side, not a fraction of it."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    from marinedata.privacy.scan import blur_faces

    calls: list[tuple[int, int]] = []
    real_gaussian_blur = cv2.GaussianBlur

    def spy(src, ksize, sigma, *args, **kwargs):
        calls.append(ksize)
        return real_gaussian_blur(src, ksize, sigma, *args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", spy)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok

    box = {"x": 20, "y": 20, "w": 63, "h": 110, "score": 1.0}
    blur_faces(encoded.tobytes(), [box], pad_frac=0.0)

    assert calls, "blur_faces did not call cv2.GaussianBlur"
    kw, kh = calls[0]
    # short side of the box is 63: the kernel must be at least that, not ~31
    # (the old, halved formula).
    assert kw >= 63 or kh >= 63
    assert kw > 31 and kh > 31


RUN = os.environ.get("MARINEDATA_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN, reason="set MARINEDATA_INTEGRATION=1 to run network tests")
def test_blur_drops_the_verifiers_own_score_below_threshold(tmp_path) -> None:
    """The brief's acceptance test: blur must be strong enough that the second-stage
    verifier's score on the BLURRED crop is below :data:`VERIFY_THRESHOLD`, on a
    real face and the real YuNet model (anonymous downloads, no login)."""
    import urllib.request

    import cv2
    import numpy as np

    from marinedata.privacy.scan import blur_faces
    from marinedata.privacy.verify import FaceVerifier

    weights = tmp_path / "face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        weights,
    )
    face_jpg = tmp_path / "lena.jpg"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        face_jpg,
    )
    image_bytes = face_jpg.read_bytes()
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    verifier = FaceVerifier(weights)
    full_box = {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0], "score": 1.0}
    before = verifier.verify(img, full_box, pad_frac=0.0)
    assert before.score >= VERIFY_THRESHOLD  # sanity: the fixture really is a detectable face

    blurred_bytes = blur_faces(image_bytes, [full_box], pad_frac=0.2)
    blurred_img = cv2.imdecode(np.frombuffer(blurred_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    after = verifier.verify(blurred_img, full_box, pad_frac=0.0)
    assert after.score < VERIFY_THRESHOLD
