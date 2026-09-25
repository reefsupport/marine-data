"""Release-time face blur decision for the v2 build (D-I2), gated by ``--privacy-blur``.

Three things live here, all release-side (never touches v1, D-X):

- :func:`blur_boxes_for_row` — which boxes get blurred: every box on an
  audited-true image (``docs/privacy-audit-2026-09-25.tsv``, ``face_true=1``),
  plus any box the :mod:`.verify` second stage confirms.
- :func:`apply_privacy_blur` — the ``--privacy-blur`` flag surface itself.
  ``enabled=False`` (the default, D-X) returns ``image_bytes`` completely
  unchanged — not just visually close, byte-identical — so a release built
  without the flag is provably the same bytes as one that never imported this
  module. ``enabled=True`` delegates the actual pixel op to
  ``scan.blur_faces`` (Gaussian, box padded 20%, unchanged from WP-5b).
- :func:`build_privacy_config_row` / :func:`write_privacy_config` — the
  per-image ``privacy_blurred`` / ``blur_boxes`` / ``face_candidate`` columns
  the brief asks for, joinable on ``image_sha256`` the same way
  ``privacy.parquet`` already is (D-M: not wired into ``hf_*`` here).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .scan import blur_faces
from .verify import VERIFY_THRESHOLD

__all__ = [
    "apply_privacy_blur",
    "blur_boxes_for_row",
    "build_privacy_config_row",
    "write_privacy_config",
]


def blur_boxes_for_row(
    face_boxes: Sequence[Mapping[str, float]],
    verify_scores: Sequence[float] | None,
    audited_true: bool,
    threshold: float = VERIFY_THRESHOLD,
) -> list[Mapping[str, float]]:
    """Boxes to blur for one image: all of them if ``audited_true``, else verifier-confirmed only.

    ``audited_true`` means this ``image_sha256`` is a ``face_true=1`` row in
    the D-I audit TSV — a human already confirmed a real face is present, so
    every face box on that image is blurred regardless of verifier score.
    Otherwise each box needs its own ``verify_scores[i] >= threshold``
    (``verify_scores`` aligned index-for-index with ``face_boxes``).
    """
    if audited_true:
        return list(face_boxes)
    if not verify_scores:
        return []
    if len(verify_scores) != len(face_boxes):
        raise ValueError("verify_scores must align with face_boxes")
    pairs = zip(face_boxes, verify_scores, strict=True)
    return [box for box, score in pairs if score >= threshold]


def apply_privacy_blur(
    image_bytes: bytes, blur_boxes: Sequence[Mapping[str, float]], enabled: bool
) -> bytes:
    """The ``--privacy-blur`` flag surface. ``enabled=False`` is a byte-identical no-op (D-X)."""
    if not enabled or not blur_boxes:
        return image_bytes
    return blur_faces(image_bytes, blur_boxes, pad_frac=0.2)


def build_privacy_config_row(
    image_sha256: str,
    face_boxes: Sequence[Mapping[str, float]],
    blur_boxes: Sequence[Mapping[str, float]],
    privacy_blur_enabled: bool,
) -> dict[str, Any]:
    """One row of the privacy config: ``privacy_blurred``, ``blur_boxes``, ``face_candidate``.

    ``face_candidate`` is ``True`` whenever the first-stage scan found any
    face box at all — set regardless of the flag, so a consumer of the
    metadata can filter out every face-flagged image even on a release built
    with ``--privacy-blur`` off (D-X: the flag controls pixels, never the
    signal). ``blur_boxes`` is only ever non-empty when the flag is on.
    """
    blurred = bool(privacy_blur_enabled and blur_boxes)
    return {
        "image_sha256": image_sha256,
        "privacy_blurred": blurred,
        "blur_boxes": [dict(b) for b in blur_boxes] if blurred else [],
        "face_candidate": bool(face_boxes),
    }


def write_privacy_config(output_path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the privacy-config parquet (schema: see :func:`build_privacy_config_row`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    box_type = pa.struct(
        [
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("w", pa.float64()),
            ("h", pa.float64()),
            ("score", pa.float64()),
        ]
    )
    schema = pa.schema(
        [
            ("image_sha256", pa.string()),
            ("privacy_blurred", pa.bool_()),
            ("blur_boxes", pa.list_(box_type)),
            ("face_candidate", pa.bool_()),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "image_sha256": r["image_sha256"],
                "privacy_blurred": r["privacy_blurred"],
                "blur_boxes": [
                    {k: b[k] for k in ("x", "y", "w", "h", "score")} for b in r["blur_boxes"]
                ],
                "face_candidate": r["face_candidate"],
            }
            for r in rows
        ],
        schema=schema,
    )
    pq.write_table(table, output_path)
