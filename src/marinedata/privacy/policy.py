"""Release privacy policy (WP-R9): what a face-detector score does to an image.

* score >= :data:`FACE_EXCLUDE_SCORE` -> the image is EXCLUDED from the release and listed in
  ``RELEASE.json`` under ``privacy_exclusions`` (image id, source, score, detector + version).
* :data:`FACE_FLAG_SCORE` <= score < :data:`FACE_EXCLUDE_SCORE` -> kept, and its ``metadata`` row
  carries ``privacy_flag`` = ``possible_face`` plus ``face_score``.
* below :data:`FACE_FLAG_SCORE` -> nothing.

The thresholds live here and nowhere else; the dataset card prints them (:func:`card_lines`).
The input is the ``privacy.parquet`` the ``privacy-scan`` step writes (``face_boxes`` per image).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FACE_EXCLUDE_SCORE = 0.85
FACE_FLAG_SCORE = 0.60
POSSIBLE_FACE = "possible_face"
DETECTOR = "yunet"

EXCLUDE, FLAG, CLEAR = "exclude", "flag", "none"


def band(score: float | None) -> str:
    """``exclude`` | ``flag`` | ``none`` for one face score (``None`` = no face found)."""
    if score is None:
        return CLEAR
    if score >= FACE_EXCLUDE_SCORE:
        return EXCLUDE
    return FLAG if score >= FACE_FLAG_SCORE else CLEAR


def max_face_score(face_boxes: Iterable[Mapping[str, Any]] | None) -> float | None:
    scores = [float(b["score"]) for b in face_boxes or () if b.get("score") is not None]
    return max(scores) if scores else None


@dataclass(frozen=True)
class PrivacyOutcome:
    exclusions: tuple[dict[str, Any], ...] = ()
    flags: Mapping[str, float] | None = None  # image_sha256 -> face_score, flag band only

    @property
    def excluded(self) -> frozenset[str]:
        return frozenset(e["image_id"] for e in self.exclusions)


def evaluate(
    privacy_rows: Iterable[Mapping[str, Any]], source_by_sha: Mapping[str, str]
) -> PrivacyOutcome:
    """The policy over scan rows. ``source_by_sha`` names each image's (primary) source."""
    exclusions: list[dict[str, Any]] = []
    flags: dict[str, float] = {}
    for row in privacy_rows:
        sha = row["image_sha256"]
        score = max_face_score(row.get("face_boxes"))
        verdict = band(score)
        if verdict == EXCLUDE:
            exclusions.append(
                {
                    "image_id": sha,
                    "source_id": source_by_sha.get(sha, "unknown"),
                    "score": round(float(score), 4),  # type: ignore[arg-type]
                    "detector": DETECTOR,
                    "detector_version": row.get("face_detector_version") or "unknown",
                }
            )
        elif verdict == FLAG:
            flags[sha] = round(float(score), 4)  # type: ignore[arg-type]
    exclusions.sort(key=lambda e: (e["source_id"], e["image_id"]))
    return PrivacyOutcome(tuple(exclusions), flags)


def load_privacy_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(
        path, columns=["image_sha256", "face_boxes", "face_detector_version"]
    ).to_pylist()


def drop_images(
    rows_by_task: Mapping[str, Sequence[Any]], excluded: frozenset[str]
) -> dict[str, list[Any]]:
    """New ``{task: rows}`` without the excluded images (rows expose ``image_sha256``)."""
    return {
        task: [r for r in rows if r.image_sha256 not in excluded]
        for task, rows in rows_by_task.items()
    }


def primary_sources(rows_by_task: Mapping[str, Sequence[Any]]) -> dict[str, str]:
    """``image_sha256 -> source_id`` (the lowest id when several tasks share the image)."""
    out: dict[str, str] = {}
    for rows in rows_by_task.values():
        for r in rows:
            if r.image_sha256 not in out or r.source_id < out[r.image_sha256]:
                out[r.image_sha256] = r.source_id
    return out


def apply_to_rows(
    rows_by_task: Mapping[str, Sequence[Any]], privacy_path: str | Path | None
) -> tuple[dict[str, list[Any]], PrivacyOutcome]:
    """The release-path entry point: ``(rows without excluded images, outcome)``. No scan file
    -> rows unchanged and an empty outcome."""
    if privacy_path is None:
        return {t: list(r) for t, r in rows_by_task.items()}, PrivacyOutcome()
    outcome = evaluate(load_privacy_rows(privacy_path), primary_sources(rows_by_task))
    return drop_images(rows_by_task, outcome.excluded), outcome


def policy_block(outcome: PrivacyOutcome) -> dict[str, Any]:
    return {
        "exclude_score": FACE_EXCLUDE_SCORE,
        "flag_score": FACE_FLAG_SCORE,
        "excluded": len(outcome.exclusions),
        "flagged": len(outcome.flags or {}),
    }


def record_in_release(release_json: Path, outcome: PrivacyOutcome) -> None:
    """Write ``privacy_exclusions`` + ``privacy_policy`` into ``RELEASE.json`` (atomic replace)."""
    if not release_json.is_file():
        return
    data = json.loads(release_json.read_text())
    data = {
        **data,
        "privacy_policy": policy_block(outcome),
        "privacy_exclusions": list(outcome.exclusions),
    }
    tmp = release_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, release_json)


def card_lines(release: Mapping[str, Any]) -> list[str]:
    """The card's privacy section: thresholds from this module, counts from RELEASE.json."""
    block = release.get("privacy_policy") or {}
    n_ex = block.get("excluded", len(release.get("privacy_exclusions") or ()))
    lines = [
        "",
        "## Privacy",
        "",
        f"Images are scanned with a face detector ({DETECTOR}). A face score of "
        f"{FACE_EXCLUDE_SCORE:.2f} or higher excludes the image from the release (listed in "
        f"`RELEASE.json` under `privacy_exclusions`); a score from {FACE_FLAG_SCORE:.2f} up to "
        f"{FACE_EXCLUDE_SCORE:.2f} keeps the image and sets `privacy_flag` = `{POSSIBLE_FACE}` "
        "with the `face_score` in the `metadata` config. Detection is automatic and can miss "
        "faces or flag non-faces.",
    ]
    if block:
        lines.append(f"This release excludes {n_ex} image(s) and flags {block.get('flagged', 0)}.")
    return lines
