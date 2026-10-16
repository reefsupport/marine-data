"""Human review of automatic face detections: ``registry/privacy/review.yaml``.

A face detector fires on things that are not faces (a close-up of bleached coral tissue scored 0.86
under YuNet, over the 0.85 exclusion score). The review list records, per ``image_sha256``, what a
person decided after looking at the image:

* ``cleared``: no person is visible, the detection is a false positive. Release gate G8 counts the
  detection under ``cleared_reviewed`` (sha, reviewer and date listed in ``GATES.json``) instead of
  failing it. A review clears the face detection only: EXIF GPS still fails the image.
* ``withhold``: the image must not ship. The v1.1.0 build drops it from every table
  (:mod:`marinedata.task_layers.v11_tables`, counted in ``V11_TABLES.json`` ``withheld_images``)
  and G8 fails ``withheld_image_present`` when it is still in the build.

The file is the audit trail: committed, reviewed in a merge request, never written by a tool.
Every entry names the detector and score that fired, the reason and who looked. A bad entry
(unknown verdict, malformed or repeated sha256, missing field) is a :class:`ValueError`, so a gate
that cannot trust the list fails instead of passing.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
CLEARED, WITHHOLD = "cleared", "withhold"
VERDICTS = (CLEARED, WITHHOLD)
REVIEW_FILE = Path("privacy") / "review.yaml"
"""The list, relative to the registry root."""
FIELDS = (
    "image_sha256", "verdict", "source_id", "detector", "score", "reason", "reviewer",
    "reviewed_at",
)  # fmt: skip
TEXT_FIELDS = ("source_id", "detector", "reason", "reviewer")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class ReviewEntry:
    image_sha256: str
    verdict: str
    source_id: str
    detector: str
    score: float
    reason: str
    reviewer: str
    reviewed_at: str  # YYYY-MM-DD


@dataclass(frozen=True)
class ReviewList:
    entries: tuple[ReviewEntry, ...] = ()

    @property
    def cleared(self) -> dict[str, ReviewEntry]:
        """``image_sha256 -> entry`` of the detections a person cleared."""
        return {e.image_sha256: e for e in self.entries if e.verdict == CLEARED}

    @property
    def withheld(self) -> dict[str, ReviewEntry]:
        """``image_sha256 -> entry`` of the images that must not ship."""
        return {e.image_sha256: e for e in self.entries if e.verdict == WITHHOLD}


def _day(value: Any, where: str) -> str:
    if type(value) is dt.date:  # YAML parses an unquoted 2026-10-16 as a date
        return value.isoformat()
    if isinstance(value, str) and _DAY.fullmatch(value):
        try:
            return dt.date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise ValueError(f"{where}: reviewed_at must be a YYYY-MM-DD date, got {value!r}")


def _entry(raw: Any, where: str) -> ReviewEntry:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    missing = [f for f in FIELDS if raw.get(f) is None]
    if missing:
        raise ValueError(f"{where}: missing field(s) {', '.join(missing)}")
    unknown = sorted(str(k) for k in raw if k not in FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}")
    sha = raw["image_sha256"]
    if not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        raise ValueError(f"{where}: image_sha256 must be 64 hex characters, got {sha!r}")
    if raw["verdict"] not in VERDICTS:
        raise ValueError(
            f"{where}: verdict must be one of {', '.join(VERDICTS)}, got {raw['verdict']!r}"
        )
    for name in TEXT_FIELDS:
        if not (isinstance(raw[name], str) and raw[name].strip()):
            raise ValueError(f"{where}: {name} must be a non-empty string")
    score = raw["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError(f"{where}: score must be a number from 0 to 1, got {score!r}")
    return ReviewEntry(
        image_sha256=sha.lower(), verdict=raw["verdict"],
        **{name: raw[name].strip() for name in TEXT_FIELDS},
        score=float(score), reviewed_at=_day(raw["reviewed_at"], where),
    )  # fmt: skip


def parse_review(data: Any, *, where: str = "privacy review") -> ReviewList:
    """Validate a loaded document: ``{schema_version: 1, entries: [...]}``."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected a mapping with schema_version and entries")
    if data.get("schema_version") != SCHEMA_VERSION or isinstance(data.get("schema_version"), bool):
        raise ValueError(f"{where}: schema_version must be {SCHEMA_VERSION}")
    raw_entries = data.get("entries")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        raise ValueError(f"{where}: entries must be a list")
    entries: list[ReviewEntry] = []
    seen: dict[str, int] = {}
    for position, raw in enumerate(raw_entries, start=1):
        entry = _entry(raw, f"{where}: entries[{position}]")
        if entry.image_sha256 in seen:
            raise ValueError(
                f"{where}: entries[{position}]: duplicate image_sha256 {entry.image_sha256} "
                f"(already entries[{seen[entry.image_sha256]}])"
            )
        seen[entry.image_sha256] = position
        entries.append(entry)
    return ReviewList(tuple(entries))


def default_review_path(registry_root: str | Path | None = None) -> Path:
    from ..licence_class import _registry_root

    return _registry_root(registry_root) / REVIEW_FILE


def load_review(
    path: str | Path | None = None, *, registry_root: str | Path | None = None
) -> ReviewList:
    """The list at ``path``, else the registry's own ``privacy/review.yaml`` (no file: empty).

    An explicit ``path`` that is not a file raises :class:`FileNotFoundError`; a file that does not
    parse or validate raises :class:`ValueError`."""
    target = Path(path) if path is not None else default_review_path(registry_root)
    if not target.is_file():
        if path is not None:
            raise FileNotFoundError(f"privacy review list not found: {target}")
        return ReviewList()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{target}: not valid YAML: {exc}") from exc
    return parse_review(data, where=str(target))
