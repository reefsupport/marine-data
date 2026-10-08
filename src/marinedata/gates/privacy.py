# ruff: noqa: E501
"""Privacy gate (P1.6): EXIF GPS is refused, people in the labels are counted, faces and plates are detected.

* ``exif_gps_present``: an embedded image whose EXIF carries a GPSInfo IFD (tag 34853) fails its
  row, unless its source is set to ``true`` under ``privacy.allow_exif_gps`` in
  ``registry/gates.yaml``. Bytes whose EXIF cannot be read fail as ``exif_unreadable``: a gate
  that could not look must not pass. Always on, no detector needed.
* Rows whose native labels name a person (human, diver, snorkeler, operator, ...) are COUNTED,
  never failed. The label list is built from the build's own label columns and class maps
  (``label``, ``native_label``, ``mask_class_map``, ``class_map``, ``instances``).
* ``face_detected`` / ``plate_detected`` come from an optional pluggable detector
  (:mod:`.privacy_detectors`) run over the embedded images; a score >= ``--privacy-threshold``
  (0.85, the privacy policy's exclusion score) fails the row.

The gate is ``skipped`` only when the build holds no image bytes. Whenever it ran it is required
for publish (:func:`marinedata.gates.summarize`)."""

from __future__ import annotations

import io
import json
import re
import struct
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path

from PIL import Image

from . import GateResult, finish, skip
from .config import exif_gps_allowed
from .phash import image_configs
from .privacy_detectors import Detector
from .shards import config_shards, iter_rows, shard_columns

NAME = "privacy"
DEFAULT_THRESHOLD = 0.85
GPS_INFO_TAG = 34853  # EXIF GPSInfo IFD pointer (0x8825)
IMAGE_COLUMNS = ("image_sha256", "image", "source_id", "source")
LABEL_COLUMNS = ("label", "native_label", "mask_class_map", "class_map", "instances")
PEOPLE = re.compile(
    r"\b(humans?|persons?|people|divers?|snorkel\w*|swimmers?|scuba|operators?|photographers?|pilots?)\b",
    re.I,
)
# What Pillow raises for bytes it cannot open or whose EXIF it cannot parse.
UNREADABLE = (
    OSError, SyntaxError, ValueError, TypeError, KeyError, IndexError, struct.error,
    Image.DecompressionBombError,
)  # fmt: skip


def _strings(value) -> Iterator[str]:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("{", "["):
            try:
                yield from _strings(json.loads(text))
                return
            except ValueError:
                pass
        if text:
            yield text
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def exif_has_gps(data: bytes) -> bool:
    """True when the image's EXIF carries a GPSInfo IFD (tag 34853); raises :data:`UNREADABLE` errors."""
    with Image.open(io.BytesIO(data)) as im:
        return GPS_INFO_TAG in im.getexif()


def _people(build_dir: Path) -> dict:
    by_label: Counter = Counter()
    by_config: Counter = Counter()
    for config, files in config_shards(build_dir).items():
        columns = [c for c in LABEL_COLUMNS if c in shard_columns(files[0])]
        for _, _, _, row in iter_rows(files, columns) if columns else ():
            hits = {s for c in columns for s in _strings(row.get(c)) if PEOPLE.search(s)}
            by_config[config] += bool(hits)
            by_label.update(hits)
    return {
        "people_rows": sum(by_config.values()),
        "people_rows_by_config": {c: n for c, n in sorted(by_config.items()) if n},
        "people_labels": dict(sorted(by_label.items())),
    }


def _scan_images(
    build_dir: Path, allowed: Collection[str], detector: Detector | None, threshold: float
) -> tuple[Counter, set[str], list[str]]:
    stats: Counter = Counter()
    seen: set[str] = set()
    failures: list[str] = []
    for config, files in image_configs(build_dir).items():
        columns = [c for c in IMAGE_COLUMNS if c in shard_columns(files[0])]
        for path, index, split, row in iter_rows(files, columns):
            data = (row.get("image") or {}).get("bytes")
            if not data:
                continue
            stats["images_scanned"] += 1
            sha, source = (
                str(row.get("image_sha256")),
                str(row.get("source_id") or row.get("source")),
            )
            who = f"{config}/{path.name}[{index}] image_sha256={sha} source={source}"
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
                    seen.add(source)
                else:
                    failures.append(f"{who} exif_gps_present: GPSInfo (tag {GPS_INFO_TAG})")
            if detector is None:
                continue
            for hit in detector.detect(data):
                if hit.score >= threshold:
                    failures.append(
                        f"{config}/{split} {sha[:12]} {hit.kind}_detected: "
                        f"score {hit.score:.2f} >= {threshold:.2f} ({detector.name} {detector.version})"
                    )
    return stats, seen, failures


def run_privacy_gate(
    build_dir: Path,
    *,
    detector: Detector | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    allow_exif_gps: Collection[str] | None = None,
    registry_root: Path | None = None,
) -> GateResult:
    """``allow_exif_gps`` (source ids) overrides ``privacy.allow_exif_gps`` from ``registry/gates.yaml``."""
    build_dir = Path(build_dir)
    allowed = frozenset(
        exif_gps_allowed(registry_root) if allow_exif_gps is None else allow_exif_gps
    )
    stats, seen, failures = _scan_images(build_dir, allowed, detector, threshold)
    counts = _people(build_dir) | {
        "images_scanned": stats["images_scanned"],
        "exif_gps_images": stats["exif_gps_images"],
        "exif_gps_allowed": stats["exif_gps_allowed"],
        "exif_gps_allowed_sources": sorted(seen),
        "exif_unreadable": stats["exif_unreadable"],
    }
    if detector is not None:
        counts |= {"detector": f"{detector.name} {detector.version}", "threshold": threshold}
    if not stats["images_scanned"]:
        return skip(
            NAME, "the build holds no image bytes to scan (EXIF GPS, faces, plates)", counts
        )
    return finish(NAME, counts, failures)
