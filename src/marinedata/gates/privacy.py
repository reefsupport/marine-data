# ruff: noqa: E501
"""Privacy gate (P1.6): people in the labels are counted, faces and plates are detected.

* Rows whose native labels name a person (human, diver, snorkeler, operator, ...) are COUNTED,
  never failed. The label list is built from the build's own label columns and class maps
  (``label``, ``native_label``, ``mask_class_map``, ``class_map``, ``instances``).
* ``face_detected`` / ``plate_detected`` come from a pluggable detector
  (:mod:`.privacy_detectors`) run over the embedded images; a score >= ``--privacy-threshold``
  (0.85, the privacy policy's exclusion score) fails the row.

Without a detector the gate is ``skipped`` (with the people counts) and not required for
publish; once a detector ran it is required."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from . import GateResult, finish, skip
from .phash import image_configs
from .privacy_detectors import Detector
from .shards import config_shards, iter_rows, shard_columns

NAME = "privacy"
DEFAULT_THRESHOLD = 0.85
LABEL_COLUMNS = ("label", "native_label", "mask_class_map", "class_map", "instances")
PEOPLE = re.compile(
    r"\b(humans?|persons?|people|divers?|snorkel\w*|swimmers?|scuba|operators?|photographers?|pilots?)\b",
    re.I,
)


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


def run_privacy_gate(
    build_dir: Path, *, detector: Detector | None = None, threshold: float = DEFAULT_THRESHOLD
) -> GateResult:
    build_dir = Path(build_dir)
    by_label: Counter = Counter()
    by_config: Counter = Counter()
    for config, files in config_shards(build_dir).items():
        columns = [c for c in LABEL_COLUMNS if c in shard_columns(files[0])]
        for _, _, _, row in iter_rows(files, columns) if columns else ():
            hits = {s for c in columns for s in _strings(row.get(c)) if PEOPLE.search(s)}
            by_config[config] += bool(hits)
            by_label.update(hits)
    counts = {
        "people_rows": sum(by_config.values()),
        "people_rows_by_config": {c: n for c, n in sorted(by_config.items()) if n},
        "people_labels": dict(sorted(by_label.items())),
    }
    if detector is None:
        return skip(
            NAME,
            "no face/plate detector configured (--detector MODULE:FACTORY); "
            "people-label rows are counted only",
            counts,
        )
    failures, scanned = [], 0
    for config, files in image_configs(build_dir).items():
        for _, _, split, row in iter_rows(files, ("image_sha256", "image")):
            data = (row.get("image") or {}).get("bytes")
            if not data:
                continue
            scanned += 1
            for hit in detector.detect(data):
                if hit.score >= threshold:
                    failures.append(
                        f"{config}/{split} {str(row.get('image_sha256'))[:12]} {hit.kind}_detected: "
                        f"score {hit.score:.2f} >= {threshold:.2f} ({detector.name} {detector.version})"
                    )
    counts |= {"detector": f"{detector.name} {detector.version}", "images_scanned": scanned}
    if not scanned:
        return skip(NAME, "a detector is configured but the build holds no image bytes", counts)
    return finish(NAME, {**counts, "threshold": threshold}, failures)
