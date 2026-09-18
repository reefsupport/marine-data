"""The three metadata files staged alongside a version's data (D1 §2, §5, step 6):
``SOURCE.json``, ``ANNOTATIONS.json`` and ``LICENSE``.

All three write once and return the sha256 of the bytes actually on disk, so
``stage_source`` (I2, ``ingest.py``) accumulates ``recorded`` for
:func:`marinedata.checksums.write_checksums` without a second read.

No staged file may carry a timestamp (D1 §3): a timestamped byte makes a second,
otherwise-identical ingest produce different bytes, which
:func:`marinedata.checksums.assert_unchanged` correctly treats as a changed file —
silently turning the mandated re-ingest no-op into a failure. :func:`write_source_json`
enforces this with an assertion rather than trusting callers to remember it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .checksums import write_digest

if TYPE_CHECKING:
    from .models import Source

_TIMESTAMP_KEY = re.compile(r"_at$|^fetched|^staged_at$")
"""D1 §3/step 6: a key ending in ``_at``, or starting with ``fetched`` (e.g. the
``fetched_at``/``fetched_bytes`` shape ``FetchResult.manifest()`` writes), or exactly
``staged_at``. ``^staged_at$`` is redundant with ``_at$`` but named explicitly in the
design, so it stays explicit here too rather than being "simplified" away."""


@dataclass(frozen=True)
class AnnotationCounts:
    """The dynamic counts ``ANNOTATIONS.json`` reports (D1 §2).

    ``geometries`` is a tuple of already-shaped mappings — one per annotation
    geometry, each carrying exactly the keys D1 §2 lists (``kind``, ``path``,
    ``format``, ``schema_id``, ``crosswalk_id``, ``classes``, ``raster_ignore_value``,
    ``images_covered``, ``rows``, ``supervises``, ``observed_indices``) — rather than a
    second named dataclass, because ``ingest.py`` (I2) is the only caller and is best
    placed to decide which of those keys a given geometry kind actually has.
    """

    images: int
    images_without_annotation: int
    geometries: tuple[Mapping[str, object], ...] = ()


def _assert_no_timestamp_keys(obj: object, where: str = "<root>") -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if _TIMESTAMP_KEY.search(str(key)):
                raise ValueError(
                    f"{where}.{key} looks like a timestamp key — staged output must "
                    "carry no timestamps (D1 §3)"
                )
            _assert_no_timestamp_keys(value, f"{where}.{key}")
    elif isinstance(obj, list | tuple):
        for index, item in enumerate(obj):
            _assert_no_timestamp_keys(item, f"{where}[{index}]")


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    return write_digest(path, text.encode("utf-8"))


def write_source_json(path: Path, source: Source, ingest: Mapping[str, object]) -> str:
    """Write ``SOURCE.json``: the registry record plus an ``_ingest`` block.

    ``checksums`` is forced to ``None`` regardless of what ``source`` carries —
    ``Checksums.root_digest`` hashes ``CHECKSUMS.sha256``, which covers ``SOURCE.json``
    itself, so embedding a value here would be circular (D1 §2). Returns the sha256 of
    the file written.
    """
    payload = source.model_dump(mode="json")
    payload["checksums"] = None
    payload["_ingest"] = dict(ingest)
    _assert_no_timestamp_keys(payload)
    return _write_json(path, payload)


def write_annotations_json(path: Path, source: Source, counts: AnnotationCounts) -> str:
    """Write ``ANNOTATIONS.json``: generated, never hand-written (D1 §2).

    Returns the sha256 of the file written.
    """
    payload = {
        "source_id": source.id,
        "version": source.version,
        "images": counts.images,
        "images_without_annotation": counts.images_without_annotation,
        "geometries": [dict(geometry) for geometry in counts.geometries],
    }
    _assert_no_timestamp_keys(payload)
    return _write_json(path, payload)


def write_license(path: Path, source: Source, text: str) -> str:
    """Write ``LICENSE``: the licence text, verbatim (D1 §2).

    ``source`` is threaded through for signature parity with the other two writers and
    so a future caller can assert the text matches ``source.licence`` without changing
    this function's shape; nothing here inspects it. Returns the sha256 of the file
    written.
    """
    _ = source  # unused: kept only for signature parity with the writers above
    return write_digest(path, text.encode("utf-8"))
