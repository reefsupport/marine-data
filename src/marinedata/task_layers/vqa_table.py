"""Visual question answering into the unified ``vqa`` table (WP-U10).

One row per question/answer pair: ``question``, ``answer``, ``qa_type``, ``lang`` plus the common
provenance columns of :mod:`marinedata.annotation_schema`. The sources are bound to a producer in
:data:`VQA_SOURCES`; every producer returns a :class:`VqaResult`:

* ``rows``: the image is staged and its sha256 is known (``image_sha256`` set);
* ``pending``: the Q/A belongs to something staged that has no single resolvable sha256 (a
  video, or an image tree without CHECKSUMS): ``image_key`` names it, as for pending boxes;
* ``skipped``: reasons the pair was dropped (image or video not staged, no answer withheld, ...).

``qa_type`` is the source's own question type. Whatever the source says about the answer lives in
``attrs``: ``licence_class`` (always), ``answer_type`` (``short-answer`` / ``multiple-choice`` /
``open-ended`` / ``exact-match``) and, for multiple choice, ``options`` (the lettered choices).

Machine-generated text is never ``human``: a source whose Q/A come from a template or a model
carries ``annotator_type`` ``pseudo`` / ``model`` and an ``annotator_detail`` naming which.

Licence class (``attrs.licence_class``) per source: the registry licence / ``lic-A`` proposed
class (``open``, ``internal-only``, ...); a source with neither is ``unknown``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    arrow_schema,
    normalise_split,
    validate_row,
    write_annotations,
)
from ..licence_class import NC, source_class

PENDING_COLUMNS = ("image_key",)
_PLACEHOLDER_SHA = "0" * 64
OPEN, INTERNAL_ONLY, UNKNOWN = "open", "internal-only", "unknown"
ANSWER_TYPES = frozenset({"short-answer", "multiple-choice", "open-ended", "exact-match"})
_OPTION = re.compile(r"^\s*(?P<letter>[A-Z])[.)]\s+(?P<text>\S.*?)\s*$")


@dataclass(frozen=True)
class VqaSource:
    """One staged Q/A source and how its text was produced."""

    source_id: str
    version: str
    producer: str  # module under task_layers.producers exposing ``staged(spec, ...)``
    annotator: str  # label-origin vocabulary: ``human`` / ``pseudo`` / ``model``
    annotator_detail: str
    ann_license: str | None  # SPDX of the Q/A text, None = none stated
    licence_class: str
    lang: str = "en"
    table: str = "vqa"


VQA_SOURCES: dict[str, VqaSource] = {
    s.source_id: s
    for s in (
        # LICV 2026-10-05: README says CC-BY-NC-4.0 (HF card field says CC-BY-4.0). Fixed templates.
        VqaSource("coralvqa", "rev-3da50a4429e4", "coralvqa_unified", "pseudo",
                  "template:coralvqa-question-templates", "CC-BY-NC-4.0", NC),
        # lic-A: open (MIT README); the QA are model-generated (has_hallucination flags kept).
        VqaSource("marineevt", "rev-37488d3c7690", "marineevt_vqa", "model",
                  "vlm:unspecified (MarineEVT pipeline; model not named in the staged files)",
                  "MIT", OPEN),
        # lic-A: internal-only (no licence stated); QA generation is not documented in the files.
        VqaSource("uwbench", "rev-84af94500308", "uwbench_vqa", "model",
                  "vlm:unverified (UWBench QA generation not documented; conservative non-human)",
                  None, INTERNAL_ONLY),
    )
}  # fmt: skip


@dataclass
class VqaResult:
    rows: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    seen: int = 0  # Q/A pairs read from the source


def _inline_options(line: str) -> list[tuple[str, str]]:
    """``(letter, text)`` pairs of a one-line ``A. x B. y C. z`` list, else ``[]``. The line must
    open with the ``A`` marker; each next marker is the *next expected letter* only, so a stray
    ``D.`` inside an option text (``D. Sub-adult D. Adult``) never splits it."""
    text = line.strip()
    marks: list[tuple[str, int, int]] = []  # (letter, marker start, marker end)
    letter, cursor = "A", 0
    while (m := re.compile(rf"(?<!\S){letter}[.)]\s+").search(text, cursor)) is not None:
        if not marks and m.start() != 0:
            return []
        marks.append((letter, m.start(), m.end()))
        cursor, letter = m.end(), chr(ord(letter) + 1)
        if letter > "Z":
            break
    options = [
        (lt, text[end : marks[i + 1][1] if i + 1 < len(marks) else len(text)].strip(" ,;"))
        for i, (lt, _start, end) in enumerate(marks)
    ]
    return options if len(options) >= 2 and all(t for _, t in options) else []


def parse_options(question: str) -> list[str]:
    """The lettered choices of a multiple-choice question, in order, as ``"A. text"``. Two layouts
    exist in the sources: one choice per line (``A. text`` lines) and all choices on one line
    (``A. x B. y C. z``; MarineEVT). ``[]`` when fewer than two consecutive choices are found."""
    found = [m for line in question.splitlines() if (m := _OPTION.match(line))]
    letters = [m["letter"] for m in found]
    if len(found) >= 2 and letters == [chr(ord("A") + i) for i in range(len(letters))]:
        return [f"{m['letter']}. {m['text']}" for m in found]
    for line in question.splitlines():
        if pairs := _inline_options(line):
            return [f"{letter}. {text}" for letter, text in pairs]
    return []


def free_row(
    *,
    spec: VqaSource,
    ordinal: int,
    sha: str | None,
    payload: Mapping[str, object],
    split: str | None = None,
    attrs: Mapping[str, object] | None = None,
) -> dict:
    """One ``vqa`` / ``captions`` row. ``payload`` holds the table's own columns; ``attrs`` always
    gets ``licence_class``. ``sha`` None = pending (the caller adds ``image_key``)."""
    typ, detail = annotator_from_origin(spec.annotator)
    return {
        "image_sha256": sha,
        "source_id": spec.source_id,
        "source_version": spec.version,
        "ann_id": f"{spec.source_id}:{ordinal}",
        "annotator_type": typ,
        "annotator_detail": spec.annotator_detail or detail,
        "ann_license": spec.ann_license,
        "ann_attribution": None,
        "confidence": None,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        **payload,
        "attrs": json.dumps(
            {
                **dict(attrs or {}),
                "licence_class": source_class(spec.source_id, spec.licence_class),
            },
            sort_keys=True,
        ),
    }


def vqa_row(
    *,
    spec: VqaSource,
    ordinal: int,
    sha: str | None,
    question: str,
    answer: str,
    qa_type: str,
    split: str | None = None,
    attrs: Mapping[str, object] | None = None,
) -> dict:
    return free_row(
        spec=spec,
        ordinal=ordinal,
        sha=sha,
        split=split,
        attrs=attrs,
        payload={
            "question": question.strip(),
            "answer": answer.strip(),
            "qa_type": qa_type,
            "lang": spec.lang,
        },
    )


def validate_pending(table: str, rows: Iterable[Mapping[str, object]]) -> list[str]:
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_COLUMNS}
        core["image_sha256"] = _PLACEHOLDER_SHA
        errs += [f"{row['ann_id']}: {e}" for e in validate_row(table, core)]
        if not row.get("image_key"):
            errs.append(f"{row['ann_id']}: image_key missing")
    return errs


def write_pending(table: str, path: Path, rows: list[dict]) -> int:
    """Rows without a resolvable ``image_sha256``: the table's columns minus the sha plus
    ``image_key``, validated against a placeholder sha."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if errs := validate_pending(table, rows):
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema(table)
    fields = [f for f in base if f.name != "image_sha256"] + [
        pa.field(c, pa.string()) for c in PENDING_COLUMNS
    ]
    tbl = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in fields},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd", write_statistics=True)
    return len(rows)


def write_vqa(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    path = annotation_path(data_dir, "vqa", source_id, source_version)
    write_annotations(path, "vqa", rows)
    return path


def pending_path(data_dir: Path, table: str, source_id: str, source_version: str) -> Path:
    return annotation_path(data_dir, table, source_id, source_version).with_suffix(
        ".pending.parquet"
    )


def licence_split(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    return dict(Counter(json.loads(r["attrs"])["licence_class"] for r in rows))


def annotator_split(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    return dict(Counter(r["annotator_type"] for r in rows))


def staged_vqa(
    spec: VqaSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> VqaResult:
    """Run the producer bound to ``spec`` (imported lazily: producers import this module)."""
    import importlib

    module = importlib.import_module(f"{__package__}.producers.{spec.producer}")
    return module.staged(spec, limit=limit, fetch=fetch, lister=lister)


__all__ = [
    "ANSWER_TYPES", "VQA_SOURCES", "VqaResult", "VqaSource", "annotator_split", "free_row",
    "licence_split", "parse_options", "pending_path", "staged_vqa", "validate_pending", "vqa_row",
    "write_pending", "write_vqa",
]  # fmt: skip
