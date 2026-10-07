"""CoralVQA train producer (WP-8c) — D-Z2 ``vqa`` payload.

Reads the fetched ``CoralVQA_train.jsonl`` (WP-8b, ``$SP/wp8b/coralvqa/``; one row per
Q/A pair: ``image`` filename + a two-turn ``conversations`` list, human question then
model-free-text answer) and writes one row per Q/A pair: ``sha256, source_id,
label_origin, question, answer, qtype, split_hint`` (D-Z2).

The images themselves were never fetched (brief Do-Not: no image downloads; CoralVQA's
image archive is a single 26.7 GB zip — see ``docs/task-layers.md``). ``sha256`` can only
be computed for whichever filenames already exist in the local verification cache
(``~/.cache/marinedata/coralvqa/*.jpg``, ~50 files, fetched during an earlier registry
verification pass, not by this producer). Every other Q/A pair is skipped, not
fabricated — a small, honestly-small ``vqa`` config beats a wrong one. ``label_origin``
is ``"human"``: the question and (per the paper) human-reviewed answer are both authored
by people, not a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from marinedata.checksums import file_digest
from marinedata.tables import _require_pyarrow

SOURCE_ID = "coralvqa"
TASK_ID = "vqa"

_QTYPE_RE = re.compile(r"\[(\w+)\]")
_IMAGE_TAG_RE = re.compile(r"^<image>\n?")


def _parse_question(value: str) -> tuple[str, str]:
    """``(qtype, question_text)`` from a ``"<image>\\n[vqa] ...">`` conversation turn."""
    text = _IMAGE_TAG_RE.sub("", value).strip()
    match = _QTYPE_RE.match(text)
    qtype = match.group(1) if match else "unknown"
    question = _QTYPE_RE.sub("", text, count=1).strip() if match else text
    return qtype, question


def produce_vqa(
    jsonl_path: str | Path,
    images_dir: str | Path,
    out_path: str | Path,
    *,
    split_hint: str = "train",
) -> int:
    """Write ``data/_tasklabels/coralvqa/vqa.parquet``. Returns the row count."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    images_dir = Path(images_dir)
    sha_cache: dict[str, str | None] = {}
    rows: list[dict] = []

    with Path(jsonl_path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_name = record.get("image")
            if not image_name:
                continue
            sha256 = sha_cache.get(image_name, "")
            if sha256 == "":  # not yet looked up (None is a cached miss)
                image_path = images_dir / image_name
                sha256 = file_digest(image_path) if image_path.is_file() else None
                sha_cache[image_name] = sha256
            if sha256 is None:
                continue  # image not staged locally — do not fabricate a key

            turns = record.get("conversations", [])
            human = next((t["value"] for t in turns if t.get("from") == "human"), None)
            answer = next((t["value"] for t in turns if t.get("from") == "gpt"), None)
            if human is None or answer is None:
                continue
            qtype, question = _parse_question(human)
            rows.append(
                {
                    "sha256": sha256,
                    "source_id": SOURCE_ID,
                    "label_origin": "human",
                    "question": question,
                    "answer": answer.strip(),
                    "qtype": qtype,
                    "split_hint": split_hint,
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return len(rows)
