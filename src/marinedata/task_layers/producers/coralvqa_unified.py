"""CoralVQA train Q/A into the unified ``vqa`` table (WP-U10).

Reads the staged ``labels/files/CoralVQA_train.jsonl`` (one record per Q/A pair: ``image``
filename + a two-turn ``conversations`` list) and resolves the image through the staged
``CHECKSUMS.sha256`` (``images/CoralVQA_Image_zip_<stem>.jpg``). The question parser is the one of
:mod:`.coralvqa_vqa` (the ``[vqa]`` tag is ``qa_type``). The ``ann_id`` ordinal is the 0-based
line number in the train file, so it is stable whatever ``limit`` is.

``CoralVQA_test.jsonl`` carries questions only (``text``, ``category``, no answer): no row can be
built from it, so its pairs are counted as skipped, never invented.

Annotator: the questions are fixed templates (the same strings recur across images and every one
ends with the same answer-format instruction), so the pair is ``pseudo`` / ``template:...``; the
earlier D-Z2 producer called it ``human`` on the strength of the paper's review claim, which the
staged files do not show.
"""


from __future__ import annotations

import json
from collections.abc import Callable

from ..image_labels_table import _stride
from ..s3_keyed import StagedTree, fetch_small
from ..vqa_table import VqaResult, VqaSource, vqa_row
from .coralvqa_vqa import _parse_question

IMAGE_PREFIX = "CoralVQA_Image_zip_"
TRAIN = "labels/files/CoralVQA_train.jsonl"
TEST = "labels/files/CoralVQA_test.jsonl"


def _turns(record: dict) -> tuple[str | None, str | None]:
    turns = record.get("conversations") or []
    human = next((t["value"] for t in turns if t.get("from") == "human"), None)
    answer = next((t["value"] for t in turns if t.get("from") == "gpt"), None)
    return human, answer


def staged(
    spec: VqaSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> VqaResult:
    tree = StagedTree(f"{spec.source_id}/{spec.version}", fetch or fetch_small)
    res = VqaResult()
    ready: list[tuple[int, str, str, str, str]] = []  # (line, sha, question, answer, qtype)
    for line_no, line in enumerate(tree.get(TRAIN).decode().splitlines()):
        if not line.strip():
            continue
        res.seen += 1
        record = json.loads(line)
        stem = str(record.get("image") or "").rsplit(".", 1)[0]
        sha = tree.shas.get(("default", IMAGE_PREFIX + stem)) if stem else None
        human, answer = _turns(record)
        if human is None or not (answer or "").strip():
            res.skipped["no question or answer"] += 1
        elif sha is None:
            res.skipped["image not staged"] += 1
        else:
            qtype, question = _parse_question(human)
            ready.append((line_no, sha, question, answer, qtype))
    test = [x for x in tree.get(TEST).decode().splitlines() if x.strip()]
    res.skipped["test split: questions only, no answer in the file"] += len(test)
    res.seen += len(test)
    for line_no, sha, question, answer, qtype in _stride(ready, limit):
        res.rows.append(
            vqa_row(
                spec=spec, ordinal=line_no, sha=sha, question=question, answer=answer,
                qa_type=qtype, split="train", attrs={"answer_type": "short-answer"},
            )
        )  # fmt: skip
    return res
