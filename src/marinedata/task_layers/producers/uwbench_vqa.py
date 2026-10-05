"""UWBench image Q/A into the unified ``vqa`` table (WP-U10).

``labels/files/unresolved-<N>.json`` is one Q/A record (``image_id``, ``question``, ``ground_truth``,
``question_id``, ``type``). Only 3,819 of the images are staged (``images/images_test_zip_<id>.<ext>``,
the upstream test zip) and the tree has no CHECKSUMS and no ``metadata.parquet``, so a pair whose
image is staged is a *pending* row (``image_key`` = the staged stem, no sha256); a pair whose image
is not staged is skipped. The ``ann_id`` ordinal is the upstream ``question_id`` (stable under
``limit``). ``qa_type`` is the source's ``type``; ``attrs.answer_type`` is ``short-answer``.
With ``limit`` only an evenly spaced ``limit * SCAN`` label files are read.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from ..boxes_table import bucket_lister
from ..image_labels_table import _stride
from ..s3_keyed import FetchFailed, fetch_small
from ..vqa_table import VqaResult, VqaSource, vqa_row

SCAN = 6  # label files read per wanted row: about a quarter of the questions have a staged image
_LABEL = re.compile(r"/labels/files/unresolved-(?P<n>\d+)\.json$")
_IMAGE_STEM = re.compile(r"^(?P<zip>images_\w+?)_zip_(?P<id>.+)$")


def staged(
    spec: VqaSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> VqaResult:
    fetch = fetch or fetch_small
    keys = (lister or bucket_lister())(f"sources/{spec.source_id}/{spec.version}/")
    staged_stems: dict[str, str] = {}
    for key in keys:
        if "/images/" in key and (m := _IMAGE_STEM.match(PurePosixPath(key).stem)):
            staged_stems[m["id"]] = PurePosixPath(key).stem
    labels = sorted((k for k in keys if _LABEL.search(k)), key=lambda k: int(_LABEL.search(k)["n"]))
    scan = labels if limit is None else _stride(labels, limit * SCAN)

    def read(key: str) -> dict | None:
        try:
            return json.loads(fetch(key))
        except (FetchFailed, OSError):
            return None  # one flaky key must not kill the run: counted as skipped below

    with ThreadPoolExecutor(16) as pool:
        loaded = list(pool.map(read, scan))
    records = [r for r in loaded if r is not None]
    res = VqaResult(seen=len(loaded))
    res.skipped["label file fetch failed"] += len(loaded) - len(records)
    for rec in records:
        stem = staged_stems.get(PurePosixPath(str(rec.get("image_id"))).stem)
        question, answer = str(rec.get("question") or ""), str(rec.get("ground_truth") or "")
        if not question.strip() or not answer.strip():
            res.skipped["no question or answer"] += 1
        elif stem is None:
            res.skipped["image not staged"] += 1
        elif limit is not None and len(res.pending) >= limit:
            res.skipped["over limit"] += 1
        else:
            row = vqa_row(
                spec=spec, ordinal=int(rec["question_id"]), sha=None, question=question,
                answer=answer, qa_type=str(rec.get("type") or "unknown"),
                split="test" if stem.startswith("images_test_zip_") else None,
                attrs={"answer_type": "short-answer", "image_id": rec["image_id"]},
            )  # fmt: skip
            res.pending.append({**{k: v for k, v in row.items() if k != "image_sha256"},
                                "image_key": stem})  # fmt: skip
    return res
