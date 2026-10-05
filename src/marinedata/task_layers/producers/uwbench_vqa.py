"""UWBench image Q/A into the unified ``vqa`` table (WP-U10).

``labels/files/unresolved-<N>.json`` is one Q/A record (``image_id``, ``question``, ``ground_truth``,
``question_id``, ``type``). Only 3,819 of the images are staged (``images/images_test_zip_<id>.<ext>``,
the upstream test zip) and the tree has no CHECKSUMS and no ``metadata.parquet``, so a pair whose
image is staged is a *pending* row (``image_key`` = the staged stem, no sha256); a pair whose image
is not staged is skipped. The ``ann_id`` ordinal is the upstream ``question_id`` (stable under
``limit``). ``qa_type`` is the source's ``type``; ``attrs.answer_type`` is ``short-answer``.
With ``limit`` the label files are read in batches of :data:`BATCH`, spread evenly over the whole
set, and reading stops as soon as ``limit`` pending rows exist (or ``limit * SCAN`` files were read,
or ``budget_s`` seconds passed). Only ``images/`` (3,819 keys) and ``labels/files/`` are listed, never
the whole prefix; a flaky GET gets 3 short tries instead of 8 long ones, so one bad key costs seconds.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from ..boxes_table import bucket_lister
from ..s3_keyed import FetchFailed, fetch_small
from ..vqa_table import VqaResult, VqaSource, vqa_row

SCAN = 6  # most label files read per wanted row: about a quarter have a staged image
BATCH = 256  # label files per round of reads; ``limit`` is checked between rounds
BUDGET_S = 420.0  # wall-clock cap on the reads of a ``limit`` run
_LABEL = re.compile(r"/labels/files/unresolved-(?P<n>\d+)\.json$")
_IMAGE_STEM = re.compile(r"^(?P<zip>images_\w+?)_zip_(?P<id>.+)$")


def _spread(keys: list[str]) -> list[str]:
    """``keys`` in a deterministic order whose every prefix is spread evenly over the whole list
    (a golden-ratio stride made coprime with ``len(keys)``), so an early stop is still a sample."""
    n = len(keys)
    step = max(1, round(n * 0.6180339887))
    while math.gcd(step, n) != 1:
        step += 1
    return [keys[(i * step) % n] for i in range(n)]


def _quick_fetch(key: str) -> bytes:
    return fetch_small(key, tries=3, timeout=15.0, base_delay=0.25, max_delay=2.0)


def staged(
    spec: VqaSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
    budget_s: float | None = BUDGET_S,
) -> VqaResult:
    fetch = fetch or _quick_fetch
    list_keys = lister or bucket_lister()
    base = f"sources/{spec.source_id}/{spec.version}/"
    staged_stems: dict[str, str] = {}
    for key in list_keys(f"{base}images/"):
        if m := _IMAGE_STEM.match(PurePosixPath(key).stem):
            staged_stems[m["id"]] = PurePosixPath(key).stem
    labels = sorted(
        (k for k in list_keys(f"{base}labels/files/") if _LABEL.search(k)),
        key=lambda k: int(_LABEL.search(k)["n"]),
    )
    order = labels if limit is None else _spread(labels)[: limit * SCAN]
    deadline = None if limit is None or budget_s is None else time.monotonic() + budget_s

    def read(key: str) -> dict | None:
        try:
            return json.loads(fetch(key))
        except (FetchFailed, OSError):
            return None  # one flaky key must not kill the run: counted as skipped below

    res = VqaResult()
    with ThreadPoolExecutor(16) as pool:
        for start in range(0, len(order), BATCH):
            if limit is not None and len(res.pending) >= limit:
                break
            if deadline is not None and time.monotonic() > deadline:
                res.skipped["read time budget reached"] += len(order) - start
                break
            loaded = list(pool.map(read, order[start : start + BATCH]))
            records = [r for r in loaded if r is not None]
            res.seen += len(loaded)
            res.skipped["label file fetch failed"] += len(loaded) - len(records)
            for rec in records:
                _add(res, spec, rec, staged_stems, limit)
    return res


def _add(
    res: VqaResult, spec: VqaSource, rec: dict, staged_stems: dict[str, str], limit: int | None
) -> None:
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
