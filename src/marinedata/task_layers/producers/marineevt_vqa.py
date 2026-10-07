"""MarineEVT video Q/A into the unified ``vqa`` table (WP-U10).

``labels/files/<split>_<Dimension>_<Subdimension>_multi_turn_data*.json`` (``{"data": [...]}``) holds
one record per question about one *video*; the staged images are frames of those videos named
``<split>_<Dimension>_<Subdimension>_videos_zip_videos_<video_id>_<frame>``. A video is not one
image, so no Q/A has a single ``image_sha256``: every pair whose video has staged frames is a
*pending* row (``image_key`` = ``<split>_<Dimension>_<Subdimension>/<video_id>``, ``attrs.frames_staged``
= how many frames). A pair whose video has no staged frame (MISSING.tsv lists the failed zips) or that
the source flags ``missing_video`` is skipped.

The ``ann_id`` ordinal is ``file_index * 1_000_000 + record_index`` (files sorted by name), stable
whatever ``limit`` is. ``qa_type`` is ``<dimension>/<subdimension>``; ``attrs.answer_type`` comes from
``question_format``; a multiple-choice question also gets ``attrs.options``.
The Q/A are model-generated (``has_hallucination`` / ``is_answer_correct`` are kept in ``attrs``).
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ..boxes_table import bucket_lister
from ..s3_keyed import StagedTree, fetch_small
from ..vqa_table import VqaResult, VqaSource, parse_options, vqa_row

_FRAME = re.compile(r"^(?P<grp>.+?)_videos_zip_videos_(?P<vid>[^_]+)_.+$")
_GROUP = re.compile(r"^(?P<grp>.+?)_multi_turn_data")
_FORMATS = {
    "MultipleChoiceQuestion": "multiple-choice",
    "Open-endQuestion": "open-ended",
    "ExactMatchQuestion": "exact-match",
}
_FLAGS = ("has_hallucination", "is_answer_correct")


def video_frames(tree: StagedTree) -> Counter:
    """``(group, video_id) -> staged frame count`` from the CHECKSUMS image stems."""
    out: Counter = Counter()
    for _partition, stem in tree.shas:
        if m := _FRAME.match(stem):
            out[(m["grp"], m["vid"])] += 1
    return out


def staged(
    spec: VqaSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> VqaResult:
    fetch = fetch or fetch_small
    tree = StagedTree(f"{spec.source_id}/{spec.version}", fetch)
    frames = video_frames(tree)
    prefix = f"sources/{spec.source_id}/{spec.version}/labels/files/"
    files = sorted(k for k in (lister or bucket_lister())(prefix) if k.endswith(".json"))
    quota = None if limit is None else -(-limit // max(1, len(files)))  # per-file share, ceil
    with ThreadPoolExecutor(8) as pool:
        docs = list(pool.map(lambda k: json.loads(fetch(k)).get("data", []), files))
    res = VqaResult()
    for file_idx, (key, records) in enumerate(zip(files, docs, strict=True)):
        name = key.rsplit("/", 1)[1]
        grp = (_GROUP.match(name) or {"grp": name.removesuffix(".json")})["grp"]
        kept = 0
        for idx, rec in enumerate(records):
            res.seen += 1
            question, answer = str(rec.get("question") or ""), str(rec.get("answer") or "")
            n_frames = frames.get((grp, str(rec.get("video_id"))), 0)
            if rec.get("missing_video") or not question.strip() or not answer.strip():
                res.skipped["no answer, no question or missing_video"] += 1
                continue
            if n_frames == 0:
                res.skipped["video has no staged frame"] += 1
                continue
            if quota is not None and kept >= quota:
                continue
            kept += 1
            fmt = _FORMATS.get(str(rec.get("question_format")), "open-ended")
            attrs: dict[str, object] = {
                "answer_type": fmt, "video_id": str(rec["video_id"]),
                "video_scene": rec.get("video_scene"), "frames_staged": n_frames,
                "dimension": rec.get("dimension"), "subdimension": rec.get("subdimension"),
                **{f: rec[f] for f in _FLAGS if f in rec},
            }  # fmt: skip
            if fmt == "multiple-choice" and (options := parse_options(question)):
                attrs["options"] = options
            row = vqa_row(
                spec=spec, ordinal=file_idx * 1_000_000 + idx, sha=None, question=question,
                answer=answer, qa_type=f"{rec.get('dimension')}/{rec.get('subdimension')}",
                split=grp.split("_", 1)[0], attrs=attrs,
            )  # fmt: skip
            res.pending.append({**{k: v for k, v in row.items() if k != "image_sha256"},
                                "image_key": f"{grp}/{rec['video_id']}"})  # fmt: skip
    return res
