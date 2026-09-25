"""CoralVQA: anonymous HF jsonl -> ``data/_tasklabels/coralvqa/vqa.parquet`` (WP-8d, D-Z2).

``CoralVQA_train.jsonl``/``CoralVQA_test.jsonl`` (HF ``CoralReefData/CoralVQA``, direct
files, no LFS, no token) hold one conversation turn pair per Q/A: a human turn tagged
``[vqa]`` and a gpt-turn answer, keyed by an ``image`` filename. The real images live only
in a single 26.7GB ``CoralVQA_Image.zip`` that is not staged by any source in this
registry (``coralvqa``'s own registry entry has no ``images_from``), so a row's ``sha256``
is left ``null`` here rather than fabricated — fetching the full zip to hash it is out of
this brief's scope (and would breach the disk floor). ``docs/task-labels-producers.md``
records this as the resume path once CoralVQA's images are separately staged.

HF anonymous downloads are rate-limited per IP; :func:`fetch_with_retries` retries at most
3 times, 5 minutes apart, and never logs in or passes a token (Do-Not list).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_ID = "coralvqa"
LABEL_ORIGIN = "model"  # CoralVQA's Q/A pairs are model-generated per its paper (arXiv 2507.10449).
_QTYPE = re.compile(r"\[(\w+)\]")
HF_BASE = "https://huggingface.co/datasets/CoralReefData/CoralVQA/resolve/main"


def parse_line(line: str) -> dict[str, Any] | None:
    """One CoralVQA jsonl row -> ``{image, question, answer, qtype}``.

    Train and test ship two different real shapes (verified 2026-09-25): train is a
    ``conversations`` human/gpt turn pair tagged ``[vqa]``; test (a held-out eval prompt
    set, no ground truth) is a flat ``{question_id, image, text, category}`` record with
    no answer at all — ``answer`` is ``None`` for those rows, not fabricated.
    """
    record = json.loads(line)
    turns = record.get("conversations")
    if turns:
        if len(turns) < 2:
            return None
        human, gpt = turns[0], turns[1]
        if human.get("from") != "human" or gpt.get("from") != "gpt":
            return None
        value = str(human.get("value", ""))
        m = _QTYPE.search(value)
        qtype = m.group(1) if m else "unknown"
        question = _QTYPE.sub("", value).replace("<image>", "").strip()
        return {
            "image": record.get("image"),
            "question": question,
            "answer": str(gpt.get("value", "")),
            "qtype": qtype,
        }
    if "text" in record and "image" in record:
        return {
            "image": record.get("image"),
            "question": str(record.get("text", "")).strip(),
            "answer": None,
            "qtype": str(record.get("category") or "unknown"),
        }
    return None


def build_rows(lines: list[str], split_hint: str) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_line(line)
        if parsed is None:
            continue
        rows.append(
            {
                "sha256": None,  # CoralVQA's own images are unstaged (26.7GB zip, out of scope)
                "source_id": SOURCE_ID,
                "label_origin": LABEL_ORIGIN,
                "image_filename": parsed["image"],
                "question": parsed["question"],
                "answer": parsed["answer"],
                "qtype": parsed["qtype"],
                "split_hint": split_hint,
            }
        )
    return rows


def write_vqa_parquet(rows: list[dict[str, Any]], out_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path, compression="zstd")
    return len(rows)


def fetch_with_retries(
    filename: str, dest: Path, *, attempts: int = 3, wait_s: float = 300.0
) -> bool:
    """Anonymous GET, no token/login. Returns True once ``dest`` holds real bytes."""
    url = f"{HF_BASE}/{filename}"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            if b"rate limit" in data[:4096].lower() or len(data) < 1000:
                raise ValueError("rate-limited or truncated response")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        except Exception as exc:
            print(f"attempt {attempt}/{attempts} for {filename} failed: {exc}")
            if attempt < attempts:
                time.sleep(wait_s)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, help="already-fetched CoralVQA_train.jsonl")
    p.add_argument("--test", type=Path, help="already-fetched CoralVQA_test.jsonl")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows: list[dict[str, Any]] = []
    if args.train and args.train.exists():
        rows += build_rows(args.train.read_text().splitlines(), "train")
    if args.test and args.test.exists():
        rows += build_rows(args.test.read_text().splitlines(), "test")
    n = write_vqa_parquet(rows, args.out)
    print(json.dumps({"rows": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
