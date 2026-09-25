"""CoralVQA: anonymous HF jsonl -> ``data/_tasklabels/coralvqa/vqa.parquet`` (WP-8d, D-Z2).

``CoralVQA_train.jsonl``/``CoralVQA_test.jsonl`` (HF ``CoralReefData/CoralVQA``, direct
files, no LFS, no token) hold one conversation turn pair per Q/A: a human turn tagged
``[vqa]`` and a gpt-turn answer, keyed by an ``image`` filename. The real images live only
in a single 26.7GB ``CoralVQA_Image.zip`` that is not staged by any source in this
registry (``coralvqa``'s own registry entry has no ``images_from``).

``sha256`` is the D-Z2 join key, so it is computed from the staged image bytes
(``--images``, one digest per distinct filename) and a Q/A row whose image is not staged
is DROPPED and counted, never written with a ``null`` key. INT-core3b: the WP-8d version
hard-coded ``"sha256": None`` on every row, so the tracked ``vqa.parquet`` (254,867 rows)
had exactly one distinct key — null — and could not join to any image; it is marked
``invalid`` in ``data/_tasklabels/MANIFEST.json`` until the 26.7GB zip is staged and this
producer is re-run. ``main`` refuses a partial write unless ``--allow-partial``.

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

from marinedata.checksums import file_digest

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


def build_rows(
    lines: list[str],
    split_hint: str,
    images_dir: Path,
    *,
    unkeyed: dict[str, int] | None = None,
    sha_cache: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """One row per Q/A pair whose image is staged under ``images_dir``, keyed by that
    image's sha256. Rows whose image is not staged are dropped and tallied in
    ``unkeyed`` (``{image_filename: dropped_rows}``) — never emitted with a null key."""
    cache: dict[str, str | None] = {} if sha_cache is None else sha_cache
    rows = []
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_line(line)
        if parsed is None:
            continue
        image = parsed["image"]
        if image and image not in cache:
            path = images_dir / image
            cache[image] = file_digest(path) if path.is_file() else None
        sha256 = cache.get(image) if image else None
        if sha256 is None:
            if unkeyed is not None:
                unkeyed[str(image)] = unkeyed.get(str(image), 0) + 1
            continue
        rows.append(
            {
                "sha256": sha256,
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
    p.add_argument(
        "--images", type=Path, required=True, help="dir holding CoralVQA's staged <image> files"
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="write even if some Q/A rows' images are not staged (they are dropped)",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows: list[dict[str, Any]] = []
    unkeyed: dict[str, int] = {}
    cache: dict[str, str | None] = {}
    for path, split_hint in ((args.train, "train"), (args.test, "test")):
        if path and path.exists():
            lines = path.read_text().splitlines()
            rows += build_rows(lines, split_hint, args.images, unkeyed=unkeyed, sha_cache=cache)
    summary = {
        "rows": len(rows),
        "images": len({r["sha256"] for r in rows}),
        "unkeyed_rows": sum(unkeyed.values()),
        "unkeyed_images": len(unkeyed),
    }
    if unkeyed and not args.allow_partial:
        print(json.dumps({**summary, "written": False}))
        return 2
    write_vqa_parquet(rows, args.out)
    print(json.dumps({**summary, "written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
