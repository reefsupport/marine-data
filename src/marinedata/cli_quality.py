"""``marinedata quality`` sub-command — score every image for rubric D9.

Split out of :mod:`marinedata.cli` for the same reason as :mod:`marinedata.cli_ingest`.
Orchestration (discovery, multiprocessing, resumability, output) lives here; the pure
per-image math lives in :mod:`marinedata.quality` so it can be unit-tested and reused
without any of the I/O or process-pool machinery below.

**Input.** ``<input>`` is a directory: either a flat/staged tree of image files (a
``sources/<id>/<version>`` root, or any directory containing image files at any depth),
or a HuggingFace ``images`` config directory of Parquet files with an ``image_sha256``
column and an ``image`` struct column (``{bytes, path}``) — the format
:mod:`marinedata.hf_parquet` writes. Detection reads only the first Parquet file's
schema, cheaply.

**Determinism.** Work units are discovered in a fixed sorted order, scored with
``Pool.map`` (order-preserving regardless of worker count), then the combined
(resumed + new) rows are sorted by ``image_sha256`` before writing. The output Parquet
uses fixed writer options and carries no timestamp, so two runs over the same input
produce byte-identical files.

**Resumability.** If ``--out`` already exists, rows whose ``image_sha256`` is already
present are not re-decoded; new rows are appended by rescoring only what's missing, and
``flags`` are recomputed for every row (cheap) against the combined ``q_blur`` percentile.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .checksums import file_digest
from .quality import (
    QUALITY_THRESHOLDS,
    SCORE_SHORT_SIDE,
    QualityScores,
    blur_percentile_threshold,
    compute_flags,
    library_versions,
    score_bytes,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("image_sha256", pa.string()),
        pa.field("decode_ok", pa.bool_()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("min_side", pa.int32()),
        pa.field("q_blur", pa.float64()),
        pa.field("q_clip_lo", pa.float64()),
        pa.field("q_clip_hi", pa.float64()),
        pa.field("q_uiqm", pa.float64()),
        pa.field("q_entropy", pa.float64()),
        pa.field("q_blank", pa.bool_()),
        pa.field("flags", pa.list_(pa.string())),
    ]
)


@dataclass(frozen=True)
class WorkUnit:
    kind: str  # "file" | "parquet"
    path: Path


def _looks_like_hf_images_dir(first_parquet: Path) -> bool:
    schema = pq.ParquetFile(first_parquet).schema_arrow
    names = set(schema.names)
    return "image_sha256" in names and "image" in names


def discover_inputs(root: Path) -> list[WorkUnit]:
    """Fixed, sorted discovery order — the basis for deterministic output."""
    root = Path(root)
    parquet_files = sorted(root.glob("*.parquet"))
    if parquet_files and _looks_like_hf_images_dir(parquet_files[0]):
        return [WorkUnit("parquet", p) for p in parquet_files]
    image_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [WorkUnit("file", p) for p in image_files]


# Populated once per worker process via the Pool initializer — avoids re-pickling a
# possibly large "already done" set on every task.
_DONE_SHAS: frozenset[str] = frozenset()


def _init_worker(done_shas: frozenset[str]) -> None:
    global _DONE_SHAS
    _DONE_SHAS = done_shas


def _row_from_scores(sha256: str, scores: QualityScores) -> dict:
    row = scores.to_dict()
    row["image_sha256"] = sha256
    return row


def _score_unit(unit: WorkUnit) -> list[dict]:
    if unit.kind == "file":
        sha256 = file_digest(unit.path)
        if sha256 in _DONE_SHAS:
            return []
        raw = unit.path.read_bytes()
        return [_row_from_scores(sha256, score_bytes(raw))]

    table = pq.read_table(unit.path, columns=["image_sha256", "image"])
    shas = table.column("image_sha256").to_pylist()
    images = table.column("image").to_pylist()
    rows = []
    for sha256, image_struct in zip(shas, images, strict=True):
        if sha256 in _DONE_SHAS:
            continue
        raw = image_struct["bytes"]
        rows.append(_row_from_scores(sha256, score_bytes(raw)))
    return rows


def _load_existing(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    table = pq.read_table(out_path)
    return table.to_pylist()


def _finalize_rows(rows: list[dict]) -> list[dict]:
    """Recompute ``flags`` for every row against the combined ``q_blur`` percentile,
    then sort by ``image_sha256`` for a deterministic write order."""
    blur_values = [
        r["q_blur"]
        for r in rows
        if r["decode_ok"] and not r["q_blank"] and r["q_blur"] is not None
    ]
    blur_p1 = blur_percentile_threshold(blur_values)
    finalized = []
    for row in rows:
        flags = compute_flags(row, blur_p1=blur_p1)
        finalized.append({**row, "flags": flags})
    finalized.sort(key=lambda r: r["image_sha256"])
    return finalized, blur_p1


def _write_output(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    pq.write_table(table, out_path, compression="zstd", use_dictionary=False, write_statistics=True)


def _write_meta(rows: list[dict], out_path: Path, blur_p1: float | None, workers: int) -> Path:
    reason_counts: dict[str, int] = {}
    for row in rows:
        for flag in row["flags"]:
            reason_counts[flag] = reason_counts.get(flag, 0) + 1
    meta = {
        "n_rows": len(rows),
        "n_flagged": sum(1 for r in rows if r["flags"]),
        "flag_counts": dict(sorted(reason_counts.items())),
        "thresholds": QUALITY_THRESHOLDS,
        "blur_p1": blur_p1,
        "score_short_side": SCORE_SHORT_SIDE,
        "library_versions": library_versions(),
        "workers": workers,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta_path


def run_quality(input_dir: Path, out_path: Path, *, workers: int) -> dict:
    units = discover_inputs(input_dir)
    existing_rows = _load_existing(out_path)
    done_shas = frozenset(r["image_sha256"] for r in existing_rows)

    new_rows: list[dict] = []
    if units:
        if workers <= 1:
            _init_worker(done_shas)
            for unit in units:
                new_rows.extend(_score_unit(unit))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(done_shas,)) as pool:
                for unit_rows in pool.map(_score_unit, units):
                    new_rows.extend(unit_rows)

    all_rows = existing_rows + new_rows
    # Existing rows already carry a "flags" column; drop it so `_finalize_rows` is the
    # single place that computes it — avoids two rows for the same sha ever disagreeing.
    all_rows = [{k: v for k, v in r.items() if k != "flags"} | {"flags": []} for r in all_rows]
    finalized, blur_p1 = _finalize_rows(all_rows)
    _write_output(finalized, out_path)
    meta_path = _write_meta(finalized, out_path, blur_p1, workers)
    return {"rows": len(finalized), "new": len(new_rows), "meta_path": str(meta_path)}


def _cmd_quality(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    out_path = Path(args.out)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    result = run_quality(input_dir, out_path, workers=workers)
    print(f"scored {result['rows']} rows ({result['new']} new) -> {out_path}")
    print(f"meta -> {result['meta_path']}")
    return 0


def add_quality_subparser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("quality", help="score per-sample image quality (D9)")
    parser.add_argument(
        "input", help="directory of images, a staged tree, or an HF images parquet dir"
    )
    parser.add_argument("--out", required=True, help="output quality.parquet path")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="process count (default: os.cpu_count()); 1 disables multiprocessing",
    )
    parser.set_defaults(func=_cmd_quality)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cmd_quality(argparse.Namespace(input=sys.argv[1], out=sys.argv[2], workers=None)))
