"""``marinedata export-wds`` — WebDataset tars beside the Parquet Hub layout (WP-3).

Reads the ``images`` config of a built ``_hf/v1``-shaped directory (:mod:`marinedata.hf_export`
layout: ``data/<config>/<split>-NNNNN-of-MMMMM.parquet``) one input shard at a time and
writes one output tar per input shard: members ``<image_sha256>.jpg|png`` (raw embedded
bytes, extension taken from the ``image.path`` field) plus ``<image_sha256>.json`` — the
image's own scalar columns plus a ``labels`` object joining every label-only config found
under ``<hf_dir>/data`` (``benthic-coarse``, ``benthic-l2``, ``bleaching-condition``,
``coral-health-binary``, ``general-pretraining``) on ``image_sha256``. ``masks`` and
``coralscop-pseudo-masks`` embed their own image column and are not joined here — a
second pass, out of WP-3's scope.

``--limit-shards N`` caps how many input images-parquet files are processed (train shards
first in file order, then validation, then test) — the disk rule for this task is
"``--limit-shards 1`` only" on real data; the unbounded full export runs server-side later.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

IMAGES_CONFIG = "images"
NON_LABEL_CONFIGS = {IMAGES_CONFIG, "masks", "coralscop-pseudo-masks"}
_SHARD_RE = re.compile(r"^(?P<split>[a-z]+)-(?P<index>\d+)-of-(?P<total>\d+)\.parquet$")


@dataclass(frozen=True)
class ShardResult:
    tar_path: Path
    n_images: int


def _shard_sort_key(path: Path) -> tuple[int, str, int]:
    m = _SHARD_RE.match(path.name)
    if not m:
        return (9, path.name, 0)
    order = {"train": 0, "validation": 1, "test": 2}.get(m["split"], 3)
    return (order, m["split"], int(m["index"]))


def _image_shards(hf_dir: Path) -> list[Path]:
    images_dir = hf_dir / "data" / IMAGES_CONFIG
    return sorted(images_dir.glob("*.parquet"), key=_shard_sort_key)


def _label_configs(hf_dir: Path) -> list[str]:
    data_dir = hf_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and p.name not in NON_LABEL_CONFIGS
    )


def build_label_index(hf_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{image_sha256: {config_name: {column: value, ...}}}`` for every label-only config."""
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for config in _label_configs(hf_dir):
        for shard in sorted((hf_dir / "data" / config).glob("*.parquet")):
            table = pq.read_table(shard)
            columns = [c for c in table.column_names if c != "image_sha256"]
            for row in table.to_pylist():
                sha = row["image_sha256"]
                index.setdefault(sha, {})[config] = {c: row[c] for c in columns}
    return index


def _ext_for(path_field: str | None) -> str:
    if path_field:
        suffix = Path(path_field).suffix
        if suffix:
            return suffix
    return ".jpg"


def _iter_image_rows(shard: Path) -> Iterator[dict[str, Any]]:
    table = pq.read_table(shard)
    yield from table.to_pylist()


def write_shard_tar(
    shard: Path, out_path: Path, label_index: dict[str, dict[str, dict[str, Any]]]
) -> ShardResult:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(out_path, "w") as tar:
        for row in sorted(_iter_image_rows(shard), key=lambda r: r["image_sha256"]):
            key = row["image_sha256"]
            image = row["image"] or {}
            ext = _ext_for(image.get("path"))
            image_bytes = image.get("bytes") or b""
            _add_bytes(tar, f"{key}{ext}", image_bytes)
            meta = {
                "image_sha256": key,
                "source_id": row.get("source_id"),
                "source_ids": row.get("source_ids"),
                "split_group": row.get("split_group"),
                "labels": label_index.get(key, {}),
            }
            _add_bytes(tar, f"{key}.json", json.dumps(meta, sort_keys=True).encode("utf-8"))
            n += 1
    return ShardResult(out_path, n)


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    import io

    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def export_wds(
    hf_dir: Path, out_dir: Path, *, limit_shards: int | None = None
) -> list[ShardResult]:
    shards = _image_shards(hf_dir)
    if limit_shards is not None:
        shards = shards[:limit_shards]
    label_index = build_label_index(hf_dir) if shards else {}
    results = []
    for shard in shards:
        tar_name = shard.stem + ".tar"
        results.append(write_shard_tar(shard, out_dir / tar_name, label_index))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m marinedata.wds_export")
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit-shards", type=int, default=None)
    args = parser.parse_args(argv)
    results = export_wds(args.hf_dir, args.out, limit_shards=args.limit_shards)
    for r in results:
        print(f"{r.tar_path}: {r.n_images} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
