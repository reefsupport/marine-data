"""Hackathon fish-boxes driver (HK-4c): staged local mirror -> ``boxes`` parquet per source.

    python scripts/hk_fish_boxes.py boxes --stage STAGE --data-dir DIR uiis uiis10k usis10k ...

Reads ``STAGE/<source_id>/`` (a mirror of ``sources/<id>/<version>/``) instead of the bucket (the
public fetch caps files at 64 MB; the COCO documents are 100 MB+), writes
``DIR/_annotations/boxes/<id>/<version>.parquet`` and an empty ``DIR/_tasklabels/`` so that
``marinedata release build --tasklabels-root DIR`` builds the ``boxes`` config. ``package`` (see
:func:`package`) then lays the hf export out as ``shards/`` + ``metadata/`` + ``annotations/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marinedata.registry import Registry, _default_root
from marinedata.task_layers.boxes_table import BOX_SOURCES, staged_boxes, write_boxes
from marinedata.task_layers.s3_keyed import StagedTree


def local_fetch(stage: Path, source_id: str, version: str):
    prefix = f"sources/{source_id}/{version}/"

    def fetch(key: str) -> bytes:
        return (stage / source_id / key.removeprefix(prefix)).read_bytes()

    return fetch


def write_source_boxes(stage: Path, data_dir: Path, source_id: str, registry) -> dict:
    spec = BOX_SOURCES[source_id]
    tree = StagedTree(spec.tree, local_fetch(stage, source_id, spec.version))
    res = staged_boxes(spec, registry, tree=tree)
    write_boxes(data_dir, source_id, spec.version, list(res.rows))
    classes = {r["label_native"] for r in res.rows}
    return {
        "source": source_id, "images": res.images, "boxes": len(res.rows),
        "classes": len(classes), "orphan": res.counts.orphan, "unparsable": list(res.unparsable),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["boxes"])
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args(argv)
    (a.data_dir / "_tasklabels").mkdir(parents=True, exist_ok=True)
    reg = Registry.load(_default_root())
    stats = [write_source_boxes(a.stage, a.data_dir, s, reg) for s in a.sources]
    (a.data_dir / "boxes_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
