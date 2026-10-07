#!/usr/bin/env python
# ruff: noqa: E501
"""Regenerate the registry-derived tables of ``docs/STORAGE.md`` and ``docs/task-layers.md``.

Only the text between ``<!-- BEGIN GENERATED: <name> -->`` and ``<!-- END GENERATED: <name> -->``
is rewritten (inserted on first run); everything else is hand-written prose. ``--check`` exits 1
when a doc is stale instead of writing. Run from the repo root:  ``python scripts/gen_storage_docs.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marinedata.models import Source
from marinedata.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
TASK_LAYER_IDS = (
    "seaview-survey-imagery",
    "reef-support-seaview-labels",
    "reef-support-benthic-own",
    "reefolution",
    "ibf",
    "coralvqa",
    "coralscapes",
    "coralseg-ucsd-mosaics",
    "coralscop-masks-rs",
    "rs-labelled-masks",
    "mermaid-aws",
)


def _n(v: int | None) -> str:
    return "—" if v is None else f"{v:,}"


def _ac(s: Source) -> str:
    return str(getattr(s.access_class, "value", s.access_class))


def _where(s: Source) -> str:
    if s.retired:
        return "retired (no read path)"
    p = s.access.params or {}
    if s.access.method.value == "s3" and p.get("bucket") and p.get("prefix"):
        return f"`s3://{p['bucket']}/{p['prefix']}`"
    return f"{s.access.method.value}: {s.access.uri or '—'}"


def _row(s: Source, task_layer: bool) -> str:
    layout = f" | `{s.loader.layout}`" if task_layer and s.loader else (" |" if task_layer else "")
    pin = "pinned" if s.checksums else "unpinned"
    return (
        f"| `{s.id}`{layout} | {_where(s)} | {s.version} | {_ac(s)} | "
        f"{_n(s.n_images)} | {_n(s.n_annotations)} | {_n(s.n_files)} | {pin} |"
    )


def storage_table(reg: Registry) -> str:
    head = (
        "| id | canonical location | version | access class | n_images | n_annotations | n_files | checksums |\n"
        "|---|---|---|---|---:|---:|---:|---|"
    )
    rows = [
        _row(s, False)
        for s in sorted(reg.sources, key=lambda s: s.id)
        if s.retired or (s.access.method.value == "s3" and (s.access.params or {}).get("prefix"))
    ]
    return (
        "### Canonical storage map (generated from `registry/sources/*.yaml`)\n\n"
        "One location per id: `sources/<id>/<version>/` (images under `images/<partition>/`, labels under\n"
        "`labels/{masks,instance_masks,exports,points}/`, then `metadata.parquet`, `CHECKSUMS.sha256`,\n"
        "`INGEST.json`). Non-open sources live in `rs-storage-private`. `—` means unknown, never zero.\n"
        "Regenerate with `python scripts/gen_storage_docs.py`; check offline with\n"
        "`marinedata registry verify --listing <bucket listing>`.\n\n"
        + head
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


def task_layer_table(reg: Registry) -> str:
    head = (
        "| id | `loader.layout` | canonical location | version | access class | n_images | n_annotations | n_files | checksums |\n"
        "|---|---|---|---|---|---:|---:|---:|---|"
    )
    rows = [_row(reg.source(i), True) for i in TASK_LAYER_IDS if i in {s.id for s in reg.sources}]
    return (
        "### Canonical locations (generated from the registry)\n\n"
        "Counts and pointers below come from `registry/sources/*.yaml` via `scripts/gen_storage_docs.py`;\n"
        "the hand-written findings that follow are WP-8b history and may cite the pre-reorg layout.\n\n"
        + head.replace("| id | `loader.layout` |", "| id | `loader.layout` |", 1)
        + "\n"
        + "\n".join(rows)
        + "\n"
    )


def splice(text: str, name: str, body: str, anchor: str | None) -> str:
    begin, end = f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    if begin in text:
        a, b = text.index(begin), text.index(end) + len(end)
        return text[:a] + block + text[b:]
    if anchor and anchor in text:
        i = text.index(anchor) + len(anchor)
        return text[:i] + "\n\n" + block + "\n" + text[i:]
    return text.rstrip("\n") + "\n\n" + block + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if a doc is stale; write nothing")
    args = ap.parse_args(argv)
    reg = Registry.load()
    targets = (
        (ROOT / "docs/STORAGE.md", "storage-map", storage_table(reg), None),
        (
            ROOT / "docs/task-layers.md",
            "task-layer-sources",
            task_layer_table(reg),
            "## Inventory (per source)",
        ),
    )
    stale = []
    for path, name, body, anchor in targets:
        old = path.read_text()
        new = splice(old, name, body, anchor)
        if new != old:
            stale.append(path.name)
            if not args.check:
                path.write_text(new)
    print(
        ("stale: " if args.check else "updated: ") + ", ".join(stale)
        if stale
        else "docs up to date"
    )
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
