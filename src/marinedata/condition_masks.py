"""v1.1.1 ``condition-masks`` patch (``hf_export --patch condition-masks``).

The 658 Tayrona bleaching rasters (``reef-support-bleaching``, pixel 0 unlabelled / 1 bleached /
2 non_bleached) were left out of every v1.1.0 config. This patch writes ONE new label-only config,
``data/condition-masks/train-reef-support-bleaching-NNNNN-of-MMMMM.parquet``, and touches nothing
else under ``data/``: no ``images`` rows (the bytes stay in their v1.0 holder, named by
``metadata.bytes_config``), no re-built label-map / sources / metadata. The split of every row comes
from the v1.1.0 SPLIT_MAP (no new allocation); ``--hidden-test`` drops hidden-test rasters, as it
does for every v1.1.0 table. The card, Croissant, CITATION and lock are v1.1.0's with the new
config added (:mod:`.condition_masks_docs`).

Columns: ``image_sha256, source_id, split, class_map, ignore_value, attrs, mask`` (palette PNG,
unlabelled -> 255 as in ``semseg``), ``mask_bytes_sha256, mask_config`` (always null).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

from .hf_export_v11 import (
    Layout,
    V11ExportError,
    _group,
    _hidden_test_drop,
    export_v11,
    hf_split,
)
from .hf_parquet import SHARD_TARGET_BYTES, ConfigSpec, EmbeddedBlob, ExportRow
from .task_layers.masks_table import CONDITION_MASK_SOURCES, staged_mask_rows
from .task_layers.s3_keyed import StagedTree

CONFIG = "condition-masks"
SOURCE_ID = "reef-support-bleaching"
COLUMNS = (
    ("image_sha256", "string"), ("source_id", "string"), ("split", "string"),
    ("class_map", "string"), ("ignore_value", "string"), ("attrs", "string"),
    ("mask", "image"), ("mask_bytes_sha256", "string"), ("mask_config", "string"),
)  # fmt: skip


def local_tree(root: str | Path, tree: str) -> StagedTree:
    """A :class:`StagedTree` over a local copy of ``sources/<tree>/`` (``root`` holds
    ``CHECKSUMS.sha256``, ``metadata.parquet`` and ``labels/``): the same reader, no network."""
    base, prefix = Path(root), f"sources/{tree.strip('/')}/"

    def fetch(key: str) -> bytes:
        return (base / key.removeprefix(prefix)).read_bytes()

    return StagedTree(tree, fetch)


def split_assignments(path: str | Path) -> dict[str, str]:
    """``{image_sha256: split}`` of ``reef-support-bleaching`` from a SPLIT_MAP v2 file."""
    assignments = json.loads(Path(path).read_text()).get("assignments", {})
    prefix = f"{SOURCE_ID}/sha:"
    return {k.removeprefix(prefix): v for k, v in assignments.items() if k.startswith(prefix)}


def build_condition_layout(
    registry,
    tree: StagedTree,
    splits: Mapping[str, str],
    *,
    hidden_test_drop: Collection[str] | None = None,
    shard_target_bytes: int = SHARD_TARGET_BYTES,
) -> tuple[Layout, dict]:
    """``({CONFIG: (spec, {train-<source>: rows})}, report)``. A raster whose image has no entry in
    ``splits`` raises (the patch never allocates a split); ``hidden_test_drop`` digests are
    withheld and counted."""
    from .task_layers.mask_encode import semantic_png

    spec_src = CONDITION_MASK_SOURCES[SOURCE_ID]
    drop = frozenset(s.lower() for s in (hidden_test_drop or ()))
    spec = ConfigSpec(CONFIG, COLUMNS, shard_target_bytes=shard_target_bytes)
    built: list[tuple[str, ExportRow]] = []
    report = {"rasters": 0, "hidden_test_excluded": 0, "rows": 0, "skipped_unstaged": 0}
    for row in staged_mask_rows(tree, spec_src, registry):
        report["rasters"] += 1
        sha = row["image_sha256"]
        if sha.lower() in drop:
            report["hidden_test_excluded"] += 1
            continue
        if sha not in splits:
            raise V11ExportError(
                f"{SOURCE_ID} {sha[:12]}: not in the SPLIT_MAP (no new allocation)"
            )
        rel = row["mask_ref"].removeprefix(f"sources/{tree.tree}/")
        png = semantic_png(
            tree.get(rel), encoding=spec_src.encoding, ignore_value=spec_src.ignore_value,
            palette=True,
        )  # fmt: skip
        split = hf_split(splits[sha])
        values = {
            "image_sha256": sha, "source_id": SOURCE_ID, "split": split,
            "class_map": row["class_map"], "ignore_value": str(row["ignore_value"]),
            "attrs": row["attrs"], "mask_bytes_sha256": hashlib.sha256(png).hexdigest(),
            "mask_config": None,
        }  # fmt: skip
        blob = EmbeddedBlob("mask", f"{SOURCE_ID}/{sha}.png", len(png), lambda data=png: data)
        built.append((split, ExportRow(values=values, blobs=(blob,))))
    report["rows"] = len(built)
    if tree.missing:
        report["skipped_unstaged"] = len(tree.missing)
    layout: Layout = {CONFIG: (spec, _group(built, True))} if built else {}
    return layout, report


def run_patch(args) -> int:
    """``hf_export --patch condition-masks`` entry. Writes the new config's shards plus the
    patched card, Croissant, CITATION and lock into ``--out``."""
    from .condition_masks_docs import write_docs
    from .registry import Registry

    if args.patch != CONFIG:
        raise V11ExportError(f"--patch {args.patch!r}: only {CONFIG!r} is defined")
    missing = [n for n in ("staged_root", "split_map") if not getattr(args, n, None)]
    if missing:
        raise V11ExportError("--patch needs --" + ", --".join(m.replace("_", "-") for m in missing))
    release = json.loads((Path(args.release_dir) / "RELEASE.json").read_text())
    flavour = args.flavour or release.get("flavour")
    if flavour != "open":
        raise V11ExportError("condition-masks are open-class rows: patch the open flavour only")
    spec_src = CONDITION_MASK_SOURCES[SOURCE_ID]
    layout, report = build_condition_layout(
        Registry.load(), local_tree(args.staged_root, spec_src.tree),
        split_assignments(args.split_map),
        hidden_test_drop=_hidden_test_drop(args, flavour),
    )  # fmt: skip
    if not layout:
        raise V11ExportError("no condition-masks rows: nothing to export")
    summary = export_v11(layout, args.out)
    docs = write_docs(args, summary, report)
    summary |= {"flavour": flavour, "patch": CONFIG, "v111": report, "docs": docs}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({"rows": report["rows"], "excluded": report["hidden_test_excluded"]}))
    print(f"{CONFIG}: {report}", file=sys.stderr)
    return 0
