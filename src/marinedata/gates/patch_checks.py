"""Checks of a patch release that stand in for the full-release gates (``release gate --patch-of``).

A patch adds configs to a published release and leaves it alone. These helpers verify only what the
patch can change and say, in the verdict's counts, what they inherit from the frozen prior release:

* :func:`frozen_card_failures` the prior release's ``configs:`` / ``dataset_info`` entries and
  ``size_categories`` stay as the Hub card has them;
* :func:`inherit_image_gate` ``dedup`` / ``privacy`` look at pixels: a patch that ships no image
  bytes inherits them from the release that published the images (the stock gate still runs the
  moment the patch carries an image column);
* :func:`run_patch_splits_gate` every patch row sits in the split the split map gives its image and
  keeps the split the frozen v1.0 tree gave it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import GateResult, finish
from .patch_base import PatchBase
from .shards import config_shards, norm_split, shard_columns
from .v11_common import MAIN_SPLITS, Collector, canonical, front_matter, shards, table_rows

IMAGE_COLUMN = "image"
SPLITS = frozenset({*MAIN_SPLITS, "test_ood"})


def _entries(readme: Path, key: str) -> dict[str, dict]:
    raw = front_matter(readme).get(key) or []
    raw = [raw] if isinstance(raw, dict) else raw
    return {str(e["config_name"]): e for e in raw if isinstance(e, dict) and "config_name" in e}


def frozen_card_failures(readme: Path, prior: PatchBase) -> list[tuple[str, str, str]]:
    """``[(kind, where, message)]``: a prior config whose card entry changed or vanished, a prior
    ``dataset_info`` entry that changed, and ``size_categories`` unlike the prior card's."""
    out: list[tuple[str, str, str]] = []
    for key, base in (("configs", prior.card_entries), ("dataset_info", prior.dataset_info)):
        now = _entries(readme, key)
        for name, entry in sorted(base.items()):
            if name not in now:
                out.append(("frozen_card_entry", name, f"prior {key} entry removed from the card"))
            elif canonical(now[name]) != canonical(entry):
                out.append(("frozen_card_entry", name, f"prior {key} entry changed in the card"))
    bands = front_matter(readme).get("size_categories") or []
    bands = tuple([bands] if isinstance(bands, str) else map(str, bands))
    if bands != prior.size_categories:
        message = f"size_categories {list(bands)} differ from the prior card's "
        out.append(("size_category", "README.md", message + str(list(prior.size_categories))))
    return out


def has_image_bytes(build_dir: Path) -> bool:
    """True when the build carries pixels: an ``images`` config or an ``image`` column anywhere."""
    found = config_shards(Path(build_dir))
    return "images" in found or any(IMAGE_COLUMN in shard_columns(f[0]) for f in found.values())


def inherit_image_gate(
    name: str, build_dir: Path, base: PatchBase, stock: Callable[[], GateResult]
) -> GateResult:
    """``stock()`` when the patch carries image bytes, else a pass that records the inheritance."""
    if has_image_bytes(build_dir):
        return stock()
    counts = {"inherited_from": base.release, "patch_image_bytes": 0}
    return finish(name, {**counts, "patch_configs": len(config_shards(build_dir))}, [])


def split_map_assignments(path: Path) -> dict[str, str]:
    """``{image_sha256: split}`` of every ``<source>/sha:<digest>`` key of a SPLIT_MAP v2 file."""
    import json

    raw = json.loads(Path(path).read_text()).get("assignments", {})
    return {k.split("/sha:", 1)[1]: norm_split(v) for k, v in raw.items() if "/sha:" in k}


def run_patch_splits_gate(
    build_dir: Path,
    *,
    split_map: Path | None,
    v1_dir: Path | None,
    configs: tuple[str, ...] = ("condition-masks",),
) -> GateResult:
    """Splits of a patch (``splits`` and ``splits_v11`` of the verdict): per patch row, the split
    is a known one, equals its shard's, equals the split map's assignment, and is a split the
    frozen v1.0 tree gave the image."""
    from .v11_common import v1_splits

    out, build = Collector(), Path(build_dir)
    assigned = split_map_assignments(split_map) if split_map and Path(split_map).is_file() else None
    if assigned is None:
        out.fail("no_split_map", "SPLIT_MAP.json not found: patch splits cannot be checked")
    v1 = v1_splits(Path(v1_dir)) if v1_dir else None
    if v1 is None:
        out.fail("no_v1_set", "no v1.0 tree (--v1-dir): a v1.0 reassignment cannot be ruled out")
    for config in configs:
        files = shards(build, config)
        if not files:
            out.fail("no_config", f"{config}: no shard in the build")
        for row in table_rows(files, ("image_sha256", "split")):
            out.counts["rows"] += 1
            sha, split, where = str(row.get("image_sha256")), row.get("split"), row["_shard"]
            out.counts[f"rows_{split}"] += 1
            if split not in SPLITS:
                out.fail("unsplit", f"{config}/{where} {sha[:12]} has split {split!r}")
                continue
            if split != row["_split"]:
                out.fail("split_shard", f"{config}/{where} {sha[:12]} has split {split}")
            if assigned is not None and assigned.get(sha) != split:
                out.fail(
                    "map_mismatch", f"{sha[:12]}: split {split}, split map {assigned.get(sha)}"
                )
            if v1 is not None and sha in v1 and split not in v1[sha]:
                out.fail("reassigned_v1", f"{sha[:12]}: v1.0 {sorted(v1[sha])} -> {split}")
    return out.result("splits")
