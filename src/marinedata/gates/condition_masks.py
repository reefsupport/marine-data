"""Gates of the v1.1.1 ``condition-masks`` patch (``python -m marinedata.gates.condition_masks``).

* ``frozen``: every v1.1.0 file stays byte-identical. The baseline is the prior release's Hub tree
  (``list_repo_tree(expand=True)``, anonymous): LFS sha256 per file, no 20 GB download. Only
  README.md, croissant.json, CITATION.cff, release.lock.yaml and provenance/* may differ; nothing
  may land in a prior config folder.
* ``join``: every row's image resolves in the v1.0 ``coral-masks`` tree (its bytes holder).
* ``masks``: the new folder exists; each mask is a mode-P PNG with values in {1, 2, 255} and the
  size of the image in the staged ``metadata`` (width x height).
* ``decon``: no hidden-test digest is in the patch.
The leak / labelmap / bytes_once gates run unchanged through ``marinedata release gate``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

from . import GateResult
from .v11_common import Collector, file_sha256, tree_files

CONFIG = "condition-masks"
ALLOW = ("README.md", "croissant.json", "CITATION.cff", "release.lock.yaml")
ALLOW_PREFIX = ("provenance/",)
VALUES = frozenset({1, 2, 255})


def hub_tree(repo_id: str, revision: str | None = None) -> dict[str, str | None]:
    """``{path: lfs sha256}`` (``None`` for a non-LFS file) of a public dataset repo."""
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        repo_id, repo_type="dataset", revision=revision, recursive=True, expand=True
    )
    return {
        e.path: (e.lfs.sha256 if getattr(e, "lfs", None) else None)
        for e in tree
        if hasattr(e, "size")
    }


def _allowed(rel: str) -> bool:
    return rel in ALLOW or rel.startswith(ALLOW_PREFIX)


def run_frozen_gate(build_dir: Path, *, prior: Mapping[str, str | None]) -> GateResult:
    out, build = Collector(), Path(build_dir)
    configs = {p.split("/")[1] for p in prior if p.startswith("data/") and p.count("/") > 1}
    if not prior:
        out.fail("no_prior", "no prior-release tree: nothing to compare against")
    for path in tree_files(build):
        rel = path.relative_to(build).as_posix()
        out.counts["build_files"] += 1
        if _allowed(rel):
            continue
        if rel in prior:
            out.counts["overlap"] += 1
            if prior[rel] is None or prior[rel] != file_sha256(path):
                out.fail("changed", f"{rel} changed_prior_file: differs from the prior release")
        elif rel.startswith("data/") and rel.split("/")[1] in configs:
            out.fail("extended", f"{rel} extends_prior_config: new file in a frozen config")
        elif not rel.startswith(f"data/{CONFIG}/"):
            out.fail("unexpected", f"{rel} unexpected_file: a patch writes only data/{CONFIG}/")
    return out.result("frozen")


def run_masks_gate(build_dir: Path, *, sizes: Mapping[str, tuple[int, int]]) -> GateResult:
    import pyarrow.parquet as pq
    from PIL import Image

    out, files = Collector(), sorted((Path(build_dir) / "data" / CONFIG).glob("*.parquet"))
    if not files:
        out.fail("no_folder", f"data/{CONFIG}/ holds no shard")
    for path in files:
        for row in pq.read_table(path, columns=["image_sha256", "mask"]).to_pylist():
            out.counts["rows"] += 1
            sha, data = row["image_sha256"], (row["mask"] or {}).get("bytes")
            if sha not in sizes or not data:
                out.fail("unresolved", f"{str(sha)[:12]}: no mask bytes or no staged size")
                continue
            with Image.open(io.BytesIO(data)) as im:
                if im.mode != "P" or im.size != tuple(sizes[sha]):
                    out.fail("size", f"{sha[:12]}: mode {im.mode} {im.size} vs {tuple(sizes[sha])}")
                    continue
                if not {v for _, v in im.getcolors(1 << 16)} <= VALUES:
                    out.fail("values", f"{sha[:12]}: mask values outside {sorted(VALUES)}")
    return out.result("masks")


def run_join_gate(build_dir: Path, *, v1_shas: Collection[str]) -> GateResult:
    from .v11_common import column_values

    out, files = Collector(), sorted((Path(build_dir) / "data" / CONFIG).glob("*.parquet"))
    for sha in column_values(files, "image_sha256"):
        out.counts["rows"] += 1
        if sha not in v1_shas:
            out.fail("unresolved", f"{str(sha)[:12]}: not in the v1.0 coral-masks tree")
    return out.result("join")


def run_decon_gate(build_dir: Path, *, hidden: Collection[str]) -> GateResult:
    from .v11_common import column_values

    out, wanted = Collector(), {h.lower() for h in hidden}
    files = sorted((Path(build_dir) / "data" / CONFIG).glob("*.parquet"))
    out.counts["hidden_requested"] = len(wanted)
    for sha in column_values(files, "image_sha256"):
        if str(sha).lower() in wanted:
            out.fail("hidden_test", f"{str(sha)[:12]}: a hidden-test image is in the patch")
    return out.result("decon")


def staged_sizes(tree) -> dict[str, tuple[int, int]]:
    """``{image_sha256: (width, height)}`` from the staged ``metadata.parquet``."""
    meta = {(r.get("partition") or "default", r["stem"]): r for r in tree.table("metadata.parquet")}
    return {tree.shas[k]: (r["width"], r["height"]) for k, r in meta.items() if k in tree.shas}


def v1_coral_masks_shas(v1_dir: Path) -> frozenset[str]:
    import pyarrow.parquet as pq

    found: set[str] = set()
    for shard in sorted((Path(v1_dir) / "data" / "coral-masks").glob("*.parquet")):
        found.update(pq.read_table(shard, columns=["image_sha256"]).column(0).to_pylist())
    return frozenset(s for s in found if s)


def main(argv: list[str] | None = None) -> int:
    from ..condition_masks import local_tree
    from ..hf_export_v11 import hidden_test_drop
    from ..task_layers.masks_table import CONDITION_MASK_SOURCES

    p = argparse.ArgumentParser(prog="python -m marinedata.gates.condition_masks")
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--staged-root", type=Path, required=True)
    p.add_argument("--v1-dir", type=Path, required=True)
    p.add_argument("--hidden-test", type=Path, required=True)
    p.add_argument("--prior-repo", default="reefsupport/marine-data")
    p.add_argument("--prior-revision", default="v1.1.0")
    p.add_argument("--prior-tree-json", type=Path, default=None, help="offline {path: sha256}")
    a = p.parse_args(argv)
    prior = (
        json.loads(a.prior_tree_json.read_text())
        if a.prior_tree_json
        else hub_tree(a.prior_repo, a.prior_revision)
    )
    tree = local_tree(a.staged_root, CONDITION_MASK_SOURCES["reef-support-bleaching"].tree)
    results = [
        run_frozen_gate(a.build, prior=prior),
        run_join_gate(a.build, v1_shas=v1_coral_masks_shas(a.v1_dir)),
        run_masks_gate(a.build, sizes=staged_sizes(tree)),
        run_decon_gate(a.build, hidden=hidden_test_drop(a.hidden_test, a.v1_dir)),
    ]
    for r in results:
        print(r.name, "PASS" if r.passed else "FAIL", *r.failures[:5], file=sys.stdout)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
