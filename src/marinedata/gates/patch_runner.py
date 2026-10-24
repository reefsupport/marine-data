"""Run the release gates of a patch release (``marinedata release gate --patch-of v1.1.0``).

A patch (v1.1.1 ``condition-masks``) adds configs to a published release. Every gate that
``release publish`` requires (``REQUIRED``, ``FAIL_CLOSED`` and ``V11_REQUIRED``) runs here under
its own name, against the patch and the frozen prior release (:class:`.patch_base.PatchBase`):

* ``patch_base``     the prior tree and card were read (what the other gates treat as frozen);
* ``licence``, ``croissant``, ``leak``  stock;
* ``dedup``, ``privacy``  stock when the patch carries image bytes, else inherited from the prior
  release, which published the pixels (the verdict's counts say so);
* ``splits`` and ``splits_v11``  split map + frozen v1.0 assignment of every patch row (one
  result under both names);
* ``masks``          stock G5 (``condition-masks`` recognised) + raster size vs staged metadata;
* ``load``, ``card``  stock, with the prior release's configs frozen (``prior=``);
* ``frozen``         every prior-release file byte-identical (Hub LFS sha256), nothing added to a
  prior config;
* ``join``           every row resolves in the v1.0 ``coral-masks`` tree;
* ``decon``          no hidden-test digest in the patch;
* ``bytes_once``     stock (``--v1-dir``);
* ``labelmap``       the ``condition-masks`` class resolution resolves in the pinned taxonomy.

A full (non-patch) release never reaches this module. A gate that raises is a failed gate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from . import GateResult, finish
from .patch_base import PatchBase

PATCH_ALL = (
    "patch_base", "licence", "dedup", "splits", "masks", "privacy", "load", "card", "croissant",
    "join", "frozen", "leak", "bytes_once", "splits_v11", "decon", "labelmap",
)  # fmt: skip


@dataclass(frozen=True)
class PatchOptions:
    base: PatchBase
    flavour: str = "open"
    release_dir: Path | None = None
    v1_dir: Path | None = None
    hidden_test: tuple[Path, ...] = ()
    staged_root: Path | None = None
    split_map: Path | None = None
    extra_nc_ids: frozenset[str] = frozenset()
    empty_threshold: float = 0.05
    sample_rows: int = 50
    taxonomy_version: str | None = None


def _need_v1(name: str, o: PatchOptions) -> GateResult | None:
    if o.v1_dir is None or not Path(o.v1_dir).is_dir():
        return finish(name, {}, [f"no_v1_set: {name} needs the v1.0 tree (--v1-dir)"])
    return None


def _combine(name: str, first: GateResult, second: GateResult, prefix: str) -> GateResult:
    counts = {k: v for k, v in first.counts.items() if k != "failures_total"}
    counts.update({f"{prefix}_{k}": v for k, v in second.counts.items() if k != "failures_total"})
    return finish(name, counts, [*first.failures, *second.failures])


def _patch_base(b: Path, o: PatchOptions) -> GateResult:
    base, failures = o.base, []
    if not base.tree:
        failures.append("no_prior: the prior release tree is empty: nothing is frozen")
    if not base.card_entries:
        failures.append("no_prior_card: the prior card lists no configs")
    counts = {
        "release": base.release,
        "repo": base.repo,
        "tree_files": len(base.tree),
        "prior_configs": len(base.configs),
        "lfs_files": sum(v is not None for v in base.tree.values()),
    }
    return finish("patch_base", counts, failures)


def _licence(b: Path, o: PatchOptions) -> GateResult:
    from .licence import run_licence_gate

    return run_licence_gate(b, flavour=o.flavour, release_dir=o.release_dir)


def _croissant(b: Path, o: PatchOptions) -> GateResult:
    from .croissant import run_croissant_gate

    return run_croissant_gate(b)


def _dedup(b: Path, o: PatchOptions) -> GateResult:
    from ..licence_class import source_families
    from .dedup import run_dedup_gate
    from .patch_checks import inherit_image_gate

    return inherit_image_gate(
        "dedup", b, o.base,
        lambda: run_dedup_gate(b, release_dir=o.release_dir, families=source_families()),
    )  # fmt: skip


def _privacy(b: Path, o: PatchOptions) -> GateResult:
    from .patch_checks import inherit_image_gate
    from .privacy import run_privacy_gate
    from .privacy_detectors import load_detector

    return inherit_image_gate(
        "privacy", b, o.base, lambda: run_privacy_gate(b, detector=load_detector(None))
    )


def _splits(b: Path, o: PatchOptions) -> GateResult:
    from .patch_checks import run_patch_splits_gate
    from .shards import find_split_map

    found = find_split_map(b, o.release_dir)
    split_map = o.split_map or (found[0] if found else None)
    return run_patch_splits_gate(b, split_map=split_map, v1_dir=o.v1_dir)


def _splits_v11(b: Path, o: PatchOptions) -> GateResult:
    return replace(_splits(b, o), name="splits_v11")


def _masks(b: Path, o: PatchOptions) -> GateResult:
    from . import condition_masks as cm
    from .masks import run_masks_gate

    stock = run_masks_gate(b, empty_threshold=o.empty_threshold)
    if o.staged_root is None:
        sizes = finish(
            "masks", {}, ["no_staged_root: raster size vs metadata is unchecked (--staged-root)"]
        )
    else:
        from ..condition_masks import local_tree
        from ..task_layers.masks_table import CONDITION_MASK_SOURCES

        tree = local_tree(o.staged_root, CONDITION_MASK_SOURCES["reef-support-bleaching"].tree)
        sizes = cm.run_masks_gate(b, sizes=cm.staged_sizes(tree))
    return _combine("masks", stock, sizes, "sizes")


def _load(b: Path, o: PatchOptions) -> GateResult:
    from .load import run_load_gate

    return run_load_gate(b, release_dir=o.release_dir, sample_rows=o.sample_rows, prior=o.base)


def _card(b: Path, o: PatchOptions) -> GateResult:
    from .card import run_card_gate

    return run_card_gate(b, release_dir=o.release_dir, prior=o.base)


def _frozen(b: Path, o: PatchOptions) -> GateResult:
    from . import condition_masks as cm

    return cm.run_frozen_gate(b, prior=o.base.tree)


def _join(b: Path, o: PatchOptions) -> GateResult:
    from . import condition_masks as cm

    return _need_v1("join", o) or cm.run_join_gate(b, v1_shas=cm.v1_coral_masks_shas(o.v1_dir))


def _decon(b: Path, o: PatchOptions) -> GateResult:
    from ..hf_export_v11 import hidden_test_drop
    from . import condition_masks as cm

    if not o.hidden_test:
        return finish("decon", {}, ["no_hidden_test: pass --hidden-test MANIFEST"])
    if (missing := _need_v1("decon", o)) is not None:
        return missing
    hidden = frozenset().union(*(hidden_test_drop(m, o.v1_dir) for m in o.hidden_test))
    return cm.run_decon_gate(b, hidden=hidden)


def _leak(b: Path, o: PatchOptions) -> GateResult:
    from .leak import run_leak_gate
    from .v11_runner import nc_source_ids

    return run_leak_gate(b, nc_ids=nc_source_ids() | o.extra_nc_ids, flavour=o.flavour)


def _bytes_once(b: Path, o: PatchOptions) -> GateResult:
    from .bytes_once import run_bytes_once_gate

    return _need_v1("bytes_once", o) or run_bytes_once_gate(b, v1_dir=o.v1_dir)


def _labelmap(b: Path, o: PatchOptions) -> GateResult:
    from .patch_labelmap import run_patch_labelmap_gate

    return run_patch_labelmap_gate(b, version=o.taxonomy_version)


RUNNERS: dict[str, Callable[[Path, PatchOptions], GateResult]] = {
    "patch_base": _patch_base, "licence": _licence, "dedup": _dedup, "splits": _splits,
    "masks": _masks, "privacy": _privacy, "load": _load, "card": _card, "croissant": _croissant,
    "join": _join, "frozen": _frozen, "leak": _leak, "bytes_once": _bytes_once,
    "splits_v11": _splits_v11, "decon": _decon, "labelmap": _labelmap,
}  # fmt: skip


def run_patch_gate(name: str, build_dir: Path, options: PatchOptions) -> GateResult:
    try:
        return RUNNERS[name](Path(build_dir), options)
    except Exception as exc:
        return finish(
            name,
            {"failed_gate_error": 1},
            [f"{name} gate_error: {type(exc).__name__}: {str(exc)[:200]}"],
        )
