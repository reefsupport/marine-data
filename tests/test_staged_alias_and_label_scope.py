"""WP-R4: one staged-id alias table (no double count), and the label gate judges only sources
that contribute rows (a rowless source is a skip with a reason, never a pass or a fail)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from test_global_split_map_flavours import _registry, _src
from test_upstream_test_split import _stage

from marinedata.labelcheck import gate_scope, release_label_gate
from marinedata.labels_check import evaluate
from marinedata.licence_class import (
    SPEC_ALIASES,
    collapse_aliases,
    source_class,
    spec_for,
    staged_id,
)
from marinedata.registry import Registry
from marinedata.release import _skipped_sources
from marinedata.taxonomy import TaxonomyGateError


# -- 1. staged-id aliases ------------------------------------------------------------------
def test_staged_id_resolves_the_alias_and_leaves_other_ids_alone() -> None:
    assert staged_id("atlantis-synthetic-depth") == "atlantis"
    assert staged_id("atlantis") == "atlantis"
    assert staged_id("coralscapes") == "coralscapes"


def test_seathru_is_not_aliased_to_seathru_nerf() -> None:
    """Different datasets (Akkaynak 2019 HF mirror, restricted-nc vs Levy 2023 NeRF scenes,
    internal-only): aliasing them would launder an unknown licence into the nc flavour."""
    assert staged_id("seathru") == "seathru"
    assert "seathru" not in SPEC_ALIASES and "seathru-nerf" not in SPEC_ALIASES.values()
    assert source_class("seathru") == "restricted-nc"
    assert source_class("seathru-nerf") == "internal-only"


def test_every_alias_points_at_a_real_spec_that_is_not_also_a_registry_source() -> None:
    registry_ids = {s.id for s in Registry.load().sources}
    for alias, target in SPEC_ALIASES.items():
        assert alias in registry_ids, alias
        assert target not in registry_ids, f"{target} would be counted twice next to {alias}"
        assert spec_for(alias) is not None, alias


def test_licence_class_of_the_alias_comes_from_the_canonical_spec() -> None:
    assert spec_for("atlantis-synthetic-depth") == spec_for("atlantis")
    assert source_class("atlantis-synthetic-depth") == "restricted-nc"


def test_collapse_aliases_drops_the_staged_id_only_when_its_registry_id_is_present() -> None:
    ids = ["atlantis", "atlantis-synthetic-depth", "zzz"]
    assert collapse_aliases(ids, ["atlantis-synthetic-depth"]) == [
        "atlantis-synthetic-depth",
        "zzz",
    ]
    assert collapse_aliases(["atlantis"], ["zzz"]) == ["atlantis"]


def test_labels_check_counts_an_aliased_dataset_once() -> None:
    registry = Registry.load()
    cache = {"atlantis": {"id": "atlantis", "meta_rows": "3600", "label_kinds": "none"}}
    rows = evaluate(registry, registry.root, audit_cache=cache, release=["atlantis"])
    atl = [r for r in rows if r.source_id.startswith("atlantis")]
    assert [r.source_id for r in atl] == ["atlantis-synthetic-depth"]
    assert (
        atl[0].ingested and atl[0].gated
    )  # the staged id's cache row and release entry carry over


# -- 2. label gate scope -------------------------------------------------------------------
def _copy_registry_without_ozfish_skip(tmp_path: Path) -> Registry:
    root = tmp_path / "registry"
    shutil.copytree(Registry.load().root, root)
    path = root / "sources" / "fish.yaml"
    lines = [
        ln
        for ln in path.read_text().splitlines(keepends=True)
        if "no-staged-images: nothing staged under sources/ozfish/" not in ln
    ]
    assert len(lines) == len(path.read_text().splitlines(keepends=True)) - 1
    path.write_text("".join(lines))
    return Registry.load(root)


def test_ozfish_and_sea_urchin_carry_a_no_staged_images_reason() -> None:
    registry = Registry.load()
    for sid in ("ozfish", "sea-urchin-detection"):
        assert (registry.source(sid).release_skip_reason or "").startswith("no-staged-images:"), sid


def test_gate_scope_skips_registry_reasons_and_rowless_sources() -> None:
    skipping = _src("a-skip", "open").model_copy(update={"release_skip_reason": "labels TBD"})
    registry = _registry([skipping, _src("b-rows", "open"), _src("c-empty", "open")])
    judged, skipped = gate_scope(
        registry, ["a-skip", "b-rows", "c-empty", "unknown-id"], empty=["c-empty"]
    )
    assert judged == ["b-rows", "unknown-id"]
    assert skipped == {"a-skip": "labels TBD", "c-empty": "no-staged-images: 0 staged rows"}


def test_rowless_bbox_source_without_crosswalk_does_not_fail_the_gate(tmp_path: Path) -> None:
    registry = _copy_registry_without_ozfish_skip(tmp_path)
    assert registry.source("ozfish").release_skip_reason is None
    # WITH rows and no crosswalk: still fails
    with pytest.raises(TaxonomyGateError, match="ozfish"):
        release_label_gate(registry, ["ozfish"])
    # 0 staged rows: skipped, so the gate neither fails nor judges it
    assert release_label_gate(registry, ["ozfish"], empty=["ozfish"])["taxonomy_version"]
    judged, skipped = gate_scope(registry, ["ozfish"], empty=["ozfish"])
    assert judged == [] and skipped["ozfish"].startswith("no-staged-images")
    # the registry reason (the shipped state) skips it with no row count at all
    assert release_label_gate(Registry.load(), ["ozfish", "sea-urchin-detection"])


def test_labels_check_release_sources_skips_rowless_and_still_fails_rowed(tmp_path: Path) -> None:
    shipped = Registry.load()
    [row] = evaluate(shipped, shipped.root, release=["ozfish"], only=["ozfish"])
    assert row.status == "n/a" and row.reasons[0].startswith("skipped: no-staged-images")
    registry = _copy_registry_without_ozfish_skip(tmp_path)
    [row] = evaluate(registry, registry.root, release=["ozfish"], only=["ozfish"])
    assert row.status == "missing-crosswalk"  # rows (or unknown) and no crosswalk: still blocks
    rowless = {"ozfish": {"id": "ozfish", "meta_rows": "0", "label_kinds": "none"}}
    [row] = evaluate(
        registry, registry.root, audit_cache=rowless, release=["ozfish"], only=["ozfish"]
    )
    assert row.status == "n/a" and "0 staged rows" in row.reasons[0]


def test_build_skips_a_rowless_source_and_records_it(tmp_path: Path) -> None:
    registry = _registry([_src("has-rows", "open"), _src("no-rows", "open")])
    empty = tmp_path / "empty"
    empty.mkdir()
    staged = _stage(tmp_path / "rows", [("s1", "g1", "train", b"x")])
    roots = {"has-rows": staged, "no-rows": empty}
    skipped = _skipped_sources(registry, ["has-rows", "no-rows"], roots, frozenset())
    assert list(skipped) == ["no-rows"]  # a source WITH rows is not skipped
    reason, allowed_by = skipped["no-rows"]
    assert reason.startswith("no-staged-images:") and allowed_by == "no-rows"
