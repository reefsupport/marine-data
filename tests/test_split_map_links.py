"""Stem-keyed V1/V2 split links (Coralscapes V2, v1.2.0).

Twin frames (a V1 frame and its V2 re-annotation, linked by stem) share one split group and keep the
split v1.0 published; V2-only frames follow the V2 split through the benchmark pins; the dedup and
splits gates treat a twin pair as one link group, not as a cross-split leak."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _gates_build import Img, default_images, make_build, smooth_png

from marinedata.bench_manifest import MANIFEST_COLUMNS
from marinedata.benchmark_pins import BenchmarkPins, apply_benchmark_pins
from marinedata.benchmarks import BenchmarkRegistry
from marinedata.gates.dedup import LINKED, run_dedup_gate
from marinedata.gates.splits import run_splits_gate
from marinedata.release.split_map_inherit import plan_inheritance
from marinedata.release.stem_links import (
    StemEntry,
    StemLinkError,
    build_stem_links,
    link_block,
    map_split,
    normalise_stem,
    parse_link_decls,
    protect_v1_pins,
    read_link_table,
)
from marinedata.splitmap import rows_to_counts
from marinedata.splitmap_inherit import resolve_inherited_splits
from marinedata.splitmap_tree import V1Tree

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = "coralscapes", "coralscapes-v2"
DECLS = parse_link_decls(V2, [{"source_id": V1, "key": "stem", "relation": "re-annotation"}])
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
NOW = "2026-10-29T00:00:00Z"
S1, S2, S4 = "site1_000001_016200", "site2_000001_000100", "site4_000001_000007"
S9, S8, S7 = "site9_000001_000001", "site8_000002_000002", "site7_000003_000003"


def world(twins=(), v1_only=(), v2_only=()):
    """``(rows, entries, v1 splits)``: one image per stem, one split group per site and source."""
    rows, entries, tree = [], [], {}

    def add(source, stem, split, file_name=None):
        sha, group = f"{source[-2:]}-{stem}", f"{source}/{stem.split('_')[0]}"
        rows.append((sha, group, source))
        entries.append(StemEntry(source, file_name or stem, sha, group, split))
        if source == V1:
            tree[sha] = frozenset({map_split(split)})

    for stem, v1, v2 in twins:
        add(V1, stem, v1, f"{stem}_leftImg8bit.png")
        add(V2, stem, v2)
    for stem, v1 in v1_only:
        add(V1, stem, v1, f"{stem}_leftImg8bit.png")
    for stem, v2 in v2_only:
        add(V2, stem, v2)
    return rows, entries, tree


def pipeline(tmp_path, rows, entries, tree, pins=None, tag="base"):
    """The generator's steps over the pieces this change owns, ending in the real resolver."""
    links = build_stem_links(entries, DECLS)
    counts, strata, merge = rows_to_counts(rows, stratified=True, links=links.pairs)
    plan = plan_inheritance(
        rows,
        inherit=None,
        never_eval_sources=(),
        canonical=merge.canonical,
        hashes={},
        near_dup=None,
        ood_rules=None,
        ood_metadata=None,
        v1_tree=V1Tree(splits=tree),
        ratios=RATIOS,
        seed=1,
        now=NOW,
        stem_links=links,
    )
    dropped, forced, overrides, withheld = (), dict(plan.forced), {}, {}
    if pins is not None:
        pins, dropped = protect_v1_pins(pins, links, rows, plan.canonical, tree)
        forced, _, overrides, withheld = apply_benchmark_pins(
            plan.forced, rows, plan.canonical, pins, prior=plan.prior.assignments
        )
    path = tmp_path / tag / "SPLIT_MAP.json"
    path.parent.mkdir(exist_ok=True)
    resolve_inherited_splits(
        path, counts, RATIOS, prior=plan.prior, prior_sha256="0" * 64, strata=strata or {},
        merge_canonical=plan.canonical, seed=1, now=NOW, release="v1.2.0", min_groups=1,
        forced=forced, withhold=withheld, ood=plan.ood, ood_header=plan.ood_header,
        override=overrides, straddles=plan.straddles, inherits=plan.inherits(plan.canonical),
        band_excluded=plan.band_excluded,
    )  # fmt: skip
    data = json.loads(path.read_text())
    assigned = data["assignments"]
    split_of = {g: assigned.get(g) or assigned.get(plan.canonical[g]) for _, g, _ in rows}
    return plan, links, dropped, split_of, data


def test_twin_in_v1_test_stays_test_and_the_conflict_is_counted(tmp_path):
    rows, entries, tree = world(
        twins=[(S1, "test", "train"), (S2, "train", "train")],
        v1_only=[(S4, "train")],
        v2_only=[(S9, "test"), (S8, "validation"), (S7, "train")],
    )
    plan, links, _, split_of, data = pipeline(tmp_path, rows, entries, tree)
    assert links.pairs == [
        (f"{V1[-2:]}-{S1}", f"{V2[-2:]}-{S1}"),
        (f"{V1[-2:]}-{S2}", f"{V2[-2:]}-{S2}"),
    ]
    assert plan.canonical[f"{V2}/site1"] == plan.canonical[f"{V1}/site1"]
    assert plan.canonical[f"{V1}/site1"] != plan.canonical[f"{V1}/site2"]
    assert split_of[f"{V1}/site1"] == split_of[f"{V2}/site1"] == "test"  # V2 says train: V1 wins
    assert split_of[f"{V1}/site2"] == split_of[f"{V2}/site2"] == "train"
    assert split_of[f"{V2}/site7"] == "train"  # a V2-only train frame follows V2
    assert [(c.kind, c.stem, c.v1, c.v2) for c in plan.stem_conflicts] == [
        ("upstream_vs_v1", S1, "test", "train")
    ]
    block = data["inherits"]["stem_links"]
    assert block["conflicts"]["count"] == 1 and block["conflicts"]["by_kind"] == {
        "upstream_vs_v1": 1
    }
    assert (block["twins"], block["declaring_only"], block["target_only"]) == (2, 3, 1)
    table = read_link_table(tmp_path / "base" / "SPLIT_MAP.json")
    assert table is not None and table.linked(f"{V2[-2:]}-{S1}", f"{V1[-2:]}-{S1}")


def test_v2_only_frames_follow_the_v2_split_through_pins(tmp_path):
    rows, entries, tree = world(
        twins=[(S1, "test", "test")],
        v1_only=[(S2, "train")],
        v2_only=[(S9, "test"), (S8, "validation"), (S7, "train")],
    )
    pins = BenchmarkPins(
        by_sha={
            f"{V2[-2:]}-{S9}": ("test", V2),
            f"{V2[-2:]}-{S8}": ("validation", V2),
            f"{V2[-2:]}-{S1}": ("test", V2),
        }
    )
    plan, _, dropped, split_of, _ = pipeline(tmp_path, rows, entries, tree, pins)
    assert dropped == () and plan.stem_conflicts == ()
    assert split_of[f"{V2}/site9"] == "test"
    assert split_of[f"{V2}/site8"] == "val"
    assert split_of[f"{V2}/site7"] == "train"
    assert split_of[f"{V1}/site1"] == split_of[f"{V2}/site1"] == "test"


def test_a_v2_pin_never_moves_a_component_holding_a_v1_image(tmp_path):
    rows, entries, tree = world(twins=[(S1, "train", "test")], v1_only=[(S2, "test")])
    pins = BenchmarkPins(by_sha={f"{V2[-2:]}-{S1}": ("test", V2)})
    plan, *_ = pipeline(tmp_path, rows, entries, tree)
    _, _, overrides, _ = apply_benchmark_pins(
        plan.forced, rows, plan.canonical, pins, prior=plan.prior.assignments
    )
    assert overrides  # unprotected, the V2 pin would move the whole v1.0 train component
    plan, _, dropped, split_of, _ = pipeline(tmp_path, rows, entries, tree, pins, tag="pinned")
    assert [(c.kind, c.stem, c.v1, c.v2) for c in dropped] == [("pin_dropped", S1, "train", "test")]
    assert split_of[f"{V1}/site1"] == split_of[f"{V2}/site1"] == "train"
    assert [c.kind for c in plan.stem_conflicts] == ["upstream_vs_v1"]


def test_declarations_and_stems():
    assert normalise_stem("d/site1_000001_016200_leftImg8bit.png") == "site1_000001_016200"
    assert normalise_stem("site1_000001_016200") == "site1_000001_016200"
    assert map_split("validation") == "val" and map_split("weird") is None
    bad = {"source_id": V1, "key": "stem", "relation": "re-annotation"}
    for patch in ({"source_id": V2}, {"key": "sha"}, {"relation": "clone"}):
        with pytest.raises(StemLinkError):
            parse_link_decls(V2, [{**bad, **patch}])
    assert (
        parse_link_decls(V1, None) == ()
        and len(link_block(build_stem_links([], DECLS))["links"]) == 1
    )


def _sha(img: Img) -> str:
    return hashlib.sha256(img.data).hexdigest()


def _link_map(build: Path, pairs, conflicts: int = 1) -> None:
    block = {
        "twins": len(pairs),
        "declaring_only": 0,
        "target_only": 0,
        "conflicts": {"count": conflicts, "by_kind": {"upstream_vs_v1": conflicts}, "listed": []},
        "pairs": [list(p) for p in pairs],
    }
    (build / "SPLIT_MAP.json").write_text(json.dumps({"inherits": {"stem_links": block}}))


def _has(result, kind: str) -> bool:
    return any(f"{kind}:" in line for line in result.failures)


V1_FRAME = Img(100, "train", "src-a", group="ga")
V2_TWIN = Img(100, "train", "src-b", group="gb", shift=1)  # other bytes, near-identical pixels


def test_dedup_twins_in_one_group_are_not_a_leak(tmp_path):
    build = make_build(tmp_path, images=[*default_images(), V1_FRAME, V2_TWIN], masks=False)
    assert _has(run_dedup_gate(build), "near_dup_across_sources")  # unlinked, the pair is flagged
    _link_map(build, [(_sha(V1_FRAME), _sha(V2_TWIN))])
    result = run_dedup_gate(build)
    assert result.passed, result.failures
    assert result.counts[LINKED] == 1 and result.counts["stem_link_conflicts"] == 1


def test_dedup_still_reports_a_v2_frame_near_an_unrelated_v1_train_image(tmp_path):
    v2_test = Img(0, "test", "src-b", group="gv2", shift=1)  # near a default train image of src-a
    build = make_build(
        tmp_path, images=[*default_images(), V1_FRAME, V2_TWIN, v2_test], masks=False
    )
    _link_map(build, [(_sha(V1_FRAME), _sha(V2_TWIN))])
    result = run_dedup_gate(build)
    assert _has(result, "near_dup_across_splits") and not _has(result, "linked_twins_split")
    assert result.counts[LINKED] == 1


def test_twins_in_different_splits_fail_both_gates(tmp_path):
    v2_test = Img(100, "test", "src-b", group="gb", shift=1)
    build = make_build(tmp_path, images=[*default_images(), V1_FRAME, v2_test], masks=False)
    _link_map(build, [(_sha(V1_FRAME), _sha(v2_test))])
    assert _has(run_dedup_gate(build), "linked_twins_split")
    assert _has(run_splits_gate(build), "linked_twins_split")


def test_splits_gate_counts_twins_that_share_a_split(tmp_path):
    build = make_build(tmp_path, images=[*default_images(), V1_FRAME, V2_TWIN], masks=False)
    _link_map(build, [(_sha(V1_FRAME), _sha(V2_TWIN))])
    result = run_splits_gate(build)
    assert not _has(result, "linked_twins_split")
    assert (
        result.counts["stem_link_pairs_checked"] >= 1
        and result.counts["stem_link_pairs_apart"] == 0
    )


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "bm_cs_v2", ROOT / "scripts" / "benchmark_manifest_coralscapes_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_script_builds_the_eval_manifest_offline(tmp_path):
    root = tmp_path / "staged"
    (root / "images").mkdir(parents=True)
    rows = []
    for i, (stem, split) in enumerate(
        [(S1, "test"), (S2, "test"), (S4, "validation"), (S9, "train")]
    ):
        (root / "images" / f"{stem}.png").write_bytes(smooth_png(i + 1))
        rows.append({"stem": stem, "image_path": f"images/{stem}.png", "upstream_split": split})
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
    script, out = _load_script(), tmp_path / "coralscapes-v2.parquet"
    base = ["--staged", str(root), "--out", str(out)]
    assert (
        script.main([*base, "--dry-run"]) == 2
    )  # a preflight: the counts are checked, nothing built
    assert script.main([*base, "--dry-run", "--allow-count-mismatch"]) == 0 and not out.exists()
    assert script.main(base) == 2 and not out.exists()  # 2 test + 1 validation != 473 + 170
    assert script.main([*base, "--allow-count-mismatch"]) == 0
    table = pq.read_table(out)
    assert table.column_names == list(MANIFEST_COLUMNS)
    assert sorted(table.column("upstream_split").to_pylist()) == ["test", "test", "validation"]
    assert set(table.column("benchmark_id").to_pylist()) == {V2}


def test_registry_entry_validates():
    entry = BenchmarkRegistry.load(ROOT / "registry" / "benchmarks.yaml").by_id(V2)
    assert (
        entry.upstream_split.eval_split == "test"
        and entry.upstream_split.heldout_val == "validation"
    )
    assert (entry.upstream_split.counts["test"], entry.upstream_split.counts["validation"]) == (
        473,
        170,
    )
    assert entry.policy == "route-to-our-test" and entry.registry_id == V2


def test_generate_split_map_wires_the_stem_links(tmp_path, monkeypatch):
    """REV F3: the real generator pairs the twins by stem exactly as the reference pipeline."""
    from marinedata.registry import Registry
    from marinedata.release import split_map as sm

    rows, entries, tree = world(
        twins=[(S1, "test", "train"), (S2, "train", "train")],
        v1_only=[(S4, "train")],
        v2_only=[(S9, "test"), (S8, "validation"), (S7, "train")],
    )
    monkeypatch.setattr(sm, "enumerate_release_rows", lambda *a, **k: iter(rows))
    monkeypatch.setattr(sm, "linked_entries", lambda *a, **k: entries)
    out = tmp_path / "gen" / "SPLIT_MAP.json"
    out.parent.mkdir()
    sm.generate_split_map(
        Registry.load(), out=out, roots={V1: tmp_path, V2: tmp_path}, ratios=RATIOS, seed=1,
        now=NOW, release="v1.2.0", min_groups=1, inherit_tree=V1Tree(splits=tree),
    )  # fmt: skip
    got = json.loads(out.read_text())
    _, _, _, _, ref = pipeline(tmp_path, rows, entries, tree)
    block = got["inherits"]["stem_links"]
    assert block["twins"] == 2
    assert block == ref["inherits"]["stem_links"]
    assert got["assignments"] == ref["assignments"]


def test_generate_split_map_drops_the_pin_v1_refuses_and_counts_new_and_moved(
    tmp_path, monkeypatch
):
    """INT-2: end to end through the real generator with a frozen v1.0 map and three V2 pins.

    Counter names: "new" is ``declaring_only`` (a V2 frame without a V1 twin), "moved" is the
    ``upstream_vs_v1`` conflict (a twin whose V2 split differs from v1.0; V1 wins) and the
    ``pins.v1_continuity_overridden`` image count (v1.0 images a pin moved: 0 while v1.0 is
    frozen)."""
    from marinedata.registry import Registry
    from marinedata.release import split_map as sm

    rows, entries, tree = world(
        twins=[(S1, "test", "test"), (S2, "train", "test")],
        v1_only=[(S4, "train")],
        v2_only=[(S9, "test")],
    )
    pins = BenchmarkPins(
        by_sha={
            f"{V2[-2:]}-{S1}": ("test", V2),  # identical splits: v1.0 accepts it
            f"{V2[-2:]}-{S2}": ("test", V2),  # v1.0 published train: refused, dropped
            f"{V2[-2:]}-{S9}": ("test", V2),  # no V1 twin: follows its pin
        }
    )
    monkeypatch.setattr(sm, "enumerate_release_rows", lambda *a, **k: iter(rows))
    monkeypatch.setattr(sm, "linked_entries", lambda *a, **k: entries)
    out = tmp_path / "gen" / "SPLIT_MAP.json"
    out.parent.mkdir()
    sm.generate_split_map(
        Registry.load(), out=out, roots={V1: tmp_path, V2: tmp_path}, ratios=RATIOS, seed=1,
        now=NOW, release="v1.2.0", min_groups=1, inherit_tree=V1Tree(splits=tree),
        benchmark_pins=pins,
    )  # fmt: skip
    got = json.loads(out.read_text())
    plan, _, dropped, _, _ = pipeline(tmp_path, rows, entries, tree, pins)
    assigned = got["assignments"]
    split_of = {g: assigned.get(g) or assigned.get(plan.canonical[g]) for _, g, _ in rows}
    assert split_of[f"{V1}/site1"] == split_of[f"{V2}/site1"] == "test"  # identical: nothing moved
    header = got.get("pins") or {}
    assert "v1_continuity_overridden" in header  # the map inherits v1.0: the block is stamped
    counted = header["benchmarks"][V2]["test"]
    assert counted == 2  # S1 and S9 are the counted pins (the S2 pin was dropped): not vacuous
    moved = header["v1_continuity_overridden"]
    assert (moved["images"], moved["groups"], moved["group_ids"]) == (0, 0, [])  # v1.0 is frozen
    assert split_of[f"{V1}/site2"] == split_of[f"{V2}/site2"] == "train"  # changed twin: V1 wins
    assert [(c.kind, c.stem) for c in dropped] == [("pin_dropped", S2)]
    assert split_of[f"{V2}/site9"] == "test"  # new frame follows its pin
    block = got["inherits"]["stem_links"]
    assert (block["twins"], block["declaring_only"], block["target_only"]) == (2, 1, 1)
    assert block["conflicts"]["by_kind"] == {"pin_dropped": 1, "upstream_vs_v1": 1}
