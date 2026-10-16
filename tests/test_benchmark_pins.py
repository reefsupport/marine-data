"""Split-map pins from the benchmark manifests (design eval-decontamination-split-v2, 2.2).

A manifest sha256 pins an upstream eval image to ``test`` and the second held-out split to ``val``,
whatever the staged ``upstream_split`` says; OOD claims, v1.0 continuity and never-eval train win.
The staged-tree fixtures are copied from ``test_upstream_test_split`` (tests is not a package).
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.benchmark_pins import (
    MAX_OVERRIDE_IDS,
    BenchmarkPins,
    apply_benchmark_pins,
    check_overridden,
    load_benchmark_pins,
    split_role,
)
from marinedata.benchmarks import BenchmarkRegistry
from marinedata.cli_release_pins import pin_kwargs, pins_summary
from marinedata.enums import (
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import Access, Coverage, Licence, LoaderSpec, Profile, Source, Verification
from marinedata.registry import Registry
from marinedata.release import generate_split_map
from marinedata.schema import Axis, LabelSchema
from marinedata.splitmap import load_split_map, save_split_map
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec

NOW = "2026-10-05T00:00:00Z"
Spec = tuple[str, str, str | None, bytes]  # stem, split_group, upstream_split, image bytes


def _source(source_id: str, tags: tuple[str, ...] = ()) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the upstream-test split rule.",
        version="v1",
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 10, 5), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(method=AccessMethod.HTTP, uri="https://example.invalid"),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout="staged-tree", params={}),
        annotations=(),
        tags=tags,
    )


def _registry(sources: dict[str, Source]) -> Registry:
    return Registry(
        sources=sources,
        licences={},
        profiles={
            "research": Profile(
                id="research", description="f", allow_tiers=(Tier.PERMISSIVE,), ships=True
            )
        },
        schemas={
            "rs-benthic-v1": LabelSchema(id="rs-benthic-v1", name="F", axes=(Axis.TAXON,), nodes=())
        },
        tasks={"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)},
    )


def _stage(root: Path, rows: list[Spec]) -> Path:
    for stem, _, _, data in rows:
        path = root / "images" / "p" / f"{stem}.img"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_metadata_table(
        root / "metadata.parquet",
        [
            StagedImage(
                stem=stem,
                partition="p",
                upstream_path=f"orig/{stem}",
                upstream_split=upstream,
                width=72,
                height=64,
                split_group=group,
            )
            for stem, group, upstream, _ in rows
        ],
    )
    return root


def _fixture_rows(prefix: str, *, test_groups: int, other_groups: int) -> list[Spec]:
    """``test_groups`` groups with TEST rows (one also holds a train row), then plain train ones."""
    rows: list[Spec] = []
    for g in range(test_groups):
        rows.append((f"t{g}a", f"{prefix}/t{g}", "TEST", f"{prefix}-t{g}-a".encode()))
        # group 0 mixes upstream train + test: the whole group must still go to test
        second = "train" if g == 0 else "TEST"
        rows.append((f"t{g}b", f"{prefix}/t{g}", second, f"{prefix}-t{g}-b".encode()))
    for g in range(other_groups):
        for k in "ab":
            rows.append((f"o{g}{k}", f"{prefix}/o{g}", "train", f"{prefix}-o{g}-{k}".encode()))
    return rows


def _map(tmp_path: Path, sources: dict[str, Source], roots: dict[str, Path], **kw):  # type: ignore[no-untyped-def]
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "SPLIT_MAP.json"
    stats = generate_split_map(
        _registry(sources), out=out, roots=roots, now=NOW, seed=3, min_groups=3, **kw
    )
    loaded = load_split_map(out)
    assert loaded is not None
    return stats, loaded.assignments


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BenchmarkRegistry.load(ROOT / "registry" / "benchmarks.yaml")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(source: str, kind: str, groups: int) -> list[Spec]:
    """``groups`` two-image groups ``<source>/<kind><i>``: unique bytes, upstream split train."""
    return [
        (f"{kind}{g}{k}", f"{source}/{kind}{g}", "train", f"{source}-{kind}{g}-{k}".encode())
        for g in range(groups)
        for k in "ab"
    ]


def _shas(rows: list[Spec]) -> list[str]:
    return [_sha(data) for _, _, _, data in rows]


def _write_manifest(root: Path, benchmark_id: str, labelled, *, column: str = "sha256") -> None:
    path = root / "benchmarks" / "manifests" / f"{benchmark_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "benchmark_id": [benchmark_id] * len(labelled),
            column: [sha for sha, _ in labelled],
            "upstream_split": [label for _, label in labelled],
        }
    )
    pq.write_table(table, path)


def _label(benchmark_id: str) -> str:
    entry = REGISTRY.by_id(benchmark_id)
    return entry.registry_id or entry.id


def _split(assignments, group: str) -> str:
    if group in assignments:
        return assignments[group]
    found = [split for key, split in assignments.items() if key.endswith(group)]
    assert len(found) == 1, f"{group}: {len(found)} matching groups in {sorted(assignments)[:4]}"
    return found[0]


def test_split_role_follows_the_registry_entry():
    by_id = REGISTRY.by_id
    assert split_role(by_id("coralscapes"), "test") == "eval"
    assert split_role(by_id("coralscapes"), "validation") == "heldout"
    assert split_role(by_id("coralscapes"), "val") == "heldout"
    assert split_role(by_id("suim"), "images") == "eval"  # staged label differs from the TEST entry
    assert split_role(by_id("uiis"), "val") == "eval"  # no upstream test split: val is the eval set
    assert split_role(by_id("usis10k"), "val") == "heldout"
    assert split_role(by_id("usis10k"), "test") == "eval"


def test_load_pins_reads_roles_counts_and_skips_exclude_entries(tmp_path):
    ev, ho, ui = _shas(_rows("x", "ev", 2)), _shas(_rows("x", "ho", 2)), _shas(_rows("x", "ui", 2))
    _write_manifest(
        tmp_path, "coralscapes", [(s, "test") for s in ev] + [(s, "validation") for s in ho]
    )
    _write_manifest(tmp_path, "uiis", [(s, "val") for s in ui])
    excluded = next(e.id for e in REGISTRY.benchmarks if e.policy == "exclude")
    _write_manifest(tmp_path, excluded, [(_sha(b"excluded"), "all")])
    pins = load_benchmark_pins(REGISTRY, tmp_path)
    assert {pins.by_sha[s][0] for s in ev + ui} == {"test"}
    assert {pins.by_sha[s][0] for s in ho} == {"validation"}
    assert _sha(b"excluded") not in pins.by_sha
    assert pins.manifest_rows[_label("coralscapes")] == {"test": 4, "validation": 4}
    assert pins.manifest_rows[_label("uiis")] == {"test": 4, "validation": 0}


def test_bytes_sha256_column_pins_too(tmp_path):
    digest = _sha(b"only-bytes")
    _write_manifest(tmp_path, "coralscapes", [(digest, "test")], column="bytes_sha256")
    assert load_benchmark_pins(REGISTRY, tmp_path).by_sha[digest][0] == "test"


def test_a_component_takes_the_pin_of_any_member_and_test_beats_val():
    rows = [("s1", "g1", "a"), ("s2", "g1", "a"), ("s3", "g2", "a"), ("s4", "g3", "a")]
    by_sha = {"s2": ("validation", "b"), "s3": ("test", "b"), "s4": ("validation", "b")}
    forced, header, _ = apply_benchmark_pins({}, rows, {"g2": "g1"}, BenchmarkPins(by_sha=by_sha))
    assert forced == {"g1": "test", "g3": "val"}
    assert header["components"] == {"test": 1, "validation": 1}
    assert header["benchmarks"] == {"b": {"test": 1, "validation": 2}}


def test_ood_claim_and_earlier_forced_rules_win_over_a_pin():
    rows = [("s1", "g1", "a"), ("s2", "g2", "a"), ("s3", "g3", "a")]
    by_sha = {"s1": ("test", "b"), "s2": ("validation", "b"), "s3": ("test", "b")}
    pins = BenchmarkPins(by_sha=by_sha)
    forced, header, overrides = apply_benchmark_pins({"g3": "train"}, rows, {}, pins, ood={"g1"})
    assert forced == {"g3": "train", "g2": "val"} and overrides == {}
    assert header["skipped_components"] == {"ood": 1, "earlier_rule": 1}


def test_a_pin_overrides_v1_continuity_and_the_override_is_recorded():
    rows = [(f"s{i}", f"g{i}", "a") for i in range(1, 6)]
    prior = {"g1": "train", "g2": "val", "g3": "test", "g4": "val", "g5": "test"}
    wanted = {"s1": "test", "s2": "validation", "s3": "test", "s4": "test", "s5": "validation"}
    pins = BenchmarkPins(by_sha={sha: (pin, "b") for sha, pin in wanted.items()})
    forced, header, overrides = apply_benchmark_pins({}, rows, {}, pins, prior=prior)
    assert forced == {"g2": "val", "g3": "test"}  # g2 and g3 already sit where their pin says
    assert overrides == {"g1": "test", "g4": "test", "g5": "val"}
    assert header["v1_continuity_overridden"] == {
        "images": 3,
        "groups": 3,
        "by_source": {"a": {"test -> val": 1, "train -> test": 1, "val -> test": 1}},
        "group_ids": ["g1", "g4", "g5"],
    }
    assert header["skipped_components"] == {"ood": 0, "earlier_rule": 0}
    assert header["components"] == {"test": 3, "validation": 2}
    assert "v1_continuity_conflicts" not in header and "never-eval" in header["rule"]


def test_override_moves_the_whole_component_but_ood_and_never_eval_still_win():
    rows = [("s1", "g1", "a"), ("s2", "g2", "a"), ("s3", "g3", "a"), ("s4", "g4", "b")]
    canonical = {"g2": "g1"}  # g1 + g2 are one component: v1.0 put g1 in train and g2 in val
    prior = {"g1": "train", "g2": "val", "g3": "train", "g4": "train"}
    pins = BenchmarkPins(by_sha={sha: ("test", "b") for sha in ("s1", "s3", "s4")})
    forced, header, overrides = apply_benchmark_pins(
        {"g4": "train"}, rows, canonical, pins, prior=prior, ood={"g3"}, train_only={"g4"}
    )
    assert overrides == {"g1": "test"} and forced == {"g4": "train"}
    over = header["v1_continuity_overridden"]
    assert (over["images"], over["groups"], over["group_ids"]) == (1, 2, ["g1", "g2"])
    assert over["by_source"] == {"a": {"train+val -> test": 1}}
    assert header["skipped_components"] == {"ood": 1, "earlier_rule": 1}


def test_group_ids_are_dropped_above_the_cap_and_the_counts_stay():
    count = MAX_OVERRIDE_IDS + 1
    rows = [(f"s{i}", f"g{i}", "a") for i in range(count)]
    pins = BenchmarkPins(by_sha={sha: ("test", "b") for sha, _, _ in rows})
    prior = {group: "train" for _, group, _ in rows}
    _, header, _ = apply_benchmark_pins({}, rows, {}, pins, prior=prior)
    over = header["v1_continuity_overridden"]
    assert (over["images"], over["groups"]) == (count, count) and "group_ids" not in over
    with pytest.raises(ValueError, match="changed split"):
        check_overridden(header, [f"g{i}" for i in range(count - 1)])
    check_overridden(header, [group for _, group, _ in rows])


def test_cli_opt_out_and_summary():
    assert pin_kwargs(argparse.Namespace(no_benchmark_pins=True), None) == {}
    pins = BenchmarkPins(by_sha={"s1": ("test", "b")})
    rows = [("s1", "g1", "a")]
    _, header, _ = apply_benchmark_pins({}, rows, {}, pins, prior={"g1": "train"})
    text = pins_summary(header)
    assert "b test=1 validation=0" in text
    assert text.splitlines()[-1] == (
        "release: benchmark pins override v1 continuity for 1 image(s) in 1 group(s): "
        "{'a': {'train -> test': 1}}"
    )
    _, quiet, _ = apply_benchmark_pins({}, rows, {}, pins, prior={"g1": "test"})
    assert pins_summary(quiet).endswith(
        "release: benchmark pins override v1 continuity for 0 image(s)"
    )


def test_split_map_pins_eval_to_test_and_heldout_to_val_from_the_manifests(tmp_path):
    ev, ho, ui = _rows("bench", "ev", 4), _rows("bench", "ho", 3), _rows("bench", "ui", 3)
    root = _stage(tmp_path / "stage", ev + ho + ui + _rows("bench", "fr", 10))
    registry_root = tmp_path / "registry"
    _write_manifest(
        registry_root,
        "coralscapes",
        [(s, "test") for s in _shas(ev)] + [(s, "validation") for s in _shas(ho)],
    )
    _write_manifest(registry_root, "uiis", [(s, "val") for s in _shas(ui)])  # val is the eval set
    pins = load_benchmark_pins(REGISTRY, registry_root)
    stats, assignments = _map(
        tmp_path / "out", {"bench": _source("bench")}, {"bench": root}, benchmark_pins=pins
    )
    assert {_split(assignments, f"bench/ev{g}") for g in range(4)} == {"test"}
    assert {_split(assignments, f"bench/ui{g}") for g in range(3)} == {"test"}
    assert {_split(assignments, f"bench/ho{g}") for g in range(3)} == {"val"}
    written = load_split_map(tmp_path / "out" / "SPLIT_MAP.json")
    assert written is not None and stats.pins == written.pins
    assert stats.pins["benchmarks"][_label("coralscapes")] == {"test": 8, "validation": 6}
    assert stats.pins["benchmarks"][_label("uiis")] == {"test": 6, "validation": 0}
    assert stats.pins["components"] == {"test": 7, "validation": 3}
    assert "v1_continuity_overridden" not in stats.pins  # no --inherit, no v1.0 to override


def test_map_without_a_matching_pin_is_byte_identical(tmp_path):
    root = _stage(tmp_path / "stage", _rows("bench", "fr", 12))
    sources, roots = {"bench": _source("bench")}, {"bench": root}
    _map(tmp_path / "plain", sources, roots)
    _map(tmp_path / "pinned", sources, roots, benchmark_pins=BenchmarkPins())
    plain = (tmp_path / "plain" / "SPLIT_MAP.json").read_bytes()
    assert plain == (tmp_path / "pinned" / "SPLIT_MAP.json").read_bytes()
    assert b'"pins"' not in plain
    loaded = load_split_map(tmp_path / "plain" / "SPLIT_MAP.json")
    assert loaded is not None
    save_split_map(tmp_path / "resaved.json", loaded)  # stamping a pins block re-saves the map
    assert (tmp_path / "resaved.json").read_bytes() == plain


def test_a_pin_beats_v1_continuity_and_the_whole_group_moves(tmp_path):
    old = _rows("old", "g", 14)
    old_root = _stage(tmp_path / "old-stage", old)
    _map(tmp_path / "v1", {"old": _source("old")}, {"old": old_root})
    v1_path = tmp_path / "v1" / "SPLIT_MAP.json"
    v1 = load_split_map(v1_path)
    assert v1 is not None
    in_train = [g for g in range(14) if _split(v1.assignments, f"old/g{g}") == "train"][:3]
    assert len(in_train) == 3
    moved = {f"old/g{g}" for g in in_train}
    pinned = [spec for spec in old if spec[1] in moved]
    new = _rows("new", "ev", 4)
    new_root = _stage(tmp_path / "new-stage", new)
    pins = BenchmarkPins(by_sha={sha: ("test", "bench") for sha in _shas(pinned + new)})
    sources = {"old": _source("old"), "new": _source("new")}
    roots = {"old": old_root, "new": new_root}
    inherit = {"inherit": v1, "inherit_sha256": _sha(v1_path.read_bytes())}
    _map(tmp_path / "v2-plain", sources, roots, **inherit)
    stats, assignments = _map(tmp_path / "v2", sources, roots, **inherit, benchmark_pins=pins)
    assert {_split(assignments, group) for group in moved} == {"test"}  # the pin wins, group whole
    assert {_split(assignments, f"new/ev{g}") for g in range(4)} == {"test"}  # fresh pinned groups
    over = stats.pins["v1_continuity_overridden"]
    assert (over["images"], over["groups"]) == (6, 3)
    assert over["by_source"] == {"old": {"train -> test": 6}} and over["group_ids"] == sorted(moved)
    assert stats.pins["skipped_components"] == {"ood": 0, "earlier_rule": 0}
    assert "v1_continuity_conflicts" not in stats.pins
    v2 = load_split_map(tmp_path / "v2" / "SPLIT_MAP.json")
    assert v2 is not None and v2.pins == stats.pins
    changed = {g for g, s in v1.assignments.items() if v2.assignments[g] != s}
    assert changed == moved  # nothing else left its v1.0 split
    plain_path = tmp_path / "v2-plain" / "SPLIT_MAP.json"
    plain = load_split_map(plain_path)
    assert plain is not None and plain.pins == {}
    assert all(plain.assignments[g] == s for g, s in v1.assignments.items())  # unpinned: unchanged
    save_split_map(tmp_path / "resaved-v2.json", plain)  # load/save is lossless for inherited maps
    assert (tmp_path / "resaved-v2.json").read_bytes() == plain_path.read_bytes()
    resaved = tmp_path / "resaved-pinned.json"
    save_split_map(resaved, v2)  # ... and for a map carrying the overridden block
    assert load_split_map(resaved) == v2
