"""D1/D2 end to end: ``generate_split_map`` and ``marinedata release split-map`` with ``--inherit``.

dHashes are faked from the staged bytes (``near`` maps an image to a base image and the number of
bits it differs by), so a near-duplicate is exact and the images need no decoding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_global_split_map_flavours import NOW, _registry, _src
from test_upstream_test_split import _stage

import marinedata.cli_release as cli_release
from marinedata.checksums import file_digest
from marinedata.cli import build_parser
from marinedata.neardup import NearDupConfig
from marinedata.registry import Registry
from marinedata.release import generate_split_map
from marinedata.splitmap import SplitMap, load_split_map, save_split_map
from marinedata.splitmap_inherit import load_inherited
from marinedata.splitv2.release_ood import load_ood_metadata
from marinedata.splitv2.rules import load_rules

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_V2 = Path(__file__).resolve().parents[1] / "registry" / "splits" / "v2.yaml"


def _img(tag: str) -> bytes:
    return f"img-{tag}".encode()


def _hash_of(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _fake_dhashes(near: dict[bytes, tuple[bytes, int]]):  # type: ignore[no-untyped-def]
    def compute(paths, *, cache_dir, workers=1):  # type: ignore[no-untyped-def]
        out = {}
        for sha, path in paths.items():
            data = Path(path).read_bytes()
            base, bits = near.get(data, (data, 0))
            out[sha] = _hash_of(base) ^ sum(1 << b for b in range(bits))
        return out

    return compute


def _fake_confirm(near: dict[bytes, tuple[bytes, int]]):  # type: ignore[no-untyped-def]
    """The confirm-hash ``release split-map`` turns on: a near-duplicate shares its base's pHash."""

    def compute(paths, *, cache_dir, workers=1):  # type: ignore[no-untyped-def]
        out = {}
        for sha, path in paths.items():
            base = near.get(Path(path).read_bytes(), (Path(path).read_bytes(), 0))[0]
            out[sha] = (hashlib.sha256(base).digest(), 8.0)
        return out

    return compute


def _relaxed_rules(tmp_path: Path) -> Path:
    """The repository's splits/v2.yaml with the size guards scaled down to the fixture."""
    text = SPLIT_V2.read_text()
    for old, new in (
        ("min_images: 500", "min_images: 4"),
        ("  min_groups: 5", "  min_groups: 2"),
        ("max_holdout_share: 0.10", "max_holdout_share: 0.50"),
        ("max_total_holdout_share: 0.25", "max_total_holdout_share: 0.60"),
    ):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "v2.yaml"
    path.write_text(text)
    return path


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    near: dict[bytes, tuple[bytes, int]] = {}

    def near_of(base: str, bits: int, tag: str) -> bytes:
        near[_img(tag)] = (_img(base), bits)
        return _img(tag)

    old = [
        (f"g{g:02d}{k}", f"old/g{g:02d}", "train", _img(f"old-g{g:02d}-{k}"))
        for g in range(12)
        for k in "ab"
    ]
    new = [
        ("h00a", "new/h00", "train", _img("old-g00-a")),  # the same bytes as an old image
        ("h00b", "new/h00", "train", _img("new-h00-b")),
        ("h01a", "new/h01", "train", near_of("old-g01-a", 3, "new-h01-a")),
        ("h02a", "new/h02", "train", near_of("old-g02-a", 2, "new-h02-a")),
        ("h02b", "new/h02", "train", near_of("old-g04-a", 2, "new-h02-b")),
        ("h03a", "new/h03", "train", _img("new-h03-a")),
        ("h04a", "new/h04", "train", near_of("new-h03-a", 3, "new-h04-a")),
        ("h05a", "new/h05", "train", near_of("new-h03-a", 3, "new-h05-a")),
        ("h06a", "new/h06", "train", near_of("old-g06-a", 6, "new-h06-a")),
        ("h07a", "new/h07", "train", near_of("old-g07-a", 6, "new-h07-a")),
        ("h08a", "new/h08", "train", near_of("old-g08-a", 6, "new-h08-a")),
        ("h08b", "new/h08", "train", near_of("old-g09-a", 6, "new-h08-b")),
        *[(f"h{n}a", f"new/h{n}", "train", _img(f"new-h{n}-a")) for n in range(10, 20)],
        *[(f"h{n}b", f"new/h{n}", "train", _img(f"new-h{n}-b")) for n in range(10, 20)],
    ]
    deep = [
        (
            f"d{i}{k}",
            f"deepfish/d{i}",
            "test" if (i, k) == (0, "a") else "train",
            _img(f"deep-{i}-{k}"),
        )
        for i in range(4)
        for k in "ab"
    ]
    roots = {
        "src-old": _stage(tmp_path / "old", old),
        "src-new": _stage(tmp_path / "new", new),
        "deepfish": _stage(tmp_path / "deep", deep),
    }
    assignments = {f"old/g{g:02d}": "train" for g in range(12)}
    assignments.update({"old/g00": "val", "old/g01": "test", "old/g04": "val", "old/g05": "test"})
    assignments.update({"old/g07": "test", "old/g09": "val"})
    prior_path = tmp_path / "SPLIT_MAP.v1.json"
    save_split_map(
        prior_path,
        SplitMap(
            by="group",
            seed=0,
            ratios=dict(RATIOS),
            assignments=assignments,
            generated_at=NOW,
            release="v1.0.0",
            stratify="source",
        ),
    )
    prior, digest = load_inherited(prior_path)
    monkeypatch.setattr("marinedata.release.split_map.compute_dhashes", _fake_dhashes(near))
    monkeypatch.setattr(
        "marinedata.release.split_map.compute_confirm_features", _fake_confirm(near)
    )
    registry = _registry(
        [_src("src-old", "open"), _src("src-new", "open"), _src("deepfish", "open")]
    )
    return SimpleNamespace(
        registry=registry,
        roots=roots,
        prior=prior,
        prior_path=prior_path,
        digest=digest,
        rules_path=_relaxed_rules(tmp_path),
        near=NearDupConfig(cache_dir=tmp_path / "cache", chain_fraction=1.0),
    )


def _generate(world: SimpleNamespace, out: Path, **kwargs):  # type: ignore[no-untyped-def]
    return generate_split_map(
        world.registry,
        out=out,
        roots=world.roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=11,
        min_groups=3,
        near_dup=world.near,
        inherit=world.prior,
        inherit_sha256=world.digest,
        **kwargs,
    )


def test_inherited_map_keeps_v1_splits_groups_near_dups_and_routes_ood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, monkeypatch)
    out = tmp_path / "SPLIT_MAP.json"
    stats = _generate(world, out, ood_rules=load_rules(world.rules_path))
    got = load_split_map(out)
    assert got is not None
    a = got.assignments
    # (1) every v1.0 group keeps its split, whatever its new near-dup neighbours say
    assert all(a[group] == split for group, split in world.prior.assignments.items())
    assert a["new/h00"] == "val"  # same image digest as an old val image
    assert a["new/h01"] == "test"  # dHash 3 from an old test image
    assert (a["old/g02"], a["old/g04"]) == ("train", "val")  # bridged by new/h02: both kept
    assert "new/h02" not in a and "new/h02" in got.withheld
    # (2) a new near-duplicate group is one split
    assert len({a["new/h03"], a["new/h04"], a["new/h05"]}) == 1
    # the 5..8 band: near a train image is never-eval, near an evaluation image never-train
    assert a["new/h06"] == "train" and a["new/h07"] == "test"
    assert "new/h08" in got.withheld and "new/h08" not in a
    # (3) the splits/v2.yaml deepfish holdout is test_ood: it outranks the upstream test split
    deep = {f"deepfish/d{i}" for i in range(4)}
    assert {g for g, s in a.items() if s == "test_ood"} == deep
    assert all(got.ood_rules[g] == ["ood-source-deepfish"] for g in deep)
    assert got.ood["rules_sha256"] and got.ood["tropical_province"].startswith("D-P holdout: none")
    assert got.inherits == {
        "release": "v1.0.0",
        "seed": 0,
        "groups": 12,
        "sha256": world.digest,
    }
    assert (stats.inherited_groups, stats.ood_groups, stats.inherit_conflict_components) == (
        12,
        4,
        1,
    )
    assert stats.withheld_groups == 2 and stats.inherit_band_pairs >= 4


def test_ood_metadata_parquet_routes_groups_to_test_ood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, monkeypatch)
    shas = [
        file_digest(world.roots["src-new"] / "images" / "p" / f"h{n}{k}.img")
        for n in (10, 11)
        for k in "ab"
    ]
    meta = tmp_path / "ood-meta.parquet"
    pq.write_table(pa.table({"image_sha256": shas, "platform": ["auv"] * len(shas)}), meta)
    assert set(load_ood_metadata(meta)) == set(shas)
    out = tmp_path / "SPLIT_MAP.json"
    _generate(
        world,
        out,
        ood_rules=load_rules(world.rules_path),
        ood_metadata=load_ood_metadata(meta),
    )
    got = load_split_map(out)
    assert got is not None
    assert {g for g, s in got.assignments.items() if s == "test_ood"} >= {"new/h10", "new/h11"}
    assert got.ood_rules["new/h10"] == ["ood-platform-auv"]


def test_new_groups_hit_the_ratios_within_five_points(tmp_path: Path) -> None:
    sources = [f"src-n{k}" for k in range(3)]
    registry = _registry([_src("src-old", "open"), *(_src(s, "open") for s in sources)])
    roots = {"src-old": _stage(tmp_path / "old", [("o0", "old/g0", "train", _img("old-0"))])}
    for source in sources:
        rows = [
            (f"i{i:03d}", f"{source}/g{i:03d}", "train", _img(f"{source}-{i}")) for i in range(60)
        ]
        roots[source] = _stage(tmp_path / source, rows)
    prior_path = tmp_path / "v1.json"
    save_split_map(
        prior_path,
        SplitMap(
            by="group",
            seed=0,
            ratios=dict(RATIOS),
            assignments={"old/g0": "train"},
            stratify="source",
        ),
    )
    prior, digest = load_inherited(prior_path)
    out = tmp_path / "v11.json"
    generate_split_map(
        registry,
        out=out,
        roots=roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=5,
        inherit=prior,
        inherit_sha256=digest,
    )
    got = load_split_map(out)
    assert got is not None
    fresh = [s for g, s in got.assignments.items() if g != "old/g0"]
    assert len(fresh) == 180
    for split, target in RATIOS.items():
        assert abs(fresh.count(split) / len(fresh) - target) <= 0.05, split


def _argv(world: SimpleNamespace, out: Path, *extra: str) -> list[str]:
    return [
        "release", "split-map", "--release", "v1.1.0", "--lock", "open.lock.yaml",
        "--lock", "nc.lock.yaml", "--inherit", str(world.prior_path), "--seed", "11",
        "--out", str(out), "--ood-rules", str(world.rules_path), "--no-near-dup", *extra,
    ]  # fmt: skip


def test_cli_accepts_lock_inherit_seed_out_and_scopes_to_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def resolve_roots(registry, profile, local, flavour=None, *, only=None):  # type: ignore[no-untyped-def]
        seen["only"] = only
        return world.roots

    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: world.registry))
    monkeypatch.setattr(cli_release, "_resolve_roots", resolve_roots)
    monkeypatch.setattr(
        cli_release, "NearDupConfig", lambda **kw: NearDupConfig(**kw, chain_fraction=1.0)
    )
    monkeypatch.setattr(
        "marinedata.cli_release_inherit.load_lock",
        lambda path, registry: SimpleNamespace(
            source_ids=lambda: {"src-old", "src-new"} if "open" in str(path) else {"deepfish"}
        ),
    )
    out = tmp_path / "out" / "SPLIT_MAP.json"
    args = build_parser().parse_args(_argv(world, out))
    assert args.func(args) == 0
    assert seen["only"] == {"src-old", "src-new", "deepfish"}
    got = load_split_map(out)
    assert got is not None and got.seed == 11 and got.inherits["release"] == "v1.0.0"
    assert got.near_dup["union_confirm"]["phash256_max"] == 32  # the v1.1.0 rule B is recorded
    assert all(got.assignments[g] == s for g, s in world.prior.assignments.items())
    assert "test_ood" in got.assignments.values()
    again = build_parser().parse_args(_argv(world, out))
    assert again.func(again) == 1  # never overwritten


def test_cli_refuses_a_missing_prior_map_and_ood_flags_without_inherit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: world.registry))
    argv = _argv(world, tmp_path / "o.json")
    argv[argv.index("--inherit") + 1] = str(tmp_path / "missing.json")
    argv = [a for a in argv if a not in ("--lock", "open.lock.yaml", "nc.lock.yaml")]
    args = build_parser().parse_args(argv)
    assert args.func(args) == 1
    assert "does not exist" in capsys.readouterr().err
    no_inherit = [a for a in _argv(world, tmp_path / "o2.json") if a != "--inherit"]
    no_inherit = [a for a in no_inherit if a != str(world.prior_path)]
    no_inherit = [a for a in no_inherit if a not in ("--lock", "open.lock.yaml", "nc.lock.yaml")]
    args = build_parser().parse_args(no_inherit)
    assert args.func(args) == 1
    assert "--ood-rules" in capsys.readouterr().err
