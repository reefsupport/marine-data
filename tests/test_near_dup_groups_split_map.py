"""WP-R6: the global split map consumes the dedup near-dup groups.

A near-duplicate pair spanning an open and an nc source gets one split, ``dedup gate`` over both
flavour builds finds no spanning group, and ``release split-map`` refuses to run without either
``--near-dup <groups>`` or an explicit ``--no-near-dup`` (recorded in the map header).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from test_global_split_map_flavours import NOW, _registry, _rows, _src
from test_upstream_test_split import _stage

from marinedata.cli import main
from marinedata.dedup.groups import load_groups, read_release_rows, run_gate
from marinedata.neardup import NearDupConfig
from marinedata.release import build_release, dedup_group_links, generate_split_map
from marinedata.splitmap import load_split_map

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

OPEN_IMG = b"near-dup-image-in-the-open-source"
NC_IMG = b"near-dup-image-in-the-nc-source-reencoded"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    registry = _registry([_src("src-open", "open"), _src("src-nc", "restricted-nc")])
    # the open copy sits in a big group (placed first, into train); the nc copy is a lone small
    # group that lands elsewhere for some seeds: only the dedup group ties them together
    fill = [(f"ndf{k}", "src-open/nd", "train", f"open-fill-{k}".encode()) for k in range(12)]
    open_rows = [
        *_rows("src-open", b"open-shared"),
        ("nd", "src-open/nd", "train", OPEN_IMG),
        *fill,
    ]
    nc_rows = [*_rows("src-nc", b"nc-shared"), ("nd", "src-nc/nd", "train", NC_IMG)]
    roots = {
        "src-open": _stage(tmp_path / "open", open_rows),
        "src-nc": _stage(tmp_path / "nc", nc_rows),
    }
    return registry, roots


def _groups_parquet(path: Path, members: dict[str, str]) -> Path:
    pq.write_table(
        pa.table(
            {
                "sha256": list(members),
                "split_group_id": list(members.values()),
                "group_upstream_splits": [[] for _ in members],
            }
        ),
        path,
    )
    return path


def _map(tmp_path: Path, registry, roots, seed: int, **kw):  # type: ignore[no-untyped-def]
    out = tmp_path / f"M{seed}-{len(kw)}.json"
    stats = generate_split_map(
        registry,
        out=out,
        roots=roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=seed,
        min_groups=3,
        **kw,
    )
    split_map = load_split_map(out)
    assert split_map is not None
    return out, stats, split_map


def test_dedup_group_links_chain_members_to_the_first_present_image() -> None:
    groups = {"c": "g1", "a": "g1", "b": "g1", "x": "g2", "y": "g3"}
    assert dedup_group_links(groups, ["c", "b", "a", "x", "z"]) == [("a", "b"), ("a", "c")]


def test_near_dup_pair_across_open_and_nc_gets_one_split_and_passes_the_gate(
    tmp_path: Path,
) -> None:
    registry, roots = _setup(tmp_path)
    # control: without the dedup groups, some seed splits the pair (so the test is sensitive)
    split_apart = [
        seed
        for seed in range(24)
        if (m := _map(tmp_path / "ctl", registry, roots, seed)[2]).assignments["src-open/nd"]
        != m.assignments["src-nc/nd"]
    ]
    assert split_apart, "fixture never separates the pair; the test would prove nothing"
    seed = split_apart[0]

    members = {_sha(OPEN_IMG): "dg-1", _sha(NC_IMG): "dg-1"}
    groups = _groups_parquet(tmp_path / "groups.parquet", members)
    out, stats, split_map = _map(
        tmp_path,
        registry,
        roots,
        seed,
        near_dup_groups=members,
        near_dup_header={"dedup_groups": {"file": groups.name}},
    )
    assert stats.dedup_group_links == 1
    assert split_map.assignments["src-open/nd"] == split_map.assignments["src-nc/nd"]
    assert split_map.near_dup["dedup_groups"] == {"file": "groups.parquet"}

    rel = tmp_path / "rel"
    for flavour, profile in (("open", "ship-open"), ("nc", "ship-noncommercial")):
        build_release(
            registry,
            release="r1",
            split_map=out,
            roots=roots,
            out_dir=rel,
            profile=profile,
            flavour=flavour,
            allow_unmapped=True,
        )
    rows = [
        row
        for flavour in ("open", "nc")
        for row in read_release_rows(rel / "releases" / "r1" / flavour)
    ]
    by_sha = dict(rows)
    assert by_sha[_sha(OPEN_IMG)] == by_sha[_sha(NC_IMG)]
    sha_to_group, upstream = load_groups(groups)
    gate = run_gate(rows, sha_to_group, upstream, allow_ungrouped=True)
    assert gate.spanning == {} and gate.ok


def _cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
    registry, roots = _setup(tmp_path)
    monkeypatch.setattr("marinedata.cli_release.Registry.load", lambda *a, **k: registry)
    monkeypatch.setattr("marinedata.cli_release.SPLIT_MAP_PROFILE", "ship-noncommercial")
    # the fixture images are not decodable pictures: give dHash a stable fake per digest, and a
    # chain guard that tolerates a 2-image component among 52 images
    monkeypatch.setattr(
        "marinedata.cli_release.NearDupConfig",
        lambda **kw: NearDupConfig(chain_fraction=1.0, **kw),
    )
    monkeypatch.setattr(
        "marinedata.release.compute_dhashes",
        lambda paths, **kw: {sha: int(sha[:16], 16) for sha in paths},
    )
    argv = ["release", "split-map", "--release", "r1", "--local-only", "--min-groups", "3"]
    argv += ["--split-map", str(tmp_path / "M.json"), *extra]
    for sid, root in roots.items():
        argv += ["--local", f"{sid}={root}"]
    return main(argv)


def test_split_map_cli_fails_without_near_dup_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli(tmp_path, monkeypatch) == 1
    assert "--near-dup" in capsys.readouterr().err
    assert load_split_map(tmp_path / "M.json") is None


def test_split_map_cli_no_near_dup_is_recorded_in_the_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _cli(tmp_path, monkeypatch, "--no-near-dup") == 0
    written = load_split_map(tmp_path / "M.json")
    assert written is not None
    assert written.near_dup["dedup_groups"] == "disabled by --no-near-dup"


def test_split_map_cli_near_dup_groups_are_consumed_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = {_sha(OPEN_IMG): "dg-1", _sha(NC_IMG): "dg-1"}
    groups = _groups_parquet(tmp_path / "groups.parquet", members)
    assert _cli(tmp_path, monkeypatch, "--near-dup", str(groups)) == 0
    written = load_split_map(tmp_path / "M.json")
    assert written is not None
    assert written.assignments["src-open/nd"] == written.assignments["src-nc/nd"]
    header = written.near_dup["dedup_groups"]
    assert isinstance(header, dict) and header["file"] == "groups.parquet" and header["images"] == 2
