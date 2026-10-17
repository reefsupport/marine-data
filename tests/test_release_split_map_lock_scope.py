"""``release split-map --inherit --lock``: a registry source the lock does not name is never
evaluated, fetched or hashed, and the inherited v1.0 groups are kept verbatim (PR-B2).

The spies sit on the four places a source could be touched: the release-scope filter, the fetch,
the per-source enumeration (where the image digests are computed) and the dHash. A run without
``--lock`` is the control: it must trip every spy for the source the lock leaves out."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_global_split_map_flavours import NOW, _registry, _src
from test_upstream_test_split import _stage

import marinedata.cli_release as cli_release
import marinedata.release.entries as entries
from marinedata.cli import main
from marinedata.neardup import NearDupConfig
from marinedata.registry import Registry
from marinedata.splitmap import SplitMap, load_split_map, save_split_map

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
LOCKED = ("kept-a", "kept-b")
ABSENT = "absent"
PRIOR = {
    **{f"kept-a/g{g}": s for g, s in enumerate(["train", "train", "val", "test", "train", "val"])},
    f"{ABSENT}/g0": "test",  # a v1.0 group of a source the v1.1 lock does not name
    f"{ABSENT}/g1": "val",
}


@pytest.fixture(autouse=True)
def _no_phash_decode(monkeypatch):  # type: ignore[no-untyped-def]
    """The staged ``.img`` files are not decodable (the test fakes their dHashes), so the gates'
    pHash rule gets no links here; ``tests/test_neardup_phash.py`` stages real images."""
    module = importlib.import_module("marinedata.release.split_map")
    monkeypatch.setattr(module, "phash_pairs", lambda *args, **kwargs: [])


def _rows(source_id: str) -> list:  # type: ignore[type-arg]
    return [
        (f"g{g}{k}", f"{source_id}/g{g}", "train", f"img-{source_id}-{g}-{k}".encode())
        for g in range(6)
        for k in "ab"
    ]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    ids = (*LOCKED, ABSENT)
    registry = _registry([_src(sid, "open") for sid in ids])
    trees = {sid: _stage(tmp_path / "trees" / sid, _rows(sid)) for sid in ids}
    prior_path = tmp_path / "SPLIT_MAP.v1.json"
    save_split_map(
        prior_path,
        SplitMap(
            by="group", seed=0, ratios=dict(RATIOS), assignments=dict(PRIOR), generated_at=NOW,
            release="v1.0.0", stratify="source",
        ),
    )  # fmt: skip
    seen = SimpleNamespace(evaluated=[], fetched=[], enumerated=[], hashed=[])

    def spy(module, name, pick):  # type: ignore[no-untyped-def]
        real = getattr(module, name)

        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            seen.evaluated.append(pick(*args))
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    spy(cli_release, "_unfetchable_reason", lambda _reg, _prof, source, *_: source.id)
    spy(entries, "evaluate", lambda source, *_: source.id)

    real_entries = entries.source_release_entries

    def source_release_entries(source, root, digest):  # type: ignore[no-untyped-def]
        seen.enumerated.append(source.id)
        return real_entries(source, root, digest)

    def fetch_sample(source, *, limit):  # type: ignore[no-untyped-def]
        seen.fetched.append(source.id)
        return SimpleNamespace(root=trees[source.id])

    def compute_dhashes(paths, *, cache_dir, workers=1):  # type: ignore[no-untyped-def]
        seen.hashed.extend(Path(p) for p in paths.values())
        return {sha: int.from_bytes(hashlib.sha256(Path(p).read_bytes()).digest()[:8], "big")
                for sha, p in paths.items()}  # fmt: skip

    monkeypatch.setattr(entries, "source_release_entries", source_release_entries)
    monkeypatch.setattr(cli_release, "fetch_sample", fetch_sample)
    monkeypatch.setattr("marinedata.release.split_map.compute_dhashes", compute_dhashes)
    monkeypatch.setattr(cli_release, "_is_pinned_staged_tree", lambda source: True)
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: registry))
    monkeypatch.setattr(
        cli_release, "NearDupConfig", lambda **kw: NearDupConfig(**kw, chain_fraction=1.0)
    )
    monkeypatch.setattr(
        "marinedata.cli_release_inherit.load_lock",
        lambda path, registry: SimpleNamespace(source_ids=lambda: set(LOCKED)),
    )
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))

    def run(name: str, *extra: str) -> SplitMap:
        out = tmp_path / name / "SPLIT_MAP.json"
        argv = ["release", "split-map", "--release", "v1.1.0", "--inherit", str(prior_path)]
        assert (
            main([*argv, "--seed", "11", "--out", str(out), "--no-ood", "--no-near-dup", *extra])
            == 0
        )
        got = load_split_map(out)
        assert got is not None
        return got

    return SimpleNamespace(run=run, seen=seen, absent_tree=trees[ABSENT])


def test_a_source_the_lock_does_not_name_is_never_evaluated_fetched_or_hashed(
    world: SimpleNamespace,
) -> None:
    got = world.run("scoped", "--lock", "open.lock.yaml")
    seen = world.seen
    assert world.absent_tree not in {p for path in seen.hashed for p in path.parents}
    assert set(seen.fetched) == set(LOCKED) and ABSENT not in seen.fetched
    assert ABSENT not in seen.evaluated and ABSENT not in seen.enumerated
    assert seen.hashed  # the locked sources were hashed: the spy is live
    # the inherited v1.0 groups are kept verbatim, the absent source's included ...
    assert {g: got.assignments[g] for g in PRIOR} == PRIOR
    assert got.inherits["release"] == "v1.0.0"
    # ... and nothing of the absent source beyond them is allocated
    assert {g for g in got.assignments if g.startswith(f"{ABSENT}/")} == {
        g for g in PRIOR if g.startswith(f"{ABSENT}/")
    }
    assert {g for g in got.assignments if g.startswith("kept-b/")}  # the locked new source is


def test_without_a_lock_the_same_source_trips_every_spy(world: SimpleNamespace) -> None:
    got = world.run("unscoped")
    seen = world.seen
    assert ABSENT in seen.fetched and ABSENT in seen.evaluated and ABSENT in seen.enumerated
    assert world.absent_tree in {p for path in seen.hashed for p in path.parents}
    assert f"{ABSENT}/g5" in got.assignments
    assert {g: got.assignments[g] for g in PRIOR} == PRIOR
