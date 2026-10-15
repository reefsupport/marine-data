"""``release build --lock`` resolves and fetches only the lock's sources (PR-F).

``_resolve_roots`` used to walk the whole registry and fetch every admitted pinned staged tree
(~39 trees / 417 GiB on the open profile) whatever the lock named. With ``--lock`` it now skips,
and never fetches, a source the lock does not name, and fails closed, before the first fetch,
when a locked source cannot be resolved. Without a lock the behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_cli_release import _image_bytes, _registry, _source

import marinedata.cli_release as cli_release
from marinedata.cli import main
from marinedata.enums import AccessClass, Tier
from marinedata.models import Checksums, Licence, Source
from marinedata.registry import Registry
from marinedata.release_lock import ReleaseLockError, build_lock, write_lock
from marinedata.tables import StagedImage, write_metadata_table

LOCKED = "lock-b"
IDS = ("lock-a", LOCKED, "lock-c")  # registry order: the order an unscoped build fetches in
SEED = {"lock-a": 0, "lock-b": 100, "lock-c": 200, "lock-nc": 300}  # distinct images per source
FIXED = {"created": "2026-10-15T00:00:00+00:00", "code_sha": "0" * 40}


def _src(source_id: str, *, pinned: bool = True, nc: bool = False) -> Source:
    update: dict[str, object] = {"id": source_id, "name": source_id}
    if pinned:
        digest = hashlib.sha256(source_id.encode()).hexdigest()
        update["checksums"] = Checksums(version="v1", root_digest=digest, files=1, size_bytes=1)
    if nc:  # not admitted under ship-open
        update["access_class"] = AccessClass.RESTRICTED_NC
        update["licence"] = Licence(id="L-nc", name="nc", tier=Tier.NONCOMMERCIAL)
    return _source().model_copy(update=update)


def _registry3(*, pinned: tuple[str, ...] = IDS, with_nc: bool = False) -> Registry:
    """Three allowed open sources (`pinned` ones carry a root_digest), optionally an NC one."""
    base = _registry()
    sources = {sid: _src(sid, pinned=sid in pinned) for sid in IDS}
    if with_nc:
        sources["lock-nc"] = _src("lock-nc", nc=True)
    return type(base)(
        sources=sources,
        licences={},
        profiles={p.id: p for p in base.profiles},
        schemas={s.id: s for s in base.schemas},
        tasks={t.id: t for t in base.tasks},
    )


def _stage(root: Path, source_id: str) -> Path:
    rows = []
    for i in range(10):
        image = root / "images" / "p" / f"a{i}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(_image_bytes(SEED[source_id] + i))
        rows.append(
            StagedImage(
                stem=f"a{i}",
                partition="p",
                upstream_path=f"o/a{i}.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group=f"{source_id}/g{i}",
            )
        )
    write_metadata_table(root / "metadata.parquet", rows)
    return root


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """``fetch_sample`` replaced by a recorder: the ids it was called with, in order."""
    calls: list[str] = []

    def _fetch(source: Source, *, limit: int) -> SimpleNamespace:
        calls.append(source.id)
        return SimpleNamespace(root=_stage(tmp_path / "fetched" / source.id, source.id))

    monkeypatch.setattr(cli_release, "fetch_sample", _fetch)
    return calls


# -- _resolve_roots ---------------------------------------------------------------------------


def test_only_fetches_just_the_locked_source(fetched: list[str]) -> None:
    roots = cli_release._resolve_roots(_registry3(), "ship-open", {}, "open", only={LOCKED})
    assert fetched == [LOCKED]
    assert set(roots) == {LOCKED}


def test_without_only_every_admitted_source_is_fetched(fetched: list[str]) -> None:
    roots = cli_release._resolve_roots(_registry3(), "ship-open", {}, "open")
    assert fetched == list(IDS)
    assert set(roots) == set(IDS)


def test_a_locked_id_absent_from_the_registry_fails_naming_it_before_any_fetch(
    fetched: list[str],
) -> None:
    with pytest.raises(ReleaseLockError, match=r"ghost \(not in the registry\)"):
        cli_release._resolve_roots(_registry3(), "ship-open", {}, "open", only={LOCKED, "ghost"})
    assert fetched == []


def test_a_locked_source_the_filters_reject_fails_naming_it_before_any_fetch(
    fetched: list[str],
) -> None:
    """Never silently dropped: the NC source is not admitted under ship-open, and the unpinned
    one has no staged tree to fetch; both are named, and nothing is fetched for lock-b either."""
    registry = _registry3(pinned=(LOCKED, "lock-c"), with_nc=True)
    with pytest.raises(ReleaseLockError) as exc:
        cli_release._resolve_roots(
            registry, "ship-open", {}, "open", only={LOCKED, "lock-a", "lock-nc"}
        )
    message = str(exc.value)
    assert "lock-nc (not admitted under profile 'ship-open')" in message
    assert "lock-a (no pinned staged tree" in message
    assert LOCKED not in message
    assert fetched == []


def test_a_locked_source_with_a_local_tree_needs_no_fetch(
    fetched: list[str], tmp_path: Path
) -> None:
    tree = _stage(tmp_path / "mine", LOCKED)
    roots = cli_release._resolve_roots(
        _registry3(pinned=()), "ship-open", {LOCKED: tree}, "open", only={LOCKED}
    )
    assert roots == {LOCKED: tree}
    assert fetched == []


def test_an_empty_lock_fetches_nothing(fetched: list[str]) -> None:
    """``only=set()`` scopes the build to nothing; it is not the same as ``only=None``."""
    assert cli_release._resolve_roots(_registry3(), "ship-open", {}, "open", only=set()) == {}
    assert fetched == []


# -- release build --lock (through the CLI) ---------------------------------------------------


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fetched: list[str]) -> SimpleNamespace:
    """Three allowed sources, a frozen split map over all of them, and a lock naming lock-b.

    ``check_build_against_lock`` makes a lock cover every *pinned* admitted source, so a pinned
    source the lock leaves out cannot reach the fetch today. Only lock-b is therefore pinned and
    ``_is_pinned_staged_tree`` is patched to True for all three: the other two stand for
    "fetchable but not in the lock", which is what an unscoped build would fetch."""
    registry = _registry3(pinned=(LOCKED,))
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: registry))
    monkeypatch.setattr(cli_release, "_is_pinned_staged_tree", lambda source: True)
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))
    trees = {sid: _stage(tmp_path / "src" / sid, sid) for sid in IDS}
    split_map = tmp_path / "SPLIT_MAP.json"
    argv = ["release", "split-map", "--release", "t1", "--split-map", str(split_map)]
    argv += ["--no-near-dup", "--local-only"]
    assert main([*argv, *[f"--local={sid}={tree}" for sid, tree in trees.items()]]) == 0
    lock = write_lock(
        build_lock(
            registry,
            release_id="t1",
            flavour="open",
            configs=["coral-masks"],
            source_ids=[LOCKED],
            **FIXED,
        ),
        tmp_path / "release.lock.yaml",
    )

    def run(*extra: str) -> int:
        argv = ["release", "build", "--release", "t1", "--split-map", str(split_map)]
        argv += ["--out", str(tmp_path / "out"), "--flavour", "open", "--profile", "ship-open"]
        return main([*argv, *extra])

    return SimpleNamespace(run=run, lock=lock, trees=trees, calls=fetched)


def test_build_with_a_lock_fetches_only_the_locked_source(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert env.run("--lock", str(env.lock)) == 0
    assert env.calls == [LOCKED]
    assert "release=t1 sources=1 " in capsys.readouterr().out


def test_build_without_a_lock_still_fetches_every_admitted_source(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert env.run() == 0
    assert env.calls == list(IDS)
    assert "release=t1 sources=3 " in capsys.readouterr().out


def test_local_only_with_a_lock_never_fetches(env: SimpleNamespace) -> None:
    local = f"--local={LOCKED}={env.trees[LOCKED]}"
    assert env.run("--lock", str(env.lock), "--local-only", local) == 0
    assert env.calls == []


def test_build_fails_naming_a_locked_id_absent_from_the_registry(
    env: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(env.lock.read_text(encoding="utf-8"))
    data["sources"].append({**data["sources"][0], "id": "ghost"})
    bad = tmp_path / "ghost.lock.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert env.run("--lock", str(bad)) != 0
    assert "ghost" in capsys.readouterr().err
    assert env.calls == []


def test_build_fails_naming_a_locked_source_that_cannot_be_fetched(
    env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_release, "_is_pinned_staged_tree", lambda source: False)
    assert env.run("--lock", str(env.lock)) == 1
    assert f"{LOCKED} (no pinned staged tree" in capsys.readouterr().err
    assert env.calls == []
