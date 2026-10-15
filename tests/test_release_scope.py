"""The release scope is a registry fact: lock, lock check and fetch share one predicate (PR-F2).

A pinned source the registry marks ``release_skip_reason`` ("deferred to v1.2: ...") is outside
the release. ``release lock`` does not write it as admitted (it lists it under ``excluded``),
``check_build_against_lock`` does not demand it and ``_resolve_roots`` never fetches it. The lock
gate is not relaxed: an *admitted* source the lock leaves out still fails the check.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_release_build_lock_scope import FIXED, IDS, _registry3, _stage

import marinedata.cli_release as cli_release
from marinedata.models import Source
from marinedata.registry import Registry
from marinedata.release_lock import (
    ReleaseLock,
    ReleaseLockError,
    build_lock,
    check_build_against_lock,
    load_lock,
    write_lock,
)
from marinedata.release_scope import release_scope, scope_problem

SKIPPED = "lock-c"
REASON = "deferred to v1.2: test fixture"
ADMITTED = ["lock-a", "lock-b"]
DEFERRED = (  # pinned, shippable, and not in the v1.1.0 fetch list (vm-build-plan 4.3)
    "marineevt",
    "koi-rgb-sonar",
    "auv-flc-fls-sunboat",
    "noaa-pifsc-esa-coral-icra",
    "salmon-cage",
    "mermaid-aws",
    "coralvqa",
    "flsea",
    "atlantis-synthetic-depth",
)
V1_0_LOCK = Path(__file__).resolve().parents[1] / "releases/marine-data/v1.0/release.lock.yaml"


def _scoped(skip: tuple[str, ...] = (SKIPPED,)) -> Registry:
    """The lock-scope tests' three pinned open sources; the ``skip`` ones get a registry reason."""
    base = _registry3()
    return type(base)(
        sources={
            s.id: s.model_copy(update={"release_skip_reason": REASON}) if s.id in skip else s
            for s in base
        },
        licences={},
        profiles={p.id: p for p in base.profiles},
        schemas={s.id: s for s in base.schemas},
        tasks={t.id: t for t in base.tasks},
    )


def _lock(registry: Registry, **kwargs: object) -> ReleaseLock:
    return build_lock(
        registry, release_id="t1", flavour="open", configs=["coral-masks"], **FIXED, **kwargs
    )


def _check(lock: ReleaseLock, registry: Registry, local: tuple[str, ...] = ()) -> None:
    check_build_against_lock(
        lock, registry, release_id="t1", flavour="open", profile="ship-open", local_ids=local
    )


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """``fetch_sample`` replaced by a recorder: the ids it was called with, in order."""
    calls: list[str] = []

    def _fetch(source: Source, *, limit: int) -> SimpleNamespace:
        calls.append(source.id)
        return SimpleNamespace(root=_stage(tmp_path / "fetched" / source.id, source.id))

    monkeypatch.setattr(cli_release, "fetch_sample", _fetch)
    return calls


# -- the shared predicate ---------------------------------------------------------------------


def test_scope_is_the_admitted_pinned_trees_minus_the_registry_skips() -> None:
    admitted, excluded = release_scope(_scoped(), "open")
    assert admitted == ADMITTED
    assert excluded == {SKIPPED: REASON}


def test_the_registry_reason_is_named_before_a_missing_tree() -> None:
    registry = _scoped()
    prof = registry.profile("ship-open")
    why = scope_problem(registry, prof, registry.source(SKIPPED), "open", rooted=False)
    assert why == f"left out of the release by the registry: {REASON}"
    assert scope_problem(registry, prof, registry.source("lock-a"), "open", rooted=True) is None
    assert "no pinned staged tree" in str(
        scope_problem(registry, prof, registry.source("lock-a"), "open", rooted=False)
    )


# -- fetch ------------------------------------------------------------------------------------


def test_a_skipped_source_is_never_fetched(fetched: list[str]) -> None:
    roots = cli_release._resolve_roots(_scoped(), "ship-open", {}, "open")
    assert fetched == ADMITTED
    assert set(roots) == set(ADMITTED)


def test_a_lock_cannot_pull_a_skipped_source_into_the_fetch(fetched: list[str]) -> None:
    reason = r"lock-c \(left out of the release by the registry: deferred to v1\.2"
    with pytest.raises(ReleaseLockError, match=reason):
        cli_release._resolve_roots(_scoped(), "ship-open", {}, "open", only={"lock-a", SKIPPED})
    assert fetched == []


# -- lock -------------------------------------------------------------------------------------


def test_the_default_lock_admits_the_scope_and_records_what_was_left_out() -> None:
    lock = _lock(_scoped())
    assert sorted(lock.source_ids()) == ADMITTED
    assert lock.excluded == {SKIPPED: REASON}


def test_naming_a_skipped_source_when_writing_the_lock_fails() -> None:
    with pytest.raises(ReleaseLockError, match=r"leaves these out of the release.*lock-c"):
        _lock(_scoped(), source_ids=["lock-a", SKIPPED])


def test_excluded_is_written_only_when_non_empty_and_round_trips(tmp_path: Path) -> None:
    plain = write_lock(_lock(_registry3()), tmp_path / "plain.lock.yaml")
    assert "excluded" not in yaml.safe_load(plain.read_text(encoding="utf-8"))
    registry = _scoped()
    scoped = write_lock(_lock(registry), tmp_path / "scoped.lock.yaml")
    assert yaml.safe_load(scoped.read_text(encoding="utf-8"))["excluded"] == {SKIPPED: REASON}
    assert load_lock(scoped, registry).excluded == {SKIPPED: REASON}


def test_a_lock_listing_a_source_as_locked_and_excluded_is_refused(tmp_path: Path) -> None:
    registry = _scoped()
    both = _lock(registry).model_copy(update={"excluded": {"lock-a": "x"}})
    bad = write_lock(both, tmp_path / "both.lock.yaml")
    with pytest.raises(ReleaseLockError, match="both locked and excluded: lock-a"):
        load_lock(bad, registry)


# -- lock check -------------------------------------------------------------------------------


def test_the_lock_check_does_not_demand_a_skipped_source() -> None:
    registry = _scoped()
    _check(_lock(registry), registry)
    _check(_lock(registry), registry, local=(SKIPPED,))  # not even when a local tree is given


def test_an_admitted_source_missing_from_the_lock_still_fails() -> None:
    registry = _scoped()
    lock = _lock(registry, source_ids=["lock-a"])  # lock-b: admitted, pinned, left out
    with pytest.raises(ReleaseLockError, match="source not in the release lock: lock-b"):
        _check(lock, registry)


def test_a_local_tree_for_an_unpinned_source_is_still_demanded() -> None:
    registry = _registry3(pinned=("lock-a", "lock-b"))
    lock = _lock(registry, source_ids=["lock-a", "lock-b"])
    _check(lock, registry)
    with pytest.raises(ReleaseLockError, match="source not in the release lock: lock-c"):
        _check(lock, registry, local=("lock-c",))


def test_lock_check_and_fetch_agree_on_the_scope(fetched: list[str]) -> None:
    registry = _scoped()
    admitted, _ = release_scope(registry, "open")
    lock = _lock(registry)
    _check(lock, registry)
    roots = cli_release._resolve_roots(registry, "ship-open", {}, "open", only=lock.source_ids())
    assert sorted(roots) == sorted(lock.source_ids()) == sorted(admitted) == sorted(fetched)
    assert set(IDS) - set(roots) == {SKIPPED}


# -- the real registry ------------------------------------------------------------------------


@pytest.mark.parametrize("flavour", ["open", "nc"])
def test_the_deferred_sources_are_outside_the_real_scope(flavour: str) -> None:
    registry = Registry.load()
    admitted, _ = release_scope(registry, flavour)
    assert not set(admitted) & set(DEFERRED)
    for source_id in DEFERRED:
        assert (registry.source(source_id).release_skip_reason or "").startswith(
            "deferred to v1.2: "
        )


def test_the_v1_0_open_trees_stay_in_the_real_scope() -> None:
    registry = Registry.load()
    admitted, _ = release_scope(registry, "open")
    assert load_lock(V1_0_LOCK, registry).source_ids() <= set(admitted)
