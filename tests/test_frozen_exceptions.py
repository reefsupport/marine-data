"""R5 / D3: the two nc v1.0 sources stay published as files and can never admit a new row."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from functools import cache

import pytest
import yaml
from _gates_build import NC_LICENCE, make_build

from marinedata.flavours import (
    FrozenException,
    flavour_source_ids,
    frozen_exception_problems,
    load_flavours,
    release_record,
    ships_in,
)
from marinedata.gates.licence import run_licence_gate
from marinedata.gates.shards import config_shards
from marinedata.registry import Registry

FROZEN = {"coralscop-masks-rs": "coral-masks-machine", "coralseg-ucsd-mosaics": "coral-masks"}


class _Source:
    licence_per_row = False


class _Registry:
    def source(self, source_id: str) -> _Source:
        return _Source()


def _flavours_yaml(tmp_path, *, open_extra: str = "", nc_extra: str = ""):
    (tmp_path / "flavours.yaml").write_text(
        "takedown_url: https://example.invalid/issues\nflavours:\n"
        f"  open:\n    repo_id: org/open\n    profile: ship-open\n{open_extra}"
        f"  nc:\n    repo_id: org/nc\n    profile: ship-noncommercial\n{nc_extra}"
    )
    return tmp_path


def _entry(**over) -> str:
    fields = {
        "source_id": "x",
        "config": "c",
        "registry_class": "internal-only",
        "reason": "because",
    } | over
    body = ", ".join(f"{k}: {v}" for k, v in fields.items())
    return f"    frozen_exceptions:\n      - {{{body}}}\n"


def test_nc_declares_the_two_v1_exceptions_and_open_none() -> None:
    flavours = load_flavours().flavours
    declared = {e.source_id: e.config for e in flavours["nc"].frozen_exceptions}
    assert declared == FROZEN
    assert flavours["open"].frozen_exceptions == ()
    assert all(
        e.registry_class == "internal-only" and e.reason for e in flavours["nc"].frozen_exceptions
    )


def test_an_exception_never_admits_a_source() -> None:
    registry = Registry.load()
    for sid in FROZEN:
        assert not ships_in(registry, sid, "nc") and not ships_in(registry, sid, "open")
    for flavour in ("open", "nc"):
        assert not set(FROZEN) & set(
            flavour_source_ids(registry, [s.id for s in registry], flavour)
        )


def test_declared_classes_match_the_registry() -> None:
    registry = Registry.load()
    assert frozen_exception_problems(registry, "nc") == []
    assert frozen_exception_problems(registry, "open") == []


def test_release_record_lists_the_exceptions_for_nc_only() -> None:
    registry = Registry.load()
    nc = release_record(registry, "nc", "ship-noncommercial", [])
    assert {e["source_id"] for e in nc["frozen_exceptions"]} == set(FROZEN)
    assert "frozen_exceptions" not in release_record(registry, "open", "ship-open", [])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"open_extra": _entry()}, "cannot declare frozen_exceptions"),
        ({"nc_extra": _entry(registry_class="restricted-nc")}, "needs no exception"),
        ({"nc_extra": _entry(reason="''")}, "needs source_id, config and a reason"),
        ({"nc_extra": _entry(files_sha256="[abc]", shards_v1_0=1)}, "64-char hex"),
        ({"nc_extra": _entry(files_sha256=f"[{'a' * 64}]", shards_v1_0=2)}, "one per v1.0 shard"),
    ],
)
def test_the_loader_refuses_malformed_exceptions(tmp_path, kwargs, message) -> None:
    root = _flavours_yaml(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        load_flavours(root)


def _nc_build(tmp_path):
    sources = [
        {"id": "src-a", "access_class": "internal-only"},
        {"id": "src-b", "access_class": "restricted-nc"},
    ]
    return make_build(
        tmp_path, flavour="nc", licence=NC_LICENCE, licence_class="restricted-nc", sources=sources
    )


def _frozen(digests=()):
    return {
        "src-a": FrozenException(
            source_id="src-a",
            config="coral-masks",
            registry_class="internal-only",
            licence="NO-LICENCE-STATED",
            reason="test",
            shards_v1_0=len(digests),
            files_sha256=frozenset(digests),
        )
    }


def _run(build, **kw):
    return run_licence_gate(build, registry=_Registry(), **kw)


def test_gate_without_an_exception_refuses_the_internal_only_source(tmp_path) -> None:
    result = _run(_nc_build(tmp_path), frozen={})
    assert any("source_class:" in line for line in result.failures)


def test_an_empty_allowlist_accepts_the_lock_entry_but_no_row(tmp_path) -> None:
    result = _run(_nc_build(tmp_path), frozen=_frozen())
    assert not any("source_class:" in line for line in result.failures)
    assert result.counts["frozen_exception_sources"] == 1
    assert any("frozen_exception_new_row:" in line for line in result.failures)
    assert not result.passed


def test_an_allowlisted_v1_shard_passes_and_any_other_shard_fails(tmp_path) -> None:
    build = _nc_build(tmp_path)
    shards = [p for files in config_shards(build).values() for p in files]
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in shards]
    result = _run(build, frozen=_frozen(digests))
    assert result.passed, result.failures
    assert result.counts["frozen_rows_allowed"] > 0
    stale = _run(build, frozen=_frozen([hashlib.sha256(b"another file").hexdigest()]))
    assert not stale.passed and any("frozen_exception_new_row:" in f for f in stale.failures)


# -- digests of the published v1.0 shards (R5) --------------------------------------------------

SHARDS_V1_0 = {"coralscop-masks-rs": 21, "coralseg-ucsd-mosaics": 4}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST_LINE = re.compile(r"^\s+- ([0-9a-f]{64})\s+# (\S+)$")
_SHARD_PATH = re.compile(
    r"^data/(?P<config>[a-z-]+)/(?P<split>test|train|validation)-(?P<source>[a-z0-9-]+)"
    r"-(?P<i>\d{5})-of-(?P<n>\d{5})\.parquet$"
)


@cache
def _flavours_text() -> str:
    return (Registry.load().root / "flavours.yaml").read_text()


def _nc_entries() -> list[dict]:
    return yaml.safe_load(_flavours_text())["flavours"]["nc"]["frozen_exceptions"]


def _commented_shards() -> dict[str, list[tuple[str, str]]]:
    """(digest, shard path) per source, from the `- <digest>  # <path>` lines of flavours.yaml."""
    out: dict[str, list[tuple[str, str]]] = {}
    current = ""
    for line in _flavours_text().splitlines():
        if match := re.match(r"^\s+- source_id: (\S+)$", line):
            current = match.group(1)
            out[current] = []
        elif current and (match := _DIGEST_LINE.match(line)):
            out[current].append((match.group(1), match.group(2)))
    return out


def test_v1_allowlists_hold_one_lowercase_digest_per_published_shard() -> None:
    entries = _nc_entries()
    assert {e["source_id"]: len(e["files_sha256"]) for e in entries} == SHARDS_V1_0
    for entry in entries:
        digests = entry["files_sha256"]
        assert entry["shards_v1_0"] == len(digests), entry["source_id"]
        assert all(isinstance(d, str) and _HEX64.fullmatch(d) for d in digests), entry["source_id"]
        assert len(set(digests)) == len(digests), entry["source_id"]
    every = [d for e in entries for d in e["files_sha256"]]
    assert len(every) == len(set(every)) == sum(SHARDS_V1_0.values())


def test_every_published_v1_shard_has_a_digest() -> None:
    declared = {e.source_id: e for e in load_flavours().flavours["nc"].frozen_exceptions}
    commented = _commented_shards()
    assert {sid: len(pairs) for sid, pairs in commented.items()} == SHARDS_V1_0
    for source_id, pairs in commented.items():
        assert {d for d, _ in pairs} == declared[source_id].files_sha256
        splits: dict[str, list[tuple[int, int]]] = {}
        for _, path in pairs:
            match = _SHARD_PATH.match(path)
            assert match, path
            assert (match["config"], match["source"]) == (FROZEN[source_id], source_id), path
            splits.setdefault(match["split"], []).append((int(match["i"]), int(match["n"])))
        for split, found in splits.items():
            totals = {n for _, n in found}
            assert len(totals) == 1, (source_id, split)
            assert sorted(i for i, _ in found) == list(range(totals.pop())), (source_id, split)


def test_release_record_carries_the_v1_digests() -> None:
    record = release_record(Registry.load(), "nc", "ship-noncommercial", [])
    assert {
        e["source_id"]: len(e["files_sha256"]) for e in record["frozen_exceptions"]
    } == SHARDS_V1_0


@pytest.mark.parametrize("source_id", sorted(SHARDS_V1_0))
def test_the_real_allowlist_admits_its_shards_and_refuses_any_other(tmp_path, source_id) -> None:
    real = {e.source_id: e for e in load_flavours().flavours["nc"].frozen_exceptions}[source_id]
    build = _nc_build(tmp_path)
    shards = [p for files in config_shards(build).values() for p in files]
    listed = frozenset(hashlib.sha256(p.read_bytes()).hexdigest() for p in shards)
    outside = _run(build, frozen={"src-a": replace(real, source_id="src-a")})
    assert not outside.passed and any("frozen_exception_new_row:" in f for f in outside.failures)
    assert not outside.counts.get("frozen_rows_allowed")
    allowlist = real.files_sha256 | listed
    inside = _run(build, frozen={"src-a": replace(real, source_id="src-a", files_sha256=allowlist)})
    assert inside.passed, inside.failures
    assert inside.counts["frozen_rows_allowed"] > 0
