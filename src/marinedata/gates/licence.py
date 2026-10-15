"""Licence gate (P1.6): nothing ships in a flavour its licence does not allow.

* ``source_class``  a lock source (``RELEASE.json`` ``lock_sources``) whose ``access_class`` the
  flavour cannot ship (open ships ``open``, nc ships ``restricted-nc``; a per-row-licence source
  is admitted to either flavour unless it is internal-only or unknown, its rows then decide);
* ``no_lock``       RELEASE.json has no ``lock_sources`` (a build always has a lock);
* ``source_not_in_lock`` a row of a source the lock does not list;
* ``null_licence``  a ``license``/``licence`` column that is null or blank;
* ``row_licence_outside_flavour`` a row licence stricter than the flavour (or unparseable);
* ``row_class_outside_flavour``   a ``licence_class`` that is not the flavour's class;
* ``empty_attribution`` an ``attribution`` column that is null or blank;
* ``frozen_exception_new_row`` a row of a flavour's frozen-exception source (D3, ``flavours.yaml``)
  in a shard whose sha256 is not on that exception's allowlist. The allowlist is files, so it can
  never admit a new row; an allowlisted shard is the byte-identical published v1.0 file and its
  rows skip the checks below. A lock entry for such a source is accepted only as the carried v1.0
  entry.

Row checks run on every config that carries one of those columns (``metadata``, the mask
configs). A build with no flavour (none given, none in RELEASE.json) fails."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from ..flavours import FrozenException, frozen_exception_map
from ..licence_class import (
    FLAVOURS,
    INTERNAL_ONLY,
    NC,
    ND,
    OPEN,
    UNKNOWN,
    coerce_class,
    licence_class_of,
)
from . import GateResult, finish
from .shards import config_shards, iter_rows, read_release, shard_columns

NAME = "licence"
_RANK = {OPEN: 0, NC: 1, ND: 2, INTERNAL_ONLY: 3, UNKNOWN: 3}
_LICENCE = ("license", "licence")
_COLUMNS = (*_LICENCE, "licence_class", "attribution", "source_id", "source", "image_sha256")


def _load_registry():
    try:
        from ..registry import Registry

        return Registry.load()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _per_row(registry, source_id: str) -> bool:
    try:
        return bool(registry.source(source_id).licence_per_row)
    except Exception:
        return False


def run_licence_gate(
    build_dir: Path,
    *,
    flavour: str | None = None,
    release_dir: Path | None = None,
    registry=None,
    frozen: Mapping[str, FrozenException] | None = None,
) -> GateResult:
    build_dir = Path(build_dir)
    release = read_release(build_dir, release_dir)
    flavour = flavour or (release or {}).get("flavour")
    if flavour not in FLAVOURS:
        msg = (
            f"no_flavour: pass --flavour open|nc or a RELEASE.json that names one (got {flavour!r})"
        )
        return finish(NAME, {}, [msg])
    want = FLAVOURS[flavour]
    registry = registry if registry is not None else _load_registry()
    frozen = frozen_exception_map(flavour) if frozen is None else dict(frozen)
    digests: dict[Path, str] = {}
    counts: Counter = Counter()
    failures: list[str] = []

    def fail(kind: str, where: str, message: str) -> None:
        counts[f"failed_{kind}"] += 1
        failures.append(f"{where} {kind}: {message}")

    sources = (release or {}).get("lock_sources") or []
    if not sources:
        fail("no_lock", "RELEASE.json", "no lock_sources; the source classes cannot be checked")
    lock_ids = {str(s.get("id")) for s in sources}
    for entry in sources:
        sid, cls = str(entry.get("id")), coerce_class(entry.get("access_class"))
        counts["lock_sources"] += 1
        shippable = cls == want or (cls not in (INTERNAL_ONLY, UNKNOWN) and _per_row(registry, sid))
        if not shippable and sid in frozen:
            counts["frozen_exception_sources"] += (
                1  # carried v1.0 entry; its shards are checked below
            )
        elif not shippable:
            fail("source_class", sid, f"access_class {cls!r} cannot ship in flavour {flavour}")
    for config, files in config_shards(build_dir).items():
        names = set(shard_columns(files[0]))
        if not names & {*_LICENCE, "licence_class", "attribution"}:
            continue
        counts["configs_checked"] += 1
        for shard, _, split, row in iter_rows(files, _COLUMNS):
            counts["rows_checked"] += 1
            where = f"{config}/{split} {str(row.get('image_sha256'))[:12]}"
            sid = row.get("source_id") or row.get("source")
            exception = frozen.get(str(sid)) if sid is not None else None
            if exception is not None:
                digests.setdefault(shard, _sha256(shard))
                if digests[shard] in exception.files_sha256:
                    counts["frozen_rows_allowed"] += 1
                else:
                    fail(
                        "frozen_exception_new_row",
                        where,
                        f"source {sid!r} is a frozen v1.0 exception; "
                        f"{shard.name} is not an allowlisted v1.0 shard",
                    )
                continue
            if sources and sid is not None and str(sid) not in lock_ids:
                fail("source_not_in_lock", where, f"source {sid!r} is not in the lock")
            for column in (c for c in _LICENCE if c in names):
                text = row.get(column)
                if text is None or not str(text).strip():
                    fail("null_licence", where, f"{column} is empty")
                elif _RANK[licence_class_of(text)] > _RANK[want]:
                    cls = licence_class_of(text)
                    fail("row_licence_outside_flavour", where, f"{column} {text!r} is {cls}")
            if "licence_class" in names and coerce_class(row.get("licence_class")) != want:
                got = coerce_class(row.get("licence_class"))
                fail(
                    "row_class_outside_flavour",
                    where,
                    f"licence_class {got!r}, flavour ships {want}",
                )
            if "attribution" in names and not str(row.get("attribution") or "").strip():
                fail("empty_attribution", where, "attribution is empty")
    return finish(NAME, {**counts, "flavour": flavour}, failures)
