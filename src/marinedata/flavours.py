"""Release flavours (WP-L1b): ``open`` and ``nc`` are two HF repos, nothing duplicated.

The per-flavour repo id, shipping profile and the takedown URL live in
``registry/flavours.yaml``. This module resolves them and decides, per source, what a flavour
ships (:func:`flavour_source_ids`) and what RELEASE.json records about it
(:func:`release_record`). The per-row filter itself is ``licence_class.flavour_filter``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .licence_class import (
    ACCESS_CLASSES,
    FLAVOURS,
    INTERNAL_ONLY,
    NC,
    OPEN,
    UNKNOWN,
    coerce_class,
    release_excluded,
    resolve_row_class,
)

SPLIT_MAP_PROFILE = "ship-noncommercial"
"""A frozen split map is shared by both flavours, so it is generated over the superset
(every open + nc source), whichever flavour asked for it."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenException:
    """Published v1.0 rows of a source whose registry class the flavour does not allow (D3).

    The exception is about *files*: ``ships_in`` ignores it, a release adds no row from the
    source, and the licence gate raises on every row of the source that is not in a shard whose
    sha256 is in ``files_sha256``. It can never admit a source or a new row."""

    source_id: str
    config: str
    registry_class: str
    licence: str
    reason: str
    shards_v1_0: int
    files_sha256: frozenset[str]


@dataclass(frozen=True)
class FlavourSpec:
    flavour: str
    repo_id: str
    profile: str
    profiles: tuple[str, ...]
    frozen_exceptions: tuple[FrozenException, ...] = ()


@dataclass(frozen=True)
class FlavourConfig:
    takedown_url: str
    flavours: dict[str, FlavourSpec]


def _frozen_exceptions(flavour: str, raw: object) -> tuple[FrozenException, ...]:
    entries = list(raw or ())
    if not entries:
        return ()
    if flavour != "nc":
        raise ValueError(
            f"flavour {flavour!r} cannot declare frozen_exceptions: "
            "only nc carries published v1.0 rows "
            "of sources its class does not allow; open ships class `open` rows only"
        )
    out: list[FrozenException] = []
    for entry in entries:
        sid = str(entry.get("source_id") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not sid or not reason or not str(entry.get("config") or "").strip():
            raise ValueError("a frozen exception needs source_id, config and a reason")
        if sid in {e.source_id for e in out}:
            raise ValueError(f"frozen exception {sid!r} declared twice")
        declared = coerce_class(entry.get("registry_class"))
        if declared == FLAVOURS[flavour]:
            raise ValueError(
                f"{sid!r} is class {declared!r}, the flavour's own: it needs no exception"
            )
        digests = [str(d).strip().lower() for d in entry.get("files_sha256") or ()]
        bad = [d for d in digests if not _HEX64.match(d)]
        shards = int(entry.get("shards_v1_0") or 0)
        if bad or len(set(digests)) != len(digests) or (digests and len(digests) != shards):
            raise ValueError(
                f"{sid!r}: files_sha256 must be unique 64-char hex digests, "
                f"one per v1.0 shard ({shards})"
            )
        out.append(
            FrozenException(
                source_id=sid,
                config=str(entry["config"]),
                registry_class=declared,
                licence=str(entry.get("licence") or ""),
                reason=reason,
                shards_v1_0=shards,
                files_sha256=frozenset(digests),
            )
        )
    return tuple(out)


def load_flavours(root: str | Path | None = None) -> FlavourConfig:
    from .licence_class import _registry_root

    data = yaml.safe_load((_registry_root(root) / "flavours.yaml").read_text()) or {}
    specs = {
        name: FlavourSpec(
            flavour=name,
            repo_id=str(entry["repo_id"]),
            profile=str(entry["profile"]),
            profiles=tuple(entry.get("profiles") or [entry["profile"]]),
            frozen_exceptions=_frozen_exceptions(name, entry.get("frozen_exceptions")),
        )
        for name, entry in (data.get("flavours") or {}).items()
    }
    if set(specs) != set(FLAVOURS):
        raise ValueError(
            f"flavours.yaml must define exactly {sorted(FLAVOURS)}, got {sorted(specs)}"
        )
    return FlavourConfig(takedown_url=str(data["takedown_url"]), flavours=specs)


def flavour_spec(flavour: str, root: str | Path | None = None) -> FlavourSpec:
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown flavour {flavour!r}; expected one of {sorted(FLAVOURS)}")
    return load_flavours(root).flavours[flavour]


def frozen_exception_map(
    flavour: str, root: str | Path | None = None
) -> dict[str, FrozenException]:
    """``{source_id: exception}`` for the flavour (empty for open)."""
    return {e.source_id: e for e in flavour_spec(flavour, root).frozen_exceptions}


def frozen_exception_problems(registry, flavour: str, root: str | Path | None = None) -> list[str]:
    """Drift between the declared exceptions and the live registry (empty = consistent)."""
    problems: list[str] = []
    for exc in flavour_spec(flavour, root).frozen_exceptions:
        try:
            live = _class(registry, exc.source_id)
        except (KeyError, ValueError):
            problems.append(f"{exc.source_id}: not in the registry")
            continue
        if live == FLAVOURS[flavour]:
            problems.append(
                f"{exc.source_id}: now class {live!r}, the flavour's own; drop the exception"
            )
        elif live != exc.registry_class:
            problems.append(
                f"{exc.source_id}: declared {exc.registry_class!r}, the registry says {live!r}"
            )
    return problems


def takedown_url(root: str | Path | None = None) -> str:
    return load_flavours(root).takedown_url


def check_profile(flavour: str, profile: str, root: str | Path | None = None) -> None:
    """A flavour may only be built under one of its shipping profiles (never ``research``)."""
    spec = flavour_spec(flavour, root)
    if profile not in spec.profiles:
        raise ValueError(
            f"flavour {flavour!r} must be built under one of {list(spec.profiles)}, not {profile!r}"
        )


def _class(registry, source_id: str) -> str:
    return coerce_class(registry.source(source_id).access_class)


def ships_in(registry, source_id: str, flavour: str) -> bool:
    """A whole source ships in ``flavour`` when its class is the flavour's. A per-row-licence
    source is ADMITTED to either flavour (WP-R2): its class is only a bound, so
    :func:`sample_ships` decides row by row (unless the source is internal-only / unknown)."""
    if release_excluded(source_id):
        return False  # never released in any flavour (registry flag)
    source = registry.source(source_id)
    if source.licence_per_row:
        return _class(registry, source_id) not in (INTERNAL_ONLY, UNKNOWN)
    return _class(registry, source_id) == FLAVOURS[flavour]


def row_licence_class(registry, source_id: str, licence: str | None) -> str:
    """One row's class: the source class bound plus the row's own licence string.

    A per-row source with no (parseable) row licence is ``unknown`` and never ships."""
    source = registry.source(source_id)
    return resolve_row_class(
        _class(registry, source_id), licence, per_row=bool(source.licence_per_row)
    )


def sample_verdict(registry, sample, flavour: str | None) -> str | None:
    """Why one built sample may NOT ship in ``flavour`` (``None`` = it ships).

    ``licence-null``: the row carries a ``license`` that is null or blank (never ships, whatever
    the source). ``licence-outside-flavour``: a per-row-licence source's row class is not the
    flavour's (CC0/CC-BY -> open, BY-NC -> nc, ND/unknown -> dropped). Every other row was
    already decided whole by :func:`ships_in`; a research build (``flavour=None``) keeps all."""
    if flavour is None:
        return None
    meta = sample.meta or {}
    licence = meta.get("license")
    if "license" in meta and not (licence and str(licence).strip()):
        return "licence-null"
    if not registry.source(sample.source_id).licence_per_row:
        return None
    cls = row_licence_class(registry, sample.source_id, licence)
    return None if cls == FLAVOURS[flavour] else "licence-outside-flavour"


def sample_ships(registry, sample, flavour: str | None) -> bool:
    """Whether one built sample may ship in ``flavour`` (see :func:`sample_verdict`)."""
    return sample_verdict(registry, sample, flavour) is None


def flavour_source_ids(registry, source_ids: Iterable[str], flavour: str) -> list[str]:
    return [s for s in source_ids if ships_in(registry, s, flavour)]


def _reason(registry, source_id: str) -> str:
    """Why a source outside this flavour's class (or a per-row source) is not in it."""
    if registry.source(source_id).licence_per_row:
        return "per-row-licence"
    cls = _class(registry, source_id)
    return "other-flavour" if cls in (OPEN, NC) else f"class-{cls}"


def release_record(registry, flavour: str, profile: str, kept: Iterable[str]) -> dict:
    """What RELEASE.json says about a flavour build: the class counts and every registry
    source that is not in this release, grouped by why."""
    spec = flavour_spec(flavour)
    kept_set = set(kept)
    want = FLAVOURS[flavour]
    excluded: dict[str, list[str]] = {}
    for source in registry:
        if source.id in kept_set:
            continue
        if _class(registry, source.id) == want and not source.licence_per_row:
            reason = "not-in-release"  # right class, but held out by the gate or not staged
        else:
            reason = _reason(registry, source.id)
        excluded.setdefault(reason, []).append(source.id)
    record = {
        "flavour": flavour,
        "profile": profile,
        "repo_id": spec.repo_id,
        "takedown_url": takedown_url(),
        "class_counts": {
            "release_sources": _counts(_class(registry, s) for s in kept_set),
            "registry_sources": _counts(_class(registry, s.id) for s in registry),
        },
        "excluded_sources": {k: sorted(v) for k, v in sorted(excluded.items())},
    }
    if spec.frozen_exceptions:  # published v1.0 files, listed for provenance; no source is admitted
        record["frozen_exceptions"] = [
            {
                "source_id": e.source_id,
                "config": e.config,
                "registry_class": e.registry_class,
                "licence": e.licence,
                "files_sha256": sorted(e.files_sha256),
            }
            for e in spec.frozen_exceptions
        ]
    return record


def _counts(classes: Iterable[str]) -> dict[str, int]:
    c = Counter(classes)
    return {k: c[k] for k in ACCESS_CLASSES if c[k]}
