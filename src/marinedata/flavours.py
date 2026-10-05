"""Release flavours (WP-L1b): ``open`` and ``nc`` are two HF repos, nothing duplicated.

The per-flavour repo id, shipping profile and the takedown URL live in
``registry/flavours.yaml``. This module resolves them and decides, per source, what a flavour
ships (:func:`flavour_source_ids`) and what RELEASE.json records about it
(:func:`release_record`). The per-row filter itself is ``licence_class.flavour_filter``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .licence_class import ACCESS_CLASSES, FLAVOURS, NC, OPEN, coerce_class, release_excluded

SPLIT_MAP_PROFILE = "ship-noncommercial"
"""A frozen split map is shared by both flavours, so it is generated over the superset
(every open + nc source), whichever flavour asked for it."""


@dataclass(frozen=True)
class FlavourSpec:
    flavour: str
    repo_id: str
    profile: str
    profiles: tuple[str, ...]


@dataclass(frozen=True)
class FlavourConfig:
    takedown_url: str
    flavours: dict[str, FlavourSpec]


def load_flavours(root: str | Path | None = None) -> FlavourConfig:
    from .licence_class import _registry_root

    data = yaml.safe_load((_registry_root(root) / "flavours.yaml").read_text()) or {}
    specs = {
        name: FlavourSpec(
            flavour=name,
            repo_id=str(entry["repo_id"]),
            profile=str(entry["profile"]),
            profiles=tuple(entry.get("profiles") or [entry["profile"]]),
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
    source never does: its class is only a bound, each row decides."""
    if release_excluded(source_id):
        return False  # never released in any flavour (registry flag)
    source = registry.source(source_id)
    return not source.licence_per_row and _class(registry, source_id) == FLAVOURS[flavour]


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
    return {
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


def _counts(classes: Iterable[str]) -> dict[str, int]:
    c = Counter(classes)
    return {k: c[k] for k in ACCESS_CLASSES if c[k]}
