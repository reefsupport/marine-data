"""Persisted split assignments — the guarantee ``assign_splits`` alone cannot make.

``assign_splits`` (see :mod:`marinedata.scan`) is a pure, stateless allocator: every
call re-sorts *every* group by ``(-count, seeded_hash)`` and re-packs them against the
target ratios. That is correct and deterministic for one fixed corpus, but a corpus is
not fixed — new samples land in existing groups and brand-new groups appear over time.
When a growing group's count crosses another group's count in that sort order, its
assigned split can flip on a later run even at the same seed, and the flip can cascade
into a second group taking the vacated slot. Stability was true "so far", not guaranteed.

``registry/SPLIT_MAP.json`` turns that into a guarantee by recording every assignment
ever made. The rule is append-only: a key, once written, keeps its value for the life
of the file. A group already in the map is never re-offered to the allocator — only
groups absent from the map are assigned, against the quota the persisted groups already
used, and the result is appended. This is the "raise, don't warn" contract the rest of
the gate uses: a call that cannot honour the append-only guarantee raises rather than
silently reinterpreting or overwriting a stale map.

The map is keyed by :func:`marinedata.scan.group_key`'s output alone — it carries no
task id — so a group lands in the same split under every task that scans it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .scan import SplitName, assign_splits

SCHEMA_VERSION = 1


class SplitMapError(ValueError):
    """A SPLIT_MAP.json read or write would violate the append-only contract."""


@dataclass(frozen=True)
class SplitMap:
    """The on-disk assignment registry: what generated it, and what it has decided."""

    by: str
    seed: int
    ratios: dict[SplitName, float]
    assignments: dict[str, SplitName] = field(default_factory=dict)
    generated_at: str = ""
    schema_version: int = SCHEMA_VERSION


def load_split_map(path: str | Path) -> SplitMap | None:
    """Read a SPLIT_MAP.json, or ``None`` if it does not exist yet.

    A missing file is the expected, ordinary state for a brand-new registry — it is not
    an error. A file that exists but carries an unrecognised schema version is.
    """
    p = Path(path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SplitMapError(
            f"{p} has schema_version={version!r}, this code understands "
            f"{SCHEMA_VERSION!r} — migrate the file before extending it."
        )
    return SplitMap(
        by=raw["by"],
        seed=raw["seed"],
        ratios=dict(raw["ratios"]),
        assignments=dict(raw["assignments"]),
        generated_at=raw.get("generated_at", ""),
        schema_version=version,
    )


def save_split_map(path: str | Path, split_map: SplitMap) -> None:
    """Write a SPLIT_MAP.json with a stable key order and a trailing newline.

    Field order matches the documented format; ``ratios`` and ``assignments`` are
    written with sorted keys so two runs over the same input are byte-identical.
    """
    payload = {
        "schema_version": split_map.schema_version,
        "by": split_map.by,
        "seed": split_map.seed,
        "ratios": dict(sorted(split_map.ratios.items())),
        "generated_at": split_map.generated_at,
        "assignments": dict(sorted(split_map.assignments.items())),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")


def resolve_splits(
    path: str | Path,
    counts: dict[str, int],
    ratios: dict[SplitName, float],
    *,
    seed: int = 0,
    by: str = "site",
    now: str | None = None,
) -> dict[str, SplitName]:
    """Assign every group in ``counts`` a split, persisting the result at ``path``.

    A group already recorded in the map at ``path`` keeps its recorded split
    unconditionally, however ``counts`` has changed for it — that is the append-only
    guarantee, and it applies whichever task's scan produced ``counts``, since the map
    is keyed by group only. Groups absent from the map are assigned by
    :func:`~marinedata.scan.assign_splits` against the quota remaining once the
    persisted groups' current contribution is counted, then appended and written back.

    With no file yet at ``path``, this reproduces exactly what
    ``assign_splits(counts, ratios, seed=seed)`` would return — persistence changes
    nothing about a first run over a fresh corpus.

    Raises ``SplitMapError`` if a map already exists at ``path`` with a different
    ``by``, ``seed``, or ``ratios`` — extending a stale map under different parameters
    would silently reinterpret it rather than fail loudly.
    """
    existing = load_split_map(path)
    if existing is not None:
        if existing.by != by or existing.seed != seed or existing.ratios != dict(ratios):
            raise SplitMapError(
                f"{path} was generated with by={existing.by!r} seed={existing.seed} "
                f"ratios={existing.ratios} — this call used by={by!r} seed={seed} "
                f"ratios={dict(ratios)}. A mismatched call cannot safely extend a stale "
                "map: use the same parameters, or start a new map deliberately."
            )
        persisted = dict(existing.assignments)
        generated_at = existing.generated_at
    else:
        persisted = {}
        generated_at = now if now is not None else datetime.now(UTC).isoformat()

    new_counts = {key: count for key, count in counts.items() if key not in persisted}

    new_assignment: dict[str, SplitName] = {}
    if new_counts:
        total = sum(counts.values())
        filled = dict.fromkeys(ratios, 0)
        for key, split in persisted.items():
            if key in counts:
                filled[split] = filled.get(split, 0) + counts[key]
        new_assignment = assign_splits(new_counts, ratios, seed=seed, total=total, filled=filled)

    merged = {**persisted, **new_assignment}

    # Structurally unreachable given the split above (persisted keys never reach
    # `assign_splits`), kept as the explicit "raise, don't warn" guard the append-only
    # contract promises rather than relying on that invariant holding silently forever.
    for key, split in persisted.items():
        if merged[key] != split:
            raise SplitMapError(
                f"append-only violation: group {key!r} would move from {split!r} to {merged[key]!r}"
            )

    if new_assignment or existing is None:
        save_split_map(
            path,
            SplitMap(
                by=by,
                seed=seed,
                ratios=dict(ratios),
                assignments=merged,
                generated_at=generated_at,
            ),
        )

    return {key: merged[key] for key in counts}
