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
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .scan import SplitName, assign_splits
from .strata import DEFAULT_MIN_GROUPS, assign_splits_stratified

SCHEMA_VERSION = 1

Row = tuple[str, str, str | None]
"""``(image_sha256, split_group, stratum)`` — one row per (image, source) pair. See
:func:`rows_to_counts`."""


class SplitMapError(ValueError):
    """A SPLIT_MAP.json read or write would violate the append-only contract."""


@dataclass(frozen=True)
class MergeInfo:
    """What :func:`rows_to_counts` did to reconcile duplicate-content ``split_group``\\ s.

    Two ``split_group`` values that ever share an image digest (same or different
    ``stratum``/source) are the same physical content under two labels — WS-D S15c found
    684 such groups in the coralscop staged tree alone. They are merged into one
    connected component and assigned as a unit; ``canonical`` is how every caller finds
    out which representative id a group's assignment now lives under.
    """

    canonical: dict[str, str]
    """Every ``split_group`` seen -> the lexicographically smallest member of its merged
    component (itself, for a group that shares no digest with another)."""

    merged_components: int
    """Count of components with 2+ distinct ``split_group`` values — real merges, not
    every group's own trivial 1-member component."""

    near_dup_unions: int = 0
    """Of the ``links`` passed to :func:`rows_to_counts` (WS-D S47 near-duplicate
    pairs), how many joined two components that were still separate — a link inside an
    already-merged component is not counted."""

    near_dup_canonicals: frozenset[str] = frozenset()
    """Canonical ids of every component at least one effective near-dup link formed."""


def _merge_components(
    parent: dict[str, str], find: Callable[[str], str]
) -> tuple[dict[str, str], int]:
    """DSU roots -> canonical (lexicographically smallest) member, plus the count of
    components that merged 2+ distinct group ids."""
    grouped: dict[str, set[str]] = {}
    for node in parent:
        grouped.setdefault(find(node), set()).add(node)
    canonical = {member: min(members) for members in grouped.values() for member in members}
    merged = sum(1 for members in grouped.values() if len(members) > 1)
    return canonical, merged


def rows_to_counts(
    rows: Iterable[Row],
    *,
    stratified: bool = False,
    links: Iterable[tuple[str, str]] = (),
) -> tuple[dict[str, int], dict[str, dict[str, int]] | None, MergeInfo]:
    """Reduce ``(image_sha256, split_group, stratum)`` rows to per-group counts.

    One count per unique image: the same physical image staged under two admitted
    sources shares one ``split_group`` (the registry's cross-source dedup rule) but
    must not be counted twice toward that group's overall size. ``stratum`` still gets
    the row once per source it was staged under, though — a cross-source duplicate is
    real evidence for *both* strata's quotas, only the overall pool must not double it.

    Shared by ``splitmap generate --in <file>`` (:mod:`marinedata.cli_splitmap`) and the
    release enumerator (:func:`marinedata.release.enumerate_release_rows`) so a
    hand-built TSV and a live staged-tree scan reduce identically.

    An image that resolves to two different ``split_group`` values no longer raises
    (WS-D S15c): the groups are merged into one connected component (see
    :class:`MergeInfo`) via union-find over the image digest, keyed by the lexically
    smallest member — deterministic regardless of row order. ``counts`` and ``strata``
    are keyed by that canonical id for every merged group; the returned ``MergeInfo``
    is how a caller (:func:`marinedata.release.generate_split_map`) maps every original
    ``split_group`` back to it, and checks a merge against a prior persisted map.

    ``links`` (WS-D S47): extra ``(image_sha256, image_sha256)`` pairs — near-duplicate
    images, not byte-identical ones — whose groups are unioned exactly like a shared
    digest. Both digests must appear in ``rows``.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    group_of: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for sha256, group, stratum in rows:
        if not sha256 or not group:
            raise ValueError("row with an empty image_sha256 or split_group")
        find(group)  # register as a DSU node even if it never merges
        prior = group_of.get(sha256)
        if prior is not None and prior != group:
            union(prior, group)
        group_of[sha256] = group
        if stratified:
            if not stratum:
                raise ValueError("--stratify needs a non-empty stratum on every row")
            members.setdefault(stratum, set()).add(sha256)

    near_dup_unions = 0
    linked_roots: list[str] = []
    for sha_a, sha_b in links:
        if sha_a not in group_of or sha_b not in group_of:
            raise ValueError(f"near-dup link ({sha_a}, {sha_b}) names a digest not in rows")
        root_a, root_b = find(group_of[sha_a]), find(group_of[sha_b])
        if root_a != root_b:
            union(root_a, root_b)
            near_dup_unions += 1
            linked_roots.append(group_of[sha_a])

    canonical, merged_components = _merge_components(parent, find)
    merge_info = MergeInfo(
        canonical=canonical,
        merged_components=merged_components,
        near_dup_unions=near_dup_unions,
        near_dup_canonicals=frozenset(canonical[group] for group in linked_roots),
    )

    counts = dict(Counter(canonical.get(group, group) for group in group_of.values()))
    if not stratified:
        return counts, None, merge_info
    strata = {
        name: dict(Counter(canonical.get(group_of[sha], group_of[sha]) for sha in shas))
        for name, shas in members.items()
    }
    return counts, strata, merge_info


@dataclass(frozen=True)
class SplitMap:
    """The on-disk assignment registry: what generated it, and what it has decided."""

    by: str
    seed: int
    ratios: dict[SplitName, float]
    assignments: dict[str, SplitName] = field(default_factory=dict)
    generated_at: str = ""
    release: str = ""
    """The release id ``splitmap generate --release`` was run for, kept for the life of
    the file — a map is never regenerated by a later release under a new id, so this is
    written once and never checked for a match the way ``by``/``seed``/``ratios`` are."""
    stratify: str = ""
    """What the map was stratified by (e.g. ``"source"``), or ``""`` for one pool. See
    :mod:`marinedata.strata`. Written only when set, so unstratified maps keep their
    exact bytes; checked like ``by``/``seed``/``ratios`` whenever a call allocates."""
    schema_version: int = SCHEMA_VERSION
    near_dup: dict[str, object] = field(default_factory=dict)
    """The near-duplicate check that shaped the map (WS-D S47): dHash definition, Pillow
    version, thresholds. Written only when set, like ``stratify``."""


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
        release=raw.get("release", ""),
        stratify=raw.get("stratify", ""),
        schema_version=version,
        near_dup=dict(raw.get("near_dup", {})),
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
        "release": split_map.release,
        **({"stratify": split_map.stratify} if split_map.stratify else {}),
        **({"near_dup": dict(sorted(split_map.near_dup.items()))} if split_map.near_dup else {}),
        "assignments": dict(sorted(split_map.assignments.items())),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")


def _component_members(merge_canonical: Mapping[str, str] | None) -> dict[str, list[str]]:
    """Canonical id -> every original ``split_group`` in its component (itself included).
    Singleton entries (a group that merged with nothing) are dropped — nothing downstream
    needs to treat them specially."""
    if not merge_canonical:
        return {}
    by_canonical: dict[str, list[str]] = {}
    for group, canon in merge_canonical.items():
        by_canonical.setdefault(canon, []).append(group)
    return {canon: members for canon, members in by_canonical.items() if len(members) > 1}


def _inherit_merged_splits(
    persisted: dict[str, SplitName],
    member_groups: dict[str, list[str]],
    counts: dict[str, int],
) -> dict[str, SplitName]:
    """D2: a merge may join a component whose members a *prior* map already split
    apart. Raise naming both if two members disagree; otherwise adopt the one split
    every persisted member already agrees on for the whole (still-unpersisted)
    component, so it is never re-offered to the allocator."""
    persisted = dict(persisted)
    for canon, members in member_groups.items():
        if canon not in counts:
            continue
        found: dict[SplitName, str] = {}
        for member in members:
            split = persisted.get(member)
            if split is not None and split not in found:
                found[split] = member
        if len(found) > 1:
            (split_a, member_a), (split_b, member_b) = list(found.items())[:2]
            raise SplitMapError(
                f"{sorted(members)!r} share an image digest and would merge into one "
                f"split_group, but a prior map already assigned {member_a!r}={split_a!r} "
                f"and {member_b!r}={split_b!r} — never silently joining a released "
                "assignment"
            )
        if found:
            (split,) = found
            persisted[canon] = split
    return persisted


def resolve_splits(
    path: str | Path,
    counts: dict[str, int],
    ratios: dict[SplitName, float],
    *,
    seed: int = 0,
    by: str = "site",
    now: str | None = None,
    frozen: bool = False,
    release: str | None = None,
    strata: Mapping[str, Mapping[str, int]] | None = None,
    stratify: str = "",
    min_groups: int = DEFAULT_MIN_GROUPS,
    merge_canonical: Mapping[str, str] | None = None,
    forced: Mapping[str, SplitName] | None = None,
    near_dup: Mapping[str, object] | None = None,
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

    ``frozen=True`` is the release-build contract: every group in ``counts`` must
    already be recorded at ``path``. A group that is not raises ``SplitMapError``
    instead of being allocated and appended — a frozen build touches nothing on disk,
    successful or not, so a release can never quietly grow the shared map that other
    tasks depend on.

    ``release``: recorded on the map the first time it is written (``existing is
    None``) — ``marinedata splitmap generate --release <id>`` passes this through so the
    map records which release it was generated for. Extending an existing map never
    overwrites it with a different value; it is not part of the ``by``/``seed``/
    ``ratios`` mismatch check above, since a shared map legitimately outlives the one
    release that first created it.

    ``strata`` + ``stratify``: allocate new groups per stratum (see
    :mod:`marinedata.strata`) instead of against one pool. ``strata`` maps stratum ->
    group -> images and must cover exactly the groups in ``counts``; ``stratify`` names
    what the strata are and is recorded on the map. A non-frozen call whose
    ``stratify`` differs from the map's raises, like a ``by``/``seed``/``ratios``
    mismatch; a frozen call allocates nothing, so it does not need the strata.

    ``merge_canonical`` (WS-D S15c): every ``split_group`` -> the canonical id of its
    merged component (see :class:`MergeInfo` — pass ``rows_to_counts``'s third return
    value's ``.canonical``). A merged component keeps the append-only guarantee at the
    *component* level: if two of its members are already persisted under different
    splits, this raises naming both, rather than silently joining a prior release's
    separated assignments; if one or more agree, that split is inherited for the whole
    component instead of being re-allocated. Every member — not just the canonical id —
    is written to the saved map, so a later ``frozen=True`` lookup by any member's own
    ``split_group`` still finds it.

    ``forced``: groups to assign directly to the named split with no lottery — the
    never-eval-only-component rule (WS-D S15c). Only applied to a group that is not
    already persisted; a persisted group keeps its recorded split regardless.

    ``near_dup``: recorded on a new map (WS-D S47); an existing map keeps its own record.
    """
    if (strata is None) != (not stratify):
        raise SplitMapError("pass `strata` and `stratify` together, or neither")
    if strata is not None:
        covered = {key for groups in strata.values() for key in groups}
        if covered != set(counts):
            raise SplitMapError(
                f"strata cover {len(covered)} groups but counts has {len(counts)} — "
                "every group must belong to at least one stratum and no stratum may "
                "name a group absent from counts"
            )
    existing = load_split_map(path)
    if existing is not None:
        if existing.by != by or existing.seed != seed or existing.ratios != dict(ratios):
            raise SplitMapError(
                f"{path} was generated with by={existing.by!r} seed={existing.seed} "
                f"ratios={existing.ratios} — this call used by={by!r} seed={seed} "
                f"ratios={dict(ratios)}. A mismatched call cannot safely extend a stale "
                "map: use the same parameters, or start a new map deliberately."
            )
        if not frozen and existing.stratify != stratify:
            raise SplitMapError(
                f"{path} was generated with stratify={existing.stratify!r} — this call "
                f"used stratify={stratify!r}. Extending it that way would allocate new "
                "groups against a different quota than the one it was built with."
            )
        persisted = dict(existing.assignments)
        generated_at = existing.generated_at
        release_value = existing.release
        near_dup_value = dict(existing.near_dup)
    else:
        if frozen:
            raise SplitMapError(
                f"{path} does not exist — a frozen build requires a split map already "
                "generated (see `marinedata splitmap generate`)."
            )
        persisted = {}
        generated_at = now if now is not None else datetime.now(UTC).isoformat()
        release_value = release or ""
        near_dup_value = dict(near_dup or {})

    member_groups = _component_members(merge_canonical)
    if existing is not None and member_groups:
        persisted = _inherit_merged_splits(persisted, member_groups, counts)

    new_counts = {key: count for key, count in counts.items() if key not in persisted}

    if frozen and new_counts:
        missing = ", ".join(sorted(new_counts))
        raise SplitMapError(
            f"{path} is frozen: group(s) not in the map: {missing}. A frozen build "
            "never allocates or appends — regenerate the map to include them first."
        )

    forced_new = {key: split for key, split in (forced or {}).items() if key in new_counts}

    new_assignment: dict[str, SplitName] = {}
    if new_counts and strata is not None:
        pinned = {key: persisted[key] for key in counts if key in persisted}
        pinned.update(forced_new)
        new_assignment = assign_splits_stratified(
            strata,
            ratios,
            seed=seed,
            pinned=pinned,
            min_groups=min_groups,
        )
        new_assignment.update(forced_new)
    elif new_counts:
        total = sum(counts.values())
        filled = dict.fromkeys(ratios, 0)
        for key, split in persisted.items():
            if key in counts:
                filled[split] = filled.get(split, 0) + counts[key]
        remaining = {key: count for key, count in new_counts.items() if key not in forced_new}
        if remaining:
            new_assignment = assign_splits(remaining, ratios, seed=seed, total=total, filled=filled)
        new_assignment.update(forced_new)

    merged = {**persisted, **new_assignment}

    # Structurally unreachable given the split above (persisted keys never reach
    # `assign_splits`), kept as the explicit "raise, don't warn" guard the append-only
    # contract promises rather than relying on that invariant holding silently forever.
    for key, split in persisted.items():
        if merged[key] != split:
            raise SplitMapError(
                f"append-only violation: group {key!r} would move from {split!r} to {merged[key]!r}"
            )

    if member_groups:
        # Every member of a merged component gets the component's split recorded under
        # its own key, not just the canonical one — a later `frozen=True` lookup keyed
        # by any original `split_group` (what a sample actually carries) must still hit.
        expanded = dict(merged)
        for canon, members in member_groups.items():
            if canon in merged:
                for member in members:
                    expanded[member] = merged[canon]
        merged = expanded

    if not frozen and (new_assignment or existing is None):
        save_split_map(
            path,
            SplitMap(
                by=by,
                seed=seed,
                ratios=dict(ratios),
                assignments=merged,
                generated_at=generated_at,
                release=release_value,
                stratify=stratify,
                near_dup=near_dup_value,
            ),
        )

    return {key: merged[key] for key in counts}
