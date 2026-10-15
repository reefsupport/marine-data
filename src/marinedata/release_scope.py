"""The release scope is a registry fact: one predicate for lock, lock check and fetch (PR-F2).

A source is in scope for a flavour when its profile admits it, its class ships in the flavour,
the registry carries no ``release_skip_reason`` for it, and a root can be had for it (a pinned
staged tree, or a ``--local`` tree). ``release lock`` writes exactly this set,
``check_build_against_lock`` demands exactly this set and ``_resolve_roots`` fetches exactly
this set, so a tree the release deliberately leaves out (``release_skip_reason: "deferred to
v1.2: ..."``) is never fetched, never demanded and never written as admitted.

Only the *registry* reason counts here. A ``needs-attribution`` source with no attribution is
skipped later, by ``build_release``, and still fails the build loudly when it has rows; keeping
it in scope means that gate is never bypassed by leaving the source out of the lock.
"""

from __future__ import annotations

from .models import Profile, Source
from .registry import Registry

NO_TREE = "no pinned staged tree (needs checksums.root_digest and a staged-tree loader)"


def class_problem(
    registry: Registry, prof: Profile, source: Source, flavour: str | None
) -> str | None:
    """Why ``source`` cannot ship under ``prof`` / ``flavour`` at all (``None``: it can)."""
    from .flavours import ships_in
    from .gate import evaluate

    if not evaluate(source, prof).allowed:
        return f"not admitted under profile {prof.id!r}"
    if flavour is not None and not ships_in(registry, source.id, flavour):
        return f"does not ship in flavour {flavour!r}"  # never fetch another flavour's bytes
    return None


def scope_problem(
    registry: Registry, prof: Profile, source: Source, flavour: str | None, *, rooted: bool
) -> str | None:
    """Why ``source`` is out of the release scope (``None``: it is in), in filter order.

    ``rooted``: a root can be had for it, i.e. a ``--local`` tree or a pinned staged tree. The
    caller decides, so a test can stand in for the staged-tree check where the fetch lives."""
    if (why := class_problem(registry, prof, source, flavour)) is not None:
        return why
    if source.release_skip_reason:
        return f"left out of the release by the registry: {source.release_skip_reason}"
    return None if rooted else NO_TREE


def release_scope(registry: Registry, flavour: str) -> tuple[list[str], dict[str, str]]:
    """``(admitted ids, excluded)`` for ``flavour`` under its own profile.

    ``admitted`` is what ``release lock`` writes, the lock check demands and the build fetches.
    ``excluded`` maps each *pinned* source the profile admits but the registry deliberately keeps
    out (``release_skip_reason``) to its reason: the audit trail for the fetch scope. Sources
    with no pin have no tree to fetch and are not listed."""
    from .fetchers_remote import _is_pinned_staged_tree
    from .flavours import flavour_spec

    prof = registry.profile(flavour_spec(flavour).profile)
    admitted: list[str] = []
    excluded: dict[str, str] = {}
    for source in registry:
        if class_problem(registry, prof, source, flavour) is not None:
            continue
        if source.release_skip_reason:
            if source.checksums is not None and source.checksums.root_digest:
                excluded[source.id] = source.release_skip_reason
        elif _is_pinned_staged_tree(source):
            admitted.append(source.id)
    return admitted, excluded
