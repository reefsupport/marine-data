"""``release.lock.yaml``: the pin set a release is built from (Phase A, P1.3).

``marinedata release lock`` writes it from the registry pins; ``release build --lock`` refuses
to run without one, and refuses any admitted source the lock does not list. The loader checks
every entry against the registry (version, ``root_digest``, access class), so a lock written
against an older registry fails loudly instead of building something else. A source without a
``root_digest`` pin cannot be locked: the generator fails and names every such id.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .gate import evaluate
from .registry import Registry
from .release_scope import release_scope, scope_problem

LOCK_FILE = "release.lock.yaml"
FLAVOURS = ("open", "nc")
_CONFIG_ID = re.compile(r"[a-z0-9][a-z0-9-]*")  # HF config ids (CONFIG_IDS holds task layers)
_REQUIRED = True  # flipped only by unlocked_for_tests(); the CLI has no opt-out flag


class ReleaseLockError(ValueError):
    """A lock that cannot be written, loaded, or satisfied (one line, names the source)."""


class LockSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    version: str
    root_digest: str
    n_images: int | None = None
    access_class: str


class ReleaseLock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    release_id: str
    created: str
    code_sha: str
    # A lock rebuilt after the fact (the release was built by hand, from a commit that cannot
    # be identified): code_sha is then not the build commit, and reconstructed_at_sha names the
    # commit whose registry pins the digests were read from. Written only on such a lock.
    reconstructed: bool = False
    reconstructed_at_sha: str | None = None
    flavour: str
    configs: tuple[str, ...]
    sources: tuple[LockSource, ...]
    # Pinned sources the registry deliberately keeps out of this release (release_skip_reason):
    # {id: reason}, an audit trail only. Optional, and written only when non-empty, so every
    # lock without exclusions is byte-identical to one written before the field existed.
    excluded: dict[str, str] = Field(default_factory=dict)

    def source_ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sources)


def lock_required() -> bool:
    return _REQUIRED


@contextlib.contextmanager
def unlocked_for_tests() -> Iterator[None]:
    """Test escape (same idiom as ``licence_class.unflavoured_for_tests``): builds in the
    pre-lock tests run without a lock. Never reachable from the CLI."""
    global _REQUIRED
    prior, _REQUIRED = _REQUIRED, False
    try:
        yield
    finally:
        _REQUIRED = prior


def current_code_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def lock_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(registry: Registry, source_id: str) -> str:
    ck = registry.source(source_id).checksums
    return (ck.root_digest if ck else "") or ""


def _shippable(registry: Registry, flavour: str) -> list[str]:
    from .flavours import flavour_spec, ships_in

    prof = registry.profile(flavour_spec(flavour).profile)
    return [
        s.id for s in registry if evaluate(s, prof).allowed and ships_in(registry, s.id, flavour)
    ]


def build_lock(
    registry: Registry,
    *,
    release_id: str,
    flavour: str,
    configs: Iterable[str],
    source_ids: Iterable[str] | None = None,
    created: str | None = None,
    code_sha: str | None = None,
    reconstructed_at_sha: str | None = None,
) -> ReleaseLock:
    """Pin ``source_ids`` (default: the flavour's release scope, see :mod:`.release_scope`).

    The scope is the pinned staged trees the profile admits and the registry does not skip; the
    pinned sources it skips are recorded under ``excluded``. Naming a skipped source fails."""
    if flavour not in FLAVOURS:
        raise ReleaseLockError(f"flavour must be one of {FLAVOURS}, got {flavour!r}")
    cfgs = tuple(configs)
    unknown = [c for c in cfgs if not _CONFIG_ID.fullmatch(c)]
    if not cfgs or unknown:
        raise ReleaseLockError(f"missing or malformed --config {unknown or ''} (kebab-case ids)")
    shippable = _shippable(registry, flavour)
    admitted, excluded = release_scope(registry, flavour)
    ids = sorted(set(source_ids)) if source_ids is not None else sorted(admitted)
    bad = [i for i in ids if i not in shippable]
    if bad:
        raise ReleaseLockError(
            f"cannot ship in flavour {flavour!r} (denied or wrong class): {', '.join(bad)}"
        )
    if left_out := [i for i in ids if registry.source(i).release_skip_reason]:
        raise ReleaseLockError(
            "the registry leaves these out of the release (release_skip_reason): "
            + "; ".join(f"{i} ({registry.source(i).release_skip_reason})" for i in left_out)
        )
    missing = [i for i in ids if not _pin(registry, i)]
    if missing:
        raise ReleaseLockError(f"no root_digest pin in the registry for: {', '.join(missing)}")
    entries = tuple(
        LockSource(
            id=i,
            version=str(registry.source(i).version),
            root_digest=_pin(registry, i),
            n_images=registry.source(i).primary_count,
            access_class=registry.source(i).access_class.value,
        )
        for i in ids
    )
    return ReleaseLock(
        release_id=release_id,
        created=created or datetime.now(UTC).isoformat(timespec="seconds"),
        code_sha=code_sha or current_code_sha(),
        reconstructed=reconstructed_at_sha is not None,
        reconstructed_at_sha=reconstructed_at_sha,
        flavour=flavour,
        configs=cfgs,
        sources=entries,
        excluded=excluded,
    )


def write_lock(lock: ReleaseLock, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = lock.model_dump(mode="json")
    if not lock.reconstructed:  # the reconstruction keys appear only on a reconstructed lock
        del data["reconstructed"], data["reconstructed_at_sha"]
    if not lock.excluded:  # ... and `excluded` only when the registry kept something out
        del data["excluded"]
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def load_lock(path: str | Path, registry: Registry) -> ReleaseLock:
    """Parse ``path`` and check every entry against ``registry`` (digest and version must match)."""
    try:
        lock = ReleaseLock.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ReleaseLockError(f"{path}: not a valid release lock ({exc})") from exc
    if lock.flavour not in FLAVOURS:
        raise ReleaseLockError(f"{path}: flavour {lock.flavour!r} is not one of {FLAVOURS}")
    if bad := [c for c in lock.configs if not _CONFIG_ID.fullmatch(c)] or not lock.configs:
        raise ReleaseLockError(f"{path}: missing or malformed config id(s) {bad or ''}")
    if both := sorted(set(lock.excluded) & lock.source_ids()):
        raise ReleaseLockError(f"{path}: source(s) both locked and excluded: {', '.join(both)}")
    for entry in lock.sources:
        try:
            src = registry.source(entry.id)
        except KeyError as exc:
            raise ReleaseLockError(f"{path}: source {entry.id} is not in the registry") from exc
        if str(src.version) != entry.version:
            raise ReleaseLockError(
                f"{path}: source {entry.id} version {entry.version} != registry {src.version}"
            )
        if _pin(registry, entry.id) != entry.root_digest:
            raise ReleaseLockError(
                f"{path}: source {entry.id} root_digest differs from the registry pin"
            )
        if src.access_class.value != entry.access_class:
            raise ReleaseLockError(
                f"{path}: source {entry.id} access_class differs from the registry"
            )
    return lock


def check_build_against_lock(
    lock: ReleaseLock,
    registry: Registry,
    *,
    release_id: str,
    flavour: str,
    profile: str,
    local_ids: Iterable[str] = (),
) -> None:
    """Refuse (one line, naming the source) a build the lock does not cover."""
    from .fetchers_remote import _is_pinned_staged_tree

    if lock.release_id != release_id or lock.flavour != flavour:
        raise ReleaseLockError(
            f"lock is for release {lock.release_id!r} flavour {lock.flavour!r}, "
            f"not {release_id!r}/{flavour!r}"
        )
    prof, local = registry.profile(profile), set(local_ids)
    # The same scope `release lock` writes and `_resolve_roots` fetches (release_scope.py): a
    # `--local` tree or a pinned staged tree, minus what the registry skips.
    wanted = [
        s.id
        for s in registry
        if scope_problem(
            registry, prof, s, flavour, rooted=s.id in local or _is_pinned_staged_tree(s)
        )
        is None
    ]
    if missing := sorted(set(wanted) - lock.source_ids()):
        raise ReleaseLockError(f"source not in the release lock: {', '.join(missing)}")
