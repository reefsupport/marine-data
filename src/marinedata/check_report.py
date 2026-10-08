"""``marinedata check``: split the gate's denials into EXPECTED and UNEXPECTED ones.

The gate stays fail-closed (``gate.evaluate`` is untouched). What this module adds is the
reading of its result. A denial is **expected** when the registry entry's own declaration
already says the source cannot ship under that profile: its ``access_class`` is
``internal-only`` / ``unknown`` / ``restricted-nd`` (those ship in no flavour), it is
``restricted-nc`` and the profile ships no nc, or the entry is ``release-excluded``. A denial is
**unexpected** when the class claims the source ships and the gate refuses anyway; that is a
registry entry contradicting itself, so ``check`` exits 1. There is no opt-out flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import AccessClass
from .gate import evaluate
from .licence_class import release_excluded
from .models import Profile, Source
from .registry import Registry

_NEVER_SHIPS = (AccessClass.INTERNAL_ONLY, AccessClass.UNKNOWN, AccessClass.RESTRICTED_ND)


@dataclass(frozen=True)
class Denial:
    source_id: str
    access_class: str
    reason: str
    expected_because: str | None  # None => unexpected


@dataclass(frozen=True)
class CheckReport:
    profile_id: str
    permitted: tuple[str, ...]
    expected: tuple[Denial, ...]
    unexpected: tuple[Denial, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.unexpected else 0

    def lines(self) -> list[str]:
        out = [f"PERMITTED ({len(self.permitted)})"]
        out += [f"✓ {sid}" for sid in self.permitted]
        out.append(
            f"\nEXPECTED DENIALS ({len(self.expected)}): the entry's own declaration already "
            f"says it cannot ship under '{self.profile_id}'"
        )
        for d in self.expected:
            out += [f"✗ {d.source_id}  [{d.expected_because}]", f"    {d.reason}"]
        out.append(
            f"\nUNEXPECTED DENIALS ({len(self.unexpected)}): the class claims it ships, "
            f"the gate refuses (fix the registry entry)"
        )
        for d in self.unexpected:
            out += [f"! {d.source_id}  [access_class {d.access_class}]", f"    {d.reason}"]
        out.append(
            f"\n{len(self.permitted)} permitted, {len(self.expected)} expected denial(s), "
            f"{len(self.unexpected)} unexpected denial(s) under '{self.profile_id}'"
        )
        return out


def profile_ships_class(profile: Profile, access_class: AccessClass) -> bool:
    """Does ``profile`` release sources of ``access_class``? A profile that names its classes
    ships exactly those; one with no class bar ships every class iff it ships at all."""
    if profile.allow_access_classes:
        return access_class.value in profile.allow_access_classes
    return profile.ships or profile.public_release


def expected_because(source: Source, profile: Profile, *, excluded: bool = False) -> str | None:
    """Why a denial of ``source`` under ``profile`` is expected, or ``None`` if it is not."""
    cls = source.access_class
    if excluded:
        return "release-excluded"
    if cls in _NEVER_SHIPS:
        return f"access_class {cls.value} ships in no flavour"
    if cls is AccessClass.RESTRICTED_NC and not profile_ships_class(profile, cls):
        return f"restricted-nc, '{profile.id}' ships no nc"
    return None


def build_report(
    registry: Registry, profile: Profile, *, legal_opinion_ref: str | None = None
) -> CheckReport:
    permitted: list[str] = []
    expected: list[Denial] = []
    unexpected: list[Denial] = []
    for src in registry:
        decision = evaluate(src, profile, legal_opinion_ref=legal_opinion_ref)
        if decision.allowed:
            permitted.append(src.id)
            continue
        why = expected_because(src, profile, excluded=release_excluded(src.id, registry.root))
        denial = Denial(src.id, src.access_class.value, " ".join(decision.reason.split()), why)
        (unexpected if why is None else expected).append(denial)
    return CheckReport(profile.id, tuple(permitted), tuple(expected), tuple(unexpected))
