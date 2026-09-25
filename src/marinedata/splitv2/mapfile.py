"""SPLIT_MAP v2 (design §3.5) — schema, hashes, append-only persistence.

Same append-only contract as v1's :mod:`marinedata.splitmap`, keyed by
``split_group_id`` instead of v1's ``group_key`` output, plus two provenance hashes
so a rule or benchmark-registry edit is visible in the map itself: ``rules_sha256``
(of ``registry/splits/v2.yaml``) and ``benchmarks_sha256`` (of
``registry/benchmarks.yaml``). ``map_sha256`` is the canonical-JSON digest of
``{seed, ratios, rules_sha256, benchmarks_sha256, assignments, dp_province,
dp_province_n_images}`` — regenerating from the same inputs must reproduce it byte
for byte. ``dp_province``/``dp_province_n_images`` record charter D-P(1)'s
data-driven tropical-province holdout choice (``None``/``0`` when none qualified),
so which province was picked is part of the map's own provenance, not just a log
line (brief P3b).

This module never touches v1's ``registry/SPLIT_MAP.json``: v2 is a new file, written
wherever the caller points ``save`` (design: the integrator's build, not this
package's dry run — brief P3 keeps its working copy out of git).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2


class SplitMapError(ValueError):
    """A SPLIT_MAP v2 read, merge, or write would violate the append-only contract."""


@dataclass(frozen=True)
class SplitMapV2:
    seed: int
    ratios: dict[str, float]
    rules_sha256: str
    benchmarks_sha256: str
    assignments: dict[str, str] = field(default_factory=dict)
    ood_tags: dict[str, list[str]] = field(default_factory=dict)
    dp_province: str | None = None  # charter D-P(1): chosen tropical-province holdout
    dp_province_n_images: int = 0  # its labelled count, at the time it was chosen
    schema_version: int = SCHEMA_VERSION
    map_sha256: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "ratios": dict(sorted(self.ratios.items())),
            "rules_sha256": self.rules_sha256,
            "benchmarks_sha256": self.benchmarks_sha256,
            "map_sha256": self.map_sha256,
            "assignments": dict(sorted(self.assignments.items())),
            "ood_tags": {k: list(v) for k, v in sorted(self.ood_tags.items())},
            "dp_province": self.dp_province,
            "dp_province_n_images": self.dp_province_n_images,
        }


def compute_map_sha256(
    *,
    seed: int,
    ratios: Mapping[str, float],
    rules_sha256: str,
    benchmarks_sha256: str,
    assignments: Mapping[str, str],
    dp_province: str | None = None,
    dp_province_n_images: int = 0,
) -> str:
    canonical = json.dumps(
        {
            "seed": seed,
            "ratios": dict(ratios),
            "rules_sha256": rules_sha256,
            "benchmarks_sha256": benchmarks_sha256,
            "assignments": dict(assignments),
            "dp_province": dp_province,
            "dp_province_n_images": dp_province_n_images,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build(
    *,
    seed: int,
    ratios: Mapping[str, float],
    rules_sha256: str,
    benchmarks_sha256: str,
    assignments: Mapping[str, str],
    ood_tags: Mapping[str, list[str]] | None = None,
    dp_province: str | None = None,
    dp_province_n_images: int = 0,
) -> SplitMapV2:
    """Pure function of its inputs: same inputs -> byte-identical ``map_sha256`` and
    JSON, any number of times (design §3.5's regeneration test). ``dp_province``/
    ``dp_province_n_images`` are charter D-P(1)'s chosen tropical-province holdout
    and its labelled count — part of the hash inputs, like everything else here."""
    digest = compute_map_sha256(
        seed=seed,
        ratios=ratios,
        rules_sha256=rules_sha256,
        benchmarks_sha256=benchmarks_sha256,
        assignments=assignments,
        dp_province=dp_province,
        dp_province_n_images=dp_province_n_images,
    )
    return SplitMapV2(
        seed=seed,
        ratios=dict(ratios),
        rules_sha256=rules_sha256,
        benchmarks_sha256=benchmarks_sha256,
        assignments=dict(assignments),
        ood_tags={k: list(v) for k, v in (ood_tags or {}).items()},
        dp_province=dp_province,
        dp_province_n_images=dp_province_n_images,
        map_sha256=digest,
    )


def to_json(m: SplitMapV2) -> str:
    return json.dumps(m.as_dict(), indent=2, sort_keys=True) + "\n"


def save(m: SplitMapV2, path: str | Path) -> None:
    Path(path).write_text(to_json(m))


def load(path: str | Path) -> SplitMapV2:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != SCHEMA_VERSION:
        got = raw.get("schema_version")
        raise SplitMapError(f"{path}: schema_version must be {SCHEMA_VERSION}, got {got!r}")
    m = SplitMapV2(
        seed=raw["seed"],
        ratios=dict(raw["ratios"]),
        rules_sha256=raw["rules_sha256"],
        benchmarks_sha256=raw["benchmarks_sha256"],
        assignments=dict(raw["assignments"]),
        ood_tags={k: list(v) for k, v in raw.get("ood_tags", {}).items()},
        dp_province=raw.get("dp_province"),
        dp_province_n_images=raw.get("dp_province_n_images", 0),
        map_sha256=raw["map_sha256"],
    )
    expected = compute_map_sha256(
        seed=m.seed,
        ratios=m.ratios,
        rules_sha256=m.rules_sha256,
        benchmarks_sha256=m.benchmarks_sha256,
        assignments=m.assignments,
        dp_province=m.dp_province,
        dp_province_n_images=m.dp_province_n_images,
    )
    if expected != m.map_sha256:
        raise SplitMapError(
            f"{path}: map_sha256 {m.map_sha256} does not match recomputed {expected}"
        )
    return m


def merge_append_only(
    existing: SplitMapV2,
    new_assignments: Mapping[str, str],
    *,
    new_ood_tags: Mapping[str, list[str]] | None = None,
    rules_sha256: str,
    benchmarks_sha256: str,
    dp_province: str | None = None,
    dp_province_n_images: int | None = None,
) -> SplitMapV2:
    """Add groups not yet in ``existing``; a key already present must keep its value —
    raise rather than silently reinterpreting a stale map (v1's contract, §3.5).
    ``dp_province``/``dp_province_n_images`` default to carrying ``existing``'s choice
    forward unchanged — the D-P(1) province, once picked, is append-only too."""
    for gid, split in new_assignments.items():
        prior = existing.assignments.get(gid)
        if prior is not None and prior != split:
            raise SplitMapError(f"append-only violation: {gid!r} was {prior!r}, now {split!r}")
    merged_assignments = {**existing.assignments, **new_assignments}
    merged_tags = {k: list(v) for k, v in existing.ood_tags.items()}
    for gid, tags in (new_ood_tags or {}).items():
        merged_tags.setdefault(gid, list(tags))
    return build(
        seed=existing.seed,
        ratios=existing.ratios,
        rules_sha256=rules_sha256,
        benchmarks_sha256=benchmarks_sha256,
        assignments=merged_assignments,
        ood_tags=merged_tags,
        dp_province=dp_province if dp_province is not None else existing.dp_province,
        dp_province_n_images=(
            dp_province_n_images if dp_province_n_images is not None
            else existing.dp_province_n_images
        ),
    )
