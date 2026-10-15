"""Applying ``registry/splits/v2.yaml`` to a release enumeration (design 2.4, precedence 2; D2).

``assign_ood`` works on :class:`Sample` rows, a release enumerates ``(image_sha256, split_group,
source)``. :func:`release_samples` joins the two: per-image metadata is optional (the yaml's
per-source fallbacks cover whole sources, and a missing field never sends an image OOD).
:func:`plan_ood` runs the data-driven province choice, the holdouts and the size guards. A holdout
over a guard is an error, never a subsample: a human narrows the yaml rule (``share_issues``).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .dp_province import (
    labelled_total,
    province_labelled_counts,
    resolve_tropical_province_rule,
    select_tropical_province,
)
from .holdouts import HoldoutOutcome, Sample, assign_ood, share_issues
from .rules import RulesConfig, rules_sha256

OOD_FIELDS = ("meow_realm", "meow_province", "depth_m", "platform", "capture_datetime")


@dataclass(frozen=True)
class OodPlan:
    group_split: dict[str, str]
    group_tags: dict[str, list[str]]
    """``split_group`` -> every constructible holdout it matches, in precedence order."""
    outcomes: tuple[HoldoutOutcome, ...]
    issues: tuple[str, ...]
    province: str
    rules_sha256: str

    def header(self) -> dict[str, object]:
        return {
            "rules_sha256": self.rules_sha256,
            "tropical_province": self.province,
            "holdouts": [
                {
                    "name": o.name,
                    "constructible": o.constructible,
                    "images": o.n_images,
                    "groups": o.n_groups,
                    "note": o.note,
                }
                for o in self.outcomes
            ],
        }


def _as_datetime(value: object) -> dt.datetime | None:
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    return None


def load_ood_metadata(path: str | Path) -> dict[str, dict[str, object]]:
    """``{image_sha256: {field: value}}`` from a parquet with an ``image_sha256`` (or ``sha256``)
    column and any of :data:`OOD_FIELDS`; null cells are dropped."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    key = "image_sha256" if "image_sha256" in table.column_names else "sha256"
    if key not in table.column_names:
        raise ValueError(f"{path}: expected an image_sha256 (or sha256) column")
    fields = [name for name in OOD_FIELDS if name in table.column_names]
    return {
        row[key]: {name: row[name] for name in fields if row[name] is not None}
        for row in table.select([key, *fields]).to_pylist()
    }


def release_samples(
    rows: Iterable[tuple[str, str, str]],
    never_eval: Iterable[str],
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> list[Sample]:
    """One :class:`Sample` per distinct ``(sha, group, source)`` row of an eval-capable source."""
    skip = set(never_eval)
    meta = metadata or {}
    samples = []
    for sha, group, source in sorted(set(rows)):
        if source in skip:
            continue
        extra = meta.get(sha, {})
        samples.append(
            Sample(
                sha256=sha,
                split_group_id=group,
                source_id=source,
                meow_realm=extra.get("meow_realm"),  # type: ignore[arg-type]
                meow_province=extra.get("meow_province"),  # type: ignore[arg-type]
                depth_m=extra.get("depth_m"),  # type: ignore[arg-type]
                platform=extra.get("platform"),  # type: ignore[arg-type]
                capture_datetime=_as_datetime(extra.get("capture_datetime")),
            )
        )
    return samples


def plan_ood(samples: Sequence[Sample], config: RulesConfig) -> OodPlan:
    """The OOD holdouts over ``samples`` (eval-capable sources only), guards included."""
    choice = select_tropical_province(
        province_labelled_counts(samples, config, set()),
        tropical_realms=config.tropical_realms,
        labelled_total=labelled_total(samples, set()),
        min_images=int(config.size_guards["min_images"]),
    )
    resolved = resolve_tropical_province_rule(config, choice)
    result = assign_ood(samples, resolved)
    return OodPlan(
        group_split=dict(result.group_split),
        group_tags={group: list(tags) for group, tags in result.group_tags.items()},
        outcomes=result.outcomes,
        issues=tuple(share_issues(result.outcomes, len(samples), resolved.size_guards)),
        province=choice.describe(),
        rules_sha256=rules_sha256(config),
    )
