"""Shared bits of the per-source resolvers: the result type and row construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .licence import LicenceDefaults, licence_columns
from .schema import ResolvedRow, Unresolved
from .staged import ResolveContext, StagedImage


@dataclass(frozen=True)
class Resolution:
    """What a resolver returns: one row per staged image, the images it could not resolve and
    free-form ``notes`` (table sha256s, rule counts) merged into ``coverage.json``."""

    rows: tuple[ResolvedRow, ...]
    unresolved: tuple[Unresolved, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)


Resolver = Callable[[Sequence[StagedImage], ResolveContext], Resolution]


def row_for(img: StagedImage, ctx: ResolveContext, **resolved: Any) -> ResolvedRow:
    """A :class:`ResolvedRow` for ``img`` carrying the identity columns plus ``resolved``."""
    return ResolvedRow(
        sample_id=f"{ctx.source_id}/{img.stem}",
        source_id=ctx.source_id,
        source_version=ctx.version,
        partition=img.partition,
        stem=img.stem,
        image_path=img.rel_path,
        image_sha256=img.sha256,
        **resolved,
    )


def miss(img: StagedImage, field_name: str, reason: str) -> Unresolved:
    return Unresolved(img.partition, img.stem, img.sha256, field_name, reason)


def finish_rows(
    rows: Sequence[ResolvedRow],
    *,
    platform: str | None,
    licence: LicenceDefaults | None,
    staged_by_sha: Mapping[str, Mapping[str, Any]],
) -> list[ResolvedRow]:
    """New rows with the columns every resolver shares filled in (a value a resolver set wins):

    * ``platform`` from the registry ``default_platform`` (``unknown`` stays ``unknown``);
    * ``date_precision``: ``day`` when ``date`` is set, ``year`` when only ``year`` is (the Tetes
      sets), otherwise null; a resolver may set ``month`` itself;
    * the eight licence columns: registry defaults, the staged row's ``upstream_id`` and the
      source's ``per_row_license`` overrides (:func:`.licence.licence_columns`).
    """
    out: list[ResolvedRow] = []
    for r in rows:
        update: dict[str, Any] = {}
        if r.platform is None and platform:
            update["platform"] = platform
        if r.date_precision is None:
            if r.date is not None:
                update["date_precision"] = "day"
            elif r.year is not None:
                update["date_precision"] = "year"
        for column, value in licence_columns(
            staged_by_sha.get(r.image_sha256, {}), licence
        ).items():
            if getattr(r, column) is None:
                update[column] = value
        out.append(replace(r, **update) if update else r)
    return out
