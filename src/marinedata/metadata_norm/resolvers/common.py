"""Shared bits of the per-source resolvers: the result type and row construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
