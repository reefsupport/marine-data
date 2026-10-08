"""Coralscapes: ``site_id`` from the frame name (``site13_000001_012400_leftImg8bit``).

The tree carries ``metadata.parquet`` (stem, split, size), ``SOURCE.json``, ``ANNOTATIONS.json``
and the masks; none holds a capture date or coordinates, so date and geo stay null.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .common import Resolution, miss, row_for
from .schema import ResolvedRow, Unresolved
from .staged import ResolveContext, StagedImage

FRAME = re.compile(r"^(?P<site>site\d+)_", re.IGNORECASE)


def resolve(images: Sequence[StagedImage], ctx: ResolveContext) -> Resolution:
    rows: list[ResolvedRow] = []
    bad: list[Unresolved] = []
    for img in images:
        m = FRAME.match(img.stem)
        if not m:
            bad.append(miss(img, "site", "frame name does not start with site<N>_"))
            rows.append(row_for(img, ctx))
        else:
            rows.append(row_for(img, ctx, site_id=m["site"].lower(), platform=ctx.default_platform))
    return Resolution(tuple(rows), tuple(bad))
