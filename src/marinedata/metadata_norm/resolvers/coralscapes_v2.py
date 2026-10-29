"""Coralscapes V2: ``site_id`` from the frame stem (``site13_000001_012400``).

The V2 tree (``sources/coralscapes-v2/2.0/``) names a frame ``<stem>_leftImg8bit.png`` with
``stem = site<N>_<frame>_<ts>``, the stem V1 carried, so the site regex fits either form.
``metadata.parquet`` holds the upstream split in ``upstream_split`` and the staged partition
(``default``) in ``partition``. The resolved row keeps that partition: it is the
``(partition, stem)`` key of the images, CHECKSUMS and the masks layer. Date and coordinates stay
null, nothing in the tree holds them.
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
