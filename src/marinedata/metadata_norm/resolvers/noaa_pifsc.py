"""NOAA PIFSC bleaching: site and year from the file stem.

Stems are ``<site>_<year>_<n>_<n>`` (``FFS-B009_2019_06_51``), ``<site>_<year>_<tag>_<n>_<n>``
(``GUA-2372_2022_A_16_1787``) or, for the Hawaii shallow sets, ``<Site>Shallow<year>IMG_<n>_<n>``
(``LaehouShallow2015IMG_2684_7313``, year glued to the site). The staged tree carries no coordinates
and no site table, and the registry entry cites none, so ``lat``/``lon``/``coord_precision_m`` stay
null. Only the year is known, so ``date`` stays null and ``year`` is filled.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .common import Resolution, miss, row_for
from .schema import ResolvedRow, Unresolved
from .staged import ResolveContext, StagedImage

STEM = re.compile(r"^(?P<site>.+?)_(?P<year>(?:19|20)\d{2})_")
GLUED = re.compile(r"^(?P<site>[A-Za-z]+?)(?P<year>(?:19|20)\d{2})IMG_")
DATE_SOURCE = "stem: year token only (no day)"


def resolve(images: Sequence[StagedImage], ctx: ResolveContext) -> Resolution:
    rows: list[ResolvedRow] = []
    bad: list[Unresolved] = []
    disagree = 0
    for img in images:
        m = STEM.match(img.stem) or GLUED.match(img.stem)
        if not m:
            bad.append(miss(img, "site+year", "stem does not match <site>_<year>_..."))
            rows.append(row_for(img, ctx))
            continue
        group = str(img.row.get("split_group") or "")
        if group and group.rsplit("/", 1)[-1] != m["site"]:
            disagree += 1
        rows.append(
            row_for(
                img,
                ctx,
                site_id=m["site"],
                year=int(m["year"]),
                platform=ctx.default_platform,
                date_source=DATE_SOURCE,
            )
        )
    return Resolution(tuple(rows), tuple(bad), {"site_differs_from_split_group": disagree})
