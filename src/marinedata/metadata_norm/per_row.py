"""Per-row licence/attribution normalisers: FathomNet, iNat-marine, planktonzilla, qut-fish.

Each maps the per-image licence + creator fields (wherever the source stages them) onto
``license`` / ``attribution``; where a source stages no per-image licence the registry string
is kept and ``licence_class`` stays ``unknown`` (fail closed) for a per-row source.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .base import NormContext, Staged, staged_rows, to_table
from .default import _blank, checksum_index, normalise_row
from .geo import fathomnet_fields, inat_fields, mermaid_fields

_INAT_PHOTO = "https://www.inaturalist.org/photos/"
_PLK_SPLIT = re.compile(r"^data_(?P<split>[a-z]+)-(?P<shard>\d+-of-\d+)_parquet_(?P<row>\d+)$")


def _int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _finish(source_id, version, rows, ctx, extras, origin):
    """``extras[i]`` (``_licences`` + field overrides) is applied over ``rows[i]``."""
    index = checksum_index(ctx.checksums)
    out = []
    for row, extra in zip(rows, extras, strict=True):
        over = {k: v for k, v in extra.items() if k not in ("_licences", "_prov") and not _blank(v)}
        lics = tuple(extra.get("_licences", ()))
        out.append(
            normalise_row(
                source_id,
                version,
                row,
                ctx,
                index,
                row_licences=lics,
                override=over,
                origin=origin,
                prov=extra.get("_prov"),
            )
        )
    return to_table(out)


def fathomnet(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """Per-image JSON (``ctx.labels[uuid]``): licence = the boxes' ``annotationLicense`` values,
    creator = the contributor's e-mail *domain* (the address itself is not copied)."""
    rows = staged_rows(staged) or [{"stem": k, "partition": "default"} for k in sorted(ctx.labels)]
    extras = []
    for row in rows:
        rec = ctx.labels.get(str(row.get("stem")), {})
        lics = sorted(
            {
                str(b["annotationLicense"])
                for b in rec.get("boundingBoxes") or []
                if b.get("annotationLicense")
            }
        )
        org = str(rec.get("contributorsEmail") or "").rpartition("@")[2]
        extra: dict[str, Any] = {"_licences": tuple(lics)}
        if lics:
            extra["license"] = "; ".join(lics)
        if org:
            extra["attribution"] = (
                f"FathomNet contributor ({org}) | {ctx.attribution} | {ctx.citation}"
            )
        extra.update(fathomnet_fields(rec))
        extra.update(
            upstream_id=rec.get("uuid"), upstream_url=rec.get("url"), width=_int(rec.get("width")),
            height=_int(rec.get("height")), image_sha256=rec.get("sha256"),
        )  # fmt: skip
        extras.append(extra)
    return _finish(source_id, version, rows, ctx, extras, "fathomnet-json")


def inat_marine(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """``staged`` = manifest rows (``photo_id``, ``license``, ``observer_id``, ``extension``)."""
    rows = [
        {"stem": str(r["photo_id"]), "partition": "default", **r}
        for r in staged_rows(staged)
        if not _blank(r.get("photo_id"))
    ]
    extras = []
    for r in rows:
        lic = str(r.get("license") or "").strip()
        who = f"iNaturalist observer {r['observer_id']}" if not _blank(r.get("observer_id")) else ""
        extra = {
            "_licences": (lic,) if lic else (),
            "license": lic or None,
            "attribution": " | ".join(
                p for p in (who, lic, f"photo {r['photo_id']}", ctx.citation) if p
            ),
            "upstream_id": str(r["photo_id"]),
            "upstream_url": f"{_INAT_PHOTO}{r['photo_id']}",
            "image_format": str(r.get("extension") or "").lower() or None,
            "width": _int(r.get("width")),
            "height": _int(r.get("height")),
            **inat_fields(r),
        }
        extras.append(extra)
    stems = [{k: r.get(k) for k in ("stem", "partition", "observation_uuid")} for r in rows]
    return _finish(source_id, version, stems, ctx, extras, "inat-manifest")


def planktonzilla(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """Only the images are staged: split and shard row come from the file stem; the
    per-sample licence (the originating instrument dataset) is not staged, so it stays
    ``unknown`` rather than inheriting the umbrella card's ``other``."""
    rows = staged_rows(staged)
    extras = []
    for r in rows:
        m = _PLK_SPLIT.match(str(r.get("stem") or ""))
        extra: dict[str, Any] = {"_licences": ()}
        if m:
            extra["upstream_split"] = m["split"]
            extra["upstream_path"] = f"data/{m['split']}-{m['shard']}.parquet#{m['row']}"
            extra["upstream_id"] = r["stem"]
        extras.append(extra)
    return _finish(source_id, version, rows, ctx, extras, "planktonzilla-stem")


def qut_fish(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """No per-image licence is staged (``LicenceAndAuthors.pdf`` only): the licence text stays
    the staged/registry string, attribution names the authors, class is the source's."""
    rows = staged_rows(staged)
    flat = replace(ctx, per_row=False)
    extras = [{"_licences": ()} for _ in rows]
    return _finish(source_id, version, rows, flat, extras, "qut-fish")


def mermaid_aws(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """Default fields plus the cached sample-event join (``ctx.events``; empty = no join).
    The join key is the image id (the stem / upstream id): MERMAID publishes no public
    image -> sample-event link, so ``events["images"]`` must come from an authenticated fetch."""
    rows = staged_rows(staged)
    extras = []
    for row in rows:
        image_id = str(row.get("upstream_id") or row.get("stem") or "")
        extras.append({"_licences": (), **mermaid_fields(image_id, ctx.events)})
    return _finish(source_id, version, rows, ctx, extras, "mermaid-sample-event")
