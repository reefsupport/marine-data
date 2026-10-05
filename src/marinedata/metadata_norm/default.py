"""The default normaliser: every field any source can fill (registry + staged data)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ..licence_class import resolve_row_class
from .base import NormContext, Staged, parse_date, staged_rows, to_table

_IMAGE_DIR = "images"


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def checksum_index(checksums: Mapping[str, str]) -> dict[tuple[str, str], tuple[str, str]]:
    """``(partition, stem) -> (relative image path, sha256)`` from ``images/`` entries;
    a flat ``images/<file>`` is partition ``default``."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for rel, sha in checksums.items():
        parts = PurePosixPath(rel).parts
        if len(parts) < 2 or parts[0] != _IMAGE_DIR:
            continue
        partition = parts[1] if len(parts) == 3 else "default"
        out[(partition, PurePosixPath(parts[-1]).stem)] = (rel, sha)
    return out


def attribution_text(ctx: NormContext) -> str | None:
    """Registry attribution + citation + upstream URL, joined; ``None`` when all are empty."""
    parts = [p.strip() for p in (ctx.attribution, ctx.citation, ctx.homepage) if p and p.strip()]
    return " | ".join(parts) if parts else None


class _Row:
    """One output row plus its provenance; fields are only set when a value is non-blank."""

    def __init__(
        self, staged: Mapping[str, Any], over: Mapping[str, Any], origin: str = "per-row"
    ) -> None:
        self.values: dict[str, Any] = {}
        self.prov: dict[str, str] = {}
        self.staged = staged
        self.over = over
        self.origin = origin

    def take(self, name: str, *candidates: tuple[Any, str]) -> None:
        """First non-blank ``(value, origin)`` wins: a per-row override, then the staged
        value, then the fallbacks in order."""
        first = ((self.over.get(name), self.origin), (self.staged.get(name), "staged"))
        for value, origin in (*first, *candidates):
            if not _blank(value):
                self.values[name] = value
                self.prov[name] = origin
                return


def normalise_row(
    source_id: str,
    version: str,
    staged: Mapping[str, Any],
    ctx: NormContext,
    index: Mapping[tuple[str, str], tuple[str, str]],
    row_licences: tuple[str, ...] = (),
    override: Mapping[str, Any] | None = None,
    origin: str = "per-row",
) -> dict[str, Any]:
    r = _Row(staged, override or {}, origin)
    stem = str(staged.get("stem") or "")
    key = (str(staged.get("partition") or "default"), stem)
    rel, sha = index.get(key) or index.get(("default", stem)) or (None, None)
    ing = ctx.ingest
    r.take("source_id", (source_id, "registry"))
    r.take("stem", (stem, "derived"))
    r.take("sample_id", (f"{source_id}/{stem}" if stem else None, "derived"))
    r.take(
        "source_version", (version or ctx.version, "registry"), (ing.get("version"), "INGEST.json")
    )
    r.take("image_path", (rel, "CHECKSUMS.sha256"))
    r.take("image_sha256", (sha, "CHECKSUMS.sha256"))
    path = r.values.get("image_path")
    ext = PurePosixPath(str(path)).suffix.lstrip(".").lower() if path else None
    r.take("image_format", (ext, "derived"))
    r.take("license", (ctx.registry_licence, "registry"), (ing.get("license"), "INGEST.json"))
    r.take("attribution", (attribution_text(ctx), "registry"))
    r.take("fetch_date", (parse_date(ing.get("fetch_date")), "INGEST.json"))
    for name in (
        "image_bytes", "image_member", "width", "height", "upstream_id", "upstream_url",
        "upstream_digest", "lineage_root_digest", "split_hint", "split_group", "upstream_split",
        "upstream_path", "capture_datetime", "lat", "lon", "gps_precision_m", "depth_m",
        "depth_source", "platform", "camera", "meow_realm", "meow_province", "meow_ecoregion",
        "depth_zone", "habitat", "label_refs",
    ):  # fmt: skip
        r.take(name)
    r.take("upstream_id", (r.values.get("upstream_path"), "upstream_path"))
    r.values["location_generalized"] = bool(staged.get("location_generalized"))
    own = row_licences or ((str(staged["license"]),) if not _blank(staged.get("license")) else ())
    r.values["licence_class"] = resolve_row_class(ctx.source_class, *own, per_row=ctx.per_row)
    r.prov["licence_class"] = "resolve_row_class" + ("(per_row)" if ctx.per_row else "")
    return {**r.values, "provenance": r.prov}


def default_normalise(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """All 37 SampleRow columns + ``licence_class`` + ``provenance``. With no staged table
    the rows are synthesised from ``CHECKSUMS.sha256`` image entries."""
    index = checksum_index(ctx.checksums)
    rows = staged_rows(staged)
    if not rows:
        rows = [{"stem": stem, "partition": part} for (part, stem), _entry in sorted(index.items())]
    return to_table([normalise_row(source_id, version, row, ctx, index) for row in rows])
