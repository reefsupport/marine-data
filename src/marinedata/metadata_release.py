"""WP-2: the pixel-free ``metadata`` Hub config — one row per ``image_sha256``, carrying
licence/provenance, position/depth, MEOW ecology and WP-1's quality scores.

Reuses :func:`marinedata.hf_export.collect_rows`/``build_layout`` to get the exact same
``image_sha256 -> (source_id, cached file)`` set the ``images`` config embeds (never a
separate, potentially-diverging selection), then joins three things per row:

1. **The registry** (:mod:`marinedata.registry`) — ``license``, ``attribution``,
   ``source_version``, ``fetch_date`` (proxy: ``Verification.verified_on`` — no source in
   this corpus predates WP-6 with a recorded true ingest timestamp, documented in
   ``docs/metadata-fields.md``), ``lineage_root_digest`` (single-parent ``images_from``
   only — multi-parent derivations leave it null, ambiguous per-image lineage).
2. **The staged ``metadata.parquet``** (D-D layout, local-first / S3 fallback) — joined by
   ``(partition, stem)`` recovered from the cached file path — for ``upstream_id``
   (``upstream_path``; no per-item URL resolves upstream, so ``upstream_url`` is always
   null here, D-K null semantics).
3. **WP-1's quality columns** (``_quality/v1/quality.parquet``), joined on
   ``image_sha256``.

``capture_datetime``/``lat``/``lon``/``depth_m``/``platform``/``camera`` are null for the
whole v1 corpus: none of the eight staged sources carry EXIF or a location column
(checked directly — see the module docstring in :mod:`marinedata.geo_meow` and the WP-2
report). ``habitat`` (WP-2b) is the exception — it comes from the registry's
per-source :attr:`~marinedata.models.Source.habitat` field, not from staged pixels, so it
is populated for every v1 row whose source declares one. The MEOW and
location-generalization machinery is implemented and tested against fixtures so it
activates with no code change once a source with real coordinates lands.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .geo_backfill import BACKFILL_ROOT, load_backfill
from .geo_meow import MeowFeature, classify
from .hf_export import (
    DEFAULT_EXCLUDE_CONFIGS,
    SPLIT_ORDER,
    _roots,
    build_layout,
    collect_rows,
    drop_excluded,
)
from .hf_export import (
    SampleRow as ExportSampleRow,
)
from .hf_parquet import ConfigSpec, ExportRow, plan_config, write_shard
from .licence_class import drop_release_excluded, flavour_filter, resolve_row_class
from .registry import Registry, Source

METADATA = "metadata"

METADATA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("image_sha256", "string"),
    ("source_id", "string"),
    ("source_version", "string"),
    ("license", "string"),
    ("licence_class", "string"),
    ("attribution", "string"),
    ("upstream_id", "string"),
    ("upstream_url", "string"),
    ("fetch_date", "string"),
    ("lineage_root_digest", "string"),
    ("capture_datetime", "string"),
    ("lat", "double"),
    ("lon", "double"),
    ("geo_precision", "string"),
    ("geo_source", "string"),
    ("gps_precision_m", "double"),
    ("depth_m", "double"),
    ("depth_source", "string"),
    ("platform", "string"),
    ("camera", "string"),
    ("meow_realm", "string"),
    ("meow_province", "string"),
    ("meow_ecoregion", "string"),
    ("depth_zone", "string"),
    ("habitat", "string"),
    ("location_generalized", "bool"),
    ("min_side", "int64"),
    ("q_blur", "double"),
    ("q_clip_lo", "double"),
    ("q_clip_hi", "double"),
    ("q_uiqm", "double"),
    ("q_entropy", "double"),
    ("q_blank", "bool"),
    ("quality_flags", "string"),
)
"""Every field the WP-2 brief lists, plus WP-1's quality join and the join key itself.
``fetch_date``/``capture_datetime`` are ISO-8601 strings (:mod:`marinedata.hf_parquet`
has no date/timestamp scalar type; adding one for two columns was not worth forking the
export's type system)."""

METADATA_SPEC = ConfigSpec(
    METADATA,
    METADATA_COLUMNS,
    "Per-image licence, provenance, position/depth and quality — one row per "
    "image_sha256, no pixels. Join on `image_sha256` against `images` or any task.",
)

LOCATION_SENSITIVE_TAG = "location-sensitive"
"""Fallback registry-tag convention, honoured for sources that predate the first-class
`Source.location_sensitive` bool field (WP-2b). No v1 source carries this tag."""

REQUIRED_NULL_REASONS = {
    "capture_datetime": (
        "no EXIF anywhere in v1; set (date precision) only where the upstream filename "
        "carries a date (WP-2c, geo_backfill.date_from_stem)"
    ),
    "lat": "set only where upstream documents a place (WP-2c); see docs/geo-provenance.md",
    "lon": "set only where upstream documents a place (WP-2c); see docs/geo-provenance.md",
    "geo_precision": "never null: 'none' when no upstream position exists",
    "geo_source": "null exactly when geo_precision is 'none'",
    "gps_precision_m": "no upstream states a GPS error; geo_precision carries the precision class",
    "depth_m": "no source in v1 records a per-image depth",
    "depth_source": "requires depth_m, which is null for all of v1",
    "platform": "not stated upstream for any v1 source and not safely inferable",
    "camera": "not stated upstream for any v1 source",
    "meow_realm": "requires lat/lon (WP-2c backfill), or the point falls in a MEOW gap",
    "meow_province": "requires lat/lon (WP-2c backfill), or the point falls in a MEOW gap",
    "meow_ecoregion": "requires lat/lon (WP-2c backfill), or the point falls in a MEOW gap",
    "depth_zone": "requires depth_m, which is null for all of v1",
    "habitat": "populated from Source.habitat; null only for a source predating the WP-2b backfill",
    "upstream_url": "no per-item URL resolves; the source-level URL is in the registry",
    "lineage_root_digest": (
        "null for first-hop ingests; also null for coralscop-masks-rs "
        "(multi-parent images_from — ambiguous per-image lineage)"
    ),
}


class MetadataBuildError(Exception):
    """The metadata rows could not be built faithfully from the release + registry."""


@dataclass(frozen=True)
class ImageRef:
    sha256: str
    source_id: str
    file: Path
    split: str


def collect_image_refs(registry: Registry, release_dir: Path, cache: Path) -> list[ImageRef]:
    """Exactly the ``image_sha256`` set + primary file the ``images`` config embeds."""
    roots = _roots(release_dir, cache)
    rows: dict[str, list[ExportSampleRow]] = collect_rows(registry, roots, release_dir)
    rows = drop_excluded(rows, DEFAULT_EXCLUDE_CONFIGS)
    release_json = release_dir / "RELEASE.json"
    release = json.loads(release_json.read_text()) if release_json.is_file() else {}
    flavour = release.get("flavour")
    layout = build_layout(rows, flavour=flavour)
    _, splits = layout["images"]
    refs = []
    for split, export_rows in splits.items():
        for row in export_rows:
            sha, sid = row.values["image_sha256"], row.values["source_id"]
            refs.append(ImageRef(sha, sid, row.file, split))
    return refs


STAGING_BUCKET = "rs-storage-open"


def _fetch_staged_metadata_from_s3(source: Source, dest: Path) -> bool:
    """D-D fallback: GET ``sources/<id>/<version>/metadata.parquet`` from the durable
    staging bucket when no local copy exists. Returns ``False`` (dest untouched) if the
    key is absent rather than raising — a source simply not staged yet is not an error
    here (its fields stay null, documented in :data:`REQUIRED_NULL_REASONS`)."""
    from .s3_upload import client_from_rclone

    key = f"sources/{source.id}/{source.version}/metadata.parquet"
    client = client_from_rclone()
    try:
        client.head_object(Bucket=STAGING_BUCKET, Key=key)
    except Exception:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(STAGING_BUCKET, key, str(dest))
    return True


def _staged_lookup(stage_root: Path, source: Source) -> dict[tuple[str, str], dict]:
    """``(partition, stem) -> staged metadata row`` for one source's version: local copy
    if present, else the D-D S3 fallback (:func:`_fetch_staged_metadata_from_s3`)."""
    import pyarrow.parquet as pq

    path = stage_root / source.id / source.version / "metadata.parquet"
    if not path.is_file():
        cached = stage_root / ".s3-cache" / source.id / source.version / "metadata.parquet"
        if cached.is_file() or _fetch_staged_metadata_from_s3(source, cached):
            path = cached
        else:
            return {}
    table = pq.read_table(path, columns=["stem", "partition", "upstream_path"])
    out = {}
    for row in table.to_pylist():
        out[(row["partition"], row["stem"])] = row
    return out


def attribution_for(source: Source) -> str:
    return f"{source.name} ({source.licence.id})"


def lineage_root_digest_for(source: Source, registry: Registry) -> str | None:
    parents = getattr(source, "images_from", None) or ()
    if len(parents) != 1:
        return None
    parent = registry.source(parents[0])
    return parent.checksums.root_digest if parent.checksums else None


def is_location_sensitive(
    source: Source,
    *,
    sample_labels: Sequence[str] = (),
    cr_en_labels: frozenset[str] = frozenset(),
) -> bool:
    """True if the sample's position must be generalized: the source itself is flagged
    (first-class `Source.location_sensitive`, falling back to the historical tag for
    sources that predate it — see :data:`LOCATION_SENSITIVE_TAG`), **or** any of this
    sample's own labels names a CR/EN taxon (WP-2b IUCN-via-WoRMS gate — a location-
    sensitive *species* makes the sample sensitive regardless of what the source as a
    whole is tagged). ``cr_en_labels`` is the label-string set built from
    :mod:`marinedata.iucn_worms`'s output joined back through the taxonomy table (the
    caller's responsibility — this function stays a pure lookup)."""
    if source.location_sensitive or LOCATION_SENSITIVE_TAG in source.tags:
        return True
    return any(label in cr_en_labels for label in sample_labels)


def _generalize(lat: float | None, lon: float | None, sensitive: bool):
    if not sensitive or lat is None:
        return lat, lon, False
    return round(lat, 1), round(lon, 1), True


def build_rows(
    refs: Sequence[ImageRef],
    registry: Registry,
    stage_root: Path,
    quality_by_sha: Mapping[str, dict],
    meow_polygons: Sequence[MeowFeature] = (),
    backfill_root: Path = BACKFILL_ROOT,
    sample_labels: Mapping[str, Sequence[str]] | None = None,
    cr_en_labels: frozenset[str] = frozenset(),
    flavour: str | None = None,
) -> list[dict]:
    """``flavour`` (``open`` | ``nc``) keeps only the rows that flavour may ship; every row
    carries ``licence_class`` (``resolve_row_class``: the source's class, stricter if its own
    licence string says so; a per-row source with no row licence is ``unknown``).
    ``sample_labels`` (``image_sha256 -> label strings``) + ``cr_en_labels`` feed the
    WP-2b CR/EN location gate per sample; v1 has no species-level per-sample labels, so the
    default is a no-op there. Geography comes from :mod:`marinedata.geo_backfill`."""
    rows: list[dict] = []
    staged_cache: dict[str, dict[tuple[str, str], dict]] = {}
    geo_cache: dict[str, dict[str, dict]] = {}
    sample_labels = sample_labels or {}
    for ref in refs:
        source = registry.source(ref.source_id)
        if ref.source_id not in staged_cache:
            staged_cache[ref.source_id] = _staged_lookup(stage_root, source)
        staged = staged_cache[ref.source_id]
        partition, stem = ref.file.parent.name, ref.file.stem
        staged_row = staged.get((partition, stem), {})
        if ref.source_id not in geo_cache:
            geo_cache[ref.source_id] = load_backfill(ref.source_id, backfill_root)
        geo = geo_cache[ref.source_id].get(f"{partition}/{stem}", {})
        lat, lon, depth_m = geo.get("lat"), geo.get("lon"), geo.get("depth_m")
        gps_precision_m = None
        sensitive = is_location_sensitive(
            source, sample_labels=sample_labels.get(ref.sha256, ()), cr_en_labels=cr_en_labels
        )
        lat, lon, generalized = _generalize(lat, lon, sensitive)
        meow = classify(lat, lon, meow_polygons) if lat is not None else None
        q = quality_by_sha.get(ref.sha256, {})
        row_licence = (
            staged_row.get("license") if getattr(source, "licence_per_row", False) else None
        ) or source.licence.id
        rows.append(
            {
                "image_sha256": ref.sha256,
                "source_id": ref.source_id,
                "source_version": source.version,
                "license": row_licence,
                "licence_class": resolve_row_class(
                    getattr(source, "access_class", None),
                    row_licence,
                    per_row=getattr(source, "licence_per_row", False),
                ),
                "attribution": attribution_for(source),
                "upstream_id": staged_row.get("upstream_path"),
                "upstream_url": None,
                "fetch_date": source.verification.verified_on.isoformat(),
                "lineage_root_digest": lineage_root_digest_for(source, registry),
                "capture_datetime": geo.get("capture_datetime"),
                "lat": lat,
                "lon": lon,
                "geo_precision": geo.get("geo_precision") or "none",
                "geo_source": geo.get("geo_source"),
                "gps_precision_m": gps_precision_m,
                "depth_m": depth_m,
                "depth_source": None,
                "platform": geo.get("platform"),
                "camera": geo.get("camera"),
                "meow_realm": meow.realm if meow else None,
                "meow_province": meow.province if meow else None,
                "meow_ecoregion": meow.ecoregion if meow else None,
                "depth_zone": None,
                "habitat": ",".join(h.value for h in source.habitat) if source.habitat else None,
                "location_generalized": generalized,
                "min_side": q.get("min_side"),
                "q_blur": q.get("q_blur"),
                "q_clip_lo": q.get("q_clip_lo"),
                "q_clip_hi": q.get("q_clip_hi"),
                "q_uiqm": q.get("q_uiqm"),
                "q_entropy": q.get("q_entropy"),
                "q_blank": q.get("q_blank"),
                "quality_flags": ",".join(q.get("flags") or []) or None,
            }
        )
    rows = flavour_filter(rows, flavour) if flavour is not None else drop_release_excluded(rows)
    return sorted(rows, key=lambda r: r["image_sha256"])


def filter_by_license(table, allow: Sequence[str]):
    """One-call licence filter over a ``metadata`` table (name matches the WP-2 brief;
    the mechanics live in :func:`marinedata.sample_schema.licence_filter`, which works on
    any table with a ``license`` column)."""
    from .sample_schema import licence_filter

    return licence_filter(table, allow)


V2_PROJECTION_FIELDS = ("lat", "lon", "depth_m", "capture_datetime", "platform", "camera")
"""WP-2b: the six fields a v2 projection reports on — the subset of WP-6's
``sample_schema`` this brief was asked to project, read generically (no hardcoded
per-source join) from whatever a staged ``metadata.parquet`` actually carries."""


def project_v2_coverage(sources: Mapping[str, Path]) -> dict[str, dict]:
    """Generic v2 coverage projection: for each ``source_id -> metadata.parquet`` path,
    read whichever of :data:`V2_PROJECTION_FIELDS` the file's columns contain (a column
    absent entirely is 0% — the file predates WP-6's schema for that field, not an
    error) and report a per-source `%` plus an overall `%` across every source's rows
    combined. Never assumes column presence, so it works unchanged for a v1-shaped
    ``{stem, partition, upstream_path, ...}`` file and a full WP-6 ``sample_schema``
    file alike."""
    import pandas as pd

    by_source: dict[str, dict] = {}
    totals = {f: 0 for f in V2_PROJECTION_FIELDS}
    total_rows = 0
    for source_id, path in sources.items():
        df = pd.read_parquet(path)
        n = len(df)
        total_rows += n
        row = {}
        for f in V2_PROJECTION_FIELDS:
            present = int(df[f].notna().sum()) if f in df.columns else 0
            totals[f] += present
            row[f] = round(100 * present / n, 1) if n else 0.0
        row["n_rows"] = n
        by_source[source_id] = row
    overall = {
        f: (round(100 * totals[f] / total_rows, 1) if total_rows else 0.0)
        for f in V2_PROJECTION_FIELDS
    }
    return {"by_source": by_source, "overall": overall, "n_rows": total_rows}


def render_v2_projection_markdown(proj: dict, meow_realm_counts: Mapping[str, int] = {}) -> str:
    """Render :func:`project_v2_coverage`'s dict as ``docs/metadata-coverage-v2-projection.md``."""
    fields = list(V2_PROJECTION_FIELDS)
    sources = list(proj["by_source"])
    lines = [
        "# Metadata coverage — v2 projection",
        "",
        f"{proj['n_rows']} rows across {len(sources)} staged source(s). Generated by "
        "`marinedata.metadata_release.project_v2_coverage` — reads whatever columns a "
        "staged `metadata.parquet` actually has; a missing column is 0%, not an error.",
        "",
        "## Overall",
        "",
        "| Field | Coverage |",
        "|---|---|",
    ]
    lines += [f"| `{f}` | {proj['overall'][f]}% |" for f in fields]
    header = "| Source | " + " | ".join(f"`{f}`" for f in fields) + " | rows |"
    lines += ["", "## By source", "", header]
    lines += ["|---|" + "---|" * (len(fields) + 1)]
    for s in sources:
        row = proj["by_source"][s]
        cells = " | ".join(f"{row[f]}%" for f in fields)
        lines.append(f"| `{s}` | {cells} | {row['n_rows']} |")
    lines += ["", "## MEOW realm counts", ""]
    if meow_realm_counts:
        lines += ["| Realm | Rows |", "|---|---|"]
        lines += [f"| {realm} | {n} |" for realm, n in sorted(meow_realm_counts.items())]
    else:
        lines += ["No rows carry a resolved MEOW realm (needs `lat`/`lon`, which are 0% here)."]
    lines.append("")
    return "\n".join(lines)


def coverage(rows: Sequence[dict]) -> dict[str, dict]:
    """Per-field non-null coverage, overall and per ``source_id`` — feeds
    ``docs/metadata-coverage-v1.md`` (generated, not hand-written)."""
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    fields = [name for name, _ in METADATA_COLUMNS if name != "image_sha256"]

    def pct(group: Sequence[dict], field: str) -> float:
        if not group:
            return 0.0
        return 100.0 * sum(1 for r in group if r.get(field) is not None) / len(group)

    return {
        "overall": {f: round(pct(rows, f), 1) for f in fields},
        "by_source": {
            sid: {f: round(pct(group, f), 1) for f in fields}
            for sid, group in sorted(by_source.items())
        },
        "n_rows": len(rows),
    }


def render_coverage_markdown(cov: dict) -> str:
    """Render ``coverage()``'s dict as ``docs/metadata-coverage-v1.md``: an overall
    table, a per-source table, and a null-reason note for every field that is not
    ~100% covered — so a reviewer never has to guess why a column is empty."""
    fields = list(cov["overall"])
    sources = list(cov["by_source"])
    lines = [
        "# Metadata coverage — v1",
        "",
        f"Generated by `python -m marinedata.metadata_release`. {cov['n_rows']} rows "
        "(one per `image_sha256`).",
        "",
        "## Overall",
        "",
        "| Field | Coverage |",
        "|---|---|",
    ]
    lines += [f"| `{f}` | {cov['overall'][f]}% |" for f in fields]
    lines += ["", "## By source", "", "| Field | " + " | ".join(sources) + " |"]
    lines += ["|---|" + "---|" * len(sources)]
    lines += [
        f"| `{f}` | " + " | ".join(f"{cov['by_source'][s][f]}%" for s in sources) + " |"
        for f in fields
    ]
    lines += ["", "## Why a field is null", ""]
    lines += [
        f"- `{f}`: {reason}"
        for f, reason in REQUIRED_NULL_REASONS.items()
        if cov["overall"].get(f, 100.0) < 100.0
    ]
    lines.append("")
    return "\n".join(lines)


def write_metadata_config(rows: Sequence[dict], out_dir: Path) -> dict:
    """Write the ``metadata`` config shards, split-aligned with ``images``."""
    by_split: dict[str, list[dict]] = {s: [] for s in SPLIT_ORDER}
    for row in rows:
        by_split.setdefault(row["_split"], []).append(row)
    splits = {
        s: [ExportRow(values={k: v for k, v in r.items() if k != "_split"}) for r in group]
        for s, group in by_split.items()
        if group
    }
    plans = plan_config(METADATA_SPEC, splits)
    written = []
    for plan in plans:
        size = write_shard(out_dir / plan.name, METADATA_SPEC, plan.rows)
        written.append({"name": plan.name, "rows": len(plan.rows), "bytes": size})
    return {"config": METADATA, "columns": [list(c) for c in METADATA_COLUMNS], "written": written}


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq

    from .fetch import cache_root

    parser = argparse.ArgumentParser(prog="python -m marinedata.metadata_release")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--meow", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--coverage-md", type=Path, default=None)
    args = parser.parse_args(argv)

    registry = Registry.load()
    refs = collect_image_refs(registry, args.release_dir, cache_root())
    quality_table = pq.read_table(args.quality)
    quality_by_sha = {r["image_sha256"]: r for r in quality_table.to_pylist()}
    from .geo_meow import load_meow_polygons

    polygons = load_meow_polygons(args.meow) if args.meow and args.meow.is_file() else ()
    flavour = json.loads((args.release_dir / "RELEASE.json").read_text()).get("flavour")
    rows = build_rows(refs, registry, args.stage_root, quality_by_sha, polygons, flavour=flavour)
    split_by_sha = {r.sha256: r.split for r in refs}  # already HF split names (build_layout)
    for row in rows:
        row["_split"] = split_by_sha[row["image_sha256"]]

    summary = write_metadata_config(rows, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True))
    cov = coverage(rows)
    args.coverage.parent.mkdir(parents=True, exist_ok=True)
    args.coverage.write_text(json.dumps(cov, indent=1, sort_keys=True))
    if args.coverage_md:
        args.coverage_md.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_md.write_text(render_coverage_markdown(cov))
    print(json.dumps({"rows": len(rows), "written": summary["written"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
