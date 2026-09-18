"""Stage a points source from an anonymous S3 prefix (D3a2).

Mirrors :mod:`marinedata.ingest_parquet`'s shape for a source whose rows live as
individually-addressable S3 objects rather than parquet shards or a zip: list the
prefix, pin the version to what was actually listed, select a deterministic
slice of it, stream each image straight into the staged tree
(:func:`marinedata.checksums.download_digest` — no archive, no double-fetch),
and write one points table instead of masks. :mod:`marinedata.s3_listing`
carries the REST-listing and stem-grouping half of this so this module stays
under its 400-line cap.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from . import checksums, gate
from .ingest import IngestError, StagedVersion
from .models import Profile, Source
from .s3_listing import (
    S3Plan,
    _object_url,
    group_by_stem,
    list_prefix,
    listing_digest,
    select_stems,
)
from .staging_finish import _finish_staging
from .tables import PointRow, StagedImage, write_points_table

_RETRY_SLEEPS = (1, 2, 4)
"""Seconds slept before each of up to 3 retries of a failed image GET (D3a2
brief §2) — 4 attempts total, the last of which raises rather than sleeping."""

_VERSION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-([0-9a-f]{12})$")


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pyarrow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return pa


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pillow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return Image


def fetch_points(plan: S3Plan, cache_dir: Path):
    """ONE ``download_digest`` of ``plan.annotations_key`` into ``cache_dir``,
    read back with pyarrow. The whole points corpus is one file — thousands of
    per-image sidecar CSVs would otherwise mean thousands of GETs for the same
    rows."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    dest = cache_dir / Path(plan.annotations_key).name
    checksums.download_digest(_object_url(plan, plan.annotations_key), dest)
    return pq.read_table(dest)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def points_for_stems(table, stems, plan: S3Plan) -> list[PointRow]:
    """``PointRow`` per row of ``table`` whose ``image_id`` is a staged stem
    (design §4): ``stem=image_id``, ``label=benthic_attribute_name``, the
    nullable ``label_id``/``form``/``region`` extras, ``schema_id=plan.
    points_schema_id``. ``id``, ``point_id``, ``growth_form_id`` and
    ``updated_on`` are dropped."""
    pa = _require_pyarrow()
    import pyarrow.compute as pc

    wanted = pa.array(sorted(set(stems)))
    filtered = table.filter(pc.is_in(table["image_id"], value_set=wanted))
    rows: list[PointRow] = []
    for record in filtered.to_pylist():
        rows.append(
            PointRow(
                stem=str(record["image_id"]),
                partition=plan.partition,
                row=int(record["row"]),
                col=int(record["col"]),
                label=str(record["benthic_attribute_name"]),
                schema_id=plan.points_schema_id,
                label_id=_opt_str(record.get("benthic_attribute_id")),
                form=_opt_str(record.get("growth_form_name")),
                region=_opt_str(record.get("region_name")),
            )
        )
    return rows


def _download_with_retry(url: str, dest: Path) -> str:
    """``download_digest``, retried up to 3x with 1/2/4s sleeps, then raised
    (D3a2 brief §2) — a dropped mid-transfer connection across thousands of
    images is expected, not exceptional."""
    last: Exception | None = None
    for attempt in range(len(_RETRY_SLEEPS) + 1):
        try:
            return checksums.download_digest(url, dest)
        except Exception as exc:  # retried, then re-raised verbatim
            last = exc
            if attempt < len(_RETRY_SLEEPS):
                time.sleep(_RETRY_SLEEPS[attempt])
    raise IngestError(f"{url}: failed after {len(_RETRY_SLEEPS) + 1} attempt(s): {last}")


def _assert_version_pinned(source: Source, digest: str) -> None:
    """``source.version`` must be ``<ISO date>-<first 12 hex of the live
    listing digest>`` — otherwise this raises, naming the hex to pin, so
    ``version: live`` (an un-pinned registry entry) raises by design until an
    operator pins it from a real listing."""
    expected = digest[:12]
    match = _VERSION_PATTERN.match(source.version)
    if match is None or match.group(1) != expected:
        raise IngestError(
            f"{source.id}: version {source.version!r} does not match the live listing "
            f"(digest {digest}) — pin version to '<ISO date>-{expected}' before ingesting"
        )


def _omitted_summary(
    groups: Mapping[str, Mapping[str, tuple[str, int, str]]], plan: S3Plan
) -> dict[str, dict[str, int]]:
    """Per sidecar suffix ``{count, bytes}``, from the listing alone —
    thumbnails, feature vectors, per-image CSVs never staged but not silently
    invisible either."""
    omitted = {suffix: {"count": 0, "bytes": 0} for suffix in plan.sidecar_suffixes}
    for fields in groups.values():
        for suffix, (_key, size, _etag) in fields.items():
            if suffix in omitted:
                omitted[suffix]["count"] += 1
                omitted[suffix]["bytes"] += size
    return omitted


def stage_s3_source(
    source: Source,
    plan: S3Plan,
    cache_root: Path,
    out_root: Path,
    profile: Profile,
    slice_cap_bytes: int | None = None,
) -> StagedVersion:
    """List -> digest -> version gate -> select -> stream images -> points
    table -> ``_finish_staging`` (D3a2 brief §2). Re-staging an already-staged
    tree never re-lists (same contract as the archive and parquet paths)."""
    decision = gate.evaluate(source, profile)
    decision.raise_if_denied()

    version_root = out_root / "sources" / source.id / source.version
    manifest_path = version_root / checksums.CHECKSUM_FILE
    if manifest_path.is_file():
        manifest = checksums.write_checksums(version_root)
        images = len(list((version_root / "images" / plan.partition).glob("*")))
        return StagedVersion(source.id, source.version, version_root, images, manifest)

    entries = list(list_prefix(plan))
    digest = listing_digest(entries)
    _assert_version_pinned(source, digest)

    groups = group_by_stem(entries, plan)
    image_groups = {
        stem: fields[plan.image_suffix]
        for stem, fields in groups.items()
        if plan.image_suffix in fields
    }
    omitted = _omitted_summary(groups, plan)

    cache_dir = cache_root / source.id
    points_table = fetch_points(plan, cache_dir)
    annotated_stems = set(points_table.column("image_id").to_pylist())

    selected = select_stems(
        image_groups,
        annotated_stems,
        annotated_only=plan.annotated_only,
        cap_bytes=slice_cap_bytes,
    )

    Image = _require_pillow()
    recorded: dict[str, str] = {}
    staged_rows: list[StagedImage] = []
    for stem in selected:
        key, _size, _etag = image_groups[stem]
        dest = version_root / "images" / plan.partition / f"{stem}{plan.image_suffix}"
        recorded[dest.relative_to(version_root).as_posix()] = _download_with_retry(
            _object_url(plan, key), dest
        )
        with Image.open(dest) as im:
            width, height = im.size
        staged_rows.append(
            StagedImage(
                stem=stem,
                partition=plan.partition,
                upstream_path=key,
                upstream_split=None,
                width=width,
                height=height,
            )
        )

    points = points_for_stems(points_table, selected, plan)
    counts: dict[str, int] = {}
    for row in points:
        counts[row.stem] = counts.get(row.stem, 0) + 1
    for stem in selected:
        n = counts.get(stem, 0)
        if n == 0:
            raise IngestError(
                f"{source.id}: {stem!r} was selected as annotated but has 0 rows in "
                f"{plan.annotations_key} — refusing to stage it with no points"
            )
        if not 1 <= n <= 25:
            raise IngestError(f"{source.id}: {stem!r} has {n} point(s), expected 1..25")

    points_path = version_root / "labels" / "points.parquet"
    recorded[points_path.relative_to(version_root).as_posix()] = write_points_table(
        points_path, points
    )

    geometry = {
        "kind": "point-label",
        "path": "labels/points.parquet",
        "format": "parquet",
        "schema_id": plan.points_schema_id,
        "crosswalk_id": source.loader.crosswalk_id if source.loader else None,
        "classes": None,
        "raster_ignore_value": None,
        "images_covered": len(selected),
        "rows": len(points),
        "coordinate_convention": "row=y,col=x, absolute pixels of the staged image",
        "supervises": list(source.annotations[0].supervises) if source.annotations else [],
        "observed_indices": [],
    }

    return _finish_staging(
        version_root,
        source,
        profile,
        decision,
        staged_rows=staged_rows,
        recorded=recorded,
        classes=0,
        license_text=plan.license_text,
        mask_count=0,
        images_without_annotation=0,
        observed_indices=frozenset(),
        upstream=[],
        ignore_index=None,
        fetched_uri=f"https://{plan.bucket}.{plan.endpoint}/{plan.prefix}",
        geometries=[geometry],
        extra_ingest={
            "listing_digest": digest,
            "annotated_only": plan.annotated_only,
            "slice_rule": (
                "lexicographic-stem, cumulative image bytes <= cap"
                if slice_cap_bytes is not None
                else None
            ),
            "slice_cap_bytes": slice_cap_bytes,
            "omitted": omitted,
        },
    )


def _mermaid_licence_text() -> str:
    """CC BY-NC-SA 4.0. The bucket carries no licence file of its own, so this
    is the canonical short notice + legalcode URL, vendored rather than the
    full legalcode (D3a2 brief §3: not fetched from the internet, and the
    official text is long — see ``licenses/cc-by-nc-sa-4.0.txt`` for how it was
    sourced)."""
    from importlib.resources import files

    return (
        files("marinedata").joinpath("licenses", "cc-by-nc-sa-4.0.txt").read_text(encoding="utf-8")
    )


_S3_PLANS: Mapping[str, S3Plan] = MappingProxyType(
    {
        "mermaid-aws": S3Plan(
            bucket="coral-reef-training",
            prefix="mermaid/",
            endpoint="s3.amazonaws.com",
            image_suffix=".png",
            sidecar_suffixes=("_thumbnail.png", "_featurevector", "_annotations.csv"),
            annotations_key="mermaid/mermaid_confirmed_annotations.parquet",
            stem_pattern=re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
            ),
            partition="default",
            annotated_only=True,
            license_text=_mermaid_licence_text(),
            points_schema_id="mermaid-attributes",
        ),
    }
)
