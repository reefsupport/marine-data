"""The parquet-path equivalent of D1 §7 f-g: write ``metadata.parquet``, the three
metadata files, then the checksum manifest last.

Split out of :mod:`marinedata.ingest_parquet` (D3a1) purely to keep that module under
its 400-line cap — :func:`_finish_staging` gained a ``geometries`` parameter and grew
past it. ``ingest_parquet.py`` re-exports the name, so every existing import site
(``ingest_parquet._finish_staging`` or ``from .ingest_parquet import _finish_staging``)
keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from . import checksums, gate
from .ingest import StagedVersion
from .manifests import AnnotationCounts, write_annotations_json, write_license, write_source_json
from .models import Profile, Source
from .tables import StagedImage, write_metadata_table


def _finish_staging(
    version_root: Path,
    source: Source,
    profile: Profile,
    decision: gate.Decision,
    *,
    staged_rows: list[StagedImage],
    recorded: dict[str, str],
    classes: int,
    license_text: str,
    mask_count: int,
    images_without_annotation: int,
    observed_indices: frozenset[int],
    upstream: list[Mapping[str, object]],
    ignore_index: int | None,
    fetched_uri: str,
    geometries: Sequence[Mapping[str, object]] | None = None,
) -> StagedVersion:
    """Write ``metadata.parquet``, the three metadata files, then the checksum
    manifest last — the parquet-path equivalent of D1 §7 f-g.

    ``geometries`` lets a caller (the points path) supply the ``ANNOTATIONS.json``
    geometry list verbatim instead of the mask-derived one built here by default.
    ``None`` — every existing caller — reproduces today's behaviour byte-for-byte,
    including that ``mask_count == 0`` still yields an empty geometries list.
    """
    metadata_path = version_root / "metadata.parquet"
    recorded[metadata_path.relative_to(version_root).as_posix()] = write_metadata_table(
        metadata_path, staged_rows
    )

    partition = staged_rows[0].partition if staged_rows else "default"
    ingest_meta = {
        "ingest_version": 1,
        "stem_rule": "hf-struct-path-basename-else-split-shard-row",
        "partition_rule": f"literal:{partition}",
        "upstream": upstream,
        "fetched_uri": fetched_uri,
        "gate": {"profile": profile.id, "allowed": decision.allowed, "reason": decision.reason},
    }
    if mask_count:
        # False provenance otherwise: a byte-for-byte copy path (or a points-only
        # stage with no masks at all) never ran pillow over anything.
        ingest_meta["mask_encoder"] = "pillow/11.0.0"
    source_json_path = version_root / "SOURCE.json"
    recorded[source_json_path.relative_to(version_root).as_posix()] = write_source_json(
        source_json_path, source, ingest_meta
    )

    annotation = source.annotations[0] if source.annotations else None
    if geometries is not None:
        resolved_geometries: Sequence[Mapping[str, object]] = geometries
    else:
        built: list[Mapping[str, object]] = []
        if mask_count:
            built.append(
                {
                    "kind": annotation.kind.value if annotation else "dense-mask",
                    "path": "labels/masks/",
                    "format": "png-indexed",
                    "schema_id": source.loader.schema_id if source.loader else "dataset-native",
                    "crosswalk_id": source.loader.crosswalk_id if source.loader else None,
                    "classes": classes,
                    "raster_ignore_value": ignore_index,
                    "images_covered": mask_count,
                    "rows": None,
                    "supervises": list(annotation.supervises) if annotation else [],
                    "observed_indices": sorted(observed_indices),
                }
            )
        resolved_geometries = built

    counts = AnnotationCounts(
        images=len(staged_rows),
        images_without_annotation=images_without_annotation,
        geometries=tuple(resolved_geometries),
    )
    annotations_json_path = version_root / "ANNOTATIONS.json"
    recorded[annotations_json_path.relative_to(version_root).as_posix()] = write_annotations_json(
        annotations_json_path, source, counts
    )

    license_path = version_root / "LICENSE"
    recorded[license_path.relative_to(version_root).as_posix()] = write_license(
        license_path, source, license_text
    )

    manifest = checksums.write_checksums(version_root, recorded=recorded)
    return StagedVersion(source.id, source.version, version_root, len(staged_rows), manifest)
