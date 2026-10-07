"""Route a source to its ingest plan: archive (``_PLANS``), parquet
(``_PARQUET_PLANS``), or S3 (``_S3_PLANS``) — split out of :mod:`marinedata.ingest`
to keep that module under the 400-line cap (D3f). Pure move: no behaviour change.

Imports :mod:`marinedata.ingest` lazily, inside :func:`stage_source`, the same way
this function already imports ``ingest_parquet``/``ingest_s3`` lazily — so this
module never creates an import cycle with ``ingest.py`` (which imports
:func:`stage_source` back from here at module load time).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import Profile, Source

if TYPE_CHECKING:
    from .ingest import StagedVersion


def stage_source(
    source: Source,
    *,
    cache_root: Path,
    out_root: Path,
    profile: Profile,
    slice_cap_bytes: int | None = None,
    workers: int | None = None,
) -> StagedVersion:
    """``_PLANS``, then ``_PARQUET_PLANS``, then ``_S3_PLANS`` (D3a2); cap and
    ``workers`` (D3d) are both S3-only, same refusal pattern for each."""
    from . import ingest_parquet, ingest_s3
    from .ingest import _PLANS, IngestError, _stage_with_plan

    if slice_cap_bytes is not None and source.id not in ingest_s3._S3_PLANS:
        raise IngestError(f"{source.id}: --slice-cap-bytes only applies to an S3 source")
    if workers is not None and source.id not in ingest_s3._S3_PLANS:
        raise IngestError(f"{source.id}: --workers only applies to an S3 source")
    if (plan := _PLANS.get(source.id)) is not None:
        return _stage_with_plan(
            source, plan, cache_root=cache_root, out_root=out_root, profile=profile
        )
    if (parquet_plan := ingest_parquet._PARQUET_PLANS.get(source.id)) is not None:
        shards = ingest_parquet.fetch_parquet_shards(source, parquet_plan, cache_root)
        return ingest_parquet._stage_with_parquet_plan(
            source, parquet_plan, shards, out_root=out_root, profile=profile
        )
    if (s3 := ingest_s3._S3_PLANS.get(source.id)) is None:
        raise IngestError(f"no ingest plan for {source.id!r}")
    return ingest_s3.stage_s3_source(
        source, s3, cache_root, out_root, profile, slice_cap_bytes, workers
    )
