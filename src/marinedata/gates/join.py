"""G1 join coverage: every row of every v1.1.0 table joins to ``metadata`` on ``image_sha256``.

* ``metadata`` is one row per image; its bytes resolve: ``bytes_config = images`` -> the sha is in
  the build's ``images`` or, with ``--v1-dir``, in the prior release's ``images`` shards (an
  additive export ships only its new pixels); otherwise a v1.0 config that holds it (and, with
  ``--v1-dir``, the sha is there).
* every ``images`` row of the build has a metadata row (prior-release images are not re-checked);
  every row of ``points`` / ``bleaching`` / ``boxes`` / ``semseg`` / ``instances`` joins (100 %, no
  tolerance, a null key is a miss).
* rows join images of their own flavour, compared by licence class and never by flavour name: a
  ``metadata`` row's ``licence_class``, and that of the image a label row joins, is the flavour's
  class (``FLAVOURS``: open -> ``open``, nc -> ``restricted-nc``), else ``not_flavour`` /
  ``joins_non_open`` / ``joins_non_nc``; in the open flavour no row, and no image it joins, is of
  a source in ``nc_ids`` either."""

from __future__ import annotations

import sys
from collections.abc import Collection
from pathlib import Path

from ..licence_class import FLAVOURS
from . import GateResult
from .v11_common import IMAGES, LABEL_TABLES, METADATA, Collector, column_values, shards, table_rows

NAME = "join"


def run_join_gate(
    build_dir: Path,
    *,
    flavour: str = "open",
    nc_ids: Collection[str] = frozenset(),
    v1_sha256: Collection[str] | None = None,
    v1_configs: Collection[str] = (),
    v1_dir: Path | None = None,
) -> GateResult:
    build, out = Path(build_dir), Collector()
    want = FLAVOURS[flavour]
    meta_files = shards(build, METADATA)
    if not meta_files:
        out.fail("no_metadata", "no metadata config: nothing to join to")
        return out.result(NAME)
    images = set(column_values(shards(build, IMAGES), "image_sha256"))
    prior_images = (  # an additive export keeps the prior release's pixels in its images shards
        {s for s in column_values(shards(Path(v1_dir), IMAGES), "image_sha256") if s}
        if v1_dir is not None
        else set()
    )
    meta: dict[str, tuple[str | None, str | None]] = {}
    cols = ("image_sha256", "source_id", "licence_class", "bytes_config")
    for row in table_rows(meta_files, cols):
        out.counts["metadata_rows"] += 1
        sha, sid, lic = row.get("image_sha256"), row.get("source_id"), row.get("licence_class")
        sid, lic = sys.intern(sid) if sid else sid, sys.intern(lic) if lic else lic
        if not sha:
            out.fail("null_key", f"metadata/{row['_shard']} row without image_sha256")
            continue
        if sha in meta:
            out.fail("duplicate_metadata", f"metadata {sha[:12]} appears twice")
        meta[sha] = (sid, lic)
        if (lic and lic != want) or (flavour == "open" and sid in nc_ids):
            out.fail("not_flavour", f"metadata {sha[:12]} ({lic}) is not an {flavour} image")
        config = row.get("bytes_config")
        if config == IMAGES:
            if sha not in images and sha not in prior_images:
                out.fail("unresolved", f"metadata {sha[:12]}: bytes_config=images, not in images")
        elif not config:
            out.fail("unresolved", f"metadata {sha[:12]}: no bytes_config")
        elif v1_configs and config not in v1_configs:
            out.fail("unresolved", f"metadata {sha[:12]}: {config} is not a v1.0 config")
        elif v1_sha256 is not None and sha not in v1_sha256:
            out.fail("unresolved", f"metadata {sha[:12]}: not in v1.0 config {config}")
    for sha in sorted(images - meta.keys()):
        out.fail("image_without_metadata", f"images {sha[:12]} has no metadata row")
    for table in LABEL_TABLES:
        for row in table_rows(shards(build, table), ("image_sha256", "source_id")):
            out.counts[f"{table}_rows"] += 1
            sha, sid = row.get("image_sha256"), row.get("source_id")
            where = f"{table}/{row['_shard']}"
            joined = meta.get(sha) if sha else None
            if joined is None:
                out.fail("orphan_row", f"{where} {str(sha)[:12] or 'null'} does not join metadata")
                continue
            out.counts[f"{table}_joined"] += 1
            if joined[1] != want or (flavour == "open" and (sid in nc_ids or joined[0] in nc_ids)):
                out.fail(f"joins_non_{flavour}", f"{where} {sha[:12]} joins a non-{flavour} image")
    return out.result(NAME)
