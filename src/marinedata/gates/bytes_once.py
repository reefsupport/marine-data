"""G4 bytes once: an image's bytes are embedded once, in ``images``, and referenced elsewhere.

Fails on a duplicated ``image_sha256`` (or one byte digest under two keys) in ``images``, an
``images`` row with no ``bytes_sha256``, an image a v1.0 config already embeds, an image-bytes
column in any new config but ``images`` (``semseg`` embeds masks, not images), and an image that is
in ``images`` while its ``metadata.bytes_config`` points at another config."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from . import GateResult
from .shards import config_shards, shard_columns
from .v11_common import IMAGES, METADATA, Collector, shards, table_rows

NAME = "bytes_once"


def run_bytes_once_gate(
    build_dir: Path, *, v1_sha256: Collection[str] | None = None, v1_dir: Path | None = None
) -> GateResult:
    build, out = Path(build_dir), Collector()
    if v1_sha256 is None and v1_dir is not None:
        from ..hf_export_v11 import v1_sha256s

        v1_sha256 = v1_sha256s(v1_dir)
    if v1_sha256 is None:
        out.fail("no_v1_set", "no v1.0 sha256 set (--v1-dir): re-uploads cannot be ruled out")
        v1_sha256 = frozenset()
    seen: dict[str, str] = {}
    by_bytes: dict[str, str] = {}
    for row in table_rows(shards(build, IMAGES), ("image_sha256", "bytes_sha256")):
        out.counts["images_rows"] += 1
        sha, digest, where = row.get("image_sha256"), row.get("bytes_sha256"), row["_shard"]
        if not sha:
            out.fail("missing_key", f"images/{where} missing image_sha256")
            continue
        if sha in seen:
            out.fail("duplicate", f"images/{where} duplicate image_sha256 {sha[:12]} ({seen[sha]})")
        seen.setdefault(sha, where)
        if sha in v1_sha256:
            out.fail("v1_reupload", f"images/{where} v1_reupload: {sha[:12]} is in a v1.0 config")
        if not digest:
            out.fail("bytes_sha256_missing", f"images/{where} bytes_sha256_missing: {sha[:12]}")
        elif by_bytes.setdefault(digest, sha) != sha:
            out.fail("bytes_twice", f"images/{where} bytes_twice: one digest under two keys")
    for config, files in config_shards(build).items():
        if config != IMAGES and "image" in shard_columns(files[0]):
            out.fail("bytes_outside_images", f"{config}: carries an `image` bytes column")
    for row in table_rows(shards(build, METADATA), ("image_sha256", "bytes_config")):
        sha = row.get("image_sha256")
        if sha in seen and row.get("bytes_config") != IMAGES:
            out.fail("embedded_twice", f"metadata {str(sha)[:12]}: in images but bytes_config="
                     f"{row.get('bytes_config')!r}")  # fmt: skip
    return out.result(NAME, images_unique=len(seen))
