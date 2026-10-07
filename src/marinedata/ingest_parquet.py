"""Stage an HF-parquet source into the same tree shape :mod:`marinedata.ingest` builds
for an archive source (D2a).

Coralscapes ships as 13 HF parquet shards (``data/{train,validation,test}-0000N-of-
0000M.parquet``), each row carrying an ``image``/``label`` pair as the HF ``Image``
struct (``{bytes: binary, path: string}``) rather than files inside a zip. This module
is the parquet-specific counterpart of :mod:`marinedata.ingest`'s archive path:
:func:`fetch_parquet_shards` downloads shards, :func:`_stage_with_parquet_plan` walks
them with ``pyarrow.parquet.ParquetFile.iter_batches`` (never a whole shard in memory)
and writes the same ``images/``, ``labels/masks/``, ``metadata.parquet``,
``SOURCE.json``, ``ANNOTATIONS.json``, ``LICENSE``, ``CHECKSUMS.sha256`` tree.

``ingest.py`` is capped at 400 lines (D2a brief) and its ``_stage_with_plan`` must not
be touched, so the write-metadata-files-then-checksum tail — which would otherwise be
>15 lines duplicated from that function — is its own private helper
(:func:`marinedata.staging_finish._finish_staging`) imported here under its original
name; it moved out of this module (D3a1) once a ``geometries`` parameter pushed this
file's own line count past 400.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import checksums, fetch, gate
from .ingest import IngestError, StagedVersion
from .models import Profile, Source
from .normalise import mask_bytes_to_indexed_png
from .staging_finish import _finish_staging
from .tables import StagedImage

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pyarrow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return pq


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pillow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return Image


@dataclass(frozen=True)
class ParquetPlan:
    """Per-source parquet-shard facts, the parquet counterpart of
    :class:`marinedata.ingest.ArchivePlan`."""

    shard_pattern: re.Pattern[str]
    """Matches a shard's filename (not its full HF path). Must carry a named group
    ``split`` — the upstream split (documentation only, same as D1 §2 for the archive
    path: split lives in ``metadata.upstream_split``, never in the staged path)."""

    image_column: str
    mask_column: str
    expected_rows: int
    partition: str
    version: str
    classes: int
    license_text: str

    ignore_index: int | None = None
    """The upstream raster's unlabelled/ignore class index, written verbatim into
    ``ANNOTATIONS.json``'s ``raster_ignore_value`` (D2c) — the parquet-path
    counterpart of :attr:`marinedata.ingest.ArchivePlan.ignore_index`."""


def _image_extension(payload: bytes, *, label: str) -> str:
    """Extension from the magic bytes, never from the struct's ``path`` — a row's
    declared filename is not proof of its actual encoding."""
    if payload.startswith(_PNG_MAGIC):
        return "png"
    if payload.startswith(_JPEG_MAGIC):
        return "jpg"
    raise IngestError(f"{label}: image bytes are neither PNG nor JPEG (magic {payload[:8]!r})")


def _resolve_hf_id(source: Source) -> str:
    """``access.params['hf_id']``, else derived from ``access.uri`` (D2a) — shared by
    :func:`fetch_parquet_shards` and the ``fetched_uri`` D2c writes into ``SOURCE.json``
    (both must name the same dataset)."""
    hf_id = str(source.access.params.get("hf_id") or "")
    if not hf_id and source.access.uri:
        hf_id = urllib.parse.urlparse(source.access.uri).path.removeprefix("/datasets/").strip("/")
    return hf_id


def fetch_parquet_shards(source: Source, plan: ParquetPlan, cache_dir: Path) -> list[Path]:
    """List and download every shard matching ``plan.shard_pattern`` under the
    dataset's ``data/`` directory, into ``cache_dir/<source.id>/_shards/``.

    A shard already cached with the size the listing declares is not re-downloaded.
    Streams to disk via :func:`marinedata.fetch.get_stream` (never :func:`marinedata.
    fetch.get_bytes`, which would buffer a ~450 MB shard whole). Calls through the
    ``fetch`` module (not bound names) so ``monkeypatch.setattr(fetch, "get_bytes"/
    "get_stream", ...)`` takes effect — the no-network path every test in this package
    uses.
    """
    hf_id = _resolve_hf_id(source)
    if not hf_id:
        raise IngestError(f"{source.id}: cannot determine the HuggingFace dataset id")

    tree_url = f"https://huggingface.co/api/datasets/{hf_id}/tree/main/data"
    listing = json.loads(fetch.get_bytes(tree_url))
    entries: list[tuple[str, str, int]] = []
    for entry in listing:
        upstream_path = str(entry.get("path", ""))
        name = Path(upstream_path).name
        if plan.shard_pattern.match(name):
            entries.append((name, upstream_path, int(entry.get("size", 0))))
    if not entries:
        raise IngestError(
            f"{source.id}: no shard under data/ matches {plan.shard_pattern.pattern!r}"
        )
    entries.sort(key=lambda e: e[0])

    shard_dir = cache_dir / source.id / "_shards"
    shards: list[Path] = []
    for name, upstream_path, size in entries:
        dest = shard_dir / name
        if not (dest.is_file() and dest.stat().st_size == size):
            url = f"https://huggingface.co/datasets/{hf_id}/resolve/main/{upstream_path}"
            fetch.get_stream(url, dest)
        shards.append(dest)
    return shards


def _row_struct(cell: object) -> Mapping[str, object]:
    return cell if isinstance(cell, Mapping) else {}


def _stage_with_parquet_plan(
    source: Source,
    plan: ParquetPlan,
    shards: Sequence[Path],
    *,
    out_root: Path,
    profile: Profile,
) -> StagedVersion:
    """The parquet pipeline, mirroring D1 §7 a-g for a row source instead of an
    archive: gate, verify the declared row count before any byte is written, then
    stage image/mask/table/manifest bytes and checksum the tree last.
    """
    decision = gate.evaluate(source, profile)
    decision.raise_if_denied()

    # ``<version>`` is ``Source.version`` verbatim (D1 §3) — same as ``_stage_with_plan``,
    # which never uses ``ArchivePlan.version`` for the path either.
    version_root = out_root / "sources" / source.id / source.version
    manifest_path = version_root / checksums.CHECKSUM_FILE

    if manifest_path.is_file():
        # Re-stage over an already-staged tree: verify, never re-fetch (D1 §3).
        manifest = checksums.write_checksums(version_root)
        images = len(list((version_root / "images" / plan.partition).glob("*")))
        return StagedVersion(source.id, source.version, version_root, images, manifest)

    pq = _require_pyarrow()

    ordered = sorted(shards, key=lambda p: p.name)
    matched: list[tuple[Path, str]] = []
    total_rows = 0
    for shard_path in ordered:
        match = plan.shard_pattern.match(shard_path.name)
        if match is None:
            raise IngestError(
                f"{source.id}: shard {shard_path.name!r} does not match "
                f"{plan.shard_pattern.pattern!r}"
            )
        pf = pq.ParquetFile(shard_path)
        total_rows += pf.metadata.num_rows
        matched.append((shard_path, match.group("split")))

    if total_rows != plan.expected_rows:
        raise IngestError(
            f"{source.id}: expected {plan.expected_rows} row(s) across {len(matched)} "
            f"shard(s), found {total_rows} — refusing to stage a short tree"
        )

    Image = _require_pillow()
    recorded: dict[str, str] = {}
    staged_rows: list[StagedImage] = []
    used_stems: set[str] = set()
    observed_indices: set[int] = set()
    mask_count = 0
    images_without_annotation = 0
    upstream: list[Mapping[str, object]] = []

    for shard_idx, (shard_path, split) in enumerate(matched):
        upstream.append(
            {
                "url": f"data/{shard_path.name}",
                "sha256": checksums.file_digest(shard_path),
                "bytes": shard_path.stat().st_size,
            }
        )
        pf = pq.ParquetFile(shard_path)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=16, columns=[plan.image_column, plan.mask_column]):
            image_cell = batch.column(plan.image_column)
            mask_cell = batch.column(plan.mask_column)
            for i in range(batch.num_rows):
                image_struct = _row_struct(image_cell[i].as_py())
                mask_struct = _row_struct(mask_cell[i].as_py())

                image_path = image_struct.get("path")
                stem = (
                    Path(str(image_path)).stem
                    if image_path
                    else f"{split}-{shard_idx:05d}-{row_idx:06d}"
                )
                if stem in used_stems:
                    raise IngestError(
                        f"{source.id}: duplicate stem {stem!r} (shard {shard_idx}, row {row_idx})"
                    )
                used_stems.add(stem)

                image_bytes = image_struct.get("bytes")
                if not isinstance(image_bytes, bytes) or not image_bytes:
                    raise IngestError(
                        f"{source.id}: row {row_idx} of shard {shard_idx} has no image bytes"
                    )
                ext = _image_extension(image_bytes, label=f"{source.id}:{stem}")
                image_dest = version_root / "images" / plan.partition / f"{stem}.{ext}"
                recorded[image_dest.relative_to(version_root).as_posix()] = checksums.write_digest(
                    image_dest, image_bytes
                )
                with Image.open(image_dest) as im:
                    width, height = im.size

                mask_bytes = mask_struct.get("bytes")
                if isinstance(mask_bytes, bytes) and mask_bytes:
                    mask_dest = version_root / "labels" / "masks" / plan.partition / f"{stem}.png"
                    mask_digest, observed = mask_bytes_to_indexed_png(
                        mask_bytes, mask_dest, classes=plan.classes
                    )
                    recorded[mask_dest.relative_to(version_root).as_posix()] = mask_digest
                    observed_indices |= observed
                    mask_count += 1
                else:
                    images_without_annotation += 1

                upstream_path = str(image_path) if image_path else f"{shard_path.name}#{row_idx}"
                staged_rows.append(
                    StagedImage(
                        stem=stem,
                        partition=plan.partition,
                        upstream_path=upstream_path,
                        upstream_split=split,
                        width=width,
                        height=height,
                        split_group=source.split_group_for(
                            stem=stem, upstream_path=upstream_path, partition=plan.partition
                        ),
                    )
                )
                row_idx += 1

    hf_id = _resolve_hf_id(source)
    return _finish_staging(
        version_root,
        source,
        profile,
        decision,
        staged_rows=staged_rows,
        recorded=recorded,
        classes=plan.classes,
        license_text=plan.license_text,
        mask_count=mask_count,
        images_without_annotation=images_without_annotation,
        observed_indices=frozenset(observed_indices),
        upstream=upstream,
        ignore_index=plan.ignore_index,
        fetched_uri=f"https://huggingface.co/datasets/{hf_id}" if hf_id else "",
        stem_rule="hf-struct-path-basename-else-split-shard-row",
    )


def _apache_2_0_text() -> str:
    from importlib.resources import files

    return files("marinedata").joinpath("licenses", "apache-2.0.txt").read_text(encoding="utf-8")


_PARQUET_PLANS: Mapping[str, ParquetPlan] = MappingProxyType(
    {
        "coralscapes": ParquetPlan(
            shard_pattern=re.compile(r"^(?P<split>train|validation|test)-\d{5}-of-\d{5}\.parquet$"),
            image_column="image",
            mask_column="label",
            expected_rows=2075,
            partition="default",
            version="1.0",
            classes=40,
            license_text=_apache_2_0_text(),
            # D2c: 255 never occurs (observed range 0-39); index 0 is the upstream
            # unlabelled/ignore value (17.15% of pixels; semantic_loss_ignore_index: 0).
            ignore_index=0,
        ),
    }
)
