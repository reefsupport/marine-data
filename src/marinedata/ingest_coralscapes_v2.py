"""Stage Coralscapes V2 (HF ``josauder/coralscapesV2``, config ``instances``) as a D-D tree.

One command turns the pinned ``instances/*.parquet`` shards into
``<out>/sources/coralscapes-v2/2.0/`` with the V1 conventions of
:mod:`marinedata.ingest_parquet` (same ``_finish``-style tail: ``SOURCE.json``,
``ANNOTATIONS.json``, ``LICENSE``, then ``CHECKSUMS.sha256`` last through
:func:`marinedata.checksums.write_checksums`, whose ``root_digest`` is printed)::

    images/default/<stem>_leftImg8bit.png
    labels/masks/default/<stem>_leftImg8bit.png        uint8 semantic ids (mode L)
    labels/instances/default/<stem>_instanceIds.png    uint16 cityscapes ids (mode I;16)
    metadata.parquet                                   37 sample-schema v2 columns + partition

Image, label and ``instance_ids`` bytes are written verbatim from the parquet (no decode and
re-encode); an ``instance_ids`` *array* column, should a revision ship one, becomes a uint16
PNG. ``stem`` in ``metadata.parquet`` is the file stem (``<hf stem>_leftImg8bit``, as V1), the
HF ``stem`` column is ``upstream_id`` and the HF ``site`` column drives ``split_group``.
Deterministic (rows sorted by stem, pinned ``fetch_date``), idempotent (a tree that already
has ``CHECKSUMS.sha256`` is only re-verified) and resumable (a file whose size and sha match
the parquet bytes is not rewritten). Runs on the VM, never the Mac (D6); tests never download.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import struct
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import checksums, gate
from .ingest import IngestError, StagedVersion
from .ingest_parquet import _apache_2_0_text
from .manifests import AnnotationCounts, write_annotations_json, write_license, write_source_json
from .models import Profile, Source
from .sample_schema import V2_FIELD_NAMES, SampleRow, normalise_split, to_table, validate_table

SOURCE_ID = "coralscapes-v2"
VERSION = "2.0"
HF_REPO = "josauder/coralscapesV2"
DEFAULT_REVISION = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"  # card dated 2026-09-15
ALLOW_PATTERNS = ("instances/*", "README.md", "LICENSE*", "extended_metadata.json")
PARTITION = "default"
SHARD_RE = re.compile(r"^(?P<split>train|validation|test)-\d{5}-of-\d{5}\.parquet$")
EXPECTED_SPLIT_ROWS: Mapping[str, int] = {"train": 1790, "validation": 170, "test": 473}
CLASSES = 96  # raster values 0..95: 0 = unlabelled (ignore), 1-95 classes
FISH_CLASS_ID, LABEL_DIVISOR = 25, 2000  # extended_metadata.json (cityscapes encoding)
FETCH_DATE = dt.date(2026, 9, 15)
"""Date of the pinned upstream revision; pinned so re-runs are byte-identical (--fetch-date)."""
ATTRIBUTION = f"Coralscapes V2, https://huggingface.co/datasets/{HF_REPO} (arXiv:2609.12826)"
_COLUMNS = ("image", "label", "instance_ids", "stem", "site")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

Progress = Callable[[int, int], None]


def _progress_stderr(done: int, total: int) -> None:
    print(f"[stage] {done}/{total} frames", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------ download


def resolve_revision(
    revision: str, *, fetch_json: Callable[[str], Mapping[str, object]] | None = None
) -> str:
    """The full 40-hex sha for ``revision``; a prefix must equal the API's ``sha`` field."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if not re.fullmatch(r"[0-9a-f]{7,39}", revision):
        raise IngestError(f"{HF_REPO}: {revision!r} is neither a commit sha nor a sha prefix")
    if fetch_json is None:
        import urllib.request

        def fetch_json(url: str) -> Mapping[str, object]:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)

    sha = str(fetch_json(f"https://huggingface.co/api/datasets/{HF_REPO}").get("sha", ""))
    if not sha.startswith(revision):
        raise IngestError(
            f"{HF_REPO}: HEAD is {sha[:12]}, not {revision}; pass the full pinned sha"
        )
    return sha


def download(
    hf_cache: Path, revision: str = DEFAULT_REVISION, *, snapshot_download: Callable | None = None
) -> Path:
    """``snapshot_download`` of the ``instances`` config + card files into ``hf_cache``."""
    sha = resolve_revision(revision)
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as _hf_snapshot_download

        snapshot_download = _hf_snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=sha,
        allow_patterns=list(ALLOW_PATTERNS),
        local_dir=str(hf_cache),
    )
    return hf_cache


def find_shards(hf_cache: Path) -> list[Path]:
    """``instances/*.parquet`` below ``hf_cache`` (never the network), sorted by name."""
    shards = sorted((hf_cache / "instances").glob("*.parquet"), key=lambda p: p.name)
    if not shards:
        raise IngestError(f"no instances/*.parquet under {hf_cache}; run with --download first")
    bad = [p.name for p in shards if not SHARD_RE.match(p.name)]
    if bad:
        raise IngestError(f"unexpected shard name(s) {bad}; expected {SHARD_RE.pattern!r}")
    return shards


# ------------------------------------------------------------------------------ decoding


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int


def png_header(payload: bytes, *, label: str) -> PngHeader:
    """IHDR of a PNG without decoding it; anything else is refused."""
    if payload[:8] != _PNG_MAGIC or payload[12:16] != b"IHDR":
        raise IngestError(f"{label}: bytes are not a PNG (magic {payload[:8]!r})")
    width, height, depth, ctype = struct.unpack(">IIBB", payload[16:26])
    return PngHeader(width, height, depth, ctype)


def _struct_bytes(cell: object, *, label: str) -> tuple[bytes, str | None]:
    payload = cell.get("bytes") if isinstance(cell, Mapping) else None
    if not isinstance(payload, bytes) or not payload:
        raise IngestError(f"{label}: Image struct carries no bytes")
    path = cell.get("path") if isinstance(cell, Mapping) else None
    return payload, (str(path) if path else None)


def instance_png(cell: object, *, label: str) -> bytes:
    """Image feature -> bytes verbatim; a 2-D array -> uint16 PNG (PIL ``I;16``)."""
    if isinstance(cell, Mapping):
        return _struct_bytes(cell, label=label)[0]
    import numpy as np
    from PIL import Image

    array = np.asarray(cell, dtype=np.uint16)
    if array.ndim != 2:
        raise IngestError(f"{label}: instance_ids array must be 2-D, got shape {array.shape}")
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class Frame:
    split: str
    shard: Path
    stem: str
    site: str
    upstream_path: str
    image: bytes
    label: bytes
    instance: bytes


def count_rows(shards: Sequence[Path]) -> dict[str, int]:
    """Rows per split from the parquet footers only."""
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for shard in shards:
        split = SHARD_RE.match(shard.name).group("split")  # type: ignore[union-attr]
        counts[split] = counts.get(split, 0) + pq.ParquetFile(shard).metadata.num_rows
    return counts


def iter_frames(shards: Sequence[Path], *, limit: int | None = None) -> Iterator[Frame]:
    """Frames in shard-name order, one batch in memory at a time; stops after ``limit``."""
    import pyarrow.parquet as pq

    seen = 0
    for shard in shards:
        split = SHARD_RE.match(shard.name).group("split")  # type: ignore[union-attr]
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=4, columns=list(_COLUMNS)):
            cols = {name: batch.column(name) for name in _COLUMNS}
            for i in range(batch.num_rows):
                if limit is not None and seen >= limit:
                    return
                stem, site = cols["stem"][i].as_py(), cols["site"][i].as_py()
                where = f"{shard.name}#{stem}"
                if not stem or not site:
                    raise IngestError(f"{where}: empty stem/site column")
                image, upstream_path = _struct_bytes(cols["image"][i].as_py(), label=where)
                label, _ = _struct_bytes(cols["label"][i].as_py(), label=where)
                instance = instance_png(cols["instance_ids"][i].as_py(), label=where)
                seen += 1
                yield Frame(
                    split, shard, stem, site, upstream_path or f"{stem}_leftImg8bit.png",
                    image, label, instance,
                )  # fmt: skip


# ------------------------------------------------------------------------------ staging


def _put(root: Path, rel: str, payload: bytes) -> str:
    """Write ``payload`` verbatim unless an identical file (size, then sha) is already there."""
    digest = hashlib.sha256(payload).hexdigest()
    dest = root / rel
    if (
        dest.is_file()
        and dest.stat().st_size == len(payload)
        and checksums.file_digest(dest) == digest
    ):
        return digest
    return checksums.write_digest(dest, payload)


def _split_group(source: Source, frame: Frame, file_stem: str) -> str:
    """The registry rule when it declares a pattern, else ``<source>/<site column>``."""
    if source.split_group.pattern is not None:
        return source.split_group_for(
            stem=file_stem, upstream_path=frame.upstream_path, partition=PARTITION
        )
    return f"{source.id}/{frame.site}"


def _metadata_table(rows: Sequence[SampleRow]):
    """37 sample-schema v2 columns + the trailing ``partition`` column, validated."""
    import pyarrow as pa

    table = to_table(rows).select(list(V2_FIELD_NAMES))
    table = table.append_column(
        pa.field("partition", pa.string()), pa.array([PARTITION] * table.num_rows, pa.string())
    )
    table = table.replace_schema_metadata({b"marinedata.sample_schema": b"2"})
    validate_table(table)
    return table


def _write_metadata(path: Path, rows: Sequence[SampleRow]) -> str:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_metadata_table(rows), path, compression="zstd", write_statistics=False)
    return checksums.file_digest(path)


def stage(
    source: Source,
    profile: Profile,
    shards: Sequence[Path],
    *,
    out_root: Path,
    revision: str = DEFAULT_REVISION,
    limit: int | None = None,
    fetch_date: dt.date = FETCH_DATE,
    license_text: str | None = None,
    expected_rows: Mapping[str, int] | None = EXPECTED_SPLIT_ROWS,
    progress: Progress = _progress_stderr,
    progress_every: int = 100,
) -> StagedVersion:
    """Stage every frame of ``shards`` into ``<out_root>/sources/<id>/<version>/``."""
    if (source.id, source.version) != (SOURCE_ID, VERSION):
        raise IngestError(f"expected {SOURCE_ID} {VERSION}, got {source.id} {source.version}")
    decision = gate.evaluate(source, profile)
    decision.raise_if_denied()
    root = out_root / "sources" / source.id / source.version
    if (root / checksums.CHECKSUM_FILE).is_file():  # finished tree: verify only, never refetch
        staged = json.loads((root / "SOURCE.json").read_text()).get("_ingest", {})
        if staged.get("limit") != limit:
            raise IngestError(
                f"{root} was staged with limit={staged.get('limit')}; use a fresh --out"
            )
        manifest = checksums.write_checksums(root)
        images = len(list((root / "images" / PARTITION).glob("*")))
        return StagedVersion(source.id, source.version, root, images, manifest)

    counts = count_rows(shards)
    if limit is None and expected_rows is not None:
        wanted = {k: v for k, v in expected_rows.items() if v}
        if {k: v for k, v in counts.items() if v} != wanted:
            raise IngestError(f"{source.id}: expected rows {wanted}, found {counts}; short tree")
    total = min(sum(counts.values()), limit) if limit is not None else sum(counts.values())

    recorded: dict[str, str] = {}
    rows: list[SampleRow] = []
    upstream: dict[str, Mapping[str, object]] = {}
    observed: set[int] = set()
    seen_stems: set[str] = set()
    from PIL import Image

    for frame in iter_frames(shards, limit=limit):
        if frame.stem in seen_stems:
            raise IngestError(f"{source.id}: duplicate stem {frame.stem!r}")
        seen_stems.add(frame.stem)
        if frame.shard.name not in upstream:
            upstream[frame.shard.name] = {
                "url": f"instances/{frame.shard.name}",
                "sha256": checksums.file_digest(frame.shard),
                "bytes": frame.shard.stat().st_size,
            }
        where = f"{source.id}:{frame.stem}"
        head = png_header(frame.image, label=f"{where} image")
        mask = png_header(frame.label, label=f"{where} label")
        inst = png_header(frame.instance, label=f"{where} instance_ids")
        if (mask.bit_depth, mask.color_type) != (8, 0) or (inst.bit_depth, inst.color_type) != (
            16,
            0,
        ):
            raise IngestError(f"{where}: label must be 8-bit gray and instance_ids 16-bit gray")
        if {(head.width, head.height)} != {(h.width, h.height) for h in (mask, inst)}:
            raise IngestError(f"{where}: image/label/instance_ids sizes differ")
        with Image.open(io.BytesIO(frame.label)) as im:
            observed.update(value for _, value in im.getcolors(maxcolors=256) or [])

        file_stem = f"{frame.stem}_leftImg8bit"
        image_rel = f"images/{PARTITION}/{file_stem}.png"
        mask_rel = f"labels/masks/{PARTITION}/{file_stem}.png"
        inst_rel = f"labels/instances/{PARTITION}/{frame.stem}_instanceIds.png"
        for rel, payload in (
            (image_rel, frame.image),
            (mask_rel, frame.label),
            (inst_rel, frame.instance),
        ):
            recorded[rel] = _put(root, rel, payload)
        rows.append(
            SampleRow(
                sample_id=f"{source.id}/{file_stem}", source_id=source.id,
                source_version=source.version, stem=file_stem, image_path=image_rel,
                image_sha256=recorded[image_rel], image_bytes=len(frame.image),
                image_format="png", license=source.licence.id, attribution=ATTRIBUTION,
                fetch_date=fetch_date, width=head.width, height=head.height,
                upstream_id=frame.stem, upstream_digest=upstream[frame.shard.name]["sha256"],
                split_hint=normalise_split(frame.split),
                label_refs=(mask_rel, inst_rel),
                split_group=_split_group(source, frame, file_stem),
                upstream_split=frame.split, upstream_path=frame.upstream_path,
            )
        )  # fmt: skip
        if len(rows) % progress_every == 0:
            progress(len(rows), total)
    if not rows:
        raise IngestError(f"{source.id}: no frames staged")

    recorded["metadata.parquet"] = _write_metadata(root / "metadata.parquet", rows)
    ingest_meta = {
        "ingest_version": 1,
        "stem_rule": "hf-instances-stem-column+_leftImg8bit",
        "partition_rule": f"literal:{PARTITION}",
        "upstream": [upstream[name] for name in sorted(upstream)],
        "fetched_uri": f"https://huggingface.co/datasets/{HF_REPO}",
        "revision": revision,
        "config": "instances",
        "limit": limit,
        "gate": {"profile": profile.id, "allowed": decision.allowed, "reason": decision.reason},
    }
    recorded["SOURCE.json"] = write_source_json(root / "SOURCE.json", source, ingest_meta)
    annotation = source.annotations[0] if source.annotations else None
    geometries = (
        {
            "kind": annotation.kind.value if annotation else "dense-mask",
            "path": "labels/masks/", "format": "png-indexed",
            "schema_id": source.loader.schema_id if source.loader else "dataset-native",
            "crosswalk_id": source.loader.crosswalk_id if source.loader else None,
            "classes": CLASSES, "raster_ignore_value": 0, "images_covered": len(rows),
            "rows": None, "supervises": list(annotation.supervises) if annotation else [],
            "observed_indices": sorted(observed),
        },
        {
            "kind": "instance-mask", "path": "labels/instances/", "format": "png-uint16",
            "encoding": "cityscapes", "label_divisor": LABEL_DIVISOR,
            "fish_class_id": FISH_CLASS_ID, "images_covered": len(rows), "rows": None,
        },
    )  # fmt: skip
    counts_json = AnnotationCounts(
        images=len(rows), images_without_annotation=0, geometries=geometries
    )
    recorded["ANNOTATIONS.json"] = write_annotations_json(
        root / "ANNOTATIONS.json", source, counts_json
    )
    recorded["LICENSE"] = write_license(
        root / "LICENSE", source, license_text or _apache_2_0_text()
    )
    manifest = checksums.write_checksums(root, recorded=recorded)  # CHECKSUMS last
    return StagedVersion(source.id, source.version, root, len(rows), manifest)


# ------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stage_coralscapes_v2", description=__doc__.split("\n")[0])
    p.add_argument("--download", action="store_true", help="snapshot_download into --hf-cache")
    p.add_argument("--stage", action="store_true", help="stage instances/*.parquet from --hf-cache")
    p.add_argument("--hf-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, help="output root; the tree is <out>/sources/<id>/<ver>/")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="full sha (or a prefix of HEAD)")
    p.add_argument("--limit", type=int, help="stage only the first N frames (trial runs)")
    p.add_argument("--profile", default="research", help="release profile for the licence gate")
    p.add_argument("--fetch-date", type=dt.date.fromisoformat, default=FETCH_DATE)
    p.add_argument("--registry", type=Path, help="registry root (default: this checkout)")
    return p


def _license_text(hf_cache: Path) -> str | None:
    found = sorted(hf_cache.glob("LICENSE*"))
    return found[0].read_text(encoding="utf-8") if found else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.download or args.stage):
        parser.error("pass --download and/or --stage")
    if args.stage and args.out is None:
        parser.error("--stage needs --out")
    if args.download:
        download(args.hf_cache, args.revision)
        print(f"downloaded {HF_REPO}@{resolve_revision(args.revision)} -> {args.hf_cache}")
    if args.stage:
        from .registry import Registry

        registry = Registry.load(args.registry)
        result = stage(
            registry.source(SOURCE_ID), registry.profile(args.profile), find_shards(args.hf_cache),
            out_root=args.out, revision=resolve_revision(args.revision), limit=args.limit,
            fetch_date=args.fetch_date, license_text=_license_text(args.hf_cache),
        )  # fmt: skip
        summary = {
            "id": result.source_id, "version": result.version, "images": result.images,
            "files": result.manifest.files, "size_bytes": result.manifest.size_bytes,
            "root_digest": result.manifest.root_digest, "root": str(result.root),
        }  # fmt: skip
        print(json.dumps(summary, sort_keys=True))
        print(f"root_digest={result.manifest.root_digest}")
    return 0
