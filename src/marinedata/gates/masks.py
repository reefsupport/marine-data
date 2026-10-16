"""Masks gate (Phase A, P1.5): refuse a build whose masks do not line up with their images.

Reads every mask-config parquet shard under a build directory (``image`` + ``mask`` plus
``class_map`` for semantic or ``instances`` for instance configs), decodes the embedded cells and
checks each row:

* ``size_mismatch``    mask size != image size (the exporter never resizes; ``--allow-resize``
  is for fixtures only);
* ``exif_orientation`` the embedded image still carries EXIF orientation != 1: the exporter applies
  ``ImageOps.exif_transpose`` and records ``exif_transposed``; the message says so when the mask
  was drawn on the transposed pixels;
* ``mask_dtype``       semantic masks are uint8, instance masks uint16;
* ``unknown_class_id`` a semantic mask id outside the class_map ids plus 255;
* ``ignore_as_class``  255 is the ignore index and never a class id;
* ``instance_ids``     ``instances[].id`` must be 1..N and cover every non-zero mask id;
* ``empty_rate``       per source, the share of masks with no labelled pixel above the threshold
  (5 % default: the registry has no per-source field for it yet).

* v1.1.0 ``semseg`` shards (``mask`` + ``palette``, no ``image``) are checked by
  :mod:`.masks_palette`: a mode-P PNG whose values and colours are the declared palette.

A build without mask rows fails: a gate that checked nothing must not pass. With ``v1_dir``
(an additive release) the v1.0 tree's mask shards are scanned when the build has none."""

from __future__ import annotations

import io
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from . import MAX_FAILURES, GateResult
from .masks_palette import check_palette_row

NAME = "masks"
IGNORE = 255
DEFAULT_EMPTY_THRESHOLD = 0.05
EXIF_ORIENTATION = 0x0112
_COLUMNS = (
    "image_sha256", "source", "source_id", "image", "mask", "class_map", "instances",
    "exif_transposed", "palette", "mask_bytes_sha256", "mask_config",
)  # fmt: skip


def _shard_rows(path: Path) -> tuple[str | None, Iterator[dict]]:
    """``(kind, rows)`` of a shard: semantic, instance, palette, or ``None`` (not a mask config)."""
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(path)
    names = set(shard.schema_arrow.names)
    if {"mask", "palette"} <= names and "image" not in names:
        kind = "palette"
    elif not {"image", "mask"} <= names:
        return None, iter(())
    elif "instances" in names:
        kind = "instance"
    elif "class_map" in names:
        kind = "semantic"
    else:
        return None, iter(())
    columns = [c for c in _COLUMNS if c in names]

    def rows() -> Iterator[dict]:
        for batch in shard.iter_batches(batch_size=16, columns=columns):
            yield from batch.to_pylist()

    return kind, rows()


def _decode(cell):
    from PIL import Image

    data = (cell or {}).get("bytes")
    if not data:
        return None
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _check_row(kind: str, row: dict) -> tuple[list[tuple[str, str]], bool]:
    """``([(failure kind, message)], mask_is_empty)`` for one row."""
    if kind == "palette":
        return check_palette_row(row)
    import numpy as np

    try:
        image, mask = _decode(row.get("image")), _decode(row.get("mask"))
    except (OSError, ValueError) as exc:
        return [("undecodable", str(exc)[:120])], False
    if image is None or mask is None:
        return [("missing_bytes", "image or mask cell holds no bytes")], False
    problems: list[tuple[str, str]] = []
    (iw, ih), (mw, mh) = image.size, mask.size
    orientation = image.getexif().get(EXIF_ORIENTATION, 1)
    if orientation not in (None, 0, 1):
        drawn = (
            " (mask is drawn on the transposed pixels)" if (mw, mh) == (ih, iw) != (iw, ih) else ""
        )
        problems.append(
            ("exif_orientation", f"image still has EXIF orientation {orientation}{drawn}")
        )
    if (mw, mh) != (iw, ih):
        problems.append(("size_mismatch", f"mask {mw}x{mh} != image {iw}x{ih}"))
    values = {int(v) for v in np.unique(np.asarray(mask))}
    if kind == "semantic":
        ids = {int(c["id"]) for c in row.get("class_map") or ()}
        if mask.mode not in ("L", "P"):
            problems.append(("mask_dtype", f"semantic mask is {mask.mode}, want uint8 (L)"))
        if IGNORE in ids:
            problems.append(("ignore_as_class", "class_map lists 255, the ignore index"))
        unknown = sorted(values - ids - {IGNORE})
        if unknown:
            problems.append(
                (
                    "unknown_class_id",
                    f"mask ids {unknown[:6]} not in class_map {sorted(ids)[:6]} + 255",
                )
            )
        return problems, not (values & ids)
    listed = sorted(int(i["id"]) for i in row.get("instances") or ())
    if not mask.mode.startswith("I;16"):
        problems.append(("mask_dtype", f"instance mask is {mask.mode}, want uint16 (I;16)"))
    if listed != list(range(1, len(listed) + 1)):
        problems.append(("instance_ids", f"instances[].id {listed[:6]} is not 1..{len(listed)}"))
    stray = sorted(values - {0} - set(listed))
    if stray:
        problems.append(("instance_ids", f"mask ids {stray[:6]} have no instances[] entry"))
    return problems, not (values - {0})


def run_masks_gate(
    build_dir: Path,
    *,
    empty_threshold: float = DEFAULT_EMPTY_THRESHOLD,
    v1_dir: Path | None = None,
) -> GateResult:
    build_dir = Path(build_dir)
    counts: Counter = Counter()
    found: list[str] = []
    per_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def scan(root: Path) -> None:
        for shard in sorted(root.rglob("*.parquet")):
            kind, rows = _shard_rows(shard)
            if kind is None:
                counts["shards_skipped"] += 1
                continue
            counts["shards"] += 1
            where = shard.relative_to(root)
            for index, row in enumerate(rows):
                problems, empty = _check_row(kind, row)
                source = str(row.get("source") or row.get("source_id") or "?")
                counts["rows"] += 1
                counts[f"{kind}_rows"] += 1
                counts["exif_transposed_rows"] += int(bool(row.get("exif_transposed")))
                per_source[source][0] += 1
                per_source[source][1] += int(empty)
                sha = str(row.get("image_sha256"))[:12]
                for failure, message in problems:
                    counts[f"failed_{failure}"] += 1
                    found.append(f"{where}[{index}] {sha} {failure}: {message}")

    scan(build_dir)
    if (
        not counts["rows"] and v1_dir is not None
    ):  # an additive release: the mask configs are v1.0's
        counts["v1_dir_scanned"] = 1
        scan(Path(v1_dir))
    by_source = {
        s: {"rows": n, "empty": e, "rate": round(e / n, 4)}
        for s, (n, e) in sorted(per_source.items())
    }
    head = []
    for source, stat in by_source.items():
        if stat["rate"] > empty_threshold:
            counts["failed_empty_rate"] += 1
            head.append(
                f"{source}: empty_rate: {stat['empty']}/{stat['rows']} masks have no labelled "
                f"pixel ({stat['rate']:.1%} > {empty_threshold:.1%})"
            )
    if not counts["rows"]:
        searched = f"{build_dir} or {v1_dir}" if v1_dir is not None else f"{build_dir}"
        head.append(f"no mask rows found under {searched}")
    failures = [*head, *found]
    counts["failures_total"] = len(failures)
    out = {**dict(sorted(counts.items())), "empty_by_source": by_source}
    return GateResult(NAME, not failures, out, failures[:MAX_FAILURES])
