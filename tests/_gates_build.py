# ruff: noqa: E501
"""Synthetic hf_export build directories for the gate tests: parquet shards with embedded HF
features, a card and a RELEASE.json. No network, no registry."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from marinedata.gates.card import size_category
from marinedata.gates.privacy_detectors import Detection

SPLITS = ("train", "validation", "test")
OPEN_LICENCE, NC_LICENCE = "CC-BY-4.0", "CC-BY-NC-4.0"
DEFAULT_SOURCES = [
    {"id": "src-a", "access_class": "open"},
    {"id": "src-b", "access_class": "open"},
]
_STR = (pa.string(), {"dtype": "string", "_type": "Value"})
_IMG = (pa.struct([("bytes", pa.binary()), ("path", pa.string())]), {"_type": "Image"})
_CLASS_MAP = (
    pa.list_(pa.struct([("id", pa.int32()), ("name", pa.string())])),
    [{"id": {"dtype": "int32", "_type": "Value"}, "name": {"dtype": "string", "_type": "Value"}}],
)
IMAGE_COLS = {"image_sha256": _STR, "image": _IMG, "source_id": _STR, "split_group": _STR}
META_COLS = {
    "image_sha256": _STR, "source_id": _STR, "license": _STR,
    "licence_class": _STR, "attribution": _STR,
}  # fmt: skip
MASK_COLS = {
    "image_sha256": _STR, "source": _STR, "image": _IMG, "mask": _IMG, "class_map": _CLASS_MAP,
    "exif_transposed": (pa.bool_(), {"dtype": "bool", "_type": "Value"}),
}  # fmt: skip


class FakeDetector:
    """A privacy detector that reports one detection per image."""

    name, version = "fake", "1"

    def __init__(self, score: float = 0.9, kind: str = "face") -> None:
        self.score, self.kind = score, kind

    def detect(self, image: bytes) -> list[Detection]:
        return [Detection(self.kind, self.score)]


def smooth_png(seed: int, shift: int = 0, size: int = 64) -> bytes:
    """A smooth structured RGB PNG; ``shift`` brightens it: same pHash, different sha256."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    channels = []
    for _ in range(3):
        fx, fy = rng.uniform(0.5, 3.0, 2)
        px, py = rng.uniform(0, 6.28, 2)
        channels.append(
            127 + 100 * np.sin(2 * np.pi * fx * x + px) * np.cos(2 * np.pi * fy * y + py)
        )
    pixels = np.clip(np.stack(channels, -1) + shift, 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "PNG")
    return buf.getvalue()


GPS_INFO_TAG = 0x8825


def gps_image(seed: int, fmt: str = "PNG", *, gps: bool = True) -> bytes:
    """``smooth_png(seed)`` re-encoded as ``fmt``; ``gps`` writes a synthetic EXIF GPSInfo IFD
    (16 55 12 S, 145 40 30 E) through Pillow."""
    save_args: dict = {}
    if gps:
        exif = Image.Exif()
        exif[GPS_INFO_TAG] = {1: "S", 2: (16.0, 55.0, 12.0), 3: "E", 4: (145.0, 40.0, 30.0)}
        save_args["exif"] = exif
    buf = io.BytesIO()
    with Image.open(io.BytesIO(smooth_png(seed))) as im:
        im.save(buf, fmt, **save_args)
    return buf.getvalue()


@dataclass(frozen=True)
class Img:
    seed: int
    split: str = "train"
    source: str = "src-a"
    group: str | None = None
    shift: int = 0
    sha: str | None = None  # override the image_sha256 column
    png: bytes | None = None  # override the embedded bytes

    @property
    def data(self) -> bytes:
        return self.png if self.png is not None else smooth_png(self.seed, self.shift)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def default_images() -> list[Img]:
    plan = ["train"] * 16 + ["validation"] * 2 + ["test"] * 2
    return [Img(i, s, "src-a" if i % 2 == 0 else "src-b") for i, s in enumerate(plan)]


def write_config(build: Path, config: str, by_split: dict[str, list[dict]], cols: dict) -> None:
    for split, rows in by_split.items():
        table = pa.table(
            {n: pa.array([r.get(n) for r in rows], type=t) for n, (t, _) in cols.items()}
        )
        features = {n: f for n, (_, f) in cols.items()}
        meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
        path = Path(build) / "data" / config / f"{split}-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(meta), path)


def meta_row(sha: str = "0" * 64, source: str = "src-a", **over) -> dict:
    row = {
        "image_sha256": sha, "source_id": source, "license": OPEN_LICENCE,
        "licence_class": "open", "attribution": "Photo: A. Smith",
    }  # fmt: skip
    return {**row, **over}


def mask_rows(names=("coral", "sand"), n: int = 4) -> list[dict]:
    rows = []
    for i in range(n):
        image = smooth_png(100 + i)
        ids = np.where(np.mgrid[0:64, 0:64][1] > 31 + i, 1, 2).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(ids).save(buf, "PNG")
        rows.append({
            "image_sha256": hashlib.sha256(image).hexdigest(), "source": "src-a",
            "image": {"bytes": image, "path": None}, "mask": {"bytes": buf.getvalue(), "path": None},
            "class_map": [{"id": 1, "name": names[0]}, {"id": 2, "name": names[1]}],
            "exif_transposed": False,
        })  # fmt: skip
    return rows


def card_text(configs: dict[str, dict[str, int]], *, categories=None, info_lines=None) -> str:
    biggest = max((sum(s.values()) for s in configs.values()), default=0)
    bands = [size_category(biggest)] if categories is None else list(categories)
    lines = ["---", "license: cc-by-4.0"]
    if bands:
        lines += ["size_categories:", *[f"- {b}" for b in bands]]
    lines.append("configs:")
    for config, splits in configs.items():
        lines += [f"- config_name: {config}", "  data_files:"]
        for split in splits:
            lines += [f"  - split: {split}", f"    path: data/{config}/{split}-*.parquet"]
    if info_lines:
        lines += ["dataset_info:", *info_lines]
    return "\n".join([*lines, "---", "", "# Fixture card", ""])


def images_info(*, sha_dtype="string", rows=None, download_size=None) -> list[str]:
    lines = ["- config_name: images", "  features:"]
    for name, dtype in (
        ("image_sha256", sha_dtype),
        ("image", "image"),
        ("source_id", "string"),
        ("split_group", "string"),
    ):
        lines += [f"  - name: {name}", f"    dtype: {dtype}"]
    if rows:
        lines.append("  splits:")
        for split, n in rows.items():
            lines += [f"  - name: {split}", f"    num_examples: {n}"]
    if download_size is not None:
        lines.append(f"  download_size: {download_size}")
    return lines


def edit_release(build: Path, **updates) -> None:
    path = Path(build) / "RELEASE.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **updates}))


def make_build(
    root: Path,
    *,
    images=None,
    extra_metadata=(),
    masks: bool = True,
    class_names=("coral", "sand"),
    card: bool = True,
    card_configs=None,
    categories=None,
    info_lines=None,
    release: bool = True,
    release_configs: bool = True,
    flavour: str | None = "open",
    sources=None,
    ratio=None,
    group_column: bool = True,
    licence: str = OPEN_LICENCE,
    licence_class: str = "open",
) -> Path:
    build = Path(root) / "build"
    build.mkdir(parents=True, exist_ok=True)
    imgs = default_images() if images is None else list(images)
    cols = (
        IMAGE_COLS if group_column else {k: v for k, v in IMAGE_COLS.items() if k != "split_group"}
    )
    by_split: dict[str, list[dict]] = {}
    for im in imgs:
        by_split.setdefault(im.split, []).append({
            "image_sha256": im.sha or im.digest, "image": {"bytes": im.data, "path": None},
            "source_id": im.source, "split_group": im.group or f"g{im.seed}",
        })  # fmt: skip
    configs: dict[str, dict[str, int]] = {}
    if by_split:
        write_config(build, "images", by_split, cols)
        configs["images"] = {s: len(by_split[s]) for s in SPLITS if s in by_split}
    meta = {
        s: [meta_row(r["image_sha256"], r["source_id"], license=licence, licence_class=licence_class) for r in rows]
        for s, rows in by_split.items()
    }  # fmt: skip
    for row in extra_metadata:
        meta.setdefault("train", []).append(meta_row(**row))
    if meta:
        write_config(build, "metadata", meta, META_COLS)
        configs["metadata"] = {s: len(meta[s]) for s in SPLITS if s in meta}
    if masks:
        rows = mask_rows(class_names)
        write_config(build, "coral-masks", {"train": rows}, MASK_COLS)
        configs["coral-masks"] = {"train": len(rows)}
    if card:
        (build / "README.md").write_text(
            card_text(card_configs or configs, categories=categories, info_lines=info_lines)
        )
    if release:
        doc = {"release": "fx", "lock_sources": DEFAULT_SOURCES if sources is None else sources}
        if flavour:
            doc["flavour"] = flavour
        if release_configs:
            doc["configs"] = configs
        if ratio:
            doc["split_ratio"] = ratio
        (build / "RELEASE.json").write_text(json.dumps(doc))
    return build
