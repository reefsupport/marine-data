"""Bleaching and condition labels into the unified ``image_labels`` table (WP-U5).

One row per (image, native label) in :mod:`marinedata.annotation_schema`'s ``image_labels`` table.
The node columns come from the source's crosswalk edge: ``condition_node_id`` (and ``form_node_id``)
from the edge's own targets, ``taxon_node_id`` plus the derived rank / AphiaID / L2 code from
:meth:`marinedata.schema.Crosswalk.resolve`. ``label_key`` is ``condition`` for the bleaching sources,
``species`` for the ICRA presence set, ``class`` for an edge that sets no condition (Roboflow's
"Non-Corals"). Nine sources (:data:`LABEL_SOURCES`), four readers:

* ``labels``: staged ``labels/image_labels.parquet`` (``stem, partition, label, schema_id, confidence``);
* ``kv``: the older ``stem, key, value`` shape (kaggle-healthy-bleached-corals);
* ``mask``: one row per (image, class present) from ``labels/masks/*.png`` (reef-support-bleaching:
  1 bleached, 2 non_bleached; ``attrs.pixel_count`` keeps the pixels);
* ``yolo``: noaa-pifsc-esa-coral-icra has no staged labels, so the YOLO ``.txt`` files are read
  anonymously from the pinned public HF revision (no login, held in memory, never stored) and
  reduced to one presence row per (image, class) with ``attrs.n_boxes``.

``attrs.licence_class`` is set on every row (the WP-U4 pattern); :func:`drop_internal_only` is the
release filter (kaggle-healthy-bleached-corals has no verifiable upstream licence: internal-only).
"""  # noqa: E501

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    normalise_split,
    write_annotations,
)
from ..registry import Registry
from ..schema import Axis
from .masks_table import INTERNAL_ONLY, drop_internal_only, is_internal_only
from .points_table import _taxon_rank, resolver_for
from .s3_keyed import StagedTree, _pixel_counts

__all__ = ["INTERNAL_ONLY", "LABEL_SOURCES", "drop_internal_only", "is_internal_only"]

HF_HOST = "https://huggingface.co"
FetchLabel = Callable[[str], "bytes | None"]


@dataclass(frozen=True)
class LabelSource:
    """How one staged source's image labels read and resolve."""

    source_id: str
    version: str
    crosswalk_id: str
    reader: str  # labels | kv | mask | yolo
    annotator: str  # a label-origin value (annotator_from_origin)
    ann_license: str | None
    licence_class: str
    label_key: str = "condition"
    label_set: str = "dataset-native"
    mask_values: Mapping[int, str] = field(default_factory=dict)
    class_names: Mapping[int, str] = field(default_factory=dict)
    hf: tuple[str, str] | None = None  # (repo, pinned revision) of an unstaged label set

    @property
    def tree(self) -> str:
        return f"{self.source_id}/{self.version}"


_RF = "roboflow-bleaching-native"
_HB, _HU = "roboflow-bleaching-condition-hb", "roboflow-bleaching-condition-hu"
# licence_class of the seven existing sources: in neither lic-A/lic-B TSV, so the registry tier
# (all PERMISSIVE: US-GOV-PD / CC-BY-4.0); the kaggle and ICRA classes are manager decisions LIC-A/B.  # noqa: E501
LABEL_SOURCES: dict[str, LabelSource] = {
    s.source_id: s
    for s in (
        LabelSource("noaa-pifsc-bleaching", "1-image-labels", "noaa-pifsc-bleaching-condition",
                    "labels", "human_expert", "US-GOV-PD", "permissive"),
        LabelSource("roboflow-coral-bleaching-final-v6i", "v6i-image-labels-r2", _HB, "labels",
                    "human_crowd", "CC-BY-4.0", "permissive", label_set=_RF),
        LabelSource("roboflow-coral-bleaching-general-v1-yolov8s", "v1-yolov8s-image-labels-r2",
                    _HB, "labels", "human_crowd", "CC-BY-4.0", "permissive", label_set=_RF),
        LabelSource("roboflow-coral-classification-copy-changed-v13i", "v13i-image-labels", _HB,
                    "labels", "human_crowd", "CC-BY-4.0", "permissive", label_set=_RF),
        LabelSource("roboflow-coral-reef-bleach-detection-v2i", "v2i-image-labels", _HB, "labels",
                    "human_crowd", "CC-BY-4.0", "permissive", label_set=_RF),
        LabelSource("roboflow-coral-reef-classification-v3i", "v3i-image-labels", _HU, "labels",
                    "human_crowd", "CC-BY-4.0", "permissive"),
        # condition raster: 0 unlabelled, 1 bleached, 2 non_bleached (loaders/staged_tree.py).
        LabelSource("reef-support-bleaching", "2026-09-24", "reef-support-bleaching-condition",
                    "mask", "human_expert", "CC-BY-4.0", "permissive",
                    mask_values={1: "bleached", 2: "non_bleached"}),
        LabelSource("kaggle-healthy-bleached-corals", "zip-5e15452254a9",
                    "kaggle-healthy-bleached-corals", "kv", "human", None, INTERNAL_ONLY),
        # data.yaml: nc 1, names ['ICRA'] (Isopora crateriformis); README: public domain (NOAA).
        LabelSource("noaa-pifsc-esa-coral-icra", "rev-b5e722f7c11e", "noaa-pifsc-esa-coral-icra",
                    "yolo", "human", "US-GOV-PD", "open", label_key="species",
                    class_names={0: "ICRA"},
                    hf=("NMFS-OSI/NOAA-PIFSC-ESD-ESA-CORAL-ICRA-Dataset",
                        "b5e722f7c11e310afd75a2bdf64c846b63955371")),
    )
}  # fmt: skip


@dataclass(frozen=True)
class Resolved:
    taxon: str | None
    form: str | None
    condition: str | None
    match_type: str
    worms_aphia_id: int | None
    rs_benthic_code: str | None


def axes_resolver_for(registry: Registry, crosswalk_id: str) -> Callable[[str], Resolved]:
    """Memoised ``native -> Resolved``: the taxon facts from ``Crosswalk.resolve`` plus the
    form / condition targets of the same edge (an ``unmapped`` label has none)."""
    walk = registry.crosswalk(crosswalk_id)
    taxon_of = resolver_for(registry, crosswalk_id)
    cache: dict[str, Resolved] = {}

    def resolve(native: str) -> Resolved:
        if native not in cache:
            taxon, match, aphia, l2 = taxon_of(native)
            edge = walk.edge(native)
            targets = edge.targets if edge is not None and match != "unmapped" else {}
            cache[native] = Resolved(
                taxon, targets.get(Axis.FORM), targets.get(Axis.CONDITION), match, aphia, l2
            )
        return cache[native]

    return resolve


def image_label_row(
    *,
    spec: LabelSource,
    registry: Registry,
    resolve: Callable[[str], Resolved],
    ordinal: int,
    image_sha256: str,
    native: str,
    split: str | None = None,
    confidence: float | None = None,
    label_native_id: str | None = None,
    label_set: str | None = None,
    attrs: Mapping[str, object] | None = None,
) -> dict:
    """One ``image_labels`` row (``attrs.licence_class`` always set)."""
    got = resolve(native)
    typ, detail = annotator_from_origin(spec.annotator)
    mapped = got.match_type != "unmapped"
    key = spec.label_key
    if key == "condition" and mapped and got.condition is None:
        key = "class"  # e.g. Roboflow "Non-Corals": a presence tag on the taxon axis, no condition
    return {
        "image_sha256": image_sha256,
        "source_id": spec.source_id,
        "source_version": spec.version,
        "ann_id": f"{spec.source_id}:{ordinal}",
        "label_native": native,
        "label_native_id": label_native_id,
        "label_set": label_set or spec.label_set,
        "taxon_node_id": got.taxon,
        "form_node_id": got.form,
        "condition_node_id": got.condition,
        "taxon_rank": _taxon_rank(registry, got.taxon),
        "worms_aphia_id": got.worms_aphia_id,
        "rs_benthic_code": got.rs_benthic_code,
        "match_type": got.match_type,
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": spec.ann_license,
        "confidence": confidence,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(
            {**dict(attrs or {}), "licence_class": spec.licence_class}, sort_keys=True
        ),
        "label_key": key,
    }  # noqa: E501, RUF100


def _splits(tree: StagedTree) -> dict[tuple[str, str], str | None]:
    try:
        rows = tree.table("metadata.parquet")
    except Exception:  # metadata is optional for the join: a missing split is a null split
        return {}
    return {
        (r.get("partition") or "default", r["stem"]): r.get("upstream_split") or r.get("split_hint")
        for r in rows
    }


def _stride(keys: list, limit: int | None) -> list:
    if limit is None or len(keys) <= limit:
        return keys
    return [keys[i * len(keys) // limit] for i in range(limit)]  # evenly spaced, deterministic


def _sha(tree: StagedTree, spec: LabelSource, key: tuple[str, str]) -> str:
    sha = tree.shas.get(key)
    if sha is None:
        raise ValueError(f"{spec.source_id}: image label references unstaged image {key!r}")
    return sha


def staged_table_rows(
    tree: StagedTree, spec: LabelSource, registry: Registry, *, limit: int | None = None
) -> Iterator[dict]:
    """``labels`` / ``kv`` readers: staged ``labels/image_labels.parquet``, at most ``limit`` images."""  # noqa: E501
    resolve = axes_resolver_for(registry, spec.crosswalk_id)
    splits = _splits(tree)
    rows = tree.table("labels/image_labels.parquet")
    by_image: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_image.setdefault((r.get("partition") or "default", r["stem"]), []).append(r)
    ordinal = 0
    for key in _stride(sorted(by_image), limit):
        for r in by_image[key]:
            native = r["label"] if spec.reader == "labels" else r["value"]
            extra = {"label_column": r["key"]} if spec.reader == "kv" else None
            yield image_label_row(
                spec=spec, registry=registry, resolve=resolve, ordinal=ordinal,
                image_sha256=_sha(tree, spec, key), native=native, split=splits.get(key),
                confidence=r.get("confidence"), label_set=r.get("schema_id") or spec.label_set,
                attrs=extra,
            )  # fmt: skip
            ordinal += 1


def mask_presence_rows(
    tree: StagedTree,
    spec: LabelSource,
    registry: Registry,
    *,
    limit: int | None = None,
    workers: int = 16,
) -> Iterator[dict]:
    """``mask`` reader: one row per (image, class value present in its mask)."""
    resolve = axes_resolver_for(registry, spec.crosswalk_id)
    splits = _splits(tree)
    items = _stride([(k, rel) for k, rel in tree.masks() if k in tree.shas], limit)

    def counts(item: tuple[tuple[str, str], str]) -> dict[int, int] | None:
        data = tree.get_or_skip(item[1])
        return None if data is None else _pixel_counts(data)

    ordinal = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (key, rel), got in zip(items, pool.map(counts, items), strict=True):
            for value, n in sorted((got or {}).items()):
                if value not in spec.mask_values or n <= 0:
                    continue
                yield image_label_row(
                    spec=spec, registry=registry, resolve=resolve, ordinal=ordinal,
                    image_sha256=tree.shas[key], native=spec.mask_values[value],
                    split=splits.get(key), label_native_id=str(value),
                    attrs={"evidence": "mask", "pixel_count": n, "mask_key": tree.key(rel)},
                )  # fmt: skip
                ordinal += 1


def hf_label_fetcher(repo: str, revision: str, *, tries: int = 4) -> FetchLabel:
    """Anonymous GET of one file of a public HF dataset at a pinned revision (no token);
    ``None`` on a 404, a retry with back-off on a transient error."""

    def fetch(path: str) -> bytes | None:
        url = f"{HF_HOST}/datasets/{repo}/resolve/{revision}/{path}"
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: RUF100, S310
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                err: Exception = exc
            except OSError as exc:
                err = exc
            time.sleep(1.5 * (attempt + 1))
        raise err

    return fetch


def parse_yolo_classes(text: str) -> dict[int, int]:
    """``{class id: n boxes}`` of one YOLO ``.txt`` (``class x y w h`` per line; blanks skipped)."""
    out: dict[int, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if parts:
            out[int(parts[0])] = out.get(int(parts[0]), 0) + 1
    return out


def yolo_label_path(upstream_id: str) -> str:
    """``dataset/images/test/x.JPG`` -> ``dataset/labels/test/x.txt``."""
    p = PurePosixPath(upstream_id)
    parts = ["labels" if part == "images" else part for part in p.parent.parts]
    return str(PurePosixPath(*parts) / f"{p.stem}.txt")


def yolo_presence_rows(
    tree: StagedTree,
    spec: LabelSource,
    registry: Registry,
    *,
    fetch: FetchLabel | None = None,
    limit: int | None = None,
    workers: int = 8,
) -> Iterator[dict]:
    """``yolo`` reader: one row per (image, class with at least one box); images with an empty
    label file get no row. The label path is derived from the staged ``metadata.upstream_id``."""
    assert spec.hf is not None, "yolo reader needs the pinned (repo, revision)"
    fetch = fetch or hf_label_fetcher(*spec.hf)
    resolve = axes_resolver_for(registry, spec.crosswalk_id)
    meta = {r["stem"]: r for r in tree.table("metadata.parquet")}
    stems = [s for s in _stride(sorted(meta), limit) if ("default", s) in tree.shas]

    def read(stem: str) -> bytes | None:
        return fetch(yolo_label_path(meta[stem]["upstream_id"]))

    ordinal = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stem, data in zip(stems, pool.map(read, stems), strict=True):
            if data is None:
                continue
            for cls, n_boxes in sorted(parse_yolo_classes(data.decode()).items()):
                yield image_label_row(
                    spec=spec, registry=registry, resolve=resolve, ordinal=ordinal,
                    image_sha256=tree.shas[("default", stem)],
                    native=spec.class_names.get(cls, f"__unknown_class_{cls}"),
                    split=meta[stem].get("split_hint"), label_native_id=str(cls),
                    attrs={"evidence": "yolo-box-presence", "n_boxes": n_boxes,
                           "label_source": f"hf:{spec.hf[0]}@{spec.hf[1][:12]}"},
                )  # fmt: skip
                ordinal += 1


def staged_rows(
    tree: StagedTree, spec: LabelSource, registry: Registry, *, limit: int | None = None, **kw
) -> Iterator[dict]:
    """Dispatch on ``spec.reader``."""
    if spec.reader in ("labels", "kv"):
        return staged_table_rows(tree, spec, registry, limit=limit)
    if spec.reader == "mask":
        return mask_presence_rows(tree, spec, registry, limit=limit)
    if spec.reader == "yolo":
        return yolo_presence_rows(tree, spec, registry, limit=limit, **kw)
    raise ValueError(f"{spec.source_id}: unknown reader {spec.reader!r}")


def write_image_labels(
    data_dir: Path, source_id: str, source_version: str, rows: list[dict]
) -> Path:
    """Write rows to ``<data_dir>/_annotations/image_labels/<source_id>/<version>.parquet``."""
    path = annotation_path(data_dir, "image_labels", source_id, source_version)
    write_annotations(path, "image_labels", rows)
    return path
