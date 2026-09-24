"""Reader for a tree this repo's own writer produced (WS-D staging, D1 §2/§5).

Shape: ``images/<partition>/<stem>.<ext>``, ``metadata.parquet`` (one row per staged
image — the sample index), optionally ``labels/points.parquet`` (sparse point
annotations), ``labels/image_labels.parquet`` (whole-image annotations, WS-D S23)
and/or ``labels/masks/<partition>/<stem>.png`` (dense masks), and ``CHECKSUMS.sha256``
(not read here — that pins bytes for ingest, not for loading).

``metadata.parquet`` is the sample index, not the ``images/`` directory tree, because
the writer (:func:`marinedata.tables.write_metadata_table`) records ``stem`` and
``partition`` but never the file extension — resolving the real file means walking
``images/`` once and matching by ``(partition, stem)``, the same style
:class:`~marinedata.loaders.generic.ImageMaskPairLoader` already uses to pair images
with masks by stem.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..models import SplitGroupRule
from ..sample import Sample
from ..tables import _require_pyarrow
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader


def _by_partition_stem(directory: Path) -> dict[tuple[str, str], Path]:
    """Every file under ``directory``, keyed by ``(partition, stem)`` — the first path
    segment below ``directory`` is the partition, ``""`` for a file sitting directly in
    it."""
    if not directory.is_dir():
        return {}
    by_key: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(directory).parts
        partition = parts[0] if len(parts) > 1 else ""
        by_key[(partition, path.stem)] = path
    return by_key


def _read_parquet(path: Path):
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(path)


@register_loader
class StagedTreeLoader(_HarmonizingLoader):
    """Reads a ``sources/<id>/<version>/`` tree exactly as this repo's ``ingest``
    writers emit it — no params, because the shape is fixed by the writer, not by a
    per-source convention a YAML entry could vary.

    ``split_group`` comes straight from the ``metadata.parquet`` column into
    ``sample.meta["split_group"]`` — the one value
    :func:`marinedata.scan.group_key` reads for ``by="group"``. It is not re-derived
    via :meth:`~marinedata.models.Source.split_group_for`: that rule already ran once,
    at staging time, and its output is pinned into the tree's checksummed bytes. Calling
    it again here would let a registry rule edited *after* staging silently disagree
    with the split a training run already used — the staged value must win.
    """

    layout = "staged-tree"

    def _metadata_path(self) -> Path:
        return self.root / "metadata.parquet"

    def _points_path(self) -> Path:
        return self.root / "labels" / "points.parquet"

    def _image_labels_path(self) -> Path:
        return self.root / "labels" / "image_labels.parquet"

    def validate(self) -> None:
        super().validate()
        metadata_path = self._metadata_path()
        if not metadata_path.is_file():
            raise LoaderError(
                f"{self.source.id}: layout 'staged-tree' expects metadata.parquet at "
                f"{metadata_path}"
            )
        # A source that explicitly overrides the default SplitGroupRule (a
        # source-specific pattern, e.g. Roboflow's `_jpg.rf.` stem prefix) is
        # declaring that cross-source leakage grouping matters for it — so a
        # staged row's split_group must never be null/empty (S28: a converter
        # that hardcoded ``None`` instead of calling ``split_group_for`` produced
        # exactly that). Sources still on the bare fallback rule are left alone:
        # many pre-D1 fixtures/trees legitimately have a null split_group column
        # (see StagedImage.split_group's docstring) and re-deriving one for them
        # is a separate migration, not this guard's job.
        if self.source.split_group != SplitGroupRule():
            null_count = 0
            for record in _read_parquet(metadata_path).to_pylist():
                value = record.get("split_group")
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    null_count += 1
            if null_count:
                raise LoaderError(
                    f"{self.source.id}: {null_count} row(s) in {metadata_path} have a "
                    "null/empty split_group, but this source declares an explicit "
                    "split_group rule — re-stage with a converter that calls "
                    "Source.split_group_for"
                )

    def _points_by_key(self) -> dict[tuple[str, str], list[dict]]:
        path = self._points_path()
        if not path.is_file():
            return {}
        grouped: dict[tuple[str, str], list[dict]] = {}
        for record in _read_parquet(path).to_pylist():
            grouped.setdefault((record["partition"], record["stem"]), []).append(record)
        return grouped

    def _image_labels_by_key(self) -> dict[tuple[str, str], list[dict]]:
        path = self._image_labels_path()
        if not path.is_file():
            return {}
        grouped: dict[tuple[str, str], list[dict]] = {}
        for record in _read_parquet(path).to_pylist():
            grouped.setdefault((record["partition"], record["stem"]), []).append(record)
        return grouped

    def _iter_samples(self) -> Iterator[Sample]:
        images_by_key = _by_partition_stem(self.root / "images")
        masks_by_key = _by_partition_stem(self.root / "labels" / "masks")
        points_by_key = self._points_by_key()
        image_labels_by_key = self._image_labels_by_key()

        for record in _read_parquet(self._metadata_path()).to_pylist():
            partition = record["partition"]
            stem = record["stem"]
            key = (partition, stem)
            image = images_by_key.get(key)
            if image is None:
                if self.partial:
                    continue  # sampled sets are legitimately incomplete
                raise LoaderError(
                    f"{self.source.id}: metadata.parquet references image '{stem}' "
                    f"(partition {partition!r}) not found under "
                    f"{self.root / 'images' / partition}"
                )

            point_rows = points_by_key.get(key, ())
            points = tuple((row["row"], row["col"]) for row in point_rows)
            native = [row["label"] for row in point_rows]
            labels, supervised = self._resolve(native[0]) if native else ({}, frozenset())

            image_label_rows = image_labels_by_key.get(key, ())
            native_image_labels = [row["label"] for row in image_label_rows]
            conflicted_axes: set = set()
            for image_label in native_image_labels:
                image_labels, image_supervised = self._resolve(image_label)
                for axis, value in image_labels.items():
                    existing = labels.get(axis)
                    if existing is not None and existing.node_id != value.node_id:
                        # Two image-level labels disagree on the same axis (a genuine
                        # multi-label conflict — e.g. a one-hot export with more than
                        # one positive class landing on the same target axis). Neither
                        # is more trustworthy than the other, so this is NOT supervised
                        # on that axis: never last-write-wins (WS-D S24 D2).
                        conflicted_axes.add(axis)
                    labels = {**labels, axis: value}
                supervised = supervised | image_supervised
            if conflicted_axes:
                labels = {a: v for a, v in labels.items() if a not in conflicted_axes}
                supervised = supervised - frozenset(conflicted_axes)

            mask = masks_by_key.get(key)
            meta: dict[str, object] = {
                "partition": partition,
                "upstream_path": record["upstream_path"],
                "split_group": record.get("split_group"),
            }
            if record.get("upstream_split"):
                meta["upstream_split"] = record["upstream_split"]
            if native:
                meta["native_labels"] = native
                meta["n_points"] = len(points)
            if native_image_labels:
                meta["native_image_labels"] = native_image_labels
            if conflicted_axes:
                meta["multi_label_conflicts"] = sorted(a.value for a in conflicted_axes)
            if mask is not None:
                meta["mask_is_dense"] = True

            yield Sample(
                source_id=self.source.id,
                key=self._relative(image),
                image=image,
                mask=mask,
                points=points,
                labels=labels,
                supervised=supervised,
                licence_tier=self.source.licence.tier,
                split=self.split,
                meta=meta,
            )
