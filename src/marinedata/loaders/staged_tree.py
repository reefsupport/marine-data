"""Reader for a tree this repo's own writer produced (WS-D staging, D1 §2/§5).

Shape: ``images/<partition>/<stem>.<ext>``, ``metadata.parquet`` (one row per staged
image — the sample index), optionally ``labels/points.parquet`` (sparse point
annotations), ``labels/image_labels.parquet`` (whole-image annotations, WS-D S23)
and/or ``labels/masks/<partition>/<stem>.png`` (dense masks), and ``CHECKSUMS.sha256``
(not read here — that pins bytes for ingest, not for loading).

A flat D-K tree (:mod:`marinedata.sample_schema`, D-AI2) is read too: no ``partition``
column means partition ``""`` (``images/<stem>.<ext>``), and with no ``labels/masks/``
the PNGs under ``labels/files/`` are the masks. ``mask_channel`` (``r``/``g``/``b``)
decodes one channel of an RGB mask into an indexed PNG cached under ``decoded_dir``
(default ``labels/masks_decoded``), refusing any value outside ``mask_values``.

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

from ..normalise import _encode_indexed_png
from ..sample import Sample
from ..sample_schema import staged_partition
from ..tables import _require_pyarrow
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader, _require_pillow_and_numpy


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

    ``split_group`` comes from the ``metadata.parquet`` column into
    ``sample.meta["split_group"]`` — the one value
    :func:`marinedata.scan.group_key` reads for ``by="group"``. A non-empty staged value is
    never re-derived: that rule already ran once, at staging time, and its output is pinned
    into the tree's checksummed bytes, so a registry rule edited *after* staging cannot
    silently disagree with the split a training run already used. Only a null/empty staged
    value on a source with a registry pattern is derived (:meth:`_split_group`).
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

    def _split_group(self, record: dict, *, partition: str, stem: str) -> str | None:
        """One row's ``split_group``: the staged column wins; a null/empty one on a source
        that declares a registry ``split_group`` pattern is derived from that pattern (the
        same order ``release._release_group`` uses). Before this, such a tree was refused,
        which blocked every pinned tree staged by a converter that wrote ``None`` (S28:
        coralscapes). A pattern that misses a stem still raises (``SplitGroupRule.resolve``),
        so the grouping is never guessed. A source on the bare fallback rule keeps ``None``:
        the release enumeration derives its group from the ``metadata_norm`` chain."""
        value = record.get("split_group")
        if isinstance(value, str) and value.strip():
            return value
        if self.source.split_group.pattern is None:
            return value
        try:
            return self.source.split_group_for(
                stem=stem, upstream_path=str(record.get("upstream_path") or ""), partition=partition
            )
        except ValueError as exc:
            raise LoaderError(f"{self.source.id}: {exc}") from exc

    def _mask_values(self) -> dict[str, str]:
        """Pixel value → native label name for a dense mask, from the ``mask_values``
        loader param (``"0=unlabelled,1=bleached,2=non_bleached"`` — same
        comma-separated-pairs style as :meth:`LabelboxNdjsonLoader._geometry_labels`).

        No layout here ever wrote a mask's pixel semantics anywhere a crosswalk could
        read them (unlike :class:`~.generic.DualConditionMaskLoader`, which builds this
        same ``meta["mask_values"]`` shape from ``positive_label``/``negative_label`` at
        read time) — this is the staged-tree equivalent, declared once in the registry
        rather than re-derived, since staging already fixed which integer means what.
        """
        raw = str(self._param("mask_values", ""))
        result: dict[str, str] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            value, _, label = part.partition("=")
            result[value.strip()] = label.strip()
        return result

    def _mask_channel(self) -> str:
        channel = str(self._param("mask_channel", "")).strip().lower()
        if channel not in ("", "r", "g", "b"):
            raise LoaderError(f"{self.source.id}: mask_channel must be r, g or b, got {channel!r}")
        return channel

    def _masks_by_key(self) -> dict[tuple[str, str], Path]:
        """Dense masks keyed like images. ``labels/masks/`` (D1) wins; without it, a flat
        D-K tree's ``labels/files/*.png`` are the masks (D-AI2: coralseg keeps them there)."""
        masks_dir = self.root / "labels" / "masks"
        if masks_dir.is_dir():
            return _by_partition_stem(masks_dir)
        files = _by_partition_stem(self.root / "labels" / "files")
        return {key: path for key, path in files.items() if path.suffix.lower() == ".png"}

    def _decode_channel(
        self, mask: Path, channel: str, partition: str, stem: str, mask_values: dict[str, str]
    ) -> Path:
        """One channel of an RGB mask -> indexed PNG (cached). Any value that is not a
        declared ``mask_values`` key raises: guessing a class would corrupt the mask."""
        decoded_dir = str(self._param("decoded_dir", "labels/masks_decoded"))
        dest = self.root / decoded_dir / partition / f"{stem}.png"
        if dest.is_file():
            return dest
        Image, np = _require_pillow_and_numpy()
        label = f"{self.source.id}/{stem}"
        allowed = sorted(int(v) for v in mask_values) if mask_values else None
        with Image.open(mask) as im:
            band = np.asarray(im.convert("RGB"))[:, :, "rgb".index(channel)].astype(np.uint8)
            size = im.size
        if allowed is not None:
            bad = ~np.isin(band, allowed)
            if bad.any():
                y, x = (int(v) for v in np.argwhere(bad)[0])
                raise LoaderError(
                    f"{label}: pixel at (x={x}, y={y}) has {channel.upper()}={int(band[y, x])}, "
                    f"not a declared mask_values key {allowed} — refusing to guess"
                )
        classes = (allowed[-1] if allowed else int(band.max())) + 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        _encode_indexed_png(
            Image.frombytes("L", size, band.tobytes()), dest, classes=classes, label=label
        )
        return dest

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

    def _row_licences(self) -> dict[str, str]:
        """``stem -> own licence`` from the source's per-row normaliser (WP-R2b); empty for
        every source whose licence is decided whole."""
        if not self.source.licence_per_row:
            return {}
        from ..metadata_norm.local import staged_row_licences

        return staged_row_licences(self.source.id, self.root)

    def _iter_samples(self) -> Iterator[Sample]:
        images_by_key = _by_partition_stem(self.root / "images")
        masks_by_key = self._masks_by_key()
        channel = self._mask_channel()
        points_by_key = self._points_by_key()
        image_labels_by_key = self._image_labels_by_key()
        mask_values = self._mask_values()
        row_licences = self._row_licences()

        for record in _read_parquet(self._metadata_path()).to_pylist():
            partition = staged_partition(record)
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
            if mask is not None and channel:
                mask = self._decode_channel(mask, channel, partition, stem, mask_values)
            meta: dict[str, object] = {
                "partition": partition,
                "upstream_path": record.get("upstream_path"),
                "split_group": self._split_group(record, partition=partition, stem=stem),
            }
            if record.get("upstream_split"):
                meta["upstream_split"] = record["upstream_split"]
            if record.get("license") or row_licences.get(stem):
                # WP-R2/R2b: the row's own licence (metadata_norm per-row normaliser output)
                meta["license"] = record.get("license") or row_licences[stem]
            if native:
                meta["native_labels"] = native
                meta["n_points"] = len(points)
            if native_image_labels:
                meta["native_image_labels"] = native_image_labels
            if conflicted_axes:
                meta["multi_label_conflicts"] = sorted(a.value for a in conflicted_axes)
            if mask is not None:
                meta["mask_is_dense"] = True
                if mask_values:
                    meta["mask_values"] = mask_values
                # A dense mask's real classes live in the raster, not in a native
                # label string this loader could resolve through `_resolve()` — the
                # same reason ImageMaskPairLoader/DualConditionMaskLoader (generic.py)
                # and every other dense-mask loader credit
                # `source.declared_supervision()` rather than leaving a masked sample
                # unsupervised. Without this, a staged-tree source whose only
                # annotation is a dense mask (coralscapes: no points/image_labels
                # rows at all) would silently report zero supervised samples.
                supervised = supervised | self.source.declared_supervision()

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
