"""Labelbox NDJSON export reader.

The format Reef Support's own annotations ship in: one JSON object per line, with
projects → labels → annotations → objects. Objects carry either a mask reference or a
geometry (line, polygon, point).

Worth noting for anyone reusing this: the export nests three levels deep before reaching
anything useful, and the same file mixes annotation *kinds* — instance masks and scale
lines sit side by side under `objects`. Splitting them by `annotation_kind` rather than
assuming homogeneity is the difference between a working loader and one that silently
treats a scale bar as a coral colony.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..sample import Sample
from ..schema import Axis
from .base import LoaderError, register_loader
from .generic import _HarmonizingLoader


@register_loader
class LabelboxNdjsonLoader(_HarmonizingLoader):
    """Labelbox v2 NDJSON export.

    Params:
        annotations: NDJSON path relative to root. Default ``"export-result.ndjson"``
        images_dir: default ``"images"``
        geometry_labels: comma-separated label names that are geometry rather than
            biota — e.g. ``"SCALE"``. These are emitted with their geometry but
            contribute no taxon supervision.
    """

    layout = "labelbox-ndjson"

    def _annotation_path(self) -> Path:
        return self.root / str(self._param("annotations", "export-result.ndjson"))

    def validate(self) -> None:
        super().validate()
        path = self._annotation_path()
        if not path.is_file():
            raise LoaderError(f"{self.source.id}: Labelbox export not found at {path}")

    def _geometry_labels(self) -> frozenset[str]:
        raw = str(self._param("geometry_labels", ""))
        return frozenset(part.strip() for part in raw.split(",") if part.strip())

    def _iter_samples(self) -> Iterator[Sample]:
        path = self._annotation_path()
        images_dir = self.root / str(self._param("images_dir", "images"))
        geometry_labels = self._geometry_labels()

        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LoaderError(
                        f"{self.source.id}: malformed NDJSON at {path}:{lineno} — {exc}"
                    ) from exc

                name = (doc.get("data_row") or {}).get("external_id") or doc.get("external_id")
                if not name:
                    continue

                biota: list[str] = []
                geometry: list[str] = []
                for project in (doc.get("projects") or {}).values():
                    for label in project.get("labels", []):
                        for obj in (label.get("annotations") or {}).get("objects", []):
                            obj_name = obj.get("name")
                            if not obj_name:
                                continue
                            (geometry if obj_name in geometry_labels else biota).append(obj_name)

                if not biota and not geometry:
                    continue

                labels, supervised = ({}, frozenset())
                if biota:
                    labels, supervised = self._resolve(biota[0], Axis.TAXON)
                elif geometry:
                    # Geometry-only frames are still supervised for taxon — the
                    # annotator looked and found no biota — but carry no label.
                    supervised = frozenset({Axis.TAXON})

                yield Sample(
                    source_id=self.source.id,
                    key=name,
                    image=images_dir / name,
                    labels=labels,
                    supervised=supervised,
                    licence_tier=self.source.licence.tier,
                    split=self.split,
                    meta={
                        "native_labels": biota,
                        "geometry_labels": geometry,
                        "n_objects": len(biota) + len(geometry),
                    },
                )
