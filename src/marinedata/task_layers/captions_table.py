"""WP-13 captions into the unified ``captions`` table (WP-U10).

The WP-13 frame (``registry/captions/schema.yaml``, checked by :mod:`marinedata.captions.schema`)
holds one row per image with a deterministic template caption (tier A, ``caption_template``) and, for
a pilot sample, a VLM caption (tier B, ``caption_vlm``). Each non-empty caption becomes one unified
row (``caption_type`` ``caption``, ``lang`` ``en``):

* template -> ``annotator_type = pseudo``, ``annotator_detail = template:<builder>``;
* VLM -> ``annotator_type = model``, ``annotator_detail = vlm:<vlm_model>@<vlm_revision>`` with the
  ``prompt_sha256`` / ``seed`` in ``attrs``.

A machine caption is never ``human``. The ``consistency_flags`` and the agent ``audit_verdict`` of
the WP-13 row ride along in ``attrs``; ``confidence`` stays null (the pipeline gives none). No staged
bucket source ships captions today (fgvc23 and marineinst20m-web: none staged), so this converter is
the only producer; ``licence_class`` is the caller's (the image source's), else ``unknown``.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ..annotation_schema import annotation_path, write_annotations
from ..captions.schema import load_schema, validate_frame
from .vqa_table import UNKNOWN, VqaSource, free_row

TEMPLATE_DETAIL = "template:marinedata.captions.template.build_caption (WP-13 tier A)"


def _clean(value: object) -> object:
    """``None`` for any pandas null (None, NaN, NA, NaT)."""
    return None if pd.isna(value) else value


def _vlm_detail(model: object, revision: object) -> str:
    return f"vlm:{model or 'unspecified'}@{revision or 'unpinned'}"


def captions_from_frame(
    frame,
    *,
    source_id: str,
    source_version: str,
    licence_class: str = UNKNOWN,
    ann_license: str | None = None,
    splits: Mapping[str, str] | None = None,
) -> list[dict]:
    """Unified ``captions`` rows for a validated WP-13 ``frame`` (pandas), ordinals in frame order."""
    if problems := validate_frame(frame, load_schema()):
        raise ValueError(f"not a WP-13 captions frame: {problems}")
    rows: list[dict] = []
    for raw in frame.to_dict("records"):
        rec = {k: _clean(v) for k, v in raw.items()}
        sha = rec["image_sha256"]
        shared = {
            "consistency_flags": json.loads(rec["consistency_flags"]),
            "audit_verdict": rec.get("audit_verdict"),
        }
        kinds = (
            ("caption_template", "pseudo", TEMPLATE_DETAIL, {}),
            (
                "caption_vlm", "model", _vlm_detail(rec.get("vlm_model"), rec.get("vlm_revision")),
                {"prompt_sha256": rec.get("prompt_sha256"), "seed": rec.get("seed")},
            ),
        )  # fmt: skip
        for column, origin, detail, extra in kinds:
            text = rec.get(column)
            if not isinstance(text, str) or not text.strip():
                continue
            spec = VqaSource(source_id, source_version, "", origin, detail, ann_license,
                             licence_class, table="captions")  # fmt: skip
            attrs = {**shared, **extra}
            if attrs.get("seed") is not None:
                attrs["seed"] = int(attrs["seed"])
            rows.append(
                free_row(
                    spec=spec,
                    ordinal=len(rows),
                    sha=sha,
                    split=(splits or {}).get(sha),
                    attrs={k: v for k, v in attrs.items() if v is not None},
                    payload={"text": text.strip(), "lang": "en", "caption_type": "caption"},
                )
            )
    return rows


def write_captions(
    data_dir: Path, source_id: str, source_version: str, rows: Sequence[dict]
) -> Path:
    path = annotation_path(data_dir, "captions", source_id, source_version)
    write_annotations(path, "captions", rows)
    return path


__all__ = ["TEMPLATE_DETAIL", "captions_from_frame", "write_captions"]
