"""Re-identification identities into the unified ``identities`` table (WP-U13).

One row per image-individual: ``individual_id`` (the source's own identity id, as a string),
``individual_scope`` (``dataset``: an id is only unique inside its dataset; ``site``: inside a site),
the image sha256 and the taxon (``species`` -> U2 crosswalk). Box columns stay null unless the source
stages a box. The first source is ``seaturtleid2022``:

* the staged tree holds the images and ``metadata.parquet`` but **no label file** (``label_staged =
  no``), so the identity of an image is read from the non-image columns of the upstream HF parquet
  (``identity``, ``year``, ``split_open`` ...) through the anonymous datasets-server ``/rows`` API
  (:func:`hf_rows`): row ``n`` of the split is the ``n``-th image in (shard, row) order of
  ``metadata.upstream_id`` (``data/train-0000K-of-00004.parquet#j``), checked against ``width`` and
  ``height``; a mismatch drops the row (counted), it is never guessed;
* the species is a dataset constant (loggerhead sea turtles, ``Caretta caretta``, arXiv:2311.05524),
  not a per-image label: ``attrs.species_origin = "dataset-level"``;
* body-part masks and orientation of the original release are not in the HF mirror: not staged.

``attrs.licence_class`` is ``unknown`` (lic-A: the HF mirror states no licence, the original's terms
are custom); nothing here releases it.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...annotation_schema import (
    TABLES,
    annotation_path,
    annotator_from_origin,
    normalise_split,
    write_annotations,
)
from ...registry import Registry
from ..image_labels_table import _stride, axes_resolver_for
from ..points_table import _taxon_rank

_UPSTREAM = re.compile(r"^data/(?P<shard>[^#]+)#(?P<row>\d+)$")
SPLIT_COLUMNS = ("split_closed", "split_closed_random", "split_open")
HF_ROWS = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class IdentitySource:
    source_id: str
    version: str
    annotator: str
    ann_license: str | None
    licence_class: str
    species: str  # native species label of the whole dataset
    crosswalk_id: str
    label_set: str
    hf_dataset: str
    scope: str = "dataset"


IDENTITY_SOURCES: dict[str, IdentitySource] = {
    "seaturtleid2022": IdentitySource(
        "seaturtleid2022", "rev-91cd6c1126c7", "human", None, "unknown", "Caretta caretta",
        "seaturtleid2022", "seaturtleid2022-species", "EnmmmmOvO/SeaTurtleID2022",
    )
}  # fmt: skip


@dataclass(frozen=True)
class IdentityResult:
    rows: tuple[dict, ...]
    individuals: int
    mismatched: int  # records whose size disagrees with the staged image
    unlabelled: int  # records with no identity
    unstaged: int  # records past the staged images


def hf_rows(
    dataset: str,
    total: int,
    *,
    get: Callable[[str], bytes] | None = None,
    page: int = 100,
) -> list[dict]:
    """The first ``total`` rows (non-image columns used) of a public HF dataset's ``train`` split."""

    def default_get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()

    out: list[dict] = []
    for offset in range(0, total, page):
        q = urllib.parse.urlencode(
            {"dataset": dataset, "config": "default", "split": "train", "offset": offset,
             "length": min(page, total - offset)}
        )  # fmt: skip
        out += [r["row"] for r in json.loads((get or default_get)(f"{HF_ROWS}?{q}"))["rows"]]
    return out


def order_meta(meta: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Staged images in the upstream (shard, row) order, i.e. the datasets-server row order."""

    def key(m: Mapping[str, object]) -> tuple[str, int]:
        hit = _UPSTREAM.match(str(m["upstream_id"]))
        if hit is None:
            raise ValueError(f"upstream_id {m['upstream_id']!r} is not data/<shard>#<row>")
        return hit["shard"], int(hit["row"])

    return sorted(meta, key=key)


def identity_row(
    *, spec: IdentitySource, registry: Registry, resolve, ordinal: int, image_sha256: str,
    individual_id: str, split: str | None, attrs: Mapping[str, object],
) -> dict:  # fmt: skip
    got = resolve(spec.species)
    typ, detail = annotator_from_origin(spec.annotator)
    row: dict = {c.name: None for c in TABLES["identities"].columns}
    row.update(
        image_sha256=image_sha256, source_id=spec.source_id, source_version=spec.version,
        ann_id=f"{spec.source_id}:{ordinal}", label_native=spec.species, label_set=spec.label_set,
        taxon_node_id=got.taxon, form_node_id=got.form, condition_node_id=got.condition,
        taxon_rank=_taxon_rank(registry, got.taxon), worms_aphia_id=got.worms_aphia_id,
        rs_benthic_code=got.rs_benthic_code, match_type=got.match_type, annotator_type=typ,
        annotator_detail=detail, ann_license=spec.ann_license, upstream_split=normalise_split(split),
        label_status="ok", individual_id=individual_id, individual_scope=spec.scope,
        attrs=json.dumps(
            {**attrs, "species_origin": "dataset-level", "licence_class": spec.licence_class},
            sort_keys=True,
        ),
    )  # fmt: skip
    return row


def staged_identities(
    spec: IdentitySource,
    registry: Registry,
    meta: Iterable[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    *,
    limit: int | None = None,
) -> IdentityResult:
    """Align ``records`` (HF rows, in split order) with the staged images and emit the rows."""
    resolve = axes_resolver_for(registry, spec.crosswalk_id)
    ordered = order_meta(meta)
    keep = set(_stride(list(range(min(len(records), len(ordered)))), limit))
    rows: list[dict] = []
    mismatched = unlabelled = 0
    for n, rec in enumerate(records[: len(ordered)]):
        m = ordered[n]
        if n not in keep:
            continue
        if (rec.get("width"), rec.get("height")) != (m.get("width"), m.get("height")):
            mismatched += 1
            continue
        if rec.get("identity") is None:
            unlabelled += 1
            continue
        attrs = {k: rec[k] for k in ("year", "date", "timestamp", *SPLIT_COLUMNS) if rec.get(k)}
        rows.append(identity_row(
            spec=spec, registry=registry, resolve=resolve, ordinal=len(rows),
            image_sha256=str(m["image_sha256"]), individual_id=str(rec["identity"]),
            split=rec.get("split_open"), attrs=attrs,
        ))  # fmt: skip
    ids = Counter(r["individual_id"] for r in rows)
    return IdentityResult(tuple(rows), len(ids), mismatched, unlabelled,
                          max(0, len(records) - len(ordered)))  # fmt: skip


def write_identities(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    path = annotation_path(data_dir, "identities", source_id, source_version)
    write_annotations(path, "identities", rows)
    return path


__all__ = [
    "IDENTITY_SOURCES", "IdentityResult", "IdentitySource", "hf_rows", "identity_row",
    "order_meta", "staged_identities", "write_identities",
]  # fmt: skip
