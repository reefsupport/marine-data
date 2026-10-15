"""Croissant 1.0 additions for a v1.1.0 export (PR-E, E4).

:func:`release_croissant.build_croissant` types every config from its parquet footers. For a
v1.1.0 export this module adds what the design asks of the document: a ``<config>_splits``
RecordSet per config (``cr:Split``) and a ``hub_split`` field read from the shard name, composite
keys (the annotation join key on the label tables), ``references`` from every ``image_sha256`` to
``metadata``, the ``match_type`` enumeration, the responsible-AI properties shared with the dataset
card, and the version and date of the export's own ``CITATION.cff``. The document is checked with
:func:`hf_export_v11.assert_public_safe` before it is returned.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from . import card_v11_text as txt
from .annotation_schema import JOIN_KEY
from .card_body import bibtex
from .hf_export_v11 import assert_public_safe
from .release_card import GITHUB

REQUIRED = frozenset({"metadata", "sources", "label-map"})
"""Configs every v1.1.0 export holds, open and nc: their presence marks an export as v1.1.0."""
SPLIT_REGEX = r"(?:^|/)(train|validation|test_ood|test)-[^/]+\.parquet$"
MATCH_TYPES = ("exact", "broader", "narrower", "related", "unmapped")
ANNOTATIONS = ("points", "bleaching", "boxes", "semseg", "instances")
KEYS: dict[str, tuple[str, ...]] = {
    "images": ("image_sha256",),
    "metadata": ("image_sha256",),
    "sites": ("source_id", "site_id"),
    "taxonomy": ("kind", "key"),
    "label-map": ("source_id", "native_label"),
    "sources": ("source_id", "version"),
}
"""The key of each config whose grain the design fixes (annotation configs use ``JOIN_KEY``)."""


def is_v11_export(facts: Mapping) -> bool:
    return set(facts["configs"]) >= REQUIRED


def _key(cfg: str, names: list[str]) -> list[str]:
    if cfg in ANNOTATIONS:
        present = [c for c in JOIN_KEY if c in names]
        return present if {"image_sha256", "source_id"} <= set(present) else []
    wanted = KEYS.get(cfg, ())
    return list(wanted) if wanted and set(wanted) <= set(names) else []


def _text_field(rid: str, column: str, description: str | None = None) -> dict:
    field = {"@type": "cr:Field", "@id": f"{rid}/{column}", "name": f"{rid}/{column}"}
    if description:
        field["description"] = description
    return field | {"dataType": "sc:Text"}


def _split_recordset(cfg: str, splits) -> dict:
    rid = f"{cfg}_splits"
    return {
        "@type": "cr:RecordSet",
        "@id": rid,
        "name": rid,
        "description": f"Hub splits of the {cfg} config.",
        "dataType": "cr:Split",
        "key": {"@id": f"{rid}/split_name"},
        "field": [_text_field(rid, "split_name")],
        "data": [{f"{rid}/split_name": s} for s in splits],
    }


def _match_types() -> dict:
    return {
        "@type": "cr:RecordSet",
        "@id": "match_types",
        "name": "match_types",
        "description": "Match types of a native label to the taxonomy (label-map).",
        "dataType": "sc:Enumeration",
        "key": {"@id": "match_types/name"},
        "field": [_text_field("match_types", "name")],
        "data": [{"match_types/name": m} for m in MATCH_TYPES],
    }


def _decorate(recordset: Mapping, cfg: str) -> dict:
    """``recordset`` with its key, its references and the ``hub_split`` field (a new dict)."""
    out = copy.deepcopy(dict(recordset))
    names = [f["@id"].split("/", 1)[1] for f in out["field"]]
    key = _key(cfg, names)
    if key:
        ids = [{"@id": f"{cfg}/{c}"} for c in key]
        out["key"] = ids if len(ids) > 1 else ids[0]
    for field in out["field"]:
        column = field["@id"].split("/", 1)[1]
        if column == "image_sha256" and cfg != "metadata":
            field["references"] = {"field": {"@id": "metadata/image_sha256"}}
        elif cfg == "label-map" and column == "match_type":
            field["references"] = {"field": {"@id": "match_types/name"}}
    out["field"].append(
        _text_field(cfg, "hub_split", "Hub split of the shard the row is stored in.")
        | {
            "source": {
                "fileSet": {"@id": f"{cfg}-parquet"},
                "extract": {"fileProperty": "fullpath"},
                "transform": {"regex": SPLIT_REGEX},
            },
            "references": {"field": {"@id": f"{cfg}_splits/split_name"}},
        }
    )
    return out


def apply_v11(
    doc: Mapping, build_dir: Path, ctx: Mapping, *, date_published: str | None = None
) -> dict:
    """The v1.1.0 form of the generic Croissant ``doc`` of the export ``build_dir`` (a new dict)."""
    facts, repo_id = ctx["facts"], ctx["repo_id"]
    flavour = (ctx.get("release") or {}).get("flavour") or ctx.get("flavour") or "open"
    own = Path(build_dir) / "CITATION.cff"
    cff = yaml.safe_load(own.read_text()) if own.is_file() else dict(ctx.get("cff") or {})
    out = copy.deepcopy(dict(doc))
    recordsets: list[dict] = []
    for recordset in out["recordSet"]:
        cfg = recordset["@id"]
        recordsets += [
            _decorate(recordset, cfg),
            _split_recordset(cfg, facts["configs"][cfg]["splits"]),
        ]
    version = str(cff.get("version") or txt.VERSION)
    out["recordSet"] = [*recordsets, _match_types()]
    out["version"] = version
    out["description"] = (
        f"Reef Support Marine Data v{version}: {len(facts['configs'])} configurations built from "
        f"{len(ctx['infos'])} source(s), each with a `<config>_splits` record set. Every row "
        f"carries its image_sha256 key; licence and attribution are in the sources record set. "
        f"Code: {GITHUB}."
    )
    if cff:
        out["citeAs"] = bibtex(cff, repo_id)
    if not (date_published or out.get("datePublished")) and cff.get("date-released"):
        out["datePublished"] = str(cff["date-released"])
    out |= txt.RAI
    assert_public_safe(json.dumps(out, ensure_ascii=False), flavour, where="croissant.json")
    return out
