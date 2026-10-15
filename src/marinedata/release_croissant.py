# ruff: noqa: E501
"""Croissant 1.0 JSON-LD of a release build.

Plain JSON-LD, so it is written without ``mlcroissant``; :func:`validate` runs the official
validator when the optional extra ``croissant`` is installed. The distribution lists every
parquet shard with its sha256 and size; each config is one ``recordSet`` whose fields are typed
from the parquet schema (the ``datasets`` feature form that :mod:`release_card` reads from the
footers): ``image`` and ``mask`` columns are ``sc:ImageObject`` (``transform.jsonPath: bytes``,
as the Hub writes them) and list-of-struct columns such as ``boxes`` become ``subField``\\s.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .card_body import _author, bibtex
from .card_fields import config_doc, field_doc
from .release_card import DEFAULT_REPO, GITHUB, _licence, display_licence

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}
PARQUET = "application/x-parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _datatype(dtype: str) -> str:
    if dtype == "image":
        return "sc:ImageObject"
    if dtype.startswith(("int", "uint")):
        return "sc:Integer"
    if dtype.startswith("float"):
        return "sc:Float"
    return "sc:Boolean" if dtype == "bool" else "sc:Text"


def _field(
    fileset: str, feature: dict, *, column: str, path: tuple, fid: str, repeated: bool
) -> dict:
    kind = feature.get("list")
    node = {
        "@type": "cr:Field",
        "@id": fid,
        "name": fid,
        "description": field_doc(feature["name"]) or feature["name"],
    }
    children = kind if isinstance(kind, list) else feature.get("struct")
    if children:
        node["subField"] = [
            _field(
                fileset,
                child,
                column=column,
                path=(*path, child["name"]),
                fid=f"{fid}/{child['name']}",
                repeated=repeated or isinstance(kind, list),
            )
            for child in children
        ]
        return node
    dtype = kind if isinstance(kind, str) else feature["dtype"]
    source = {"fileSet": {"@id": fileset}, "extract": {"column": column}}
    if dtype == "image":
        source["transform"] = {"jsonPath": "bytes"}
    elif path:
        source["transform"] = {"jsonPath": ".".join(path)}
    node |= {"dataType": _datatype(dtype), "source": source}
    if repeated or isinstance(kind, str):
        node["repeated"] = True
    return node


def _licences(infos: dict, sids) -> list[str]:
    found = []
    for sid in sids:
        lic = _licence(infos.get(sid) or {})
        label = lic.get("url") or (display_licence(lic["id"]) if lic.get("id") else None)
        if label:
            found.append(label)
    return list(dict.fromkeys(found))


def build_croissant(
    build_dir: Path, ctx: dict, *, date_published: str | None = None, name: str | None = None
) -> dict:
    """The Croissant document of ``build_dir`` (``ctx`` is :func:`release_card.load_context`)."""
    facts, infos, cff, repo_id = ctx["facts"], ctx["infos"], ctx["cff"], ctx["repo_id"]
    base = f"https://huggingface.co/datasets/{repo_id}"
    shards: list[dict] = []
    filesets: list[dict] = []
    recordsets: list[dict] = []
    for cfg, spec in facts["configs"].items():
        files = sorted((Path(build_dir) / "data" / cfg).glob("*.parquet"))
        for path in files:
            shards.append(
                {
                    "@type": "cr:FileObject",
                    "@id": f"{cfg}/{path.name}",
                    "name": f"{cfg}/{path.name}",
                    "contentUrl": f"{base}/resolve/main/data/{cfg}/{path.name}",
                    "contentSize": f"{path.stat().st_size} B",
                    "encodingFormat": PARQUET,
                    "sha256": sha256_file(path),
                }
            )
        fileset = f"{cfg}-parquet"
        filesets.append(
            {
                "@type": "cr:FileSet",
                "@id": fileset,
                "name": fileset,
                "containedIn": {"@id": "repo"},
                "encodingFormat": PARQUET,
                "includes": f"data/{cfg}/*.parquet",
            }
        )
        rows = sum(e["num_examples"] for e in spec["splits"].values())
        sids = list(spec.get("sources") or {})
        recordset = {
            "@type": "cr:RecordSet",
            "@id": cfg,
            "name": cfg,
            "description": f"{config_doc(cfg).task}: {rows:,} rows from {len(sids)} source(s).",
        }
        lic = _licences(infos, sids)
        if lic:
            recordset["license"] = lic
        recordset["field"] = [
            _field(fileset, f, column=f["name"], path=(), fid=f"{cfg}/{f['name']}", repeated=False)
            for f in spec["features"]
        ]
        recordsets.append(recordset)
    lines = sorted(f"{s['sha256']}  {s['@id']}" for s in shards)
    repo = {
        "@type": "cr:FileObject",
        "@id": "repo",
        "name": "repo",
        "description": "The dataset repository. sha256 is the digest of the sorted per-shard "
        "'sha256  path' lines listed below.",
        "contentUrl": f"{base}/tree/main",
        "encodingFormat": "git+https",
        "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }
    release = facts.get("release") or {}
    published = date_published or release.get("built") or str(cff.get("date-released") or "")
    creators = [
        {"@type": "sc:Person", "name": _author(a)} for a in cff.get("authors", []) if _author(a)
    ]
    creators.append(
        {"@type": "sc:Organization", "name": "Reef Support", "url": "https://reef.support"}
    )
    doc = {
        "@context": copy.deepcopy(CONTEXT),
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": name or repo_id.split("/")[-1],
        "description": (
            f"{len(facts['configs'])} configuration(s) built from {len(infos)} source(s); every "
            f"row carries its licence and attribution. Code: {GITHUB or DEFAULT_REPO}."
        ),
        "url": base,
        "license": _licences(infos, infos),
        "citeAs": bibtex(cff, repo_id) if cff else base,
        "creator": creators,
    }
    if published:
        doc["datePublished"] = published
    if ctx.get("release_id"):
        doc["version"] = ctx["release_id"]
    doc |= {"distribution": [repo, *shards, *filesets], "recordSet": recordsets}
    from . import release_croissant_v11 as v11  # lazy: only the v1.1.0 export needs it

    if v11.is_v11_export(facts):
        doc = v11.apply_v11(doc, Path(build_dir), ctx, date_published=date_published)
    return doc


def validate(doc: dict) -> tuple[str, str]:
    """``("valid", "")``, ``("invalid", reason)`` or ``("not-validated", reason)`` (no validator)."""
    try:
        import mlcroissant as mlc
    except ImportError:
        return "not-validated", "mlcroissant is not installed (extra `croissant`)"
    try:
        # mlcroissant adds "@base" to the dict it is given: validate a copy, never the original
        mlc.Dataset(jsonld=copy.deepcopy(doc))
    except Exception as err:  # mlcroissant raises ValidationError with every issue listed
        return "invalid", str(err)[:1500]
    return "valid", ""
