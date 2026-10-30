"""v1.1.0 export (PR-E, E1/E2): the Hub data folder of a ``release build --tasks v2 --v11`` build.

Additive to v1.0: nothing here rewrites a v1.0 config, row or path. The export writes the NEW
configs only, under ``data/<config>/<split>-<source>-NNNNN-of-MMMMM.parquet``:

* ``images`` (E1): the bytes of every image whose ``metadata.bytes_config`` is ``images``, once
  per ``image_sha256``. An image a v1.0 config already embeds (``bytes_config`` names that config,
  or its sha256 is in ``skip_sha256``) is never re-uploaded.
* the six build tables (``metadata`` ... ``instances``, plus ``benthic-cover`` / ``sites`` when the
  build wrote them), label-only, typed from their parquet schema. ``semseg`` embeds the palette
  PNG of every mask no v1.0 config holds, and every embedded bytes column gets its sha256
  (``images.bytes_sha256``, ``semseg.mask_bytes_sha256``, ``metadata.bytes_sha256``).
* ``taxonomy`` (open flavour only), ``label-map`` and ``sources`` (E2), from the pinned taxonomy
  release, the crosswalks and the registry. A label with no mapping is carried as
  ``mapped = false``.

Shard names are ``<split>-<source>-NNNNN-of-MMMMM.parquet`` (MMMMM is this export's count). An
additive run (``--v1-dir``) ships only the pixels the prior tree lacks, so an ``images`` shard of a
prior name would be a smaller file that ``release publish`` (upload by path, no delete) writes over
the prior one. Every ``images`` shard of such a run is named
``<split>-<source>-<release-id>-NNNNN-of-MMMMM.parquet``, e.g.
``train-noaa-pifsc-bleaching-v1.2.0-00000-of-00001.parquet``. ``--release-id vX.Y.Z`` is required
with ``--v1-dir`` and ``bootstrap.sh`` passes it. The infix sits before the index, so the split
stays the first token and ``data/images/<split>-*.parquet`` still matches. No other config is
renamed: each is a full re-export that replaces its prior shards by name, and a run without
``--v1-dir`` changes nothing. ``gates.shards.parse_shard`` reads both shapes.

A row whose licence class is not the flavour's is dropped and counted; a row whose class can never
ship (internal-only, unknown, nd) raises. The key everywhere is ``image_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from .checksums import CHECKSUM_FILE, file_digest
from .gates.shards import RELEASE_ID
from .hf_parquet import (
    SHARD_TARGET_BYTES,
    ConfigSpec,
    EmbeddedBlob,
    ExportRow,
    files_per_folder,
    plan_config,
    write_shard,
)
from .licence_class import FLAVOURS, coerce_class, release_excluded

IMAGES, TAXONOMY, LABEL_MAP, SOURCES = "images", "taxonomy", "label-map", "sources"
TABLE_CONFIGS = (
    "metadata", "points", "bleaching", "boxes", "semseg", "instances", "benthic-cover", "sites",
)  # fmt: skip
V11_TABLES_FILE = "V11_TABLES.json"
SPLIT_NAME = {
    "train": "train", "val": "validation", "valid": "validation", "validation": "validation",
    "probe": "validation", "test": "test", "test_ood": "test_ood",
}  # fmt: skip
IMAGE_COLUMNS = (
    ("image_sha256", "string"), ("image", "image"), ("bytes_sha256", "string"),
    ("source_id", "string"), ("source_ids", "string"), ("split_group", "string"),
    ("split", "string"),
)  # fmt: skip
LABEL_MAP_COLUMNS = (
    ("source_id", "string"), ("crosswalk_id", "string"), ("native_label", "string"),
    ("native_label_id", "string"), ("pixel_value", "int64"), ("taxon_node_id", "string"),
    ("form_node_id", "string"), ("condition_node_id", "string"), ("match_type", "string"),
    ("fidelity", "string"), ("mapped", "bool"), ("note", "string"), ("taxonomy_version", "string"),
)  # fmt: skip
TAXONOMY_COLUMNS = (
    ("kind", "string"), ("key", "string"), ("name", "string"), ("parent", "string"),
    ("rank", "string"), ("axis", "string"), ("aphia_id", "int64"), ("non_taxon", "bool"),
    ("fidelity", "string"), ("targets", "string"), ("taxonomy_version", "string"),
)  # fmt: skip
SOURCES_COLUMNS = (
    ("source_id", "string"), ("version", "string"), ("root_digest", "string"),
    ("licence", "string"), ("licence_url", "string"), ("ann_licence", "string"),
    ("attribution", "string"), ("rights_holder", "string"), ("verified_by", "string"),
    ("verified_on", "string"), ("provenance_url", "string"), ("licence_class", "string"),
    ("n_images", "int64"), ("configs", "string"), ("n_labels", "int64"),
    ("mapped_pct", "double"), ("known_defects", "string"), ("taxonomy_version", "string"),
)  # fmt: skip
PRIVATE_NAMES = ("marine-data-nc", "rs-storage-private", "rs-hel1", "s3://")
"""Strings no public (open-flavour) text may carry: the private nc repo, the private bucket, the
rclone remote and any bucket URI."""
Layout = dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]]


class V11ExportError(Exception):
    """The v1.1.0 export would not be a faithful, licence-clean copy of the build."""


def is_v11_release(release_dir: str | Path) -> bool:
    return (Path(release_dir) / "task_layers" / V11_TABLES_FILE).is_file()


def hf_split(name: str | None) -> str:
    try:
        return SPLIT_NAME[str(name)]
    except KeyError:
        raise V11ExportError(
            f"split {name!r} has no Hub split name (need train/val/test/test_ood)"
        ) from None


def assert_public_safe(text: str, flavour: str, *, where: str = "text") -> None:
    """Raise when ``text`` names a private repo or bucket: the open card, croissant and tables may
    not name the nc repo or any bucket; the nc card may not name a bucket or bucket URI. The open
    flavour also refuses the id of a source that does not ship in it (a whole token, as the leak
    gate reads it); only the id's ``nc_id#`` label is echoed."""
    banned = PRIVATE_NAMES if flavour == "open" else PRIVATE_NAMES[1:]
    hit = [n for n in banned if n in text]
    if flavour == "open":
        from .gates.leak import id_tags
        from .gates.v11_runner import nc_source_ids

        hit += id_tags(text, nc_source_ids())
    if hit:
        raise V11ExportError(f"{where} ({flavour}) names a private resource: {', '.join(hit)}")


# ------------------------------------------------------------------------------ reading inputs


def read_table(release_dir: str | Path, config: str) -> list[dict]:
    path = Path(release_dir) / "task_layers" / f"{config}.parquet"
    if not path.is_file():
        return []
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def read_checksums(root: str | Path) -> dict[str, str]:
    """``sha256 -> relative path`` of one staged tree's ``CHECKSUMS.sha256`` (first path wins)."""
    path = Path(root) / CHECKSUM_FILE
    if not path.is_file():
        raise V11ExportError(f"{root}: no {CHECKSUM_FILE}; the image key cannot be resolved")
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, _, rel = line.lstrip("\\").partition("  ")
        if len(digest) == 64 and rel:
            out.setdefault(digest.lower(), rel.removeprefix("*"))
    return out


def v1_sha256s(v1_dir: str | Path) -> frozenset[str]:
    """Every ``image_sha256`` the v1.0 shards under ``<v1_dir>/data/*/*.parquet`` hold."""
    import pyarrow.parquet as pq

    found: set[str] = set()
    for shard in sorted((Path(v1_dir) / "data").glob("*/*.parquet")):
        if "image_sha256" in pq.ParquetFile(shard).schema_arrow.names:
            found.update(pq.read_table(shard, columns=["image_sha256"]).column(0).to_pylist())
    return frozenset(s for s in found if s)


def hidden_test_drop(manifest: str | Path, exempt_v1_dir: str | Path) -> frozenset[str]:
    """The digests the open release withholds: every sha256 the hidden-test ``manifest`` lists
    that the v1.0 shards under ``exempt_v1_dir`` do not already publish (lower-cased)."""
    from .gates.decon_gate import manifest_shas

    if not Path(manifest).exists():
        raise V11ExportError(f"--hidden-test {Path(manifest).name}: not found")
    listed = {s.lower() for s in manifest_shas(Path(manifest))}
    if not listed:
        raise V11ExportError(f"--hidden-test {Path(manifest).name}: no sha256 found")
    public = {s.lower() for s in v1_sha256s(exempt_v1_dir)}
    if not public:
        raise V11ExportError(f"--hidden-test-exempt-v1 {exempt_v1_dir}: no v1.0 shard rows found")
    return frozenset(listed - public)


def drop_hidden_test(
    tables: Mapping[str, list[Mapping]], drop: Collection[str]
) -> tuple[dict[str, list[Mapping]], dict]:
    """``(tables without the rows keyed by a digest in drop, counts)``: a row goes when its
    ``image_sha256`` or ``bytes_sha256`` is listed. ``counts`` holds numbers only: the digests
    asked for, the distinct ones found, and the rows removed per table (never a digest)."""
    wanted, found = frozenset(drop), set()
    kept: dict[str, list[Mapping]] = {}
    rows_removed: dict[str, int] = {}
    for config, rows in tables.items():
        kept[config] = []
        for row in rows:
            keys = {str(row.get(k) or "").lower() for k in ("image_sha256", "bytes_sha256")}
            if keys & wanted:
                found |= keys & wanted
            else:
                kept[config].append(row)
        rows_removed[config] = len(rows) - len(kept[config])
    return kept, {"requested": len(wanted), "images": len(found), "rows": rows_removed}


# ------------------------------------------------------------------------------ licence filter


def row_class(registry, row: Mapping, class_by_sha: Mapping[str, str]) -> str:
    """The licence class of one table row: its own, else its image's (the metadata row, by
    ``image_sha256``), else its source's registry class. The export writes it into every row."""
    raw = row.get("licence_class") or class_by_sha.get(row.get("image_sha256"))
    return coerce_class(raw or registry.source(row.get("source_id")).access_class)


def keep_for_flavour(
    registry, rows: Iterable[Mapping], flavour: str, class_by_sha: Mapping[str, str]
):
    """``(kept rows, dropped count)``: the rows whose licence class is the flavour's. A row whose
    class can never ship in any flavour raises (the build must not have kept it)."""
    want, shippable = FLAVOURS[flavour], set(FLAVOURS.values())
    kept: list[Mapping] = []
    dropped = 0
    for row in rows:
        sid = row.get("source_id")
        if sid and release_excluded(sid):
            dropped += 1
            continue
        cls = row_class(registry, row, class_by_sha)
        if cls not in shippable:
            raise V11ExportError(
                f"{sid}: licence class {cls!r} can never ship; the build kept a row"
            )
        if cls == want:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


# ------------------------------------------------------------------------------ typed tables


def _kind(arrow_type) -> str:
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_integer(arrow_type):
        return "int64"
    return "double" if pa.types.is_floating(arrow_type) else "string"


def table_spec(config: str, rows: list[Mapping], schema=None) -> ConfigSpec:
    """The Hub spec of one label table: its parquet columns as ``string`` / ``bool`` / ``int64`` /
    ``double`` (a list or struct column becomes a JSON string)."""
    if schema is None:
        import pyarrow as pa

        names = list(dict.fromkeys(k for r in rows for k in r))
        schema = pa.schema([(n, pa.array([r.get(n) for r in rows]).type) for n in names])
    return ConfigSpec(config, tuple((f.name, _kind(f.type)) for f in schema))


def _cell(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "string":
        return value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    return bool(value) if kind == "bool" else value


def _as_row(spec: ConfigSpec, values: Mapping, **extra: Any) -> ExportRow:
    kinds = dict(spec.columns)
    cells = {n: _cell(values.get(n), k) for n, k in kinds.items() if k != "image"}
    return ExportRow(values={**cells, **extra})


def _group(
    rows: list[tuple[str, ExportRow]], by_source: bool, release: str | None = None
) -> dict[str, list[ExportRow]]:
    """``{<split>[-<source>][-<release>]: rows}`` in a stable order; the key is the shard-name
    stem. ``release`` (an additive ``images`` export) ends every stem, after the order is set, so
    the order is the one the plain stems have."""
    groups: dict[str, list[ExportRow]] = defaultdict(list)
    for split, row in rows:
        source = row.values.get("source_id") if by_source else None
        groups[f"{split}-{source}" if source else split].append(row)
    ordered = sorted(groups, key=lambda s: (_split_rank(s), s))
    return {f"{k}-{release}" if release else k: groups[k] for k in ordered}


def _split_of(row: Mapping, split_by_sha: Mapping[str, str | None]) -> str:
    """The Hub split of a table row: its own ``split``, else its image's; a row with no image key
    (``sites``, ``benthic-cover`` by survey) is ``train``. An image with no split raises."""
    sha = row.get("image_sha256")
    if not sha:
        return "train"
    split = row.get("split") or split_by_sha.get(sha)
    if not split:
        raise V11ExportError(f"image {sha} has no split (an unsplit rehearsal build cannot export)")
    return hf_split(split)


def _split_rank(stem: str) -> int:
    order = ("train", "validation", "test_ood", "test")
    return next((i for i, s in enumerate(order) if stem == s or stem.startswith(s + "-")), 9)


# ------------------------------------------------------------------------------ E1: images


def staged_index(registry, source_id: str, root: str | Path) -> dict[str, str]:
    """``sha256 -> relative path`` of one staged tree: its ``CHECKSUMS.sha256``, else (a build
    fetch leaves none) the split map's enumeration hashed, the same key the build gave the image."""
    if (Path(root) / CHECKSUM_FILE).is_file():
        return read_checksums(root)
    from .release.entries import source_release_entries

    entries = source_release_entries(registry.source(source_id), root, file_digest)
    return {e.sha256: e.key for e in entries}


def image_rows(
    meta_rows: Iterable[Mapping], roots: Mapping[str, Path], *,
    skip_sha256: Collection[str] = frozenset(), stats: Counter | None = None,
    locate: Callable[[str], Mapping[str, str]] | None = None,
) -> tuple[list[tuple[str, ExportRow]], dict[str, str]]:  # fmt: skip
    """``([(split, row)], {image_sha256: bytes_sha256})``: one row per image whose bytes no v1.0
    config holds, each hashed from the bytes about to be embedded (and checked against the key).
    ``locate(source_id)`` (``sha256 -> relative path``) replaces the ``CHECKSUMS.sha256`` lookup."""
    stats = Counter() if stats is None else stats
    skip = {s.lower() for s in skip_sha256}
    members: dict[str, list[Mapping]] = defaultdict(list)
    for row in sorted(meta_rows, key=lambda r: (str(r.get("source_id")), str(r["image_sha256"]))):
        sha = row["image_sha256"]
        if row.get("bytes_config") != IMAGES:
            stats["skipped_v1_config"] += 1
        elif sha in skip:
            stats["skipped_v1_sha256"] += 1
        else:
            members[sha].append(row)
    index: dict[str, dict[str, str]] = {}
    out: list[tuple[str, ExportRow]] = []
    bytes_sha: dict[str, str] = {}
    for sha in sorted(members):
        primary = members[sha][0]
        sid = primary["source_id"]
        if sid not in roots:
            raise V11ExportError(f"{sid}: no staged tree for image {sha}")
        if sid not in index:
            index[sid] = dict(locate(sid) if locate else read_checksums(roots[sid]))
        rel = index[sid].get(sha)
        path = Path(roots[sid]) / rel if rel else None
        if path is None or not path.is_file():
            raise V11ExportError(f"{sid}: image {sha} not found in its staged tree")
        digest = file_digest(path)
        if digest != sha:
            raise V11ExportError(f"{sid}: {rel} hashes to {digest}, the key is {sha}")
        bytes_sha[sha] = digest
        values = {
            "image_sha256": sha, "bytes_sha256": digest, "source_id": sid,
            "source_ids": ",".join(sorted({m["source_id"] for m in members[sha]})),
            "split_group": primary.get("split_group"), "split": hf_split(primary.get("split")),
        }  # fmt: skip
        row = ExportRow(
            values=values, file=path, file_path=f"{sid}/{sha}{path.suffix}",
            file_bytes=path.stat().st_size,
        )  # fmt: skip
        out.append((values["split"], row))
        stats["embedded"] += 1
    return out, bytes_sha


# ------------------------------------------------------------------------------ semseg masks


def _attrs(row: Mapping) -> dict:
    raw = row.get("attrs")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def palette_mask(row: Mapping, roots: Mapping[str, Path]) -> bytes | None:
    """The palette PNG (mode ``P``, declared palette) of a semseg row no v1.0 config holds, else
    ``None`` (the mask lives in its ``mask_config``, or the row has no mask).

    A ``mask_ref`` may be a bucket key (unified masks table) or tree-relative (legacy tables)."""
    ref = row.get("mask_ref") or row.get("mask_key")
    if row.get("mask_config") or not ref:
        return None
    from .task_layers.mask_encode import encoding_for, semantic_png

    sid, root = row["source_id"], Path(roots[row["source_id"]])
    # a bucket key, sources/<sid>/<version>/<path>, names the file by its path in the staged tree
    parts = ref.split("/", 3)
    rel = parts[3] if len(parts) == 4 and parts[:2] == ["sources", sid] else ref
    found = next((p for p in (root / rel, root / "labels" / rel) if p.is_file()), None)
    if found is None:
        raise V11ExportError(f"{sid}: mask {ref} not found under its staged tree")
    encoding = encoding_for(sid, row.get("mask_encoding") or _attrs(row).get("mask_encoding"))
    return semantic_png(
        found.read_bytes(), encoding=encoding, ignore_value=row.get("ignore_value"), palette=True
    )


# ------------------------------------------------------------------------------ E2: configs


def taxonomy_rows(manifest: Mapping, version: str, crosswalk_ids: Collection[str]) -> list[dict]:
    """Node rows (the whole pinned vocabulary) and edge rows (only the flavour's crosswalks)."""
    rows = [
        {
            "kind": "node", "key": key, "name": n.get("name"), "parent": n.get("parent"),
            "rank": n.get("rank"), "axis": n.get("axis"), "aphia_id": n.get("aphia"),
            "non_taxon": bool(n.get("non_taxon")), "taxonomy_version": version,
        }
        for key, n in sorted(manifest["nodes"].items())
    ]  # fmt: skip
    for key, e in sorted(manifest["edges"].items()):
        if key.split(":", 1)[0] in crosswalk_ids:
            rows.append(
                {
                    "kind": "edge", "key": key, "fidelity": e.get("fidelity"),
                    "targets": json.dumps(e.get("targets") or {}, sort_keys=True),
                    "taxonomy_version": version,
                }
            )  # fmt: skip
    return rows


def _axis(edge, name: str) -> str | None:
    return next((v for k, v in edge.targets.items() if getattr(k, "value", str(k)) == name), None)


def table_crosswalk_ids() -> dict[str, str]:
    """``source -> crosswalk`` the task-layer table builders resolve its labels with (boxes, image
    labels, semantic and instance masks). Some of these sources (the box and instance sets) declare
    no ``loader.crosswalk_id`` in the registry, and their tables still carry the nodes."""
    from .task_layers.boxes_table import BOX_SOURCES
    from .task_layers.image_labels_table import LABEL_SOURCES
    from .task_layers.masks_table import MASK_SOURCES
    from .task_layers.sources.masks_instance import INSTANCE_SOURCES

    out: dict[str, str] = {}
    for specs in (BOX_SOURCES, LABEL_SOURCES, MASK_SOURCES, INSTANCE_SOURCES):
        for sid, spec in specs.items():
            if spec.crosswalk_id:
                out.setdefault(sid, spec.crosswalk_id)
    return out


def source_crosswalk_id(
    registry, source_id: str, bound: Mapping[str, str] | None = None
) -> str | None:
    """The crosswalk ``source_id``'s labels resolve with: its loader's, else the one its table
    builder binds (``bound``, :func:`table_crosswalk_ids` by default)."""
    loader = registry.source(source_id).loader
    cid = loader.crosswalk_id if loader is not None else None
    return cid or (table_crosswalk_ids() if bound is None else bound).get(source_id)


def export_crosswalk_ids(registry, source_ids: Iterable[str]) -> set[str]:
    """The crosswalks of ``source_ids``: the edges the label map and the taxonomy table carry."""
    bound = table_crosswalk_ids()
    return {c for s in source_ids if (c := source_crosswalk_id(registry, s, bound))}


def label_map_rows(
    registry, source_ids: Iterable[str], version: str, label_rows: Iterable[Mapping] = ()
) -> list[dict]:  # fmt: skip
    """One row per ``(source, native label)``: every crosswalk edge (unmappable ones included) plus
    every label seen in ``label_rows`` with no edge, carried as ``mapped = false``. A mask
    ``class_map`` fills ``pixel_value``. A source's crosswalk is its loader's, else the one its
    table builder binds (:func:`table_crosswalk_ids`)."""
    pixels: dict[tuple[str, str], int] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    for r in label_rows:
        sid, native = r.get("source_id"), r.get("label_native")
        if sid and native is not None:
            seen[sid].add(str(native))
        try:
            entries = json.loads(r["class_map"]) if isinstance(r.get("class_map"), str) else []
        except ValueError:
            entries = []
        for c in entries if isinstance(entries, list) else ():
            if (
                isinstance(c, dict)
                and c.get("label_native") is not None
                and c.get("id") is not None
            ):
                pixels.setdefault((sid, str(c["label_native"])), int(c["id"]))
    out: list[dict] = []
    bound = table_crosswalk_ids()
    for sid in sorted(source_ids):
        cid = source_crosswalk_id(registry, sid, bound)
        edges = registry.crosswalk(cid).edges if cid else ()
        labels = set()
        for e in edges:
            labels.add(e.source_label)
            out.append(
                {
                    "source_id": sid, "crosswalk_id": cid, "native_label": e.source_label,
                    "native_label_id": e.source_label_id, "taxon_node_id": _axis(e, "taxon"),
                    "pixel_value": pixels.get((sid, e.source_label)),
                    "form_node_id": _axis(e, "form"), "condition_node_id": _axis(e, "condition"),
                    "match_type": e.match_type, "fidelity": e.fidelity.value,
                    "mapped": bool(e.targets), "note": e.note, "taxonomy_version": version,
                }
            )  # fmt: skip
        for native in sorted(seen[sid] - labels):  # carried, never dropped
            out.append(
                {
                    "source_id": sid, "crosswalk_id": cid, "native_label": native,
                    "pixel_value": pixels.get((sid, native)), "match_type": "unmapped",
                    "fidelity": "unmappable", "mapped": False, "note": "no crosswalk edge",
                    "taxonomy_version": version,
                }
            )  # fmt: skip
    return out


def _first(*values: Any) -> Any:
    return next((v for v in values if v not in (None, "", [], {})), None)


def source_rows(
    registry, source_ids: Iterable[str], version: str, *, lock: Mapping[str, Mapping] | None = None,
    counts: Mapping[str, Mapping] | None = None,
) -> list[dict]:  # fmt: skip
    """One row per source of the flavour: licence, attribution, rights holder, verification,
    counts, the configs it appears in and its mapped share."""
    from .release_card import source_info

    lock, counts, out = lock or {}, counts or {}, []
    for sid in sorted(source_ids):
        info = source_info(registry, sid)
        lic, ver = info.get("licence") or {}, info.get("verification") or {}
        ann = next((a for a in info.get("annotations") or [] if isinstance(a, dict)), {})
        c = counts.get(sid, {})
        labels, mapped = c.get("labels", 0), c.get("mapped", 0)
        out.append(
            {
                "source_id": sid, "version": info.get("version"),
                "root_digest": (lock.get(sid) or {}).get("root_digest") or info.get("root_digest"),
                "licence": lic.get("id"), "licence_url": lic.get("url"),
                "ann_licence": _first(ann.get("licence"), ann.get("license"), lic.get("id")),
                "attribution": info.get("_attribution"), "rights_holder": info.get("rights_holder"),
                "verified_by": ver.get("method"),
                "verified_on": str(ver.get("verified_on") or "") or None,
                "provenance_url": info.get("homepage"),
                "licence_class": coerce_class(info.get("access_class")),
                "n_images": c.get("images", 0), "configs": ",".join(sorted(c.get("configs", ()))),
                "n_labels": labels,
                "mapped_pct": round(100.0 * mapped / labels, 2) if labels else None,
                "known_defects": info.get("known_defects"), "taxonomy_version": version,
            }
        )  # fmt: skip
    return out


# ------------------------------------------------------------------------------ orchestration


def build_v11_layout(
    registry, release_dir: str | Path, roots: Mapping[str, Path], *, flavour: str,
    skip_sha256: Collection[str] = frozenset(), lock: Mapping[str, Mapping] | None = None,
    shard_target_bytes: int = SHARD_TARGET_BYTES, registry_root: str | Path | None = None,
    hidden_test_drop: Collection[str] | None = None, release_id: str | None = None,
) -> tuple[Layout, dict]:  # fmt: skip
    """``({config: (spec, {<split>[-<source>]: rows})}, report)`` for ``flavour``'s new configs.
    ``hidden_test_drop`` (open flavour only): digests whose rows in every table are removed before
    anything is counted or embedded; ``report["hidden_test_excluded"]`` records the counts.
    ``release_id`` (an additive export, ``--v1-dir``): ends the stem of every ``images`` shard, so
    none of them takes the path of a prior release's shard; no other config is renamed."""
    if flavour not in FLAVOURS:
        raise V11ExportError(f"unknown flavour {flavour!r}")
    report = json.loads((Path(release_dir) / "task_layers" / V11_TABLES_FILE).read_text())
    version = report["taxonomy_version"]
    tables = {c: read_table(release_dir, c) for c in TABLE_CONFIGS}
    meta, stats = tables["metadata"], Counter()
    class_by_sha = {
        r["image_sha256"]: r.get("licence_class") for r in meta if r.get("licence_class")
    }
    dropped: dict[str, int] = {}
    for config in tables:
        tables[config], dropped[config] = keep_for_flavour(
            registry, tables[config], flavour, class_by_sha
        )
    hidden = None
    if hidden_test_drop is not None:
        if flavour != "open":
            raise V11ExportError("the hidden-test drop is for the open flavour; nc ships them")
        tables, hidden = drop_hidden_test(tables, hidden_test_drop)
    meta = tables["metadata"]
    split_by_sha = {m["image_sha256"]: m.get("split") for m in meta}
    images, bytes_sha = image_rows(
        meta, roots, skip_sha256=skip_sha256, stats=stats,
        locate=lambda sid: staged_index(registry, sid, roots[sid]),
    )  # fmt: skip
    layout: Layout = {}
    spec = ConfigSpec(
        IMAGES, IMAGE_COLUMNS, "Image bytes, once per image_sha256.", shard_target_bytes
    )
    layout[IMAGES] = (spec, _group(images, True, release_id)) if images else (spec, {})
    counts: dict[str, dict] = defaultdict(
        lambda: {"configs": set(), "images": 0, "labels": 0, "mapped": 0}
    )
    for config in TABLE_CONFIGS:
        rows = tables[config]
        if not rows:
            continue
        if config == "metadata":
            rows = [{**r, "bytes_sha256": bytes_sha.get(r["image_sha256"])} for r in rows]
        spec = table_spec(config, rows)
        embedded = config == "semseg"
        if embedded:
            spec = ConfigSpec(
                config,
                (*spec.columns, ("mask", "image"), ("mask_bytes_sha256", "string")),
                shard_target_bytes=shard_target_bytes,
            )
        else:
            spec = ConfigSpec(config, spec.columns, shard_target_bytes=shard_target_bytes)
        built: list[tuple[str, ExportRow]] = []
        for r in rows:
            split = _split_of(r, split_by_sha)
            # the Hub split name and the resolved class, on every table that has those columns
            stamped = {**r, "split": split, "licence_class": row_class(registry, r, class_by_sha)}
            row = _as_row(spec, stamped)
            png = palette_mask(r, roots) if embedded else None
            if png is not None:
                name = f"{r['source_id']}/{r['image_sha256']}.png"
                blob = EmbeddedBlob("mask", name, len(png), lambda data=png: data)
                row = ExportRow(
                    values={**row.values, "mask_bytes_sha256": hashlib.sha256(png).hexdigest()},
                    blobs=(blob,),
                )
            built.append((split, row))
            sid = r.get("source_id")
            if sid:
                counts[sid]["configs"].add(config)
                if config == "metadata":
                    counts[sid]["images"] += 1
                elif r.get("label_native") is not None or r.get("label_native_id") is not None:
                    counts[sid]["labels"] += 1
                    counts[sid]["mapped"] += 1 if r.get("taxon_node_id") else 0
        layout[config] = (spec, _group(built, "source_id" in dict(spec.columns)))
    source_ids = sorted({r["source_id"] for r in meta})
    for sid in source_ids:
        if images and any(r.values["source_id"] == sid for _, r in images):
            counts[sid]["configs"].add(IMAGES)
    labels = [r for c in TABLE_CONFIGS if c != "metadata" for r in tables[c]]
    crosswalks = export_crosswalk_ids(registry, source_ids)
    extra: dict[str, list[dict]] = {
        LABEL_MAP: label_map_rows(registry, source_ids, version, labels),
        SOURCES: source_rows(registry, source_ids, version, lock=lock, counts=counts),
    }
    if flavour == "open":
        from .licence_class import _registry_root

        path = (
            Path(registry_root or _registry_root(None))
            / "taxonomy"
            / "releases"
            / f"{version}.json"
        )
        extra[TAXONOMY] = taxonomy_rows(json.loads(path.read_text()), version, crosswalks)
    for config, columns in (
        (TAXONOMY, TAXONOMY_COLUMNS),
        (LABEL_MAP, LABEL_MAP_COLUMNS),
        (SOURCES, SOURCES_COLUMNS),
    ):
        if extra.get(config):
            spec = ConfigSpec(config, columns, shard_target_bytes=shard_target_bytes)
            layout[config] = (
                spec,
                _group([("train", _as_row(spec, r)) for r in extra[config]], False),
            )
    if flavour == "open":  # fail closed: the open tables never name a private resource
        assert_public_safe(
            json.dumps(extra, default=str), "open", where="label-map/sources/taxonomy"
        )
    layout = {c: v for c, v in layout.items() if v[1]}
    out_report = {
        "taxonomy_version": version, "flavour": flavour, "licence_rows_dropped": dropped,
        "images": dict(stats), "sources": source_ids,
    }  # fmt: skip
    if hidden is not None:
        out_report["hidden_test_excluded"] = hidden
    return layout, out_report


def export_v11(layout: Layout, out_dir: str | Path) -> dict:
    """Write every shard and return the summary (same shape as :func:`hf_export.export`)."""
    out_dir, summary = Path(out_dir), {"configs": {}, "files": []}
    for config, (spec, groups) in layout.items():
        plans = plan_config(spec, groups)
        entry: dict = {"columns": [list(c) for c in spec.columns], "splits": {}, "written": []}
        for plan in plans:
            split = plan.split.split("-", 1)[0]
            stats = entry["splits"].setdefault(split, {"rows": 0, "shards": 0, "embedded_bytes": 0})
            stats["rows"] += len(plan.rows)
            stats["shards"] += 1
            stats["embedded_bytes"] += sum(r.embedded_bytes for r in plan.rows)
            size = write_shard(out_dir / plan.name, spec, plan.rows)
            entry["written"].append({"name": plan.name, "rows": len(plan.rows), "bytes": size})
        summary["configs"][config] = entry
        summary["files"].extend(p.name for p in plans)
    extra = ("README.md", "LICENSE", "NOTICE", "CITATION.cff", "croissant.json")
    summary["file_count"] = len(summary["files"]) + len(extra)
    summary["files_per_folder"] = files_per_folder([*summary["files"], *extra])
    summary["layout"] = "v1.1"
    return summary


def _hidden_test_drop(args, flavour: str | None) -> frozenset[str] | None:
    """The digests ``--hidden-test`` withholds, ``None`` without the flag. The flags come as a
    pair and are for the open flavour (the nc release ships the hidden-test images)."""
    manifest = getattr(args, "hidden_test", None)
    exempt = getattr(args, "hidden_test_exempt_v1", None)
    if not manifest and not exempt:
        return None
    if flavour != "open":
        raise V11ExportError(f"--hidden-test is for the open flavour, not {flavour!r}")
    if not (manifest and exempt):
        raise V11ExportError("--hidden-test needs --hidden-test-exempt-v1 DIR, and the reverse")
    return hidden_test_drop(manifest, exempt)


def additive_release_id(args) -> str | None:
    """The release id an additive export (``--v1-dir``) puts in its ``images`` shard names, ``None``
    for a plain export (``--release-id`` is then not read). It is required and must be
    ``vX.Y.Z``: a missing or free-form id would let two releases name a shard alike."""
    if not getattr(args, "v1_dir", None):
        return None
    release_id = getattr(args, "release_id", None)
    if not release_id or not RELEASE_ID.fullmatch(release_id):
        raise V11ExportError(
            f"--v1-dir needs --release-id vX.Y.Z (the release being exported), got {release_id!r}"
        )
    return release_id


def run(args) -> int:
    """``hf_export`` entry for a v1.1.0 build dir (``task_layers/V11_TABLES.json`` exists)."""
    import yaml

    from .fetch import cache_root
    from .hf_export import _roots, record_export_counts
    from .registry import Registry

    release_dir = Path(args.release_dir)
    release = json.loads((release_dir / "RELEASE.json").read_text())
    flavour = args.flavour or release.get("flavour")
    if flavour != release.get("flavour"):
        raise V11ExportError(
            f"--flavour {flavour} but the release was built as {release.get('flavour')!r}"
        )
    release_id = additive_release_id(args)
    skip: set[str] = set()
    if getattr(args, "v1_sha256", None):
        skip |= {s.strip().lower() for s in Path(args.v1_sha256).read_text().split() if s.strip()}
    if getattr(args, "v1_dir", None):
        skip |= set(v1_sha256s(args.v1_dir))
    lock_path = release_dir / "release.lock.yaml"
    lock = (
        {s["id"]: s for s in (yaml.safe_load(lock_path.read_text()) or {}).get("sources", [])}
        if lock_path.is_file()
        else {}
    )
    drop = _hidden_test_drop(args, flavour)
    layout, report = build_v11_layout(
        Registry.load(), release_dir, _roots(release_dir, cache_root()), flavour=flavour,
        skip_sha256=skip, lock=lock, hidden_test_drop=drop, release_id=release_id,
    )  # fmt: skip
    if "hidden_test_excluded" in report:
        h = report["hidden_test_excluded"]
        tables = ", ".join(f"{c}={n}" for c, n in h["rows"].items() if n)
        print(
            f"hidden_test_excluded={h['images']} images, {sum(h['rows'].values())} rows "
            f"({tables or 'none'}); {h['requested']} not in v1.0",
            file=sys.stderr,
        )
    summary = export_v11(layout, args.out)
    record_export_counts(release_dir / "RELEASE.json", summary)  # what the load/card gates check
    summary |= {"flavour": flavour, "repo_id": release.get("repo_id"), "v11": report}
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({"file_count": summary["file_count"], "configs": sorted(summary["configs"])}))
    return 0
