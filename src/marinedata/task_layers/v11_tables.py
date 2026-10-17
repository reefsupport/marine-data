"""v1.1.0 build stage (PR-B): the stamped task tables and the ``metadata`` spine.

``release build --tasks v2 --v11`` turns the task-layer configs into the v1.1.0 task tables:

* ``metadata`` (B7): one row per staged image of every admitted source (v1.0 images included),
  keyed on the build's digest of the staged file. Both staged ``metadata.parquet`` schemas count
  (PR-M): the narrow image-labels index (no ``image_sha256``) and the wide ``sample_schema``
  (its stored ``image_sha256`` is compared, never trusted). Rows come from the v1.0 ``metadata``
  machinery (:func:`marinedata.metadata_release.build_rows`) over the split map's own enumeration
  (:func:`marinedata.release.entries.source_release_entries`), the tree's provenance columns fill
  the gaps, and an admitted source with images or task rows but no row fails the build
  (:class:`MetadataSpineError`). Each row has the ``metadata`` Hub-config columns plus
  ``bytes_config`` (the config holding the pixels: the v1.0 config that embeds the image, else
  ``images``), ``bytes_sha256`` (null here; the export fills it when it rewrites bytes),
  ``split`` and ``taxonomy_version``. A row never carries bytes.
* ``points`` (B1), ``bleaching`` (B2), ``boxes`` (B5, B4), ``semseg`` and ``instances`` (B3, B4):
  the existing builders' rows stamped with ``split``, ``taxonomy_version``, ``fidelity`` (the
  crosswalk fidelity of ``match_type``) and ``bytes_config`` (``mask_config`` / ``mask_ref`` on
  masks: ``mask_config`` names the frozen v1.0 mask config only when that config holds the
  image, else null and the export embeds the mask), typed by the WP-U1 table of the same name
  (:mod:`marinedata.annotation_schema`).

Taxonomy 3.0.0 validates a row against its stamped release, so every row of a new table carries
``taxonomy_version`` = the build's release; the v1.0 tables stay under 2.9.0, never rebuilt.
An image with no split in the split map fails the build (:class:`UnsplitImagesError`: the count and
the first five ``image_sha256``), because the Hub configs are per split. ``V11Inputs.allow_unsplit``
(rehearsal builds only) writes its rows with ``split = null`` and counts them (``unsplit``,
``unsplit_images``). An image of a withheld near-dup group is dropped from every table, and so is
an image a person withheld in ``registry/privacy/review.yaml`` (``V11Inputs.privacy_withheld``,
:mod:`marinedata.privacy.review`); ``withheld_images`` counts both and ``withheld_images_privacy``
the privacy ones. A frozen build knows the split map's withheld groups (``resolve_splits`` leaves
them out of every task manifest): the report adds ``withheld_groups`` (how many distinct groups
the drop covered) and ``withheld_by_source`` (``{source: {images, groups}}``, by the source that
described the image); they never count as ``unsplit``. An image the split map lists under
``band_excluded`` (schema 2: the per-image prior-release band; its group keeps its split) is
dropped the same way and reported as ``band_excluded_images`` and ``band_excluded_by_source``
(``{source: {images, groups}}``). A row of a source outside
``V11Inputs.roots`` (the flavour's admitted set) is dropped, so an nc source cannot reach the open
tables through this layer.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..annotation_schema import TABLES
from ..label_schemes import current_taxonomy_version
from ..licence_class import coerce_class, resolve_row_class
from ..registry import Registry
from .configs.common import ConfigResult, _attrs
from .configs.constants import MASK_CONFIG_BY_SOURCE
from .mask_classmap import palette_json

logger = logging.getLogger(__name__)

IMAGES_CONFIG = "images"
V1_BYTES_CONFIG: dict[str, str] = {**MASK_CONFIG_BY_SOURCE, "roboflow-aquarium": "fish-boxes"}
"""Source -> the frozen v1.0 config that embeds its images. The UIIS family sits in both
``instance-masks`` and ``fish-boxes`` (grandfathered double embed): the mask config wins."""
V1_EMBEDDING_CONFIGS = frozenset(V1_BYTES_CONFIG.values())
"""The frozen v1.0 configs that embed image pixels: the only configs a ``bytes_config`` can name."""
_V1_CONFIG_ORDER = tuple(sorted(V1_EMBEDDING_CONFIGS))
V11_CONFIG_IDS = ("metadata", "points", "bleaching", "boxes", "semseg", "instances")
ANNOTATION_TABLE = {
    "points": "points", "bleaching": "image_labels", "boxes": "boxes", "semseg": "masks",
    "instances": "masks",
}  # fmt: skip
FIDELITY = {
    "exact": "exact", "broader": "coarsened", "narrower": "approximate", "related": "approximate",
    "unmapped": "unmappable",
}  # fmt: skip
"""``match_type`` (WP-U1) -> the crosswalk fidelity word. ``narrower`` is stored for audit and never
used as a positive, so it reads as approximate."""
_STAMP = (
    ("split", "str"),
    ("taxonomy_version", "str"),
    ("fidelity", "str"),
    ("bytes_config", "str"),
)
_EXTRA_CORE = {
    "bleaching": (("near_dup_group", "str"), ("near_dup_images", "int32")),
    "boxes": (("modality", "str"),),
    "semseg": (("mask_config", "str"), ("palette", "str")),
    "instances": (("modality", "str"), ("mask_config", "str")),
}  # fmt: skip


def bytes_config_of(
    source_id: str | None, sha: str | None, v1_config_shas: Mapping[str, Collection[str]]
) -> str:
    """The config holding one image's bytes: the frozen v1.0 config whose sha set holds ``sha``,
    else ``images``. v1.0 configs never gain a row, so an image the v1.0 tree does not hold (a new
    image of a v1.0 source included) is exported into ``images``. ``source_id`` only ranks the
    configs: its own v1.0 config first (the UIIS family sits in both ``instance-masks`` and
    ``fish-boxes``: the mask config wins), then the others by name."""
    own = V1_BYTES_CONFIG.get(source_id or "")
    for config in (own, *_V1_CONFIG_ORDER):
        if config is not None and sha in v1_config_shas.get(config, ()):
            return config
    return IMAGES_CONFIG


def mask_config_of(
    source_id: str | None, sha: str | None, v1_config_shas: Mapping[str, Collection[str]]
) -> str | None:
    """The frozen v1.0 mask config holding one image's mask, else ``None`` (the v1.1 export embeds
    the mask: :func:`marinedata.hf_export_v11.palette_mask`). The mask twin of
    :func:`bytes_config_of`: v1.0 configs never gain a row, so a mask of an image the source's v1.0
    mask config does not hold (a new image of a v1.0 source included) is not stamped with it. Only
    the source's own mask config counts: a mask is that source's labels, not shared pixels, so
    another config holding the sha says nothing about it."""
    own = MASK_CONFIG_BY_SOURCE.get(source_id or "")
    return own if own is not None and sha in v1_config_shas.get(own, ()) else None


def source_bytes_config(
    source_id: str | None, v1_config_shas: Mapping[str, Collection[str]]
) -> str:
    """The v1.0 config the v1.0 tree embeds ``source_id``'s images in, else ``images``: the
    source-level view of :func:`bytes_config_of`, used to order sources (a row is decided per
    image)."""
    config = V1_BYTES_CONFIG.get(source_id or "")
    return config if config is not None and v1_config_shas.get(config) else IMAGES_CONFIG


class MissingV1TreeError(ValueError):
    """The build admits v1.0 sources but has no v1.0 tree (``--v1-dir``) to say which of their
    images v1.0 already embeds; guessing would stamp a v1.0 config on images it does not hold."""


def require_v1_tree(source_ids: Iterable[str]) -> None:
    """Raise :class:`MissingV1TreeError`, naming ``--v1-dir``, when ``source_ids`` has a v1.0
    source."""
    held = sorted(s for s in source_ids if s in V1_BYTES_CONFIG)
    if held:
        raise MissingV1TreeError(
            f"the lock admits v1.0 source(s) {', '.join(held)} but no v1.0 tree was given: pass "
            "--v1-dir DIR (the frozen v1.0 release dir with data/<config>/*.parquet, as for "
            "`release gate --v11 --v1-dir`) so their v1.0 images keep the v1.0 config and their "
            "new images go to `images`"
        )


def load_v1_config_shas(
    v1_dir: str | Path | None, source_ids: Iterable[str]
) -> dict[str, frozenset[str]]:
    """``{v1.0 config: {image_sha256}}`` for the configs that embed pixels, read from ``v1_dir``
    with the gates' reader. No ``v1_dir`` is fine only when no v1.0 source is admitted; a tree with
    no ``image_sha256`` in any embedding config (a wrong path) is as bad as no tree."""
    sources = list(source_ids)
    if v1_dir is None:
        require_v1_tree(sources)
        return {}
    from ..gates.v11_common import v1_config_shas

    found = {
        config: frozenset(shas)
        for config, shas in v1_config_shas(Path(v1_dir), V1_EMBEDDING_CONFIGS).items()
    }
    if not any(found.values()) and any(s in V1_BYTES_CONFIG for s in sources):
        raise MissingV1TreeError(
            f"--v1-dir {v1_dir}: no image_sha256 under data/<config>/*.parquet for "
            f"{', '.join(sorted(V1_EMBEDDING_CONFIGS))}: this is not the frozen v1.0 release dir"
        )
    return found


def fidelity_of(match_type: str | None) -> str | None:
    return FIDELITY.get(match_type) if match_type else None


class UnsplitImagesError(ValueError):
    """The split map leaves admitted images without a split; a release build refuses."""


class MetadataSpineError(ValueError):
    """An admitted source has staged images or task rows but no ``metadata`` row (PR-M)."""


@dataclass(frozen=True)
class V11Inputs:
    """What the v1.1.0 layer needs besides the config rows: the build's taxonomy release, the
    admitted source roots, the frozen split map and the near-dup groups."""

    taxonomy_version: str
    roots: Mapping[str, Path]
    split_of: Mapping[str, str] = field(default_factory=dict)
    """``image_sha256 -> split`` from the build's task manifests (wins over ``assignments``)."""
    assignments: Mapping[str, str] = field(default_factory=dict)
    """``split_group -> split``: the frozen (inherited) split map."""
    withheld_groups: frozenset[str] = frozenset()
    near_dup_group_of: Mapping[str, str] = field(default_factory=dict)
    """``image_sha256 -> dedup group`` (``groups.parquet``); else the image's ``split_group``."""
    allow_unsplit: bool = False
    """Rehearsal only: keep the rows of images the split map does not cover (``split = null``)."""
    digest: Callable[[Path], str] | None = None
    """The build's (memoized) file digest, else :func:`marinedata.checksums.file_digest`: the spine
    keys every staged image on it, as the task manifests do."""
    privacy_withheld: frozenset[str] = frozenset()
    """``image_sha256`` of the images the privacy review list says to withhold: dropped from every
    table, whatever their split group."""
    phash_excluded: frozenset[str] = frozenset()
    """The map's ``phash_excluded`` images (schema 2): no row anywhere, like the band."""
    band_excluded: frozenset[str] = frozenset()
    """``image_sha256`` of the images the split map lists under ``band_excluded`` (schema 2): a new
    image inside the prior-release near-duplicate band, on the side its component did not keep.
    Dropped from every table, as ``release build`` drops it from every config; its group keeps its
    split and the group's other images ship."""
    v1_config_shas: Mapping[str, Collection[str]] = field(default_factory=dict)
    """``v1.0 config -> image_sha256`` of the frozen v1.0 tree (``--v1-dir``): an image keeps a v1.0
    ``bytes_config`` only when its sha is here; every other image goes to ``images``. A mask keeps
    its v1.0 ``mask_config`` on the same test, else the export embeds it."""


def inputs_from_build(
    *, taxonomy_version: str | None, roots: Mapping[str, str | Path], manifests: Iterable[Any],
    split_map: Any, dedup_groups: str | Path | None = None, allow_unsplit: bool = False,
    digest: Callable[[Path], str] | None = None, privacy_withheld: Iterable[str] = (),
    v1_config_shas: Mapping[str, Collection[str]] | None = None,
) -> V11Inputs:  # fmt: skip
    """:class:`V11Inputs` from what ``build_release`` already holds. ``v1_config_shas`` is the
    v1.0 tree's ``{config: shas}`` (:func:`load_v1_config_shas`); ``None`` (no tree) is refused
    when a v1.0 source is admitted (:func:`require_v1_tree`)."""
    if v1_config_shas is None:
        require_v1_tree(roots)
        v1_config_shas = {}
    near: Mapping[str, str] = {}
    if dedup_groups is not None:
        from ..dedup.groups import load_groups

        near = load_groups(dedup_groups)[0]
    return V11Inputs(
        taxonomy_version=taxonomy_version or current_taxonomy_version(),
        roots={sid: Path(root) for sid, root in roots.items()},
        split_of={sha: split for m in manifests for sha, split in m.rows},
        assignments=dict(split_map.assignments),
        withheld_groups=frozenset(split_map.withheld),
        privacy_withheld=frozenset(privacy_withheld),
        band_excluded=frozenset(getattr(split_map, "band_excluded", ())),
        phash_excluded=frozenset(getattr(split_map, "phash_excluded", ())),
        near_dup_group_of=dict(near),
        allow_unsplit=allow_unsplit,
        digest=digest,
        v1_config_shas={config: frozenset(shas) for config, shas in v1_config_shas.items()},
    )


def metadata_columns() -> tuple[tuple[str, str], ...]:
    """The ``metadata`` Hub-config columns (SampleRow v3 names) plus the v1.1.0 additions."""
    from ..metadata_norm.resolvers.join import EXTRA_COLUMNS
    from ..metadata_release import METADATA_COLUMNS

    cols = dict(METADATA_COLUMNS)
    for name, kind in EXTRA_COLUMNS:
        cols.setdefault(name, kind)
    cols.setdefault("split", "string")
    cols.update(bytes_config="string", bytes_sha256="string", taxonomy_version="string")
    return tuple(cols.items())


def core_columns(config_id: str) -> tuple[tuple[str, str], ...]:
    """The typed columns of one v1.1.0 table (the rest are inferred from the rows)."""
    if config_id == "metadata":
        return metadata_columns()
    base = tuple((c.name, c.kind) for c in TABLES[ANNOTATION_TABLE[config_id]].columns)
    have = {n for n, _ in base}
    more = (*_EXTRA_CORE.get(config_id, ()), *_STAMP)
    return base + tuple(c for c in more if c[0] not in have)


_ARROW_KINDS = {
    "str": "string", "string": "string", "int16": "int16", "int32": "int32", "int64": "int64",
    "float32": "float32", "double": "float64", "float64": "float64", "bool": "bool_",
}  # fmt: skip


def _arrow_type(kind: str):
    """The pyarrow type of a declared column kind; ``None`` for a kind this layer does not know
    (the column is then typed from its rows)."""
    import pyarrow as pa

    name = _ARROW_KINDS.get(kind)
    return getattr(pa, name)() if name else None


def _inferred_type(rows: Sequence[Mapping[str, Any]], key: str):
    import pyarrow as pa

    kind = pa.array([r.get(key) for r in rows]).type
    return pa.string() if pa.types.is_null(kind) else kind


def table_schema(config_id: str, rows: Sequence[Mapping[str, Any]] = ()):
    """The pyarrow schema of one v1.1.0 table: :func:`core_columns` in declared order and type,
    then every other column of ``rows`` typed from its values (an all-null column is a string)."""
    import pyarrow as pa

    core = core_columns(config_id)
    fields = [pa.field(n, _arrow_type(k) or _inferred_type(rows, n)) for n, k in core]
    known = {n for n, _ in core}
    extra = list(dict.fromkeys(k for r in rows for k in r if k not in known))
    return pa.schema(fields + [pa.field(k, _inferred_type(rows, k)) for k in extra])


def v11_schemas(results: Mapping[str, ConfigResult]) -> dict[str, Any]:
    return {
        cid: table_schema(cid, results[cid].rows if cid in results else ())
        for cid in V11_CONFIG_IDS
    }


def _plain(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


_NOT_FROM_TREE = frozenset(
    {
        "image_sha256", "source_id", "split", "lat", "lon", "location_generalized",
        "licence_class", "bytes_config", "bytes_sha256", "taxonomy_version",
    }
)  # fmt: skip
"""Columns a staged record never fills: the key, the build's stamps, and the location columns the
v1.0 rows already resolved under the location rule (geo backfill, generalisation)."""


def _spine_row(base: Mapping[str, Any], rec: Mapping[str, Any], source_id: str, inputs) -> dict:
    """A v1.0 ``build_rows`` row on the ``metadata`` columns; the staged record fills what it left
    null (the v1.0 join's fill-only rule): a wide tree's upstream_url, camera, capture time..."""
    names = [n for n, _ in metadata_columns()]
    row = {n: _plain(base.get(n)) for n in names}
    for n in names:
        if row[n] is None and n not in _NOT_FROM_TREE and rec.get(n) is not None:
            row[n] = _plain(rec[n])
    row.update(
        source_id=source_id,
        bytes_config=bytes_config_of(source_id, row["image_sha256"], inputs.v1_config_shas),
        split=None, taxonomy_version=inputs.taxonomy_version, bytes_sha256=None,
    )  # fmt: skip
    return row


def _sha_keyed_row(rec: Mapping[str, Any], source: Any, source_id: str, inputs) -> dict:
    """A bare ``image_sha256``-keyed index (no ``stem`` + ``partition`` / ``image_path``, so no
    file to hash): the record as stored, the pre-PR-M behaviour."""
    names = [n for n, _ in metadata_columns()]
    row = {n: _plain(rec.get(n)) for n in names}
    row.update(
        image_sha256=rec["image_sha256"], source_id=source_id,
        bytes_config=bytes_config_of(source_id, rec["image_sha256"], inputs.v1_config_shas),
        taxonomy_version=inputs.taxonomy_version, bytes_sha256=None,
        location_generalized=bool(rec.get("location_generalized")),
        licence_class=resolve_row_class(
            coerce_class(source.access_class), rec.get("license"),
            per_row=bool(source.licence_per_row),
        ),
    )  # fmt: skip
    return row


def _tree_rows(
    registry: Registry, source_id: str, root: Path, inputs: V11Inputs
) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """One admitted tree: its spine rows, ``sha -> walk group`` and its counts. A staged sample
    index (narrow or wide) is enumerated as the split map enumerates it and hashed with the
    build's digest; a stored ``image_sha256`` that differs from the bytes is counted."""
    import pyarrow.parquet as pq

    from ..checksums import file_digest
    from ..metadata_release import ImageRef, build_rows
    from ..release.entries import _staged_meta, source_release_entries

    counts = {"images": 0, "sha256_mismatches": 0, "licence_dropped": 0}
    path = Path(root) / "metadata.parquet"
    if not path.is_file():
        return [], {}, counts
    source = registry.source(source_id)
    records = pq.read_table(path).to_pylist()
    if not _staged_meta(path):
        rows = [
            _sha_keyed_row(r, source, source_id, inputs) for r in records if r.get("image_sha256")
        ]
        counts["images"] = len(rows)
        return rows, {}, counts
    entries = source_release_entries(source, root, inputs.digest or file_digest)
    if len(entries) != len(records) or any(
        e.path.stem != r["stem"] for e, r in zip(entries, records, strict=True)
    ):
        raise MetadataSpineError(
            f"{source_id}: the staged enumeration and metadata.parquet disagree"
        )
    refs = [ImageRef(e.sha256, source_id, e.path, "", e.group) for e in entries]
    base = {
        r["image_sha256"]: r
        for r in build_rows(refs, registry, Path(root), {}, source_roots={source_id: Path(root)})
    }
    rows, mismatched = [], []
    for entry, rec in zip(entries, records, strict=True):
        stored = str(rec.get("image_sha256") or "").lower()
        if stored and stored != entry.sha256:
            mismatched.append(str(rec["stem"]))
        if entry.sha256 not in base:  # the release licence rule dropped it (v1.0 rows alike)
            counts["licence_dropped"] += 1
            continue
        rows.append(_spine_row(base[entry.sha256], rec, source_id, inputs))
    counts.update(images=len(entries), sha256_mismatches=len(mismatched))
    if mismatched:
        logger.warning(
            "%s: %d stored image_sha256 value(s) differ from the staged bytes (keyed on the bytes);"
            " first: %s", source_id, len(mismatched), ", ".join(mismatched[:5]),
        )  # fmt: skip
    return rows, {e.sha256: e.group for e in entries}, counts


def _metadata_rows(
    registry: Registry, inputs: V11Inputs
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    """``(sha -> row, sha -> walk group, source -> counts)``; an image staged by two sources is
    described once, by the first (``shared`` counts it for the other)."""
    seen: dict[str, dict] = {}
    walk: dict[str, str] = {}
    by_source: dict[str, dict[str, int]] = {}
    v1 = inputs.v1_config_shas
    order = sorted(inputs.roots, key=lambda s: (source_bytes_config(s, v1) == IMAGES_CONFIG, s))
    for source_id in order:  # v1.0-embedded sources first: their config holds the bytes
        rows, groups, counts = _tree_rows(
            registry, source_id, Path(inputs.roots[source_id]), inputs
        )
        for sha, group in groups.items():
            walk.setdefault(sha, group)
        added = 0
        for row in rows:
            if row["image_sha256"] not in seen:
                seen[row["image_sha256"]] = row
                added += 1
        shared = len({r["image_sha256"] for r in rows}) - added
        by_source[source_id] = {**counts, "rows": added, "shared": shared}
    return seen, walk, by_source


def mark_near_dup_conflicts(
    rows: Iterable[Mapping[str, Any]], group_of: Mapping[str, str]
) -> tuple[dict, ...]:
    """B2: every row gets its ``near_dup_group`` (the image's own sha when it has none) and
    ``near_dup_images``; when the parties of a group — one per (source, image) — disagree on the
    canonical condition, every row of the group gets ``label_status = conflict``. All rows are kept.
    One image carrying several labels (a mask's bleached and healthy regions) is one party."""
    rows = list(rows)
    parties: dict[str, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        sha = r.get("image_sha256") or r.get("sha256")
        party = parties[group_of.get(sha, sha)][(r.get("source_id"), sha)]
        if r.get("canonical_condition"):
            party.add(r["canonical_condition"])
    conflicted = {g for g, p in parties.items() if len({frozenset(v) for v in p.values() if v}) > 1}
    out = []
    for r in rows:
        sha = r.get("image_sha256") or r.get("sha256")
        group = group_of.get(sha, sha)
        out.append(
            {
                **r, "near_dup_group": group,
                "near_dup_images": len({s for _, s in parties[group]}),
                "label_status": "conflict" if group in conflicted else r.get("label_status", "ok"),
            }
        )  # fmt: skip
    return tuple(out)


def _withheld_group(
    group: str | None, walk_group: str | None, withheld: frozenset[str]
) -> str | None:
    """The split-map-withheld group an image sits in, else ``None``: its ``metadata`` row's
    ``split_group``, then the group the release walk gave it (the two keys the split lookup
    tries)."""
    return next((g for g in (group, walk_group) if g is not None and g in withheld), None)


def _stamp_row(config_id: str, row: Mapping[str, Any], inputs: V11Inputs, split_of) -> dict:
    sha = row.get("image_sha256") or row.get("sha256")
    source_id = row.get("source_id")
    out = {
        **row, "image_sha256": sha, "split": split_of.get(sha),
        "taxonomy_version": inputs.taxonomy_version, "fidelity": fidelity_of(row.get("match_type")),
        "bytes_config": bytes_config_of(source_id, sha, inputs.v1_config_shas),
    }  # fmt: skip
    if config_id in ("boxes", "instances"):
        out["modality"] = _attrs(row).get("modality") or row.get("modality")
    if config_id in ("semseg", "instances"):
        # mask_config: the v1.0 config embedding this mask (null: the export embeds it)
        out["mask_config"] = mask_config_of(source_id, sha, inputs.v1_config_shas)
        out["mask_ref"] = row.get("mask_ref") or row.get("mask_key")
    if config_id == "semseg":
        out["palette"] = palette_json(row.get("hf_class_map"))
    return out


def build_v11_tables(
    registry: Registry, results: Mapping[str, ConfigResult], inputs: V11Inputs
) -> dict[str, ConfigResult]:
    """The six v1.1.0 tables (``metadata`` plus the five stamped configs) from the built
    ``results``; the metadata result's ``notes`` carries the build report."""
    meta, walk, by_source = _metadata_rows(registry, inputs)
    split_of = dict(inputs.split_of)
    withheld: set[str] = set()
    privacy: set[str] = set()
    banded_images: dict[str, set[str]] = defaultdict(set)  # source -> images the band excludes
    phashed_images: dict[str, set[str]] = defaultdict(set)  # source -> phash_excluded images
    banded_groups: dict[str, set[str]] = defaultdict(set)  # source -> their split groups
    held_images: dict[str, set[str]] = defaultdict(set)  # source -> images of withheld groups
    held_groups: dict[str, set[str]] = defaultdict(set)  # source -> those withheld groups
    meta_rows: list[dict] = []
    for sha, row in meta.items():
        group = row.get("split_group")
        if sha in inputs.privacy_withheld:  # a person said: this image does not ship
            withheld.add(sha)
            privacy.add(sha)
            continue
        held = _withheld_group(group, walk.get(sha), inputs.withheld_groups)
        if held is not None:  # the split map withholds its group: no split, so no row anywhere
            withheld.add(sha)
            held_images[str(row.get("source_id"))].add(sha)
            held_groups[str(row.get("source_id"))].add(held)
            continue
        if sha in inputs.phash_excluded:  # one image of a cross-split pHash-64 pair: no row
            phashed_images[str(row.get("source_id"))].add(sha)
            continue
        if sha in inputs.band_excluded:  # the per-image prior-release band: no row anywhere
            source = str(row.get("source_id"))
            banded_images[source].add(sha)
            banded_groups[source].add(str(group or walk.get(sha) or sha))
            continue
        split = (
            split_of.get(sha) or inputs.assignments.get(group)
            or inputs.assignments.get(walk.get(sha))
        )  # fmt: skip
        if split:
            split_of[sha] = split
        meta_rows.append({**row, "split": split})
    group_of = {sha: r["split_group"] for sha, r in meta.items() if r.get("split_group")}
    group_of.update(inputs.near_dup_group_of)
    tables: dict[str, ConfigResult] = {}
    report: dict[str, Any] = {"taxonomy_version": inputs.taxonomy_version, "tables": {}}
    task_rows: Counter = Counter()
    # + task rows off the spine
    dropped = withheld | inputs.privacy_withheld | inputs.band_excluded | inputs.phash_excluded
    for cid in V11_CONFIG_IDS[1:]:
        base = results.get(cid)
        rows = base.rows if base else ()
        if cid == "bleaching":
            rows = mark_near_dup_conflicts(rows, group_of)
        rows = tuple(_stamp_row(cid, r, inputs, split_of) for r in rows)
        rows = tuple(r for r in rows if r.get("source_id") in inputs.roots)
        task_rows.update(r.get("source_id") for r in rows)
        rows = tuple(r for r in rows if r["image_sha256"] not in dropped)
        tables[cid] = ConfigResult(cid, rows, dict(base.unmapped_by_source) if base else {})
    tables = {"metadata": ConfigResult("metadata", tuple(meta_rows), {}), **tables}
    by_source = {s: {**c, "task_rows": task_rows.get(s, 0)} for s, c in sorted(by_source.items())}
    missing = [
        s
        for s, c in by_source.items()
        if (c["images"] or c["task_rows"]) and not (c["rows"] or c["shared"])
    ]
    if missing:
        detail = "; ".join(
            f"{s}: {by_source[s]['images']} staged image(s), "
            f"{by_source[s]['task_rows']} task row(s)"
            for s in missing
        )
        raise MetadataSpineError(
            f"admitted source(s) with no metadata row: {detail}. The images / sources configs and "
            "gate G1 join on metadata; check the tree's metadata.parquet against its images/"
        )
    unsplit = sorted(
        {
            str(r.get("image_sha256") or "(no image_sha256)")
            for res in tables.values()
            for r in res.rows
            if r.get("split") is None
        }
    )
    if unsplit and not inputs.allow_unsplit:
        raise UnsplitImagesError(
            f"{len(unsplit)} admitted image(s) have no split in the split map (the Hub configs "
            f"are per split); first {min(5, len(unsplit))} image_sha256: {', '.join(unsplit[:5])}. "
            "Extend the map with `marinedata release split-map --inherit ... --lock ...`; "
            "--allow-unsplit is for rehearsal builds only"
        )
    for cid, res in tables.items():
        report["tables"][cid] = {
            "rows": len(res.rows), "images": res.n_images,
            "unsplit": sum(1 for r in res.rows if r.get("split") is None),
        }  # fmt: skip
    report["metadata_by_source"] = by_source
    report["withheld_images"] = len(withheld)
    report["withheld_images_privacy"] = len(privacy)
    report["withheld_groups"] = len(set().union(*held_groups.values()))
    report["withheld_by_source"] = {
        s: {"images": len(held_images[s]), "groups": len(held_groups[s])}
        for s in sorted(held_images)
    }
    report["band_excluded_images"] = sum(len(shas) for shas in banded_images.values())
    report["band_excluded_by_source"] = {
        s: {"images": len(banded_images[s]), "groups": len(banded_groups[s])}
        for s in sorted(banded_images)
    }
    report["phash_excluded_images"] = sum(len(shas) for shas in phashed_images.values())
    report["phash_excluded_by_source"] = {s: len(phashed_images[s]) for s in sorted(phashed_images)}
    report["unsplit_images"] = len(unsplit)
    report["allow_unsplit"] = inputs.allow_unsplit
    notes = {"report": json.loads(json.dumps(report))}
    return {**tables, "metadata": ConfigResult("metadata", tuple(meta_rows), {}, notes=notes)}


def write_v11_report(
    results: Mapping[str, ConfigResult],
    out_dir: str | Path,
    *,
    staged_layers: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """``V11_TABLES.json`` next to the tables: taxonomy version, rows / images / unsplit rows per
    table and the withheld counts: images (split-map and privacy), split-map groups and images by
    source (the build's own accounting for release gate G6). ``staged_layers`` (a ``--v11`` build:
    :func:`~marinedata.task_layers.staged_annotations.layer_record` per overlay layer) adds the
    rows / images / reused / detail of every task annotation built from the staged trees."""
    meta = results.get("metadata")
    report = dict(meta.notes.get("report", {})) if meta is not None else {}
    if staged_layers is not None:
        report["staged_layers"] = [dict(layer) for layer in staged_layers]
    path = Path(out_dir) / "V11_TABLES.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path
