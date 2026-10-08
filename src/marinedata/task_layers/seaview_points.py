"""Seaview benthic points: the staged tree -> the complete ``points`` table (P3.1).

``marinedata tasks build-points --source seaview-survey-imagery`` reads, anonymously and by single
key (never a listing, never an image), the staged tree's ``CHECKSUMS.sha256``, ``metadata.parquet``
and the nine ``labels/points/tabular-data/annotations_<REGION>.csv``. It writes ``points.parquet``
(the unified ``points`` schema, one row per annotated point, keyed by ``image_sha256``),
``summary.json`` and ``unmatched.tsv``. The rules, each checkable in the output:

* the 11-digit quadrat id is the JPG stem. The image (``image_sha256``, ``width``,
  ``height``) is the region-partition row of ``metadata.parquet``. The ``labelled_data`` rows
  repeat region stems with different bytes (other sha256), so they never key a point;
* an image with no annotation row and an annotation id with no image are never guessed: both go to
  ``unmatched.tsv`` with a reason. A single constant id offset between the two sets is shown in
  ``hint`` and not applied;
* ``ann_license`` / ``ann_attribution`` are the registry entry's licence id and citation;
* ``worms_aphia_id`` comes from the taxonomy node. A taxon node without one is looked up once by
  scientific name in the WoRMS REST API (``AphiaIDByName``, at most 200 calls) and cached in
  ``registry/taxonomy/worms-cache.json``;
* a head sample of the 1.59 GB master CSV (one ranged GET) is compared with the region CSVs.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..annotation_schema import write_annotations
from ..checksums import parse_checksums
from ..metadata_norm.resolvers.join import join_points, load_resolved, location_sensitive_sources
from ..metadata_norm.resolvers.staged import (
    RangeLedger,
    StagedImage,
    http_range,
    make_getter,
    read_rows,
    staged_images,
)
from ..registry import Registry, _default_root
from .points_table import (
    resolver_for,
    seaview_rows,
    source_attribution,
    source_licence,
    write_points,
)
from .s3_keyed import fetch_small

SOURCE_ID = "seaview-survey-imagery"
CROSSWALK_ID = "seaview-point-labels"
TARGET_SCHEMA = "rs-benthic-v1"
TABLES = "labels/points/tabular-data/"
MASTER = TABLES + "seaviewsurvey_annotations.csv"
LABELLED_PARTITION = "labelled_data"
MASTER_SAMPLE_BYTES = 8 * (1 << 20)
WORMS_MAX_CALLS = 200
WORMS_URL = "https://www.marinespecies.org/rest/AphiaIDByName/{name}?marine_only=true"
UNMATCHED_COLUMNS = ("kind", "quadrat_id", "region", "n_points", "image_path", "reason", "hint")
NO_IMAGE = "no staged region image has this quadrat id (labelled_data copies are not used)"
NO_ROWS = "staged region image with no row in any annotations_<REGION>.csv"


@dataclass(frozen=True)
class QuadratImage:
    quadrat_id: str
    image_sha256: str
    width: int
    height: int
    partition: str
    image_path: str


def quadrat_index(images: list[StagedImage]) -> tuple[dict[str, QuadratImage], dict[str, int]]:
    """``{quadrat id: image}`` over the region partitions, plus counted notes. The ``labelled_data``
    rows are left out: they repeat region stems under other bytes."""
    index: dict[str, QuadratImage] = {}
    notes: Counter[str] = Counter()
    for img in sorted(images, key=lambda i: (i.partition, i.stem)):
        if img.partition == LABELLED_PARTITION:
            notes["labelled_data_rows_excluded"] += 1
            continue
        width, height = img.row.get("width"), img.row.get("height")
        if not (img.sha256 and width and height):
            notes["region_rows_without_sha256_or_size"] += 1
        elif img.stem in index:
            notes["duplicate_region_stems_skipped"] += 1
        else:
            index[img.stem] = QuadratImage(
                img.stem, img.sha256, int(width), int(height), img.partition, img.rel_path
            )
    for img in images:
        twin = index.get(img.stem)
        if img.partition == LABELLED_PARTITION and twin and twin.image_sha256 != img.sha256:
            notes["labelled_data_rows_with_other_sha256_than_region_twin"] += 1
    return index, dict(notes)


def _http_aphia(name: str) -> int | None:
    req = urllib.request.Request(
        WORMS_URL.format(name=urllib.parse.quote(name)),
        headers={"Accept": "application/json", "User-Agent": "marinedata-points/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        if exc.code in (204, 404):
            return None
        raise
    value = int(body) if body else 0
    return value if value > 0 else None  # -999 = several matches: not a match


class WormsCache:
    """``{scientific name: aphia id | null}`` on disk; at most ``max_calls`` REST calls per run."""

    def __init__(
        self,
        path: Path | None,
        *,
        fetch: Callable[[str], int | None] | None = None,
        max_calls: int = WORMS_MAX_CALLS,
        enabled: bool = True,
    ) -> None:
        self.path, self.fetch, self.max_calls, self.enabled = (
            path,
            fetch or _http_aphia,
            max_calls,
            enabled,
        )
        self.calls = self.failed = 0
        self.names: dict[str, dict] = {}
        if path is not None and path.is_file():
            self.names = dict(json.loads(path.read_text(encoding="utf-8")).get("names", {}))

    def aphia(self, name: str) -> int | None:
        if name in self.names:
            return self.names[name]["aphia_id"]
        if not self.enabled or self.calls >= self.max_calls:
            return None
        self.calls += 1
        try:
            value = self.fetch(name)
        except (OSError, ValueError):
            self.failed += 1  # transient: not cached, so a rerun asks again
            return None
        self.names[name] = {"aphia_id": value, "queried_on": dt.date.today().isoformat()}
        return value

    def save(self) -> None:
        if self.path is None:
            return
        doc = {
            "api": "AphiaIDByName?marine_only=true",
            "schema": 1,
            "names": dict(sorted(self.names.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "calls": self.calls,
            "failed": self.failed,
            "max_calls": self.max_calls,
            "names_cached": len(self.names),
        }


def aphia_fallback(registry: Registry, cache: WormsCache) -> Callable[[str], int | None]:
    """``taxon node id -> aphia id`` for a node without one: a taxon node is looked up by its
    scientific name; a non-taxon node (substrate, unknown) never is."""
    schema = registry.label_schema(TARGET_SCHEMA)
    memo: dict[str, int | None] = {}

    def fallback(node_id: str) -> int | None:
        if node_id not in memo:
            node = schema.node(node_id)
            if node is None or node.non_taxon:
                memo[node_id] = None
            elif node.worms_aphia_id is not None:
                memo[node_id] = node.worms_aphia_id
            else:
                memo[node_id] = cache.aphia(node.worms_scientificname or node.name)
        return memo[node_id]

    return fallback


def _norm_label(label: str) -> str:
    return label.replace("-", "_").strip().lower()


def _point_key(r: Mapping[str, str]) -> tuple[int, int, str]:
    return round(float(r["y"])), round(float(r["x"])), _norm_label(r["label"])


def parse_master_sample(sample: bytes) -> dict[str, Counter]:
    """``{quadrat id: Counter((y, x, label))}`` over the complete ids of a head sample of the master
    CSV (``quadratid,y,x,label``). The cut-off last line and the last id are dropped."""
    lines = sample.decode("utf-8", errors="ignore").splitlines()[:-1]
    out: dict[str, Counter] = {}
    order: list[str] = []
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        qid = r["quadratid"]
        if qid not in out:
            order.append(qid)
            out[qid] = Counter()
        out[qid][_point_key(r)] += 1
    if order:
        del out[order[-1]]
    return out


def compare_master(
    master: Mapping[str, Counter], region: Mapping[str, Counter], ids: set[str]
) -> dict:
    def split(c: Counter) -> tuple[Counter, Counter]:
        xy: Counter = Counter()
        lab: Counter = Counter()
        for (y, x, label), n in c.items():
            xy[(y, x)] += n
            lab[label] += n
        return xy, lab

    compared = same_xy = same_lab = equal = absent = 0
    bad: list[dict] = []
    for qid, mc in master.items():
        if qid not in ids:
            absent += 1
            continue
        compared += 1
        (mxy, mlab), (rxy, rlab) = split(mc), split(region.get(qid, Counter()))
        a, b = mxy == rxy, mlab == rlab
        same_xy, same_lab, equal = same_xy + a, same_lab + b, equal + (a and b)
        if not (a and b) and len(bad) < 20:
            bad.append(
                {
                    "quadrat_id": qid,
                    "master_points": sum(mc.values()),
                    "region_points": sum(region.get(qid, Counter()).values()),
                    "same_coordinates": a,
                    "same_labels": b,
                }
            )
    return {
        "read": True,
        "complete_ids_in_sample": len(master),
        "compared": compared,
        "ids_not_in_region_csvs": absent,
        "same_coordinates": same_xy,
        "same_labels": same_lab,
        "fully_equal": equal,
        "mismatching_ids": compared - equal,
        "mismatches_first_20": bad,
    }


def _offset_hints(images_only: list[str], ids_only: list[str]) -> dict[str, str]:
    """If the id-less images and the image-less ids pair up (sorted) under ONE constant offset, say
    so. Never applied."""
    a, b = sorted(images_only), sorted(ids_only)
    if len(a) != len(b) or len(a) < 2:
        return {}
    try:
        diffs = {int(y) - int(x) for x, y in zip(a, b, strict=True)}
    except ValueError:
        return {}
    if len(diffs) != 1:
        return {}
    d = diffs.pop()
    hints: dict[str, str] = {}
    for x, y in zip(a, b, strict=True):
        hints[x] = f"candidate annotation id {y} (id {d:+d}); not applied"
        hints[y] = f"candidate image {x} (id {-d:+d}); not applied"
    return hints


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 3) if total else None


@dataclass
class BuildResult:
    rows: list[dict]
    summary: dict
    unmatched: list[dict]


def build_seaview_points(
    registry: Registry,
    *,
    version: str,
    meta_rows: list[dict],
    checksums: Mapping[str, str],
    csv_texts: Mapping[str, str],
    fallback: Callable[[str], int | None] | None = None,
    worms: WormsCache | None = None,
    master_sample: bytes | None = None,
) -> BuildResult:
    index, index_notes = quadrat_index(staged_images(meta_rows, checksums))

    def info(qid: str):
        im = index.get(qid)
        return None if im is None else (im.image_path, im.width, im.height, im.image_sha256)

    master, master_error = {}, None
    if master_sample:
        try:
            master = parse_master_sample(master_sample)
        except (KeyError, ValueError) as exc:
            master_error = f"{type(exc).__name__}: {exc}"
    rows: list[dict] = []
    regions: dict[str, dict] = {}
    csv_points: Counter[str] = Counter()
    first_region: dict[str, str] = {}
    region_cmp: dict[str, Counter] = {}
    for region in sorted(csv_texts):
        per_id: Counter[str] = Counter()
        for r in csv.DictReader(io.StringIO(csv_texts[region])):
            qid = r["quadratid"]
            per_id[qid] += 1
            if qid in master:
                region_cmp.setdefault(qid, Counter())[_point_key(r)] += 1
        new = seaview_rows(
            csv_texts[region],
            region,
            registry,
            info,
            source_version=version,
            start=len(rows),
            aphia_fallback=fallback,
        )
        rows.extend(new)
        csv_points.update(per_id)
        for qid in per_id:
            first_region.setdefault(qid, region)
        regions[region] = {
            "csv_rows": sum(per_id.values()),
            "points": len(new),
            "unmatched_points": sum(per_id.values()) - len(new),
            "images_in_tree": sum(1 for i in index.values() if i.partition == region),
            "images_with_points": sum(1 for q in per_id if q in index),
            "points_on_images_of_another_region": sum(
                n for q, n in per_id.items() if q in index and index[q].partition != region
            ),
        }
    ids_no_image = sorted(q for q in first_region if q not in index)
    imgs_no_rows = sorted(q for q in index if q not in first_region)
    hints = _offset_hints(imgs_no_rows, ids_no_image)
    unmatched = [
        {
            "kind": "annotation_id_without_image",
            "quadrat_id": q,
            "region": first_region[q],
            "n_points": csv_points[q],
            "image_path": "",
            "reason": NO_IMAGE,
            "hint": hints.get(q, ""),
        }
        for q in ids_no_image
    ] + [
        {
            "kind": "image_without_annotation_rows",
            "quadrat_id": q,
            "region": index[q].partition,
            "n_points": 0,
            "image_path": index[q].image_path,
            "reason": NO_ROWS,
            "hint": hints.get(q, ""),
        }
        for q in imgs_no_rows
    ]
    plain = resolver_for(registry, CROSSWALK_ID)
    codes = Counter(r["label_native"] for r in rows)
    schema = registry.label_schema(TARGET_SCHEMA)
    missing: dict[str, dict] = {}
    for code in codes:
        taxon, _mt, aphia, _l2 = plain(code)
        if taxon is not None and aphia is None and taxon not in missing:
            node = schema.node(taxon)
            missing[taxon] = {
                "name": node.worms_scientificname or node.name,
                "non_taxon": bool(node.non_taxon),
                "aphia_id_after_lookup": fallback(taxon) if fallback else None,
            }
    n = len(rows)
    mt = Counter(r["match_type"] for r in rows)
    with_aphia = sum(1 for r in rows if r["worms_aphia_id"] is not None)
    from_taxonomy = sum(c for code, c in codes.items() if plain(code)[2] is not None)
    if master_sample is None:
        check: dict = {"read": False, "reason": "disabled (--master-sample-bytes 0)"}
    elif master_error:
        check = {"read": False, "reason": master_error}
    else:
        check = compare_master(master, region_cmp, set(first_region)) | {
            "bytes_read": len(master_sample),
            "note": "head sample of the master only; its full totals are not read "
            "(1.59 GB > the read cap)",
        }
    summary = {
        "source_id": SOURCE_ID,
        "source_version": version,
        "crosswalk": CROSSWALK_ID,
        "ann_license": source_licence(registry, SOURCE_ID),
        "ann_attribution": source_attribution(registry, SOURCE_ID),
        "annotator_type": dict(Counter(r["annotator_type"] for r in rows)),
        "regions": regions,
        "totals": {
            "csv_rows": sum(csv_points.values()),
            "points": n,
            "unmatched_points": sum(csv_points[q] for q in ids_no_image),
            "images_in_tree": len(index),
            "images_with_points": len(index) - len(imgs_no_rows),
            "images_without_annotation_rows": len(imgs_no_rows),
            "annotation_ids_without_image": len(ids_no_image),
        },
        "match_type": dict(sorted(mt.items())),
        "crosswalked_points": n - mt.get("unmapped", 0),
        "crosswalked_pct": _pct(n - mt.get("unmapped", 0), n),
        "taxon_node_pct": _pct(sum(1 for r in rows if r["taxon_node_id"]), n),
        "worms_aphia_pct": _pct(with_aphia, n),
        "worms_aphia_pct_from_taxonomy_nodes": _pct(from_taxonomy, n),
        "worms_nodes_without_aphia": missing,
        "worms_lookup": worms.stats() if worms else {"enabled": False},
        "distinct_codes": len(codes),
        "unmapped_codes": {c: k for c, k in sorted(codes.items()) if plain(c)[1] == "unmapped"},
        "upstream_split": dict(Counter(r["upstream_split"] for r in rows)),
        "clamped_points": sum(1 for r in rows if r["attrs"] and '"clamped": true' in r["attrs"]),
        "index_notes": index_notes,
        "master_check": check,
    }
    return BuildResult(rows, summary, unmatched)


def write_outputs(
    out_dir: str | Path,
    result: BuildResult,
    *,
    version: str,
    annotations_root: str | Path | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_annotations(out / "points.parquet", "points", result.rows)
    (out / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = ["\t".join(UNMATCHED_COLUMNS)] + [
        "\t".join(str(u[c]) for c in UNMATCHED_COLUMNS) for u in result.unmatched
    ]
    (out / "unmatched.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if annotations_root is not None:  # the layout build_points_config reads
        write_points(Path(annotations_root), SOURCE_ID, version, result.rows)
    return out


def run_build_points(
    out_dir: str | Path,
    *,
    version: str | None = None,
    registry_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    master_sample_bytes: int = MASTER_SAMPLE_BYTES,
    worms_cache_path: str | Path | None = None,
    worms: bool = True,
    annotations_root: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    fetch: Callable[..., bytes] = fetch_small,
    range_fetch: Callable[[str, int], bytes] | None = None,
    worms_fetch: Callable[[str], int | None] | None = None,
) -> dict:
    """Read the staged tree (single-key anonymous GETs), build, write; returns ``summary``."""
    from ..metadata_norm.resolvers.run import default_version

    root = Path(registry_root) if registry_root else _default_root()
    registry = Registry.load(root)
    version = version or default_version(SOURCE_ID, registry_root)
    resolved = load_resolved(metadata_dir, SOURCE_ID) if metadata_dir else {}
    if metadata_dir and not resolved:
        raise ValueError(
            f"{metadata_dir}: no {SOURCE_ID}/*/samples.parquet (metadata resolve --out)"
        )
    tree = f"sources/{SOURCE_ID}/{version}/"
    checksums = parse_checksums(fetch(tree + "CHECKSUMS.sha256").decode())
    get = make_getter(tree, checksums, cache_dir, fetch)
    meta_rows = read_rows(get("metadata.parquet"))
    csv_texts = {
        Path(rel).stem.removeprefix("annotations_"): get(rel).decode("utf-8")
        for rel in sorted(checksums)
        if rel.startswith(TABLES + "annotations_") and rel.endswith(".csv")
    }
    if not csv_texts:
        raise ValueError(f"{tree}{TABLES}: no annotations_<REGION>.csv in CHECKSUMS.sha256")
    sample = None
    if master_sample_bytes and MASTER in checksums:
        ranged = range_fetch or http_range(
            tree, RangeLedger(max_requests=1, max_bytes=master_sample_bytes)
        )
        sample = ranged(MASTER, master_sample_bytes)
    cache = WormsCache(
        Path(worms_cache_path) if worms_cache_path else root / "taxonomy" / "worms-cache.json",
        fetch=worms_fetch,
        enabled=worms,
    )
    result = build_seaview_points(
        registry,
        version=version,
        meta_rows=meta_rows,
        checksums=checksums,
        csv_texts=csv_texts,
        fallback=aphia_fallback(registry, cache),
        worms=cache,
        master_sample=sample,
    )
    cache.save()
    out = write_outputs(out_dir, result, version=version, annotations_root=annotations_root)
    if not resolved:
        return result.summary
    # P2.5: points + the resolved samples (fill-only, by image_sha256) -> points_joined.parquet
    n = join_points(
        out / "points.parquet",
        out / "points_joined.parquet",
        resolved,
        location_sensitive_sources(registry),
    )
    n = n | {f"{k}_pct": _pct(v, n["points"]) for k, v in n.items()}
    text = json.dumps(n, indent=2, sort_keys=True) + "\n"
    (out / "metadata_join.json").write_text(text, encoding="utf-8")
    return {**result.summary, "metadata_join": n}
