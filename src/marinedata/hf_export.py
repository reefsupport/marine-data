"""Export a frozen release to a Hub-ready folder (WS-D S55). Prepares; never publishes.

Layout (decided in S55, see the layout report): pixels are embedded **once**, in an
``images`` config keyed by ``image_sha256``; every task is a label-only config joined on
that key. Six task configs embedding their own copies would push ~6x the bytes for the
same pixels. Ground-truth masks live in ``masks`` (one row per ``(image, source)`` — the
658 Tayrona images carry a benthic and a bleaching mask), CoralSCOP model output in its
own ``coralscop-pseudo-masks`` config so it can never be mistaken for ground truth.

Rows are rebuilt the way :func:`marinedata.release.build_release` builds them (same
builder, same frozen split map, same two never-eval exclusions) and then **asserted equal**
to the release's own ``tasks/<task>.tsv`` multiset — the export can attach labels, but it
can never change a split or a membership. Split names map ``val``/``probe`` →
``validation`` (``probe`` is ``general-pretraining``'s name for the validation images), and
one image must land in one split across every task or the export raises.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .builder import DatasetBuilder
from .checksums import file_digest
from .flavours import row_licence_class, sample_ships
from .hf_parquet import ConfigSpec, ExportRow, files_per_folder, plan_config, write_shard
from .licence_class import drop_release_excluded, flavour_filter, require_flavour
from .privacy import policy as privacy_policy
from .registry import Registry
from .release import (
    DEFAULT_SCHEMA_ID,
    SplitGroupError,
    _admitted_source_ids,
    _loader_layout,
    _never_eval_source_ids,
    _with_release_group,
    require_split_groups,
    source_release_entries,
)
from .strata import TRAIN
from .task_layers import hf_wiring as _tl

HF_SPLITS = {"train": "train", "val": "validation", "probe": "validation", "test": "test"}
SPLIT_ORDER = ("train", "validation", "test")
IMAGES, MASKS, PSEUDO_MASKS = "images", "masks", "coralscop-pseudo-masks"
PSEUDO_TAG = "pseudo-label"
REPO_EXTRA_FILES = ("README.md", "LICENSE", ".gitattributes")

DEFAULT_REPO_ID = "reefsupport/marine-data"
"""D-A (2026-09-25, delegated): the public Hub repo id for this dataset."""

DEFAULT_EXCLUDE_CONFIGS = ("coral-genus-caribbean",)
"""D-A: dropped from the v1 Hub configs — the release TSVs themselves are untouched."""

TASK_LAYER_CONFIGS = _tl.CONFIG_IDS
"""WP-8c (D-Z2), ``--tasks v2`` only: the 6 additional HF config entries for whatever
``<release>/task_layers/<config_id>.parquet`` :func:`marinedata.release.build_release`
wrote (v1's :data:`IMAGES`/:data:`MASKS`/:data:`PSEUDO_MASKS` configs are unaffected —
this is a sibling list, not a replacement, and v1 exports never read it)."""

EXCLUDE_REASONS: dict[str, str] = {
    "coral-genus-caribbean": (
        "0 labels in v1 — every `label` and `mask_class_map` value is null. "
        "The release `tasks/coral-genus-caribbean.tsv` is unchanged; only this Hub "
        "export omits the config (D-A, 2026-09-25)."
    ),
}

IMAGE_SPEC = ConfigSpec(
    IMAGES,
    (
        ("image_sha256", "string"),
        ("image", "image"),
        ("source_id", "string"),
        ("source_ids", "string"),
        ("split_group", "string"),
    ),
    "Every v1 image once, keyed by image_sha256.",
)
LABEL_COLUMNS = (
    ("image_sha256", "string"),
    ("source_id", "string"),
    ("sample_key", "string"),
    ("label", "string"),
    ("native_label", "string"),
    ("label_reason", "string"),
    ("mask_class_map", "string"),
)
PRETRAIN_COLUMNS = (("image_sha256", "string"), ("source_id", "string"), ("sample_key", "string"))
MASK_COLUMNS = (
    ("image_sha256", "string"),
    ("source_id", "string"),
    ("mask", "image"),
    ("mask_values", "string"),
)


class HFExportError(Exception):
    """The export would not be a faithful copy of the frozen release."""


def hf_split(name: str) -> str:
    try:
        return HF_SPLITS[name]
    except KeyError:
        raise HFExportError(f"release split {name!r} has no Hub split name") from None


@dataclass(frozen=True)
class SampleRow:
    task_id: str
    raw_split: str
    image_sha256: str
    source_id: str
    sample_key: str
    image: Path
    split_group: str | None = None
    mask: Path | None = None
    mask_values: str | None = None
    label: str | None = None
    native_label: str | None = None
    label_reason: str | None = None
    mask_class_map: str | None = None
    licence_class: str | None = None
    license: str | None = None

    @property
    def split(self) -> str:
        return hf_split(self.raw_split)


def read_manifest(path: Path) -> Counter:
    """``Counter((image_sha256, split))`` of one release ``tasks/<task>.tsv``."""
    counts: Counter = Counter()
    lines = path.read_text().splitlines()
    if not lines or lines[0].split("\t") != ["image_sha256", "split"]:
        raise HFExportError(f"{path}: not a release task manifest")
    for line in lines[1:]:
        sha, split = line.split("\t")
        counts[(sha, split)] += 1
    return counts


def _mask_values(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    pairs = (part.partition("=") for part in str(raw or "").split(",") if part.strip())
    return {k.strip(): v.strip() for k, _, v in pairs}


def _resolve_mask(root: Path, mask) -> Path | None:
    if mask is None:
        return None
    path = Path(mask)
    for candidate in (path, root / "labels" / path, root / path):
        if candidate.is_file():
            return candidate
    raise HFExportError(f"mask {mask} not found under {root}")


def _pseudo_mask(root: Path, image: Path) -> Path | None:
    candidate = root / "labels" / "masks" / f"{image.stem}.png"
    return candidate if candidate.is_file() else None


def _labels(registry: Registry, task, projector, sample) -> dict[str, str | None]:
    out: dict[str, str | None] = {"label": None, "label_reason": None, "mask_class_map": None}
    native = sample.meta.get("native_image_labels")
    out["native_label"] = None if native is None else json.dumps(native, default=str)
    if projector is None or task.axis is None:
        return out
    value = sample.labels.get(task.axis)
    if value is not None:
        projection = projector.project(value.node_id)
        out["label"], out["label_reason"] = projection.target_class, projection.reason
    values = _mask_values(sample.meta.get("mask_values"))
    loader = registry.source(sample.source_id).loader
    if sample.mask is not None and values and loader is not None and loader.crosswalk_id:
        crosswalk = registry.crosswalk(loader.crosswalk_id)
        mapping: dict[str, str | None] = {}
        for pixel, native_name in values.items():
            edge = crosswalk.edge(native_name)
            node = edge.targets.get(task.axis) if edge is not None else None
            mapping[pixel] = projector.project(node).target_class if node else None
        out["mask_class_map"] = json.dumps(mapping, sort_keys=True)
    return out


def collect_rows(
    registry: Registry, roots: dict[str, Path], release_dir: Path, profile: str = "research"
) -> dict[str, list[SampleRow]]:
    """Every task's rows, labels attached, asserted equal to the release manifests."""
    release = json.loads((release_dir / "RELEASE.json").read_text())
    split_map = release_dir / "SPLIT_MAP.json"
    excluded = frozenset(release["never_eval_near_dup_excluded"]["sha256"])
    flavour = release.get("flavour")
    admitted = {sid: Path(roots[sid]) for sid in _admitted_source_ids(registry, roots, profile)}
    never_eval = _never_eval_source_ids(registry, admitted)
    pseudo = {sid for sid in admitted if PSEUDO_TAG in registry.source(sid).tags}
    digests: dict[Path, str] = {}
    walk_groups = {  # the same group lookup build_release applies to non-staged-tree layouts
        sid: {
            e.key: e.group for e in source_release_entries(registry.source(sid), root, file_digest)
        }
        for sid, root in admitted.items()
        if _loader_layout(registry.source(sid)) != "staged-tree"
    }
    out: dict[str, list[SampleRow]] = {}
    for task_id in release["tasks"]:
        task = registry.task(task_id)
        dataset = DatasetBuilder(
            registry,
            profile=profile,
            roots=admitted,
            schema_id=task.schema_id or DEFAULT_SCHEMA_ID,
            task_id=task_id,
            exclude_unmapped=True,  # the release left these sources out of the task (RELEASE.json)
        ).build()
        dataset.samples = [_with_release_group(s, walk_groups, admitted) for s in dataset.samples]
        try:
            require_split_groups(dataset.samples, task_id)
        except SplitGroupError as exc:
            raise HFExportError(str(exc)) from exc
        dataset.split(by="group", split_map=split_map, frozen=True, tolerance=None)
        projector = dataset.projector
        rows: list[SampleRow] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None or not sample_ships(registry, sample, flavour):
                    continue  # per-row-licence sources: the rows the release manifest kept
                row_licence = sample.meta.get("license")
                image = Path(sample.image)
                sha = digests.get(image) or digests.setdefault(image, file_digest(image))
                if sample.source_id in never_eval and (sha in excluded or split_name != TRAIN):
                    continue  # the two never-eval exclusions build_release applies
                root = admitted[sample.source_id]
                mask = _resolve_mask(root, sample.mask)
                if mask is None and sample.source_id in pseudo:
                    mask = _pseudo_mask(root, image)
                values = _mask_values(sample.meta.get("mask_values"))
                rows.append(
                    SampleRow(
                        task_id=task_id,
                        raw_split=split_name,
                        image_sha256=sha,
                        source_id=sample.source_id,
                        sample_key=sample.key,
                        image=image,
                        split_group=sample.meta.get("split_group"),
                        licence_class=row_licence_class(registry, sample.source_id, row_licence),
                        license=row_licence,
                        mask=mask,
                        mask_values=",".join(f"{k}={v}" for k, v in values.items()) or None,
                        **_labels(registry, task, projector, sample),
                    )
                )
        got = Counter((r.image_sha256, r.raw_split) for r in rows)
        expected = read_manifest(release_dir / "tasks" / f"{task_id}.tsv")
        if got != expected:
            raise HFExportError(
                f"{task_id}: rebuilt rows differ from the release manifest "
                f"(+{sum((got - expected).values())} / -{sum((expected - got).values())})"
            )
        out[task_id] = sorted(rows, key=lambda r: (r.image_sha256, r.source_id, r.sample_key))
    return out


def drop_excluded(
    rows_by_task: dict[str, list[SampleRow]], exclude: Sequence[str]
) -> dict[str, list[SampleRow]]:
    """Task configs to omit from the Hub export (D-A). Images already shared with a
    kept config (e.g. ``general-pretraining``) stay in ``images``/``masks`` — this
    only drops the excluded config's own label-only files."""
    skip = frozenset(exclude)
    return {task: rows for task, rows in rows_by_task.items() if task not in skip}


def _by_split(rows: Sequence[ExportRow], splits: Sequence[str]) -> dict[str, list[ExportRow]]:
    grouped: dict[str, list[ExportRow]] = {s: [] for s in SPLIT_ORDER}
    for row, split in zip(rows, splits, strict=True):
        grouped[split].append(row)
    return {s: grouped[s] for s in SPLIT_ORDER if grouped[s]}


def _file_row(values: dict, file: Path, name: str) -> ExportRow:
    return ExportRow(values=values, file=file, file_path=name, file_bytes=file.stat().st_size)


def build_layout(
    rows_by_task: dict[str, list[SampleRow]],
    pseudo_sources: frozenset[str] = frozenset(),
    task_layers: dict[str, list[dict]] | None = None,
    flavour: str | None = None,
) -> dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]]:
    """``{config: (spec, {hf_split: rows})}`` — images once, tasks label-only, masks apart.
    ``flavour`` (``open`` | ``nc``, WP-L1a) keeps only the rows that flavour may ship, in the
    images, masks and task configs alike."""
    require_flavour(flavour, "build_layout")
    keep = (lambda r: flavour_filter(r, flavour)) if flavour else drop_release_excluded
    rows_by_task = {t: keep(r) for t, r in rows_by_task.items()}
    task_layers = {t: keep(r) for t, r in (task_layers or {}).items()}
    by_sha: dict[str, list[SampleRow]] = defaultdict(list)
    for rows in rows_by_task.values():
        for row in rows:
            by_sha[row.image_sha256].append(row)
    image_rows, image_splits = [], []
    for sha in sorted(by_sha):
        members = by_sha[sha]
        splits = {m.split for m in members}
        if len(splits) != 1:
            raise HFExportError(f"image {sha} lands in splits {sorted(splits)} across tasks")
        primary = min(members, key=lambda m: (m.source_id, m.sample_key))
        values = {
            "image_sha256": sha,
            "source_id": primary.source_id,
            "source_ids": ",".join(sorted({m.source_id for m in members})),
            "split_group": primary.split_group,
        }
        image_rows.append(
            _file_row(values, primary.image, f"{primary.source_id}/{primary.sample_key}")
        )
        image_splits.append(splits.pop())
    layout = {IMAGES: (IMAGE_SPEC, _by_split(image_rows, image_splits))}

    for config, wanted in ((MASKS, False), (PSEUDO_MASKS, True)):
        seen: dict[tuple[str, str], SampleRow] = {}
        for rows in rows_by_task.values():
            for row in rows:
                if row.mask is not None and (row.source_id in pseudo_sources) is wanted:
                    seen.setdefault((row.image_sha256, row.source_id), row)
        mask_rows = [
            _file_row(
                {
                    "image_sha256": r.image_sha256,
                    "source_id": r.source_id,
                    "mask_values": r.mask_values,
                },
                r.mask,
                f"{r.source_id}/{r.mask.name}",
            )
            for _, r in sorted(seen.items())
        ]
        if mask_rows:
            spec = ConfigSpec(config, MASK_COLUMNS)
            layout[config] = (
                spec,
                _by_split(mask_rows, [r.split for _, r in sorted(seen.items())]),
            )

    for task_id, rows in sorted(rows_by_task.items()):
        supervised = any(r.label or r.mask_class_map for r in rows)
        columns = LABEL_COLUMNS if supervised else PRETRAIN_COLUMNS
        names = [name for name, _ in columns]
        export = [ExportRow(values={n: getattr(r, n) for n in names}) for r in rows]
        layout[task_id] = (ConfigSpec(task_id, columns), _by_split(export, [r.split for r in rows]))
    layout.update(_tl.layout_entries(task_layers or {}, {k: v[0].split for k, v in by_sha.items()}))
    return layout


def per_task_embedding_bytes(rows_by_task: dict[str, list[SampleRow]]) -> dict[str, int]:
    """What each task config would weigh if it embedded its own pixels (rejected layout)."""
    out: dict[str, int] = {}
    for task, rows in rows_by_task.items():
        first = {}
        for row in rows:
            first.setdefault(row.image_sha256, row.image)
        out[task] = sum(path.stat().st_size for path in first.values())
    return out


def export(layout, out_dir: Path, *, sample: bool = False) -> dict:
    """Write the shards (``sample``: only shard 0 of each config's train split, or its
    first split) and return the per-config/split summary."""
    summary: dict = {"configs": {}, "files": []}
    for config, (spec, splits) in layout.items():
        plans = plan_config(spec, splits)
        chosen = plans[:1] if sample else plans
        entry: dict = {"columns": [list(c) for c in spec.columns], "splits": {}}
        for split, rows in splits.items():
            entry["splits"][split] = {
                "rows": len(rows),
                "shards": sum(1 for p in plans if p.split == split),
                "embedded_bytes": sum(r.file_bytes for r in rows),
            }
        written = []
        for plan in chosen:
            size = write_shard(out_dir / plan.name, spec, plan.rows)
            written.append(
                {
                    "name": plan.name,
                    "rows": len(plan.rows),
                    "embedded_bytes": sum(r.file_bytes for r in plan.rows),
                    "bytes": size,
                }
            )
        entry["written"] = written
        summary["configs"][config] = entry
        summary["files"].extend(p.name for p in plans)
    all_files = [*summary["files"], *REPO_EXTRA_FILES]
    summary["file_count"] = len(all_files)
    summary["files_per_folder"] = files_per_folder(all_files)
    return summary


def _roots(release_dir: Path, cache: Path) -> dict[str, Path]:
    release = json.loads((release_dir / "RELEASE.json").read_text())
    roots = {}
    for source in release["sources"]:
        root = cache / source["id"]
        fetched = json.loads((root / "_fetch.json").read_text())
        if fetched.get("truncated"):
            raise HFExportError(f"{source['id']}: cached tree is a truncated sample")
        roots[source["id"]] = root
    return roots


def main(argv: list[str] | None = None) -> int:
    from .fetch import cache_root

    parser = argparse.ArgumentParser(prog="python -m marinedata.hf_export")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sample", action="store_true", help="one shard per config only")
    parser.add_argument(
        "--flavour",
        choices=("open", "nc"),
        default=None,
        help="open -> reefsupport/marine-data, nc -> reefsupport/marine-data-nc; must match "
        "the --release-dir's RELEASE.json (default: that file's flavour; collect_rows stays "
        "unfiltered for its manifest check)",
    )
    parser.add_argument(
        "--profile", default=None, help="default: the profile recorded in RELEASE.json"
    )
    parser.add_argument(
        "--exclude-configs",
        default=",".join(DEFAULT_EXCLUDE_CONFIGS),
        help="comma-separated task configs to drop from the Hub export (D-A); '' for none",
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="skip the per-image `metadata` config"
    )
    parser.add_argument(
        "--quality", type=Path, default=None, help="WP-1 quality.parquet joined into `metadata`"
    )
    parser.add_argument(
        "--privacy",
        type=Path,
        default=None,
        help="privacy.parquet from `privacy-scan`: apply the release privacy policy (score >= "
        "0.85 excludes the image and lists it in RELEASE.json; 0.60-0.85 flags it in `metadata`)",
    )
    args = parser.parse_args(argv)

    registry = Registry.load()
    release_json = args.release_dir / "RELEASE.json"
    release = json.loads(release_json.read_text()) if release_json.is_file() else {}
    flavour = args.flavour or release.get("flavour")
    if args.flavour and release.get("flavour") != args.flavour:
        raise HFExportError(
            f"--flavour {args.flavour} but {args.release_dir} was built as flavour "
            f"{release.get('flavour')!r}"
        )
    profile = args.profile or release.get("profile") or "research"
    roots = _roots(args.release_dir, cache_root())
    rows = collect_rows(registry, roots, args.release_dir, profile)
    exclude = [c for c in args.exclude_configs.split(",") if c]
    rows = drop_excluded(rows, exclude)
    rows, privacy = privacy_policy.apply_to_rows(rows, args.privacy)
    if args.privacy:
        privacy_policy.record_in_release(release_json, privacy)
    pseudo = frozenset(s for s in roots if PSEUDO_TAG in registry.source(s).tags)
    layout = build_layout(
        rows, pseudo, task_layers=_tl.read_task_layers(args.release_dir), flavour=flavour
    )
    if not args.no_metadata:  # WP-R2b: licence_class / geo / split_group per image, same flavour
        import pyarrow.parquet as pq

        from .metadata_release import add_metadata_config

        quality = pq.read_table(args.quality).to_pylist() if args.quality else []
        layout = add_metadata_config(
            layout, registry, roots, flavour=flavour,
            quality_by_sha={r["image_sha256"]: r for r in quality},
            privacy_flags=privacy.flags,
        )  # fmt: skip
    summary = export(layout, args.out, sample=args.sample)
    if flavour:
        summary["flavour"] = flavour
        summary["repo_id"] = release["repo_id"]
    summary["per_task_embedding_bytes"] = per_task_embedding_bytes(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                "file_count": summary["file_count"],
                "configs": {c: e["splits"] for c, e in summary["configs"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
