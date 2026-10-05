"""Dataset card (``README.md``) and ``LICENSE`` for the Hub export (WS-D S55).

Licences are recorded exactly as the registry states them (Yohan 2026-09-25: licence
concerns are out of scope for this prep — record, do not gate). The YAML ``configs:`` block
is generated from the same export summary that named the shards, so the card can never
point at a file the export did not plan.
"""

from __future__ import annotations

import json

from .hf_card_flavour import license_header, tail_sections, top_notice
from .hf_export import (
    DEFAULT_EXCLUDE_CONFIGS,
    DEFAULT_REPO_ID,
    EXCLUDE_REASONS,
    IMAGES,
    MASKS,
    PSEUDO_MASKS,
    SPLIT_ORDER,
)
from .privacy import policy as privacy_policy
from .task_layers import hf_wiring as _tl

METADATA = "metadata"

CONFIG_BLURB = {
    IMAGES: "every image once (pixels embedded), keyed by `image_sha256`",
    MASKS: "ground-truth dense masks, one row per (image, source)",
    PSEUDO_MASKS: "CoralSCOP **model output** masks — weak supervision only, never ground truth",
    "general-pretraining": "pretraining membership (train + validation; test excluded)",
    METADATA: (
        "no pixels — licence, provenance, position/depth, MEOW ecology and WP-1's quality "
        "scores, one row per `image_sha256` (WP-2, D-K)"
    ),
}


def decon_limitations(release: dict) -> list[str]:
    """The card's limitations lines for benchmarks decon could not verify (from
    ``RELEASE.json``'s ``decon.exempt``); empty when every benchmark was checked."""
    exempt = (release.get("decon") or {}).get("exempt") or {}
    if not exempt:
        return []
    return [
        "",
        "## Limitations",
        "",
        "Decontamination not verified against: "
        + "; ".join(f"`{bid}` ({reason})" for bid, reason in sorted(exempt.items()))
        + ".",
    ]


def _yaml_configs(summary: dict) -> list[str]:
    lines = ["configs:"]
    for config in summary["configs"]:
        lines.append(f"- config_name: {config}")
        if config == IMAGES:
            lines.append("  default: true")
        lines.append("  data_files:")
        for split in SPLIT_ORDER:
            if split in summary["configs"][config]["splits"]:
                lines.append(f"  - split: {split}")
                lines.append(f"    path: data/{config}/{split}-*.parquet")
    return lines


def _sources_table(sources: list[dict]) -> list[str]:
    rows = [
        "| Source | Licence (registry) | Tier | Images in v1 | Citation |",
        "|---|---|---|---|---|",
    ]
    for s in sources:
        citation = (s.get("citation") or "—").replace("\n", " ").replace("|", "/")
        rows.append(
            f"| `{s['id']}` ({s['version']}) | {s['licence']} | {s['tier']} | "
            f"{s.get('images', '—')} | {citation} |"
        )
    notes = [f"- `{s['id']}`: {s['licence_note']}" for s in sources if s.get("licence_note")]
    return rows + (["", *notes] if notes else [])


def _split_table(summary: dict) -> list[str]:
    rows = ["| Config | " + " | ".join(SPLIT_ORDER) + " |", "|---|" + "---|" * len(SPLIT_ORDER)]
    for config, entry in summary["configs"].items():
        cells = [str(entry["splits"].get(s, {}).get("rows", "—")) for s in SPLIT_ORDER]
        rows.append(f"| `{config}` | " + " | ".join(cells) + " |")
    return rows


def render_card(
    summary: dict,
    release: dict,
    sources: list[dict],
    *,
    pretty_name: str,
    repo_id: str = DEFAULT_REPO_ID,
    unsupervised: tuple[str, ...] = (),
    excluded: dict[str, str] | None = None,
    metadata_licence_rows: list[str] | None = None,
    task_layers: dict | None = None,
    flavour: str | None = None,
    takedown_url: str | None = None,
) -> str:
    near = release["near_dup"]
    empty_note = [
        f"- `{c}` has **no supervision in this release**: every `label` and every "
        "`mask_class_map` value is null. It is published for split membership only."
        for c in unsupervised
    ]
    excluded = excluded or {}
    not_included = (
        ["", "## Not included", ""]
        + [f"- `{c}` — {reason}" for c, reason in sorted(excluded.items())]
        if excluded
        else []
    )
    n_images = sum(v["rows"] for v in summary["configs"][IMAGES]["splits"].values())
    ids = sorted({s["licence_hf"] for s in sources})
    yaml = [
        "---",
        f"pretty_name: {json.dumps(pretty_name)}",
        *(
            license_header(flavour)
            if flavour
            else ["license: other", "license_name: mixed-per-source", "license_link: LICENSE"]
        ),
        "task_categories:",
        "- image-classification",
        "- image-segmentation",
        "tags:",
        "- coral-reef",
        "- marine",
        "- benthic",
        "- coral-bleaching",
        "size_categories:",
        "- 10K<n<100K",
        *_yaml_configs(summary),
        "---",
    ]
    body = [
        f"# {pretty_name}",
        *(top_notice(flavour, repo_id) if flavour else []),
        "",
        f"Release `{release['release']}` of the Reef Support coral-reef imagery corpus: "
        f"{n_images} unique images from {len(sources)} sources, one frozen split shared by "
        "every task "
        f"(`SPLIT_MAP.json` sha256 `{release['split_map_sha256']}`).",
        "",
        "## Layout",
        "",
        "Pixels are stored once, in `images`; every task config carries labels only and joins "
        "on `image_sha256`. Load a task and attach pixels with a dict lookup or a join:",
        "",
        "```python",
        "from datasets import load_dataset",
        f'labels = load_dataset("{repo_id}", "coral-health-binary", split="train")',
        f'images = load_dataset("{repo_id}", "images", split="train")',
        "```",
        "",
        *[
            f"- `{c}` — {CONFIG_BLURB.get(c, 'task labels (label-only, join on `image_sha256`)')}"
            for c in summary["configs"]
        ],
        "",
        *_split_table(summary),
        "",
        "A task row is one *(image, source)* sample: identical bytes published by two sources "
        "appear once in `images` and once per source in a task. `label` is the task class "
        "(null = the source abstains or does not supervise this axis), `label_reason` says how "
        "it was reached, `native_label` is the source's own label. For mask sources, "
        "`mask_class_map` maps each mask pixel value to the task class (null = unlabelled or "
        "abstain); the mask itself is in `masks`.",
        *not_included,
        "",
        "## Sources",
        "",
        f"Licence identifiers present: {', '.join(ids)}. Each source keeps its own licence; "
        "see `LICENSE`.",
        "",
        *_sources_table(sources),
        "",
        *(
            [
                "## Per-sample metadata",
                "",
                "`metadata` carries no pixels: licence/provenance, position/depth, MEOW "
                "ecology and WP-1's quality scores, one row per `image_sha256`. `license` "
                "is never null (D-C: record, never a storage filter) — counted here from "
                "the config's own rows, not just declared per-source:",
                "",
                *metadata_licence_rows,
                "",
                "Filter by licence in one call:",
                "",
                "```python",
                "from datasets import load_dataset",
                "from marinedata.metadata_release import filter_by_license",
                "",
                f'meta = load_dataset("{repo_id}", "metadata", split="train")',
                'allowed = filter_by_license(meta.data.table, allow=["CC-BY-4.0", "CC0-1.0"])',
                "```",
                "",
            ]
            if metadata_licence_rows
            else []
        ),
        "## Splits",
        "",
        "Split by `split_group` (site/station/transect for our own imagery, the upstream "
        "grouping for others), stratified by source, frozen in `SPLIT_MAP.json`. Every image "
        "has one split across every task. `general-pretraining` is train + validation (the "
        "release calls its validation part `probe`); test is never used for pretraining. "
        "Never-eval sources (CoralSCOP pseudo-labels) only ever appear in train.",
        "",
        "## Near-duplicate handling",
        "",
        f"Perceptual hash: {near['algorithm']} (Pillow {near['pil_version']}).",
        f"- Rule A: a never-eval image within Hamming {near['never_eval_exclude_max_hamming']} "
        f"of any eval-capable image is excluded from every task "
        f"({release['never_eval_near_dup_excluded']['count']} images).",
        f"- Rule B: images within Hamming {near['union_max_hamming']} are unioned into one "
        "split group before the split, so near-copies cannot straddle train and test.",
        f"- Chain guard: a union component larger than {near['chain_guard_fraction']:.0%} "
        "of the corpus fails the build.",
        "",
        "**Known limitation.** A 64-bit dHash sees structure, not content: low-texture frames "
        "(flat sand against open water, blue-water shots) hash close together and can be "
        "merged though they are different scenes. The error is conservative — it only ever "
        "moves images into the same split, never leaks one across splits — but it can shrink "
        "evaluation sets. A confirm-hash (second, content-aware check on each merge) is "
        "planned for v2.",
        "",
        "## Label notes",
        "",
        "- Image-level condition labels (Roboflow, NOAA PIFSC) roll up to "
        "`coral-health-binary`; `bleaching-condition` drops sources whose `Unhealthy` label "
        "cannot be split into its finer classes (listed in `RELEASE.json`).",
        "- Our Colombian benthic masks label biota only; benthic tasks see them via "
        "`mask_class_map`.",
        "- Point annotations are not part of v1; no config carries points.",
        "- CoralSCOP masks are model output (`coralscop-pseudo-masks`) and never an "
        "evaluation target.",
        *empty_note,
        *decon_limitations(release),
        *privacy_policy.card_lines(release),
        "",
        *_tl.card_section(task_layers),
    ]
    if flavour:
        body += tail_sections(flavour, repo_id, sources, takedown_url or "")
    return "\n".join(yaml + body)


def _supervises(column: str, value: object) -> bool:
    if value is None:
        return False
    if column == "mask_class_map":
        return any(v is not None for v in json.loads(str(value)).values())
    return True


def unsupervised_configs(out, summary: dict) -> tuple[str, ...]:
    """Task configs where no row carries a non-null ``label`` or mask-map target."""
    import pyarrow.parquet as pq

    empty = []
    for config in summary["configs"]:
        paths = sorted((out / "data" / config).glob("*.parquet"))
        names = set(pq.read_schema(paths[0]).names) if paths else set()
        cols = [c for c in ("label", "mask_class_map") if c in names]
        if cols and not any(
            _supervises(c, v)
            for p in paths
            for c in cols
            for v in pq.read_table(p, columns=[c]).column(0).to_pylist()
        ):
            empty.append(config)
    return tuple(empty)


def metadata_config_summary(out) -> dict | None:
    """A ``summary["configs"]["metadata"]``-shaped entry read back from the shards WP-2's
    ``metadata_release`` already wrote — so the card's existing config/split rendering
    (``_yaml_configs``, ``_split_table``) documents it with no per-config special-casing."""
    import pyarrow.parquet as pq

    paths = sorted((out / "data" / METADATA).glob("*.parquet"))
    if not paths:
        return None
    splits: dict[str, dict] = {}
    for path in paths:
        split = path.name.split("-", 1)[0]
        n = pq.read_metadata(path).num_rows
        entry = splits.setdefault(split, {"rows": 0, "shards": 0, "embedded_bytes": 0})
        entry["rows"] += n
        entry["shards"] += 1
    from .metadata_release import METADATA_COLUMNS

    return {"columns": [list(c) for c in METADATA_COLUMNS], "splits": splits, "written": []}


def metadata_licence_table(out) -> list[str]:
    """A licence table **computed from the `metadata` config's own rows** (not the
    registry's declared licence) — counts every ``license`` value actually shipped."""
    import pyarrow.parquet as pq

    paths = sorted((out / "data" / METADATA).glob("*.parquet"))
    counts: dict[str, int] = {}
    for path in paths:
        for value in pq.read_table(path, columns=["license"]).column(0).to_pylist():
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return []
    rows = ["| Licence (as shipped in `metadata`) | Rows |", "|---|---|"]
    rows += [f"| `{lic}` | {n} |" for lic, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows


def render_licence(sources: list[dict]) -> str:
    lines = [
        "This dataset combines sources under different licences. Each image and label keeps",
        "the licence of the source it came from (`source_id` column):",
        "",
    ]
    lines += [
        f"- {s['id']}: {s['licence']} — {s.get('licence_url') or 'no URL recorded'}"
        + (f" ({s['licence_note']})" if s.get("licence_note") else "")
        for s in sources
    ]
    return "\n".join(lines) + "\n"


def _attribution(source) -> str:
    from .metadata_release import attribution_for

    return attribution_for(source)


def source_rows(registry, release: dict, image_counts: dict[str, int]) -> list[dict]:
    """Card/licence rows for every release source, licence exactly as the registry states."""
    rows = []
    for entry in release["sources"]:
        source = registry.source(entry["id"])
        licence = source.licence
        rows.append(
            {
                "id": entry["id"],
                "version": entry["version"],
                "licence": licence.id,
                "licence_hf": licence.id.lower() if licence.id.startswith("CC-") else "other",
                "licence_url": licence.url,
                "licence_note": licence.notes,
                "tier": licence.tier.value,
                "citation": source.citation or source.homepage,
                "attribution": _attribution(source),
                "images": image_counts.get(entry["id"], 0),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write ``README.md`` + ``LICENSE`` into an exported folder from its summary."""
    import argparse
    from pathlib import Path

    import pyarrow.parquet as pq

    from .registry import Registry

    parser = argparse.ArgumentParser(prog="python -m marinedata.hf_card")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretty-name", default="Reef Support coral-reef imagery v1")
    parser.add_argument("--repo-id", default=None, help="default: the flavour's repo id")
    parser.add_argument(
        "--flavour", choices=("open", "nc"), default=None, help="default: RELEASE.json's flavour"
    )
    parser.add_argument(
        "--exclude-configs",
        default=",".join(DEFAULT_EXCLUDE_CONFIGS),
        help="comma-separated configs dropped from the export (D-A); '' for none",
    )
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    release = json.loads((args.release_dir / "RELEASE.json").read_text())
    flavour = args.flavour or release.get("flavour")
    if flavour and release.get("flavour") not in (None, flavour):
        parser.error(f"--flavour {flavour} but the release was built as {release['flavour']}")
    repo_id = args.repo_id or release.get("repo_id") or DEFAULT_REPO_ID
    counts: dict[str, int] = {}
    for path in sorted((args.out / "data" / IMAGES).glob("*.parquet")):
        for ids in pq.read_table(path, columns=["source_ids"]).column(0).to_pylist():
            for sid in ids.split(","):
                counts[sid] = counts.get(sid, 0) + 1
    sources = source_rows(Registry.load(), release, counts)
    empty = unsupervised_configs(args.out, summary)
    exclude = [c for c in args.exclude_configs.split(",") if c]
    excluded = {c: EXCLUDE_REASONS.get(c, "excluded from this release") for c in exclude}
    metadata_entry = metadata_config_summary(args.out)
    if metadata_entry is not None:
        summary["configs"][METADATA] = metadata_entry
    licence_rows = metadata_licence_table(args.out)
    (args.out / "README.md").write_text(
        render_card(
            summary,
            release,
            sources,
            pretty_name=args.pretty_name,
            repo_id=repo_id,
            flavour=flavour,
            takedown_url=release.get("takedown_url"),
            unsupervised=empty,
            excluded=excluded,
            metadata_licence_rows=licence_rows,
        )
    )
    (args.out / "LICENSE").write_text(render_licence(sources))
    print(json.dumps({"images_per_source": counts, "unsupervised": empty}, sort_keys=True))
    return 0


def task_layer_config_table(results: dict) -> str:
    """Markdown table for the 5 WP-8c v2 task-layer configs (per config: images,
    annotation rows, class count where the config has a fixed vocabulary, and human vs
    model row counts) — appended to the card when a ``v2`` release built any of them.
    ``results`` is ``{config_id: marinedata.task_layers.configs.ConfigResult}``.
    """
    classes_by_config = {
        "benthic-coarse": 6,  # HC, MIL, SC, ALGAE, ABIOTIC, OTHER_FAUNA (D-Z)
        "benthic-cover": 6,
    }
    lines = [
        "| config | images | annotation rows | classes | human rows | model rows |",
        "|---|---|---|---|---|---|",
    ]
    for config_id in sorted(results):
        result = results[config_id]
        human = sum(1 for row in result.rows if row.get("label_origin") == "human")
        model = sum(1 for row in result.rows if row.get("label_origin") == "model")
        classes = classes_by_config.get(config_id, "-")
        lines.append(
            f"| {config_id} | {result.n_images} | {len(result.rows)} | {classes} | "
            f"{human} | {model} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
