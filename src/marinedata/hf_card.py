"""Dataset card (``README.md``) and ``LICENSE`` for the Hub export (WS-D S55).

Licences are recorded exactly as the registry states them (Yohan 2026-09-25: licence
concerns are out of scope for this prep — record, do not gate). The YAML ``configs:`` block
is generated from the same export summary that named the shards, so the card can never
point at a file the export did not plan.
"""

from __future__ import annotations

import json

from .hf_export import IMAGES, MASKS, PSEUDO_MASKS, SPLIT_ORDER

CONFIG_BLURB = {
    IMAGES: "every image once (pixels embedded), keyed by `image_sha256`",
    MASKS: "ground-truth dense masks, one row per (image, source)",
    PSEUDO_MASKS: "CoralSCOP **model output** masks — weak supervision only, never ground truth",
    "general-pretraining": "pretraining membership (train + validation; test excluded)",
}


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
    return rows


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
    repo_id: str = "<repo>",
    unsupervised: tuple[str, ...] = (),
) -> str:
    near = release["near_dup"]
    empty_note = [
        f"- `{c}` has **no supervision in this release**: every `label` and every "
        "`mask_class_map` value is null. It is published for split membership only."
        for c in unsupervised
    ]
    n_images = sum(v["rows"] for v in summary["configs"][IMAGES]["splits"].values())
    ids = sorted({s["licence_hf"] for s in sources})
    yaml = [
        "---",
        f"pretty_name: {json.dumps(pretty_name)}",
        "license: other",
        "license_name: mixed-per-source",
        "license_link: LICENSE",
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
        "",
        "## Sources",
        "",
        f"Licence identifiers present: {', '.join(ids)}. Each source keeps its own licence; "
        "see `LICENSE`.",
        "",
        *_sources_table(sources),
        "",
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
        "",
    ]
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


def render_licence(sources: list[dict]) -> str:
    lines = [
        "This dataset combines sources under different licences. Each image and label keeps",
        "the licence of the source it came from (`source_id` column):",
        "",
    ]
    lines += [
        f"- {s['id']}: {s['licence']} — {s.get('licence_url') or 'no URL recorded'}"
        for s in sources
    ]
    return "\n".join(lines) + "\n"


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
                "tier": licence.tier.value,
                "citation": source.citation or source.homepage,
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
    parser.add_argument("--repo-id", default="<repo>")
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    release = json.loads((args.release_dir / "RELEASE.json").read_text())
    counts: dict[str, int] = {}
    for path in sorted((args.out / "data" / IMAGES).glob("*.parquet")):
        for ids in pq.read_table(path, columns=["source_ids"]).column(0).to_pylist():
            for sid in ids.split(","):
                counts[sid] = counts.get(sid, 0) + 1
    sources = source_rows(Registry.load(), release, counts)
    empty = unsupervised_configs(args.out, summary)
    (args.out / "README.md").write_text(
        render_card(
            summary,
            release,
            sources,
            pretty_name=args.pretty_name,
            repo_id=args.repo_id,
            unsupervised=empty,
        )
    )
    (args.out / "LICENSE").write_text(render_licence(sources))
    print(json.dumps({"images_per_source": counts, "unsupervised": empty}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
