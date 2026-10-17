"""v1.1.0 dataset card and root files (PR-E, E3/E4): ``README.md``, ``LICENSE``, ``NOTICE`` and
``CITATION.cff`` of a ``hf_export_v11`` export folder.

The card follows the Hub order of design section 4: summary, supported tasks, quick start, dataset
structure, label space, splits, dataset creation, personal and sensitive information, limitations,
licensing, excluded sources, takedown, versions and changelog, citation, contact.

Front matter: the v1.0 ``configs:`` entries are carried verbatim (:func:`v1_entries`), the new
configs follow with ``metadata`` as the default, and ``dataset_info`` is written for the new configs
only (a ``features`` block on a frozen v1.0 config would change how ``datasets`` casts its rows).
Every text goes through :func:`hf_export_v11.assert_public_safe` before it is written: the open
files never name the private repo or a bucket, the nc files never name a bucket.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

import yaml

from . import card_body as cb
from . import card_v11_text as txt
from .hf_card_flavour import EXCLUDED_CLASSES, NC_CHECKBOX, NC_GATE_PROMPT, top_notice
from .hf_export_v11 import V11ExportError, assert_public_safe
from .release_card import (
    BASE_TAGS,
    CONTACT,
    DASH,
    DEFAULT_REPO,
    GITHUB,
    display_licence,
    facts_from_build,
    hf_dataset_id,
    size_category,
    source_info,
)
from .release_provenance import render_citation_cff, render_license, render_notice

PRETTY_NAME = "Reef Support Marine Data"
REPO_CFF = Path(__file__).resolve().parents[2] / "CITATION.cff"
V1_SPLITS = ("train", "validation", "test")
V1_CONFIGS = {
    "open": ("coral-masks", "scene-masks", "instance-masks", "fish-boxes"),
    "nc": ("coral-masks", "coral-masks-machine"),
}
"""The frozen v1.0 configs of each flavour (their ``configs:`` entries are never rewritten)."""
SPLIT_COLUMNS = ("train", "validation", "test", "test_ood")
TASK_CATEGORIES = ("image-segmentation", "object-detection", "image-classification")
TASK_IDS = ("semantic-segmentation", "instance-segmentation", "multi-class-image-classification")
TAGS = (
    *BASE_TAGS, "coral", "benthic", "fish", "bleaching", "semantic-segmentation",
    "instance-segmentation", "object-detection", "image-classification", "taxonomy",
    "reef-support",
)  # fmt: skip
TAGLINE = (
    "Underwater imagery with masks, boxes, points and image labels for coral-reef computer "
    "vision, harmonised into one label space, with the licence and attribution of every row "
    "recorded."
)
MAPPED_FLOOR = 95.0
MAX_OVERLAP_ROWS = 40
MAX_COLUMNS_SHOWN = 14


# ------------------------------------------------------------------------------ front matter


def v1_entries(flavour: str) -> list[dict]:
    """The v1.0 ``configs:`` entries of ``flavour``, exactly as the Hub README carries them."""
    return [
        {
            "config_name": name,
            "data_files": [{"split": s, "path": f"data/{name}/{s}-*.parquet"} for s in V1_SPLITS],
        }
        for name in V1_CONFIGS[flavour]
    ]


def ordered_configs(configs: Iterable[str]) -> list[str]:
    names = list(configs)
    known = [c for c in txt.V11_ORDER if c in names]
    return [*known, *sorted(set(names) - set(known))]


def new_entries(configs: Mapping[str, dict]) -> list[dict]:
    out = []
    for name in ordered_configs(configs):
        entry: dict = {"config_name": name}
        if name == "metadata":
            entry["default"] = True
        entry["data_files"] = [
            {"split": s, "path": f"data/{name}/{e.get('prefix', s)}-*.parquet"}
            for s, e in configs[name]["splits"].items()
        ]
        out.append(entry)
    return out


def front_matter_dict(
    facts: dict, infos: Mapping[str, dict], *, flavour: str, repo_id: str, pretty_name: str
) -> dict:
    configs = facts["configs"]
    images = sum(e["num_examples"] for e in configs.get("metadata", {}).get("splits", {}).values())
    hub = list(dict.fromkeys(filter(None, (hf_dataset_id(infos[sid]) for sid in sorted(infos)))))
    front: dict = {
        "pretty_name": pretty_name,
        "license": "other",
        "license_name": "mixed-open" if flavour == "open" else "mixed-non-commercial",
        "license_link": "LICENSE",
    }
    if flavour != "open":
        front["extra_gated_prompt"] = NC_GATE_PROMPT
        front["extra_gated_fields"] = {
            "Name": "text", "Affiliation": "text", "Intended use": "text", NC_CHECKBOX: "checkbox",
        }  # fmt: skip
    front |= {
        "language": ["en"],
        "task_categories": list(TASK_CATEGORIES),
        "task_ids": list(TASK_IDS),
        "annotations_creators": ["expert-generated", "found"],
        "source_datasets": ["original", *(f"extended|{h}" for h in hub)],
        "size_categories": [size_category(images)],
        "tags": list(TAGS),
        "configs": [*v1_entries(flavour), *new_entries(configs)],
        "dataset_info": [
            {
                "config_name": name,
                "features": configs[name]["features"],
                "splits": [
                    {"name": s, "num_bytes": e["num_bytes"], "num_examples": e["num_examples"]}
                    for s, e in configs[name]["splits"].items()
                ],
                "download_size": configs[name]["download_size"],
                "dataset_size": sum(e["num_bytes"] for e in configs[name]["splits"].values()),
            }
            for name in ordered_configs(configs)
        ],
    }  # fmt: skip
    return front


def front_matter(*args, **kwargs) -> str:
    return yaml.safe_dump(
        front_matter_dict(*args, **kwargs),
        sort_keys=False, allow_unicode=True, default_flow_style=False, width=10_000,
    ).rstrip("\n")  # fmt: skip


# ------------------------------------------------------------------------------ tables


def _c(value: object) -> str:
    text = DASH if value in (None, "") else str(value)
    return " ".join(text.split()).replace("|", "/")


def _table(header: tuple[str, ...], rows: Iterable[Iterable[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_c(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def read_rows(export: Path, config: str) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted((Path(export) / "data" / config).glob("*.parquet"))
    return [r for f in files for r in pq.read_table(f).to_pylist()]


def tasks_table(configs: Mapping[str, dict]) -> str:
    rows = []
    for name in ordered_configs(configs):
        task, note = txt.CONFIG_TASKS.get(name, (name, ""))
        cfg = configs[name]
        rows.append(
            (
                f"`{name}`",
                task,
                f"{sum(e['num_examples'] for e in cfg['splits'].values()):,}",
                ", ".join(f"`{s}`" for s in cfg.get("sources", {})),
                note,
            )
        )
    return _table(("Config", "Task", "Rows", "Sources", "Notes"), rows)


def splits_table(configs: Mapping[str, dict]) -> str:
    rows = []
    for name in ordered_configs(configs):
        splits = configs[name]["splits"]
        total = sum(e["num_examples"] for e in splits.values())
        if total:
            cells = [
                f"{splits[s]['num_examples']:,} ({100 * splits[s]['num_examples'] / total:.1f} %)"
                if s in splits
                else DASH
                for s in SPLIT_COLUMNS
            ]
            rows.append((f"`{name}`", *cells, f"{total:,}"))
    return _table(("Config", *SPLIT_COLUMNS, "Total"), rows)


def overlap_table(label_rows: Iterable[Mapping]) -> str:
    """Taxonomy nodes that more than one source of this repo reaches (the cross-source classes)."""
    by_node: dict[str, set[str]] = defaultdict(set)
    for r in label_rows:
        if r.get("mapped") and r.get("taxon_node_id"):
            by_node[r["taxon_node_id"]].add(r["source_id"])
    multi = sorted(
        ((n, s) for n, s in by_node.items() if len(s) > 1), key=lambda kv: (-len(kv[1]), kv[0])
    )
    rows = [
        (f"`{n}`", len(s), ", ".join(f"`{x}`" for x in sorted(s)))
        for n, s in multi[:MAX_OVERLAP_ROWS]
    ] or [(DASH, 0, "no taxonomy node is reached by more than one source")]
    note = f"{len(by_node) - len(multi):,} node(s) are reached by a single source."
    return _table(("Taxonomy node", "Sources", "Reached by"), rows) + "\n\n" + note


def mapped_table(sources: Iterable[Mapping]) -> str:
    rows = [
        (
            f"`{s['source_id']}`",
            f"{s.get('n_labels') or 0:,}",
            DASH if s.get("mapped_pct") is None else f"{s['mapped_pct']:.1f} %",
        )
        for s in sorted(sources, key=lambda s: s["source_id"])
    ]
    return _table(("Source", "Labels", "Mapped to the taxonomy"), rows)


def sources_table(sources: Iterable[Mapping]) -> str:
    rows = [
        (
            f"`{s['source_id']}`",
            s.get("version"),
            s.get("rights_holder"),
            " ".join(filter(None, [s.get("verified_by"), s.get("verified_on")])),
            s.get("provenance_url"),
            f"{s.get('n_images') or 0:,}",
            s.get("configs"),
        )
        for s in sorted(sources, key=lambda s: s["source_id"])
    ]
    return _table(
        ("Source", "Version", "Rights holder", "Verified", "Provenance", "Images", "Configs"), rows
    )


def licence_matrix(sources: Iterable[Mapping]) -> str:
    def lic(s: Mapping) -> str:
        label = display_licence(s.get("licence")) if s.get("licence") else DASH
        return f"[{label}]({s['licence_url']})" if s.get("licence_url") else label

    rows = [
        (
            f"`{s['source_id']}`",
            lic(s),
            s.get("ann_licence"),
            s.get("attribution"),
            s.get("licence_class"),
        )
        for s in sorted(sources, key=lambda s: s["source_id"])
    ]
    return _table(("Source", "Image licence", "Annotation licence", "Attribution", "Class"), rows)


def structure_lines(configs: Mapping[str, dict]) -> str:
    lines = []
    for name in ordered_configs(configs):
        cols = [f["name"] for f in configs[name]["features"]]
        shown = ", ".join(f"`{c}`" for c in cols[:MAX_COLUMNS_SHOWN])
        more = (
            f", and {len(cols) - MAX_COLUMNS_SHOWN} more" if len(cols) > MAX_COLUMNS_SHOWN else ""
        )
        lines.append(f"- `{name}` ({len(cols)} columns): {shown}{more}.")
    return "\n".join(lines)


def quick_start(repo_id: str, configs: Mapping[str, dict]) -> str:
    code = ["from datasets import load_dataset", ""]
    code += [f'meta = load_dataset("{repo_id}", "metadata", split="train")  # default, no pixels']
    if "images" in configs:
        code += [f'images = load_dataset("{repo_id}", "images", split="train", streaming=True)']
    sql = []
    if "points" in configs:
        sql += [
            "import duckdb",
            "",
            'duckdb.sql("""',
            "    SELECT p.image_sha256, p.label_native, m.bytes_config",
            f"    FROM 'hf://datasets/{repo_id}/data/points/*.parquet' p",
            f"    JOIN 'hf://datasets/{repo_id}/data/metadata/*.parquet' m USING (image_sha256)",
            "    LIMIT 5",
            '""").show()',
        ]
    if "label-map" in configs:
        sql += [
            "",
            'duckdb.sql("""',
            f"    SELECT * FROM 'hf://datasets/{repo_id}/data/label-map/*.parquet'",
            "    WHERE match_type = 'unmapped' LIMIT 5",
            '""").show()',
        ]
    text = "```python\n" + "\n".join(code) + "\n```"
    if sql:
        text += (
            "\n\nEvery label table joins to `metadata` and `images` on `image_sha256`; "
            "`metadata.bytes_config` names the config that holds the pixels.\n\n"
            "```python\n" + "\n".join(sql) + "\n```"
        )
    return text


# ------------------------------------------------------------------------------ the card


def excluded_section(flavour: str) -> str:
    """Every registered source that is not in the release, by reason: the licence classes, the
    named exclusions and, in the open flavour, the source of the non-commercial flavour."""
    named = txt.NOT_INCLUDED + (txt.NOT_INCLUDED_OPEN if flavour == "open" else ())
    lines = [f"- **{name}**: {why}" for name, why in (*EXCLUDED_CLASSES, *named)]
    return "## Excluded sources\n\n" + txt.EXCLUDED_INTRO + "\n\n" + "\n".join(lines)


def render_v11_card(
    facts: dict, infos: Mapping[str, dict], sources: list[dict], label_rows: list[dict], *,
    flavour: str, repo_id: str, pretty_name: str = PRETTY_NAME, cff: dict | None = None,
    takedown_url: str | None = None, lock_sha: str | None = None, built: str | None = None,
    hidden_test_withheld: int | None = None,
) -> str:  # fmt: skip
    def open_only(text: str, sep: str = " ") -> str:
        """The v1.1.0 open-run statements (chain-7 gates) belong to the open card only."""
        return sep + text if flavour == "open" else ""

    configs = facts["configs"]
    names = ordered_configs(configs)
    images = sum(e["num_examples"] for e in configs.get("metadata", {}).get("splits", {}).values())
    gb = sum(c["download_size"] for c in configs.values()) / 1e9
    low = [
        f"`{s['source_id']}` ({s['mapped_pct']:.1f} %)"
        for s in sorted(sources, key=lambda s: s["source_id"])
        if s.get("mapped_pct") is not None and s["mapped_pct"] < MAPPED_FLOOR
    ]
    limits = [f"- {t}" for t in txt.LIMITATIONS]
    if flavour == "open" and hidden_test_withheld is not None:
        limits.append("- " + txt.HIDDEN_TEST_WITHHELD.format(n=f"{hidden_test_withheld:,}"))
    if low:
        limits.append(f"- Sources below the {MAPPED_FLOOR:.0f} % mapped floor: {', '.join(low)}.")
    summary = (
        f"{pretty_name} v{txt.VERSION} adds {len(names)} configurations to the frozen v1.0 "
        "release, additively: image bytes once per `image_sha256`, per-image metadata, point, "
        "box, mask and bleaching labels, benthic cover and sites, and the tables that describe "
        "the label space (`taxonomy`, `label-map`) and the sources. Every row carries its "
        "`image_sha256`, so any label reaches its pixels in one join, and `licence` and "
        "attribution stay attached to the pixels."
    )
    facts_rows = [
        ("Configurations", f"{len(names)}: " + ", ".join(f"`{n}`" for n in names)),
        ("Images (`metadata` rows)", f"{images:,}"),
        ("Sources", f"{len(sources)}"),
        ("Format", "Parquet shards, streamable; images and new masks embedded once"),
        ("Size of the new configs", f"{gb:.1f} GB"),
        ("Version", f"v{txt.VERSION}" + (f", built {built}" if built else "")),
        ("Code", f"[{GITHUB.split('//')[1]}]({GITHUB})"),
    ]
    versions = _table(
        ("Version", "Date", "Notes"),
        [
            ("v1.0", txt.V1_DATE, txt.V1_NOTES[flavour]),
            (f"v{txt.VERSION}", built, f"Additive: {len(names)} new configurations."),
        ],
    )
    lock_line = (
        f"The release is pinned by `provenance/release.lock.yaml` (sha256 `{lock_sha}`)."
        if lock_sha
        else "The release is pinned by `provenance/release.lock.yaml`."
    )
    front = front_matter(facts, infos, flavour=flavour, repo_id=repo_id, pretty_name=pretty_name)
    pin = f'load_dataset("{repo_id}", "metadata", split="train", revision="v{txt.VERSION}")'
    sections = [
        f"---\n{front}\n---",
        f"# {pretty_name}\n\n{TAGLINE}",
        "\n".join(top_notice(flavour, repo_id)).strip(),
        "## Dataset summary\n\n" + summary + "\n\n" + _table(("Field", "Value"), facts_rows),
        "## Supported tasks\n\n" + tasks_table(configs),
        "## Quick start\n\n" + quick_start(repo_id, configs),
        "## Dataset structure\n\n" + structure_lines(configs) + "\n\n"
        "Keys: the image key is `image_sha256` (64 lowercase hex, the staged checksum, never "
        "recomputed from re-encoded bytes); `metadata.bytes_sha256` is the hash of the embedded "
        "bytes. The annotation key is `image_sha256` + `source_id` + `source_version` + `ann_id`. "
        "`metadata.bytes_config` names the config that holds each image's pixels: a v1.0 config "
        "or `images`. The Hub splits are `train`, `validation`, `test` and `test_ood`.\n\n"
        + txt.MASKS,
        "## Label space\n\n" + txt.LABEL_SPACE + "\n\n### Label overlap\n\n"
        "Taxonomy nodes that more than one source reaches. Cross-source training should use "
        "these, or the coarse training views.\n\n" + overlap_table(label_rows)
        + "\n\n### Mapped share per source\n\n" + mapped_table(sources),
        "## Splits\n\n" + txt.SPLITS_TEXT + open_only(txt.SPLITS_OPEN, "\n\n") + "\n\n"
        + splits_table(configs),
        "## Dataset creation\n\n### Sources\n\n" + sources_table(sources)
        + "\n\n### Build and gates\n\nBuilt with the " + f"[marine-data]({GITHUB}) registry; "
        "every source is admitted only after its licence is recorded, and the licence gate "
        "raises, never warns. " + lock_line + " Gate results are published as "
        "`provenance/GATES.json`. " + txt.DEDUP_FAMILY + open_only(txt.DEDUP_OPEN),
        "## Personal and sensitive information\n\n" + txt.PERSONAL
        + open_only(txt.PERSONAL_OPEN),
        "## Limitations and known defects\n\n" + "\n".join(limits),
        "## Licensing\n\nThe data are not under one licence. Every row records its `licence`, "
        "`annotation_licence` and `attribution`; the matrix lists each source. Combine sources "
        "only in ways each licence allows. `LICENSE` and `NOTICE` list the same notices. The "
        "table records the licences as Reef Support read them from the primary sources and is "
        "not legal advice.\n\n" + licence_matrix(sources)
        + f"\n\nCredits: {txt.MEOW} {txt.WORMS}",
        excluded_section(flavour),
        "## Takedown\n\nRights holders and data subjects: open an issue at "
        f"{takedown_url or GITHUB + '/issues'}. Accepted requests are published as a new version.",
        "## Versions and changelog\n\n" + versions + "\n\n"
        + "\n".join(f"- {c}" for c in txt.CHANGELOG)
        + "\n\nFiles of a released version are not edited. Pin a version with `revision`:\n\n"
        f"```python\n{pin}\n```",
        "## Citation\n\n" + cb.citation_block(cff or {}, dict(infos), repo_id),
        f"## Contact\n\n{CONTACT}",
    ]  # fmt: skip
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# ------------------------------------------------------------------------------ files


def write_card_files(
    export: Path, registry, *, flavour: str, repo_id: str, pretty_name: str | None = None,
    cff: dict | None = None, takedown_url: str | None = None, lock_sha: str | None = None,
    released: str | None = None, hidden_test_withheld: int | None = None,
) -> list[str]:  # fmt: skip
    """Write ``README.md``, ``LICENSE``, ``NOTICE`` and ``CITATION.cff`` into ``export``; nothing is
    written when any of them names a private resource."""
    export = Path(export)
    name = pretty_name or (PRETTY_NAME if flavour == "open" else f"{PRETTY_NAME} (non-commercial)")
    facts, sources = facts_from_build(export), read_rows(export, "sources")
    if not facts["configs"] or not sources:
        raise V11ExportError(f"{export}: no v1.1.0 export (no data/ configs or no sources config)")
    infos = {s["source_id"]: source_info(registry, s["source_id"]) for s in sources}
    citation = render_citation_cff(cff or {}, version=txt.VERSION, released=released)
    files = {
        "README.md": render_v11_card(
            facts,
            infos,
            sources,
            read_rows(export, "label-map"),
            flavour=flavour,
            repo_id=repo_id,
            pretty_name=name,
            cff=yaml.safe_load(citation),
            takedown_url=takedown_url,
            lock_sha=lock_sha,
            built=released,
            hidden_test_withheld=hidden_test_withheld,
        ),
        "LICENSE": render_license(sources, pretty_name=name, flavour=flavour),
        "NOTICE": render_notice(sources, pretty_name=name),
        "CITATION.cff": citation,
    }
    for fname, text in files.items():
        assert_public_safe(text, flavour, where=fname)
    for fname, text in files.items():
        (export / fname).write_text(text)
    return list(files)


def run(args, summary: dict) -> int:
    """``hf_card`` entry for a v1.1.0 export (``summary['layout'] == 'v1.1'``)."""
    from .registry import Registry

    release_dir = Path(args.release_dir)
    release = json.loads((release_dir / "RELEASE.json").read_text())
    flavour = args.flavour or release.get("flavour")
    if flavour not in ("open", "nc") or release.get("flavour") not in (None, flavour):
        raise V11ExportError(
            f"--flavour {flavour} but the release was built as {release.get('flavour')!r}"
        )
    cff_path = Path(args.citation) if getattr(args, "citation", None) else REPO_CFF
    if not cff_path.is_file():
        raise V11ExportError(f"no CITATION.cff at {cff_path}; pass --citation <file>")
    lock = release_dir / "release.lock.yaml"
    written = write_card_files(
        args.out, Registry.load(), flavour=flavour,
        repo_id=args.repo_id or release.get("repo_id") or summary.get("repo_id") or DEFAULT_REPO,
        pretty_name=getattr(args, "pretty_name", None),
        cff=yaml.safe_load(cff_path.read_text()),
        takedown_url=release.get("takedown_url"),
        lock_sha=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        released=str(release.get("built") or release.get("created") or date.today())[:10],
        hidden_test_withheld=(
            ((summary.get("v11") or {}).get("hidden_test_excluded") or {}).get("images")
        ),
    )  # fmt: skip
    print(json.dumps({"layout": "v1.1", "written": written}))
    return 0
