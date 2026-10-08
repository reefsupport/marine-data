# ruff: noqa: E501
"""``marinedata release card``: the Hugging Face dataset card (README.md) of a build (PA-P1.9).

Everything numeric or per-config is read from the build (parquet footers and schema, so the
``Data fields`` follow the export layout and never a hand-typed list) or from the registry; the
prose lives in ``docs/card-template.md.tmpl`` (``string.Template`` slots: jinja2 is only reachable
through the optional ``baselines`` extra, so it is not a dependency of this command).

``facts`` is the normal form between the build and the renderer. ``facts_from_build`` produces it
from ``<build>/data/<config>/*.parquet``; a JSON file of the same shape reproduces a release whose
build is no longer on disk (tests/data/v1.0-card-facts.json is the published v1.0).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from string import Template

import yaml

from .card_fields import config_doc

SPLITS = ("train", "validation", "test")
SPLIT_ALIAS = {"val": "validation", "valid": "validation", "probe": "validation"}
SHORT = {"validation": "val"}
SOURCE_COLUMNS = ("source", "source_id", "source_ids")
DASH = "–"
DEFAULT_REPO = "reefsupport/marine-data"
GITHUB = "https://github.com/reefsupport/marine-data"
CONTACT = "hello@reef.support"
DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "docs" / "card-template.md.tmpl"
BASE_TAGS = ("marine", "coral-reef", "underwater", "biology", "ecology", "conservation")
SPECIFIC_CATEGORY_TAGS = frozenset({"object-detection"})
HF_LICENCES = frozenset(
    {"apache-2.0", "mit", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-nc-4.0", "cc0-1.0"}
)
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
BANDS = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
)
_CC = re.compile(r"^CC-(BY(?:-[A-Z]+)*)-(\d(?:\.\d)?)(.*)$")
_HF_ID = re.compile(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)")
PROVENANCE_FIELDS = (
    "access_class",
    "licence_evidence",
    "root_digest",
    "n_images",
    "checksums",
    "default_platform",
    "per_row_license",
    "rights_holder",
    "other_versions",
    "superseded_by",
)


def size_category(rows: int) -> str:
    """The Hub ``size_categories`` band of a row count (same bands as the card gate)."""
    return next((label for limit, label in BANDS if rows < limit), "n>10M")


def display_licence(licence_id: str | None) -> str:
    match = _CC.match(licence_id or "")
    return f"CC {match[1]} {match[2]}{match[3]}" if match else (licence_id or DASH)


def words(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def _cell(value: object) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_cell(v)}" for k, v in value.items())
    return str(value).replace("\n", " ").replace("|", "/").strip() or DASH


# ---------------------------------------------------------------------------------- facts


def _is_image(t) -> bool:
    import pyarrow.types as pt

    return pt.is_struct(t) and {f.name for f in t} == {"bytes", "path"}


def _scalar(t) -> str:
    import pyarrow.types as pt

    if pt.is_string(t) or pt.is_large_string(t):
        return "string"
    if pt.is_boolean(t):
        return "bool"
    return {"double": "float64", "float": "float32", "halffloat": "float16"}.get(str(t), str(t))


def _feature(name: str, t, hf: dict | None) -> dict:
    import pyarrow.types as pt

    hf = hf if isinstance(hf, dict) else {}  # a list column's entry is a one-element list
    if hf.get("_type") == "Image" or hf.get("dtype") == "image" or _is_image(t):
        return {"name": name, "dtype": "image"}
    if pt.is_list(t) or pt.is_large_list(t):
        inner = t.value_type
        if pt.is_struct(inner):
            return {"name": name, "list": [_feature(f.name, f.type, None) for f in inner]}
        return {"name": name, "list": _scalar(inner)}
    if pt.is_struct(t):
        return {"name": name, "struct": [_feature(f.name, f.type, None) for f in t]}
    return {"name": name, "dtype": _scalar(t)}


def arrow_features(schema) -> list[dict]:
    """The ``dataset_info.features`` list of a parquet schema (HF ``datasets`` YAML shape)."""
    hf: dict = {}
    raw = (schema.metadata or {}).get(b"huggingface")
    if raw:
        try:
            hf = json.loads(raw).get("info", {}).get("features", {}) or {}
        except ValueError:
            hf = {}
        if isinstance(hf, list):  # the `datasets` YAML shape: [{name, dtype}, ...]
            hf = {f["name"]: f for f in hf if isinstance(f, dict) and "name" in f}
    return [_feature(f.name, f.type, hf.get(f.name)) for f in schema]


def _ordered(names) -> list[str]:
    return [s for s in SPLITS if s in names] + sorted(n for n in names if n not in SPLITS)


def facts_from_build(build_dir: Path) -> dict:
    """Configs, features, split counts, shard bytes and rows per source of ``<build>/data``."""
    import pyarrow.parquet as pq

    data = Path(build_dir) / "data"
    configs: dict[str, dict] = {}
    for folder in sorted(p for p in data.iterdir() if p.is_dir()) if data.is_dir() else ():
        files = sorted(folder.glob("*.parquet"))
        if not files:
            continue
        features, splits, counts = None, {}, {}
        for path in files:
            prefix = path.name.split("-")[0]
            split = SPLIT_ALIAS.get(prefix, prefix)
            shard = pq.ParquetFile(path)
            names = shard.schema_arrow.names
            features = features or arrow_features(shard.schema_arrow)
            entry = splits.setdefault(split, {"num_examples": 0, "num_bytes": 0, "prefix": prefix})
            entry["num_examples"] += shard.metadata.num_rows
            entry["num_bytes"] += path.stat().st_size
            column = next((c for c in SOURCE_COLUMNS if c in names), None)
            for value in shard.read(columns=[column]).column(0).to_pylist() if column else ():
                for sid in str(value).split(","):
                    if sid:
                        counts.setdefault(sid, Counter())[split] += 1
        configs[folder.name] = {
            "features": features,
            "splits": {s: splits[s] for s in _ordered(splits)},
            "download_size": sum(e["num_bytes"] for e in splits.values()),
            "sources": {
                sid: {"rows": sum(c.values()), "splits": {s: c[s] for s in _ordered(c)}}
                for sid, c in sorted(counts.items())
            },
        }
    return {"configs": configs}


def _read_json(path: Path | None) -> dict:
    return json.loads(path.read_text()) if path and path.is_file() else {}


def _first_file(folders: list[Path | None], *names: str) -> Path | None:
    for folder in folders:
        for name in names:
            if folder and (Path(folder) / name).is_file():
                return Path(folder) / name
    return None


# ----------------------------------------------------------------------------- registry


def source_info(registry, sid: str) -> dict:
    """The registry entry of ``sid`` as a dict (``{"id": sid}`` alone when it has none)."""
    try:
        source = registry.source(sid)
    except Exception:
        return {"id": sid}
    info = source.model_dump(mode="json")
    try:
        from .metadata_release import attribution_for

        info["_attribution"] = attribution_for(source)
    except Exception:
        info["_attribution"] = None
    return info


def hf_dataset_id(info: dict, override: str | None = None) -> str | None:
    if override:
        return override
    access = info.get("access") or {}
    for text in (info.get("homepage"), access.get("uri"), access.get("url"), info.get("citation")):
        found = _HF_ID.search(str(text or ""))
        if found:
            return found[1]
    return None


def _licence(info: dict) -> dict:
    return info.get("licence") or {}


# ---------------------------------------------------------------------------- front matter


def front_matter_dict(
    facts: dict, infos: dict[str, dict], *, flavour: str, repo_id: str, pretty_name: str
) -> dict:
    configs = facts["configs"]
    docs = [config_doc(c) for c in configs]

    def union(attr: str) -> list[str]:
        return list(dict.fromkeys(x for d in docs for x in getattr(d, attr)))

    own = tuple(facts.get("own_prefixes", ("reef-support",)))
    hf_ids = facts.get("source_datasets_hf", {})
    datasets = ["original"] if any(s.startswith(own) for s in infos) else []
    for sid in sorted(infos):
        hf = hf_dataset_id(infos[sid], hf_ids.get(sid))
        if hf:
            datasets.append(f"extended|{hf}")
    licences = {_licence(i).get("id") for i in infos.values() if _licence(i)}
    one = next(iter(licences)).lower() if len(licences) == 1 else None
    total = sum(e["num_examples"] for c in configs.values() for e in c["splits"].values())
    task_tags = [*union("ids"), *(c for c in union("categories") if c in SPECIFIC_CATEGORY_TAGS)]
    front: dict = {"pretty_name": pretty_name}
    if one in HF_LICENCES:
        front["license"] = one
    else:
        front |= {
            "license": "other",
            "license_name": f"per-source-{'open' if flavour == 'open' else flavour}-licences",
            "license_link": f"https://huggingface.co/datasets/{repo_id}#licensing",
        }
    front |= {
        "language": list(facts.get("language", ["en"])),
        "task_categories": union("categories"),
        "task_ids": union("ids"),
        "annotations_creators": list(
            facts.get("annotations_creators", ["expert-generated", "found"])
        ),
        "source_datasets": datasets,
        "size_categories": [size_category(total)],
        "tags": [*BASE_TAGS, *union("tags"), *task_tags, "reef-support"],
        "configs": [
            {
                "config_name": name,
                "data_files": [
                    {"split": s, "path": f"data/{name}/{e.get('prefix', s)}-*.parquet"}
                    for s, e in cfg["splits"].items()
                ],
            }
            for name, cfg in configs.items()
        ],
        "dataset_info": [
            {
                "config_name": name,
                "features": cfg["features"],
                "splits": [
                    {"name": s, "num_bytes": e["num_bytes"], "num_examples": e["num_examples"]}
                    for s, e in cfg["splits"].items()
                ],
                "download_size": cfg["download_size"],
                "dataset_size": sum(e["num_bytes"] for e in cfg["splits"].values()),
            }
            for name, cfg in configs.items()
        ],
    }
    return front


def front_matter(*args, **kwargs) -> str:
    """The YAML front matter (without the ``---`` fences), as ``huggingface_hub`` writes it."""
    return yaml.safe_dump(
        front_matter_dict(*args, **kwargs),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=10_000,
    ).rstrip("\n")


# ------------------------------------------------------------------------------------- body

TAGLINE = (
    "Underwater imagery with segmentation masks and bounding boxes for coral-reef computer "
    "vision, with the licence and attribution of every row recorded."
)
LICENSING_INTRO = (
    "The data are not under one licence. Every row records its own `licence`, "
    "`annotation_licence` and `attribution`; the table lists each source with the licence its "
    "authors chose and the attribution it requires. Combine sources only in ways each licence "
    "allows. The table records the licences as Reef Support read them from the primary sources "
    "and is not legal advice."
)
PIN = (
    "Pin a version with `revision`:\n\n```python\nfrom datasets import load_dataset\n\n"
    'ds = load_dataset("{repo_id}", "{config}", split="train", revision="{rid}")\n```'
)


def _release_id(release: dict, release_dir: Path | None) -> str | None:
    for key in ("release_id", "id", "release", "name", "version"):
        if isinstance(release.get(key), str) and release[key]:
            return release[key]
    if release_dir and Path(release_dir).parent.parent.name == "releases":
        return Path(release_dir).parent.name
    return None


def _task_map(configs: dict) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in configs:
        doc = config_doc(name)
        for task in doc.ids or doc.categories:
            found.setdefault(task.replace("-", " "), []).append(name)
    return found


def _demote(markdown: str, by: int = 2) -> str:
    """Push the headings of an embedded document ``by`` levels down (code fences untouched)."""
    out, fence = [], False
    for line in markdown.splitlines():
        fence ^= line.startswith("```")
        if not fence and re.match(r"#{1,6} ", line):
            line = "#" * min(6, len(line) - len(line.lstrip("#")) + by) + line.lstrip("#")
        out.append(line)
    return "\n".join(out)


def render_card(
    facts: dict,
    infos: dict[str, dict],
    *,
    flavour: str = "open",
    repo_id: str = DEFAULT_REPO,
    pretty_name: str | None = None,
    gates: dict | None = None,
    release_id: str | None = None,
    lock_sha: str | None = None,
    changelog: str | None = None,
    cff: dict | None = None,
    template: Path | None = None,
) -> str:
    """The dataset card (README.md) of a build: ``facts`` + registry ``infos`` -> text."""
    from urllib.parse import quote

    from . import card_body as cb

    configs, rel = facts["configs"], facts.get("release") or {}
    rid = release_id or rel.get("id")
    label = rel.get("label") or rid or "unreleased build"
    built = rel.get("built")
    name = pretty_name or facts.get("pretty_name") or "Reef Support Marine Data"
    overrides = facts.get("licence_display") or {}
    shown = {sid: {**i, "_licence_display": overrides.get(sid)} for sid, i in infos.items()}
    splits = cb._split_columns(configs)
    totals = {
        s: sum(c["splits"].get(s, {}).get("num_examples", 0) for c in configs.values())
        for s in splits
    }
    rows = sum(totals.values())
    task_map = _task_map(configs)
    licences = list(
        dict.fromkeys(display_licence(_licence(i).get("id")) for i in infos.values() if _licence(i))
    )
    fields, missing = cb.data_fields(configs)
    if missing:
        print(f"card: no description for {', '.join(dict.fromkeys(missing))}", file=sys.stderr)
    date = (
        f"{rel['date_label']} (built {built})"
        if rel.get("date_label") and built
        else (built or rel.get("date_label") or DASH)
    )
    versions = "\n".join(
        [
            "| Version | Date | Notes |",
            "|---|---|---|",
            f"| {rid or DASH} | {date} | {rel.get('note') or DASH} |",
        ]
    )
    tail = [
        "Files of a released version are not edited. Corrections and removals, including accepted "
        "takedown requests, are published as a new version, and the changelog lists them."
    ]
    if rid:
        tail.append(PIN.format(repo_id=repo_id, config=next(iter(configs)), rid=rid))
    if changelog and changelog.strip():
        tail.append(_demote(changelog.strip()))
    mapping = {
        "front_matter": front_matter(
            facts, infos, flavour=flavour, repo_id=repo_id, pretty_name=name
        ),
        "pretty_name": name,
        "tagline": facts.get("tagline") or TAGLINE,
        "github": GITHUB,
        "contact": CONTACT,
        "repo_id": repo_id,
        "rows": f"{rows:,}",
        "rows_badge": quote(f"{rows:,}", safe=""),
        "summary": (
            f"{name} {label} collects {words(len(infos))} source{'' if len(infos) == 1 else 's'} "
            f"into {words(len(configs))} configuration{'' if len(configs) == 1 else 's'}. Every "
            "row carries its `licence` and `attribution`, so provenance stays attached to the "
            f"pixels. The release is built with the [marine-data]({GITHUB}) registry, which "
            "records the licence of every source before it is admitted.\n\n"
            f"The licences differ per source ({', '.join(licences) or DASH}), so read "
            "[Licensing](#licensing) before you share or publish results."
        ),
        "facts_table": "\n".join(
            [
                "| | |",
                "|---|---|",
                f"| Configurations | {len(configs)}: {', '.join(f'`{c}`' for c in configs)} |",
                f"| Rows | {rows:,} ({', '.join(f'{s} {totals[s]:,}' for s in splits)}) |",
                f"| Tasks | {', '.join(task_map) or DASH} |",
                "| Format | Parquet shards with images and masks embedded, streamable |",
                f"| Size | {sum(c['download_size'] for c in configs.values()) / 1e9:.1f} GB |",
                f"| Sources | {len(infos)}: {', '.join(i.get('name') or s for s, i in infos.items())} |",
                f"| Version | {label}{f', built {built}' if built else ''} |",
                f"| Code | [{GITHUB.split('//')[1]}]({GITHUB}) |",
            ]
        ),
        "tasks": "\n".join(
            f"- {t.capitalize()}: {', '.join(f'`{n}`' for n in names)}."
            for t, names in task_map.items()
        ),
        "quick_start": cb.quick_start(configs, repo_id),
        "configurations_table": cb.configurations_table(configs),
        "masks_note": cb.masks_note(configs, repo_id),
        "config_sections": cb.config_sections(configs, shown),
        "data_fields": fields,
        "sources_table": cb.provenance_table(
            configs, shown, {s: i.get("version") for s, i in infos.items()}
        ),
        "gates_table": cb.gates_table(gates or {}),
        "lock_line": (
            f"Pinned by `release.lock.yaml` (sha256 `{lock_sha}`)."
            if lock_sha
            else "No `release.lock.yaml` accompanied this build."
        ),
        "licensing_intro": LICENSING_INTRO,
        "licensing_table": cb.licensing_table(configs, shown),
        "versions": versions + "\n\n" + "\n\n".join(tail),
        "citation": cb.citation_block(cff or {}, infos, repo_id),
    }
    text = Template(Path(template or DEFAULT_TEMPLATE).read_text()).substitute(mapping)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


def find_release_dir(build_dir: Path) -> Path | None:
    root = Path(build_dir) / "releases"
    found = sorted(root.glob("*/*/RELEASE.json")) + sorted(root.glob("*/RELEASE.json"))
    return found[-1].parent if found else None


def load_context(
    build_dir: Path | None,
    *,
    facts_path: Path | None = None,
    release_dir: Path | None = None,
    citation: Path | None = None,
    repo_id: str | None = None,
    flavour: str | None = None,
) -> dict:
    """Everything the card and Croissant renderers read: facts, registry infos, release files."""
    from .registry import Registry

    build = Path(build_dir) if build_dir else None
    facts = json.loads(Path(facts_path).read_text()) if facts_path else facts_from_build(build)
    rdir = Path(release_dir) if release_dir else (find_release_dir(build) if build else None)
    release = _read_json(rdir / "RELEASE.json" if rdir else None)
    lock = _first_file([rdir, build], "release.lock.yaml")
    log = _first_file([rdir, rdir.parent if rdir else None], "CHANGELOG.md")
    cff_path = Path(citation or "CITATION.cff")
    registry = Registry.load()
    sids = dict.fromkeys(s for c in facts["configs"].values() for s in c.get("sources", {}))
    return {
        "facts": facts,
        "infos": {sid: source_info(registry, sid) for sid in sids},
        "release": release,
        "release_id": _release_id(release, rdir),
        "gates": _read_json(_first_file([rdir, build], "GATES.json")),
        "lock_sha": hashlib.sha256(lock.read_bytes()).hexdigest() if lock else None,
        "changelog": log.read_text() if log else None,
        "cff": yaml.safe_load(cff_path.read_text()) if cff_path.is_file() else {},
        "repo_id": repo_id or DEFAULT_REPO,
        "flavour": flavour or (rdir.name if rdir and rdir.name in ("open", "nc") else "open"),
    }


def run_card(args) -> int:
    ctx = load_context(
        args.build_dir,
        facts_path=args.facts,
        release_dir=args.release_dir,
        citation=args.citation,
        repo_id=args.repo_id,
        flavour=args.flavour,
    )
    text = render_card(
        ctx["facts"],
        ctx["infos"],
        flavour=ctx["flavour"],
        repo_id=ctx["repo_id"],
        pretty_name=args.pretty_name,
        gates=ctx["gates"],
        release_id=ctx["release_id"],
        lock_sha=ctx["lock_sha"],
        changelog=ctx["changelog"],
        cff=ctx["cff"],
        template=args.template,
    )
    out = args.out or (Path(args.build_dir or ".") / "README.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0
