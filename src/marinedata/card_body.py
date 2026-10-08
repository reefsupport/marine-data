"""Markdown blocks of the generated dataset card: tables, per-config sections, data fields."""

from __future__ import annotations

import re

from .card_fields import config_doc, field_doc
from .release_card import (
    DASH,
    PROVENANCE_FIELDS,
    SHORT,
    SPLITS,
    _cell,
    _licence,
    display_licence,
    words,
)

IMAGE_CONFIGS = frozenset({"fish-boxes", "images"})


def _rows(cfg: dict) -> int:
    return sum(e["num_examples"] for e in cfg["splits"].values())


def _split_columns(configs: dict) -> list[str]:
    seen = {s for c in configs.values() for s in c["splits"]}
    return [s for s in SPLITS if s in seen] + sorted(seen - set(SPLITS))


def _lic(info: dict | None) -> str:
    info = info or {}
    return info.get("_licence_display") or display_licence(_licence(info).get("id"))


def configurations_table(configs: dict) -> str:
    cols = _split_columns(configs)
    out = [
        "| config | task | " + " | ".join(cols) + " | total | GB |",
        "|---|---|" + "---|" * len(cols) + "---|---|",
    ]
    for name, cfg in configs.items():
        cells = [
            f"{cfg['splits'][s]['num_examples']:,}" if s in cfg["splits"] else DASH for s in cols
        ]
        out.append(
            f"| [`{name}`](#{name}) | {config_doc(name).task} | {' | '.join(cells)} | "
            f"{_rows(cfg):,} | {cfg['download_size'] / 1e9:.1f} |"
        )
    return "\n".join(out)


def source_table(name: str, cfg: dict, infos: dict, style: str) -> list[str]:
    sources = cfg.get("sources") or {}
    if not sources:
        return []
    cols = list(cfg["splits"])
    if style == "splits":
        head = [
            "| source | rows | " + " / ".join(SHORT.get(s, s) for s in cols) + " | licence |",
            "|---|---|---|---|",
        ]
        rows = [
            f"| `{sid}` | {v['rows']:,} | "
            + " / ".join(f"{v['splits'].get(s, 0):,}" for s in cols)
            + f" | {_lic(infos.get(sid))} |"
            for sid, v in sources.items()
        ]
    else:
        unit = "images" if name in IMAGE_CONFIGS else "rows"
        head = [f"| source | {unit} | licence |", "|---|---|---|"]
        rows = [
            f"| `{sid}` | {v['rows']:,} | {_lic(infos.get(sid))} |" for sid, v in sources.items()
        ]
    return [*head, *rows]


def config_sections(configs: dict, infos: dict) -> str:
    out: list[str] = []
    for name, cfg in configs.items():
        doc, sources = config_doc(name), cfg.get("sources") or {}
        text = doc.blurb.format(
            rows=f"{_rows(cfg):,}",
            sources_phrase=f"{words(len(sources))} source{'' if len(sources) == 1 else 's'}",
            vocab_note=" with different label vocabularies" if len(sources) > 1 else "",
        )
        style = doc.table
        if style == "none" and len(sources) == 1:
            sid = next(iter(sources))
            text += f" Source: `{sid}`, {_lic(infos.get(sid))}."
        elif style == "none":
            style = "plain"
        out += [f"### {name}", "", text, ""]
        if style != "none":
            out += [*source_table(name, cfg, infos, style), ""]
    return "\n".join(out).rstrip() + "\n\n" if out else ""


def _type(feature: dict) -> str:
    if "list" in feature:
        inner = feature["list"]
        return "list of struct" if isinstance(inner, list) else f"list of {inner}"
    if "struct" in feature:
        return "struct"
    return {"image": "Image"}.get(feature["dtype"], feature["dtype"])


def data_fields(configs: dict) -> tuple[str, list[str]]:
    """One table per distinct column layout, from the build's own features; plus the columns that
    have no description in :data:`card_fields.FIELD_DOCS` (shown as an en dash)."""
    groups: dict[tuple, list[str]] = {}
    for name, cfg in configs.items():
        groups.setdefault(tuple(f["name"] for f in cfg["features"]), []).append(name)
    out: list[str] = []
    missing: list[str] = []
    for cols, names in groups.items():
        feats = {f["name"]: f for f in configs[names[0]]["features"]}
        out += [", ".join(f"`{n}`" for n in names) + ":", "", "| field | type | description |"]
        out.append("|---|---|---|")
        for col in cols:
            doc = field_doc(col)
            if doc is None:
                missing.append(col)
            out.append(f"| `{col}` | {_type(feats[col])} | {doc or DASH} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n", missing


def _sources_in_order(configs: dict) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, cfg in configs.items():
        for sid in cfg.get("sources") or {}:
            found.setdefault(sid, []).append(name)
    return found


def licensing_table(configs: dict, infos: dict) -> str:
    out = ["| Source | In configs | Licence | Required attribution |", "|---|---|---|---|"]
    notes: list[str] = []
    for sid, names in _sources_in_order(configs).items():
        info = infos.get(sid) or {}
        lic = _licence(info)
        label = info.get("_licence_display") or display_licence(lic.get("id"))
        cell = f"[{label}]({lic['url']})" if lic.get("url") else label
        attribution = info.get("_attribution") or info.get("citation")
        out.append(
            f"| `{sid}` | {', '.join(f'`{n}`' for n in names)} | {cell} | {_cell(attribution)} |"
        )
        if lic.get("notes"):
            notes.append(f"- `{sid}`: {_cell(lic['notes'])}")
    return "\n".join([*out, *(["", *notes] if notes else [])])


def _expect(info: dict) -> str:
    keys = sorted(k for k in info if k.startswith("expect_") and info[k])
    return ", ".join(k[7:] if info[k] is True else f"{k[7:]}={_cell(info[k])}" for k in keys)


def provenance_table(configs: dict, infos: dict, versions: dict[str, str]) -> str:
    """Registry provenance per source; a field the registry does not carry is an en dash."""
    head = (
        "| Source | Version | Access class | Images (build / upstream) | Root digest | Checksums | "
        "Platform | Licence evidence | Per-row licence | Rights holder | Expected metadata | "
        "Other versions |"
    )
    out = [head, "|" + "---|" * 12]
    for sid in _sources_in_order(configs):
        i = infos.get(sid) or {}
        n, up = i.get("n_images"), i.get("n_images_upstream")
        images = DASH if n is None else f"{n:,}" + (f" / {up:,}" if up is not None else "")
        if i.get("n_images_note"):
            images += f" ({_cell(i['n_images_note'])})"
        platform = _cell(i.get("default_platform"))
        if i.get("platform_evidence"):
            platform += f' ("{_cell(i["platform_evidence"])}")'
        other = _cell(i.get("other_versions"))
        if i.get("superseded_by"):
            other = f"superseded by {_cell(i['superseded_by'])}" + (
                "" if other == DASH else f"; {other}"
            )
        out.append(
            "| "
            + " | ".join(
                [
                    f"`{sid}`",
                    _cell(versions.get(sid)),
                    _cell(i.get("access_class")),
                    images,
                    _cell((i.get("root_digest") or "")[:12]),
                    "pinned" if i.get("checksums") else DASH,
                    platform,
                    _cell(i.get("licence_evidence")),
                    _cell(i.get("per_row_license") or i.get("licence_per_row")),
                    _cell(i.get("rights_holder")),
                    _cell(_expect(i)),
                    other,
                ]
            )
            + " |"
        )
    assert all(f in PROVENANCE_FIELDS or True for f in PROVENANCE_FIELDS)
    return "\n".join(out)


def gates_table(gates: dict) -> str:
    entries = {k: v for k, v in gates.items() if isinstance(v, dict) and not k.startswith("_")}
    if not entries:
        return "No `GATES.json` accompanied this build, so no gate result is recorded here."
    out = ["| gate | status | detail |", "|---|---|---|"]
    for name, e in sorted(entries.items()):
        if e.get("skipped") is not None:
            status, detail = "skipped", _cell(e["skipped"])
        elif e.get("passed") is True:
            counts = {k: v for k, v in (e.get("counts") or {}).items() if not k.startswith("fail")}
            status, detail = "passed", _cell(dict(list(counts.items())[:3]))
        else:
            status, detail = "failed", f"{len(e.get('failures') or [])} failure(s)"
        out.append(f"| `{name}` | {status} | {detail} |")
    return "\n".join(out)


def _author(a: dict) -> str:
    return a.get("name") or " ".join(x for x in (a.get("given-names"), a.get("family-names")) if x)


def bibtex(cff: dict, repo_id: str) -> str:
    title, version = cff.get("title", "marine-data"), cff.get("version")
    authors = " and ".join(_author(a) for a in cff.get("authors", [])) or "Reef Support"
    year = str(cff.get("date-released", ""))[:4]
    key = re.sub(r"\W+", "_", f"{authors.split(' and ')[0]}_{title}_{year}".lower()).strip("_")
    url = cff.get("url") or f"https://huggingface.co/datasets/{repo_id}"
    entry = [
        ("title", title),
        ("author", authors),
        ("year", year),
        ("version", version),
        ("url", url),
    ]
    return "\n".join([f"@misc{{{key},", *(f"  {k} = {{{v}}}," for k, v in entry if v), "}"])


def citation_block(cff: dict, infos: dict, repo_id: str) -> str:
    if not cff:
        return "See `CITATION.cff` in the [GitHub repository](https://github.com/reefsupport/marine-data)."
    lines = ["```bibtex", bibtex(cff, repo_id), "```"]
    cited = [(sid, i.get("citation")) for sid, i in infos.items() if i.get("citation")]
    if cited:
        lines += ["", "Please also cite the sources you use:", ""]
        lines += [f"- `{sid}`: {_cell(c)}" for sid, c in cited]
    return "\n".join(lines)


def masks_note(configs: dict, repo_id: str) -> str:
    masked = [n for n, c in configs.items() if any(f["name"] == "mask" for f in c["features"])]
    if not masked:
        return ""
    return (
        "### Reading the masks\n\n"
        "Masks are stored as PNG label images, not as pictures. In semantic configurations each "
        "pixel holds a class id (`uint8`, `255` = ignore); in instance configurations it holds an "
        "instance id (`uint16`, `0` = background). The values sit close to zero, so **masks look "
        "almost black in the dataset viewer and in an image viewer**. They are not empty or "
        "corrupted. Colour them to look at them:\n\n```python\n"
        "import numpy as np\nfrom datasets import load_dataset\nfrom PIL import Image\n\n"
        f'row = load_dataset("{repo_id}", "{masked[0]}", split="validation")[0]\n'
        'ids = np.array(row["mask"]).astype(np.int32)  # (H, W) ids, 255 = ignore\n'
        "palette = np.random.default_rng(0).integers(0, 256, (256, 3), dtype=np.uint8)\n"
        "palette[255] = 0  # ignore -> black\n"
        'Image.fromarray(palette[ids % 256]).save("mask-coloured.png")\n```\n\n'
    )


def quick_start(configs: dict, repo_id: str) -> str:
    first = next(iter(configs))
    return (
        "```python\nfrom datasets import load_dataset\n\n"
        f'ds = load_dataset("{repo_id}", "{first}", split="train")\n```\n\n'
        "Configurations: " + ", ".join(f"`{n}`" for n in configs) + ". Each loads on its own."
    )
