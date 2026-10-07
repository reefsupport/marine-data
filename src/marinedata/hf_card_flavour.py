"""Per-flavour pieces of the dataset card (WP-L1b): licence header, gating, tables, recipes.

``open`` -> ``reefsupport/marine-data`` (ungated, commercial use OK). ``nc`` ->
``reefsupport/marine-data-nc`` (gated, the restricted-nc DELTA). The NC flavour is the open
repo and the nc repo used together (composed of both, non-commercial as a whole); the nc card
says so in its gate prompt, its top notice and its body.
"""

from __future__ import annotations

import json

NC_GATE_PROMPT = (
    "This dataset is NON-COMMERCIAL. By requesting access you agree that you will not use it, "
    "or anything derived from it (models, weights, embeddings, annotations), for any commercial "
    "purpose. Every source keeps its own licence (see LICENSE and the sources table in the "
    "README); you must also comply with each of them, including attribution. The combined set "
    "(reefsupport/marine-data + reefsupport/marine-data-nc) is non-commercial as a whole. "
    "Sources outside this repo (no-derivatives, internal-only, unknown-licence) are not "
    "distributed here."
)
NC_CHECKBOX = "I will use this dataset for non-commercial purposes only"

EXCLUDED_CLASSES = (
    (
        "restricted-nd",
        "no derivatives: a verbatim-only licence cannot cover cropped, resized or "
        "re-labelled releases",
    ),
    (
        "internal-only / unknown",
        "no licence granting redistribution was established for these "
        "sources (research-use-only or nothing on record)",
    ),
    (
        "per-row licence",
        "sources whose every item carries its own licence are held out until "
        "each row's licence is resolved",
    ),
)


def license_header(flavour: str) -> list[str]:
    """YAML ``license`` lines (and, for nc, the Hub gating fields)."""
    if flavour == "open":
        return ["license: other", "license_name: mixed-open", "license_link: LICENSE"]
    return [
        "license: other",
        "license_name: mixed-non-commercial",
        "license_link: LICENSE",
        f"extra_gated_prompt: {json.dumps(NC_GATE_PROMPT)}",
        "extra_gated_fields:",
        "  Name: text",
        "  Affiliation: text",
        "  Intended use: text",
        f"  {NC_CHECKBOX}: checkbox",
    ]


def top_notice(flavour: str, repo_id: str) -> list[str]:
    if flavour == "open":
        return [
            "",
            "> **Open flavour**: every source here allows commercial use "
            "(per-source licences and attribution below).",
            "",
        ]
    return [
        "",
        "> **NON-COMMERCIAL, gated.** The NC flavour is composed of both the open and the "
        "restricted options together: `reefsupport/marine-data` (open) + this repo "
        "(restricted-nc), used together under the non-commercial terms as a whole. This repo "
        "physically holds only the restricted-nc delta; alone or combined with "
        "`reefsupport/marine-data` it may be used for non-commercial purposes only.",
        "",
    ]


def _attribution_table(sources: list[dict]) -> list[str]:
    rows = ["| Source | Licence | Attribution |", "|---|---|---|"]
    for s in sources:
        cell = lambda v: str(v or "—").replace("\n", " ").replace("|", "/")  # noqa: E731
        rows.append(f"| `{s['id']}` | {cell(s['licence'])} | {cell(s.get('attribution'))} |")
    return rows


def tail_sections(flavour: str, repo_id: str, sources: list[dict], takedown_url: str) -> list[str]:
    out = ["", "## Licence and attribution per source", "", *_attribution_table(sources)]
    other = "reefsupport/marine-data"
    if flavour != "open":  # restricted flavour only; the open card names no other repo
        out += [
            "",
            "## The NC flavour: open + restricted-nc together",
            "",
            f"The NC flavour is composed of both the open and the restricted options together: "
            f"`{other}` (open) + `{repo_id}` (restricted-nc, this repo), used together under the "
            "non-commercial terms as a whole. The two repos share one frozen split and never "
            "duplicate an image (`image_sha256` is disjoint). Load both and concatenate:",
            "",
            "```python",
            "from datasets import concatenate_datasets, load_dataset",
            f'open_ = load_dataset("{other}", "images", split="train")',
            f'restricted = load_dataset("{repo_id}", "images", split="train")  # gated',
            "nc_flavour = concatenate_datasets([open_, restricted])  # NON-COMMERCIAL use only",
            "```",
        ]
    is_open = flavour == "open"
    # The open card must not name or link the gated repo, so its headings stay single-repo.
    out += ["", "## Excluded sources" if is_open else "## Excluded from both repos", ""]
    out += [f"- **{name}**: {why}" for name, why in EXCLUDED_CLASSES]
    out += [
        "",
        "## Contact" if is_open else "## Takedown and contact",
        "",
        f"Rights holders and data subjects: open an issue at {takedown_url}.",
        "",
    ]
    return out
