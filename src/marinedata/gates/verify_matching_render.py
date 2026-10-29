"""Markdown for the generated blocks of ``docs/label-matching.md`` (see ``verify_matching``)."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

LONG = 12  # a crosswalk with more unmappable edges than this is summarised


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _status(rec: Mapping[str, Any]) -> str:
    return {"pass": "pass", "fail": "FAIL", "no-crosswalk": "no crosswalk"}.get(
        rec["status"], rec["status"]
    )


def sources_block(report: Mapping[str, Any]) -> str:
    rows = []
    for r in report["sources"]:
        c = r["counts"]
        rows.append(
            [
                f"`{r['id']}`",
                r["origin"],
                f"`{r['crosswalk']}`" if r["crosswalk"] else "-",
                r["basis"],
                c["labels"],
                c["edges"],
                c["exact"],
                c["coarsened"],
                c["approximate"] + c["narrower"],
                c["ignored"],
                c["unresolved"],
                c["ambiguous"],
                c["missing"],
                _status(r),
            ]
        )
    head = [
        "Source",
        "Origin",
        "Crosswalk",
        "Basis",
        "Labels",
        "Edges",
        "Exact",
        "Coarsened",
        "Approx.",
        "Ignored",
        "Unresolved",
        "Ambiguous",
        "Missing",
        "Status",
    ]
    gaps = [r for r in report["sources"] if r["status"] == "no-crosswalk"]
    lines = [
        table(head, rows),
        "",
        "**Sources without a crosswalk** (labels cannot be resolved here):",
        "",
    ]
    lines += [f"- `{r['id']}` ({r['schema']}): {r['reason']}" for r in gaps]
    return "\n".join(lines)


def families_block(report: Mapping[str, Any]) -> str:
    agree = report["agreement"]
    rows = [
        [f, v["nodes"], v["fed"], v["nodes"] - v["fed"], len(v["sources"]), v["labels"]]
        for f, v in agree["families"].items()
    ]
    lines = [table(["Family", "Nodes", "Fed", "Unfed", "Sources", "Native labels"], rows), ""]
    for family, v in agree["families"].items():
        lines.append(
            f"- **{family}** ({len(v['sources'])} sources): "
            + ", ".join(f"`{s}`" for s in v["sources"])
        )
    lines += [
        "",
        f"Nodes no source feeds: {len(agree['unfed_nodes'])}. "
        f"Nodes fed by a single source: {len(agree['single_source_nodes'])}.",
    ]
    return "\n".join(lines)


def unmappable_block(report: Mapping[str, Any]) -> str:
    items = report["unmappable"]
    by_cw: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_cw.setdefault(item["crosswalk"], []).append(item)
    cs95 = [i["label"] for i in by_cw.get("coralscapes-95", [])]
    lines = [
        f"The registry holds {len(items)} unmappable edges in {len(by_cw)} crosswalks. "
        f"Coralscapes-95 has {len(cs95)}: {', '.join(f'`{x}`' for x in cs95)}. "
        "Each carries a note saying why the label has no benthic class; the pixels or points it "
        "labels become 255 (ignore).",
        "",
    ]
    rows = []
    for cw, edges in sorted(by_cw.items()):
        if len(edges) > LONG:
            rows.append(
                [f"`{cw}`", f"{len(edges)} edges", "listed in `results/label-matching-v1.2.0.json`"]
            )
        else:
            rows += [[f"`{cw}`", f"`{e['label']}`", e["note"]] for e in edges]
    return "\n".join([*lines, table(["Crosswalk", "Label", "Note"], rows)])


def legacy_block(report: Mapping[str, Any]) -> str:
    out = []
    for new_id, v in report["legacy"].items():
        kinds = ", ".join(f"{k}: {n}" for k, n in v["kinds"].items())
        out.append(
            f"`{new_id}` against `{v['legacy_crosswalk']}`: "
            f"{v['paired']} of {v['edges']} edges paired "
            f"through the legacy column ({kinds}); "
            f"condition differences: {len(v['condition_differences'])}; "
            f"form differences: {len(v['form_differences'])}; untagged: {len(v['untagged'])}; "
            f"unpaired: {len(v['unpaired'])}."
        )
        rows = [
            [
                r["v2_id"] or "",
                f"`{r['v2']}`",
                f"`{r['legacy']}`",
                r["kind"],
                r["v2_target"] or "ignore",
                r["legacy_target"] or "ignore",
            ]
            for r in v["refinements"]
        ]
        out += [
            "",
            table(
                ["V2 id", "V2 label", "Legacy label", "Relation", "V2 target", "Legacy target"],
                rows,
            ),
        ]
    return "\n".join(out)


def review_block(report: Mapping[str, Any]) -> str:
    s, a, c = report["summary"], report["agreement"], report["condition"]
    legacy = report["legacy"].get("coralscapes-95", {})
    rows = [
        [
            "4 agreement",
            "exact taxon edges whose native label shares no word with the node",
            a["review"]["no_shared_word"]["count"],
        ],
        ["4 agreement", "rs-benthic-v1 nodes no source feeds", len(a["unfed_nodes"])],
        [
            "5 legacy",
            "Coralscapes-95 edges that refine, coarsen or retarget the 39-class target",
            len(legacy.get("refinements", [])),
        ],
        [
            "5 legacy",
            "condition differences against the 39-class edge",
            len(legacy.get("condition_differences", [])),
        ],
        ["6 condition", "sources that carry the condition axis", len(c["sources_with_condition"])],
        [
            "6 condition",
            "sources with a crosswalk and no condition edge",
            len(c["sources_without_condition"]),
        ],
        [
            "6 condition",
            "health-word labels with no condition target",
            c["review"]["health_label_without_condition"]["count"],
        ],
        [
            "6 condition",
            "health-word labels on a different condition node",
            c["review"]["health_label_other_condition"]["count"],
        ],
        [
            "1 coverage",
            "schema nodes with no edge in any crosswalk (declared, not harvested)",
            s["declared_unmapped"],
        ],
        [
            "3 fidelity",
            "lossy edges without a note (pre-existing; ratcheted)",
            sum(s["lossy_without_note"].values()),
        ],
        ["1 coverage", "sources without a crosswalk", len(s["no_crosswalk"])],
        ["7 ignore", "ignore-to-255 assertions evaluated", report["ignore"]["checks"]],
    ]
    rows.append(
        [
            "7 ignore",
            "unmappable labels a scheme trains as a class (override)",
            len(report["ignore"]["overrides"]),
        ]
    )
    rows.append(
        [
            "4 agreement",
            "exact edges whose native label is a bare code or id (not compared)",
            a["review"]["code_like_labels"],
        ]
    )
    return table(["Check", "Review item", "Count"], rows)


BLOCKS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "sources": sources_block,
    "families": families_block,
    "unmappable": unmappable_block,
    "legacy": legacy_block,
    "review": review_block,
}


def blocks(report: Mapping[str, Any]) -> dict[str, str]:
    return {name: fn(report) for name, fn in BLOCKS.items()}


def splice(text: str, rendered: Mapping[str, str]) -> str:
    """Replace the body between ``<!-- BEGIN GENERATED: name -->`` and the matching END marker."""
    for name, body in rendered.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n).*?"
            rf"(<!-- END GENERATED: {re.escape(name)} -->)",
            re.S,
        )
        if not pattern.search(text):
            raise ValueError(f"docs/label-matching.md has no generated block {name!r}")
        text = pattern.sub(lambda m, b=body: f"{m.group(1)}{b.rstrip()}\n{m.group(2)}", text)
    return text
