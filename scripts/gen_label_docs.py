#!/usr/bin/env python
"""Regenerate the tables of ``docs/LABELS.md`` from ``registry/label-schemes/*.yaml``.

Only the text between ``<!-- BEGIN GENERATED: <name> -->`` and ``<!-- END GENERATED: <name> -->``
is rewritten; the surrounding prose is hand-written. ``--check`` exits 1 when the doc is stale
instead of writing (``tests/test_label_docs.py`` runs the same comparison). From the repo root:

    python scripts/gen_label_docs.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from marinedata import label_schemes as ls
from marinedata import labels
from marinedata.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "LABELS.md"
PUBLIC_REPO = "reefsupport/marine-data"
EXCLUDE_EXAMPLE = ("dead", "bleached")

# Meaning of the coral-binary classes (benthic-coarse names come from the rs-benthic-v1 schema,
# scene descriptions from the scheme YAML).
COARSE_MEANING = {
    "NOT_CORAL": "Every other mapped biotic, abiotic or transition label",
    "CORAL": "Hard coral, soft coral and fire coral (dead and bleached included by default)",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _names(items: Iterable[str]) -> str:
    return ", ".join(f"`{_cell(i)}`" for i in items) or "none"


def _table(header: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _meaning(scheme: str, name: str, registry: Registry, raw: dict) -> str:
    if scheme == "benthic-coarse":
        return registry.label_schema("rs-benthic-v1").node(name).name
    if scheme == "scene":
        return str(raw["schemes"][scheme]["class_descriptions"][name])
    return COARSE_MEANING[name]


def _public(source: str, scheme: str) -> bool:
    return ls.source_spec(source, scheme).repo == PUBLIC_REPO


def _config(source: str, scheme: str) -> str:
    return f"`{ls.source_spec(source, scheme).config}`" if _public(source, scheme) else "not public"


def _native_by_class(source: str, scheme: str) -> dict[str | None, list[str]]:
    """Native labels of ``source`` grouped by scheme class (``None`` = ignored)."""
    out: dict[str | None, list[str]] = {}
    for label in ls.source_spec(source, scheme).native_labels:
        out.setdefault(ls.resolve_class(source, label, scheme), []).append(label)
    return out


def classes_block(registry: Registry, raw: dict) -> str:
    parts = []
    for scheme in labels.schemes():
        rows = [
            (str(i), f"`{name}`", _meaning(scheme, name, registry, raw))
            for i, name in enumerate(labels.class_names(scheme))
        ]
        rows.append(("255", "ignore", "Not a class: excluded from the loss and from metrics"))
        parts.append(f"#### `{scheme}`\n\n{_table(('Id', 'Class', 'Meaning'), rows)}")
    return "\n\n".join(parts)


def sources_block() -> str:
    parts = []
    for scheme in labels.schemes():
        rows = []
        for source in ls.scheme_spec(scheme).sources:
            by_class = _native_by_class(source, scheme)
            ignored = by_class.get(None, [])
            total = len(ls.source_spec(source, scheme).native_labels)
            rows.append(
                (
                    f"`{source}`",
                    _config(source, scheme),
                    "dense" if labels.is_dense(source) else "partial",
                    ", ".join(f"`{c}`" for c in labels.supervised_classes(source, scheme)),
                    str(total - len(ignored)),
                    str(len(ignored)),
                )
            )
        header = ("Source", "Config", "Annotation", "Supervised classes", "Mapped", "Ignored")
        parts.append(f"#### `{scheme}`\n\n{_table(header, rows)}")
    return "\n\n".join(parts)


def mapping_block() -> str:
    parts = []
    for scheme in labels.schemes():
        names = labels.class_names(scheme)
        rows = []
        for source in ls.scheme_spec(scheme).sources:
            by_class = _native_by_class(source, scheme)
            for name in (*names, None):
                if name in by_class:
                    cls = "255 (ignored)" if name is None else f"`{name}` ({names.index(name)})"
                    rows.append((f"`{source}`", cls, _names(by_class[name])))
        parts.append(
            f"#### `{scheme}`\n\n{_table(('Source', 'Class (id)', 'Native labels'), rows)}"
        )
    return "\n\n".join(parts)


def conditions_block() -> str:
    options = ls.make_options(exclude_conditions=EXCLUDE_EXAMPLE)
    rows = []
    for scheme in labels.schemes():
        if ls.scheme_spec(scheme).labels is not None:
            continue  # a by-name scheme has no condition axis
        for source in ls.scheme_spec(scheme).sources:
            sent = [
                label
                for label in ls.source_spec(source, scheme).native_labels
                if ls.resolve_class(source, label, scheme) is not None
                and ls.resolve_class(source, label, scheme, options) is None
            ]
            if sent:
                rows.append((f"`{scheme}`", f"`{source}`", str(len(sent)), _names(sent)))
    return _table(("Scheme", "Source", "Labels", "Sent to 255"), rows)


BLOCKS: dict[str, Callable[[Registry, dict], str]] = {
    "label-classes": classes_block,
    "label-sources": lambda _r, _raw: sources_block(),
    "label-mapping": lambda _r, _raw: mapping_block(),
    "label-conditions": lambda _r, _raw: conditions_block(),
}


def generate() -> dict[str, str]:
    """Block name -> generated markdown, derived from the registry as it is now."""
    registry = Registry.load()
    raw = {"schemes": ls._load_yaml("schemes.yaml")["schemes"]}
    return {name: build(registry, raw).rstrip() for name, build in BLOCKS.items()}


def markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"


def splice(text: str, name: str, body: str) -> str:
    """Replace the body between the markers of ``name``; raise if the markers are missing."""
    begin, end = markers(name)
    if begin not in text or end not in text:
        raise ValueError(f"docs/LABELS.md has no {begin} ... {end} block")
    a, b = text.index(begin), text.index(end) + len(end)
    return f"{text[:a]}{begin}\n\n{body}\n\n{end}{text[b:]}"


def render(text: str) -> str:
    for name, body in generate().items():
        text = splice(text, name, body)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if the doc is stale; write nothing"
    )
    args = ap.parse_args(argv)
    old = DOC.read_text()
    new = render(old)
    if new == old:
        print("docs/LABELS.md up to date")
        return 0
    if args.check:
        print("docs/LABELS.md is stale: run python scripts/gen_label_docs.py")
        return 1
    DOC.write_text(new)
    print("updated: docs/LABELS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
