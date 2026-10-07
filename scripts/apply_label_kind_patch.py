"""Apply the WP-U15 label-kind triage patch (``registry-patch.yaml``) to ``registry/``.

Text-level edits (comments and layout survive): ``measured.label_type`` into
``registry/ingest-specs/<id>.yaml``, ``annotations`` into ``registry/sources/<file>.yaml``.
Sources the patch marks as excluded from releases (floating-marine-debris, noaa-dsc-csv) or
quarantined (large-scale-fish) get ``measured.release_excluded`` / ``measured.quarantined`` in
the ingest spec (the loader's free ``measured`` block) and, where a registry source exists, the
tags ``release-excluded`` / ``quarantined``. Idempotent: a second run changes nothing.

``python scripts/apply_label_kind_patch.py <registry-patch.yaml> [--root <repo>]``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

EXCLUDED = {
    "floating-marine-debris": "tabular Sentinel-2 pixel spectra, no imagery",
    "noaa-dsc-csv": "tabular occurrence records, no imagery",
}
QUARANTINED = {"large-scale-fish": "out-of-water market photos and no licence stated"}


def _scalar(key: str, value: object, indent: int) -> list[str]:
    text = yaml.safe_dump({key: value}, default_flow_style=False, allow_unicode=True, width=10**6)
    return [" " * indent + line for line in text.rstrip("\n").splitlines()]


def _block_end(lines: list[str], start: int) -> int:
    """Index after the top-level block opened at ``lines[start]`` (its indented or blank lines)."""
    i = start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
        i += 1
    while i > start + 1 and not lines[i - 1].strip():
        i -= 1
    return i


MEASURED_DEFAULTS = {"method": "u15-label-kind-triage (registry-patch.yaml)", "date": "2026-10-05"}


def set_measured(text: str, items: dict[str, object]) -> str:
    """Set ``items`` in the spec's ``measured`` block (created if absent). ``check_spec`` and the
    ``test_ingest_subset`` gate need every ``measured`` block to carry a method and a date, so a
    block without them gets :data:`MEASURED_DEFAULTS`; existing values are never overwritten."""
    lines = text.splitlines()
    head = next((i for i, ln in enumerate(lines) if re.match(r"^measured:", ln)), None)
    if head is None:
        lines += ["", "measured:"]
        head = len(lines) - 1
    elif lines[head].strip() != "measured:":
        lines[head] = "measured:"
    for k, v in items.items():
        end = _block_end(lines, head)
        at = next(
            (i for i in range(head + 1, end) if re.match(rf"^  {re.escape(k)}:", lines[i])), None
        )
        new = _scalar(k, v, 2)
        if at is None:
            lines[end:end] = new
        else:
            stop = at + 1
            while stop < end and lines[stop].startswith("    "):
                stop += 1
            lines[at:stop] = new
    for k, v in MEASURED_DEFAULTS.items():
        end = _block_end(lines, head)
        if not any(re.match(rf"^  {k}:", lines[i]) for i in range(head + 1, end)):
            lines[end:end] = _scalar(k, v, 2)
    return "\n".join(lines) + "\n"


def _entry_range(lines: list[str], source_id: str) -> tuple[int, int, str]:
    """``(start, end, pad)`` of a source entry: ``pad`` is the indent of its fields."""
    pat = re.compile(rf"^( *)- id: {re.escape(source_id)}\s*$")
    start = next(i for i, ln in enumerate(lines) if pat.match(ln))
    base = len(pat.match(lines[start]).group(1))
    nxt = re.compile(rf"^ {{{base}}}- id:")
    end = next((i for i in range(start + 1, len(lines)) if nxt.match(lines[i])), len(lines))
    return start, end, " " * (base + 2)


def set_entry_fields(
    text: str, source_id: str, annotations: list[dict] | None, tags: list[str]
) -> str:
    lines = text.splitlines()
    start, end, pad = _entry_range(lines, source_id)
    if annotations is not None:
        dumped = yaml.safe_dump(
            annotations, default_flow_style=False, allow_unicode=True, width=10**6
        )
        block = [f"{pad}annotations:"] + [pad + ln for ln in dumped.splitlines()]
        at = next((i for i in range(start, end) if lines[i] == f"{pad}annotations:"), None)
        if at is None:
            lines[end:end] = block
        else:
            stop = at + 1
            while (
                stop < end
                and lines[stop].startswith(pad)
                and not re.match(rf"^{pad}[A-Za-z_]+:", lines[stop])
            ):
                stop += 1
            lines[at:stop] = block
        start, end, pad = _entry_range(lines, source_id)
    for tag in tags:
        flow = next(
            (i for i in range(start, end) if re.match(rf"^{pad}tags: \[.*\]\s*$", lines[i])), None
        )
        head = next((i for i in range(start, end) if lines[i] == f"{pad}tags:"), None)
        if flow is not None:
            items = [
                t.strip()
                for t in lines[flow].split("[", 1)[1].rstrip("] \n").split(",")
                if t.strip()
            ]
            if tag not in items:
                lines[flow] = f"{pad}tags: [" + ", ".join([*items, tag]) + "]"
        elif head is not None:
            stop = head + 1
            while stop < end and lines[stop].startswith(f"{pad}- "):
                stop += 1
            if f"{pad}- {tag}" not in lines[head + 1 : stop]:
                lines.insert(stop, f"{pad}- {tag}")
        else:
            lines[end:end] = [f"{pad}tags:", f"{pad}- {tag}"]
        start, end, pad = _entry_range(lines, source_id)
    return "\n".join(lines) + "\n"


def apply(patch_path: Path, root: Path) -> dict[str, int]:
    patches = yaml.safe_load(patch_path.read_text())["patches"]
    stats = {"specs": 0, "sources": 0}
    for p in patches:
        items: dict[str, object] = {"label_type": p["spec_label_type"]}
        tags: list[str] = []
        if p["id"] in EXCLUDED:
            items |= {"release_excluded": True, "release_exclusion_reason": EXCLUDED[p["id"]]}
            tags.append("release-excluded")
        if p["id"] in QUARANTINED:
            items |= {
                "release_excluded": True,
                "quarantined": True,
                "release_exclusion_reason": QUARANTINED[p["id"]],
            }
            tags += ["release-excluded", "quarantined"]
        spec = root / p["spec_file"]
        spec.write_text(set_measured(spec.read_text(), items))
        stats["specs"] += 1
        if p.get("source_file"):
            src = root / p["source_file"]
            src.write_text(
                set_entry_fields(src.read_text(), p["id"], p.get("source_annotations"), tags)
            )
            stats["sources"] += 1
    return stats


if __name__ == "__main__":
    args = sys.argv[1:]
    root = Path(args[args.index("--root") + 1]) if "--root" in args else Path.cwd()
    print(apply(Path(args[0]), root))
