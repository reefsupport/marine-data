#!/usr/bin/env python3
"""Fold every ingest-spec id with no registry entry into ``registry/sources/spec-<domain>.yaml``.

P1.2 (registry as the single truth). Idempotent: an id that already has a registry entry (or is the
``SPEC_ALIASES`` target of one) is skipped, so a second run finds nothing to do and writes nothing.

Each new entry states only what its ingest spec states:
  * licence: the spec's ``license`` string mapped onto a registry licence id when it names one
    (``CC-BY-4.0 (IMOS data licence)`` -> ``CC-BY-4.0``). Anything else (``unstated``,
    ``NOASSERTION``, ``CC-BY (per child)`` ...) stays ``NO-LICENCE-STATED`` / ``PER-SOURCE-VARIES``
    with the verbatim string kept in ``notes``. Nothing is guessed.
  * access_class: the spec's, verbatim, except ``open`` without an established licence, which is
    tightened to ``internal-only`` in the entry and in the spec (the two stay equal): an open class
    no licence backs is an unexpected denial in ``marinedata check``. Loosen it once verified.
  * n_images / n_files: ``expected_images`` or ``measured.items|images`` when a clean integer.
  * ``access: {method: s3}``, ``version`` and ``loader: staged-tree`` only when ``bucket-trees.csv``
    (the 2026-10-08 flat listing, one row per versioned tree) holds a tree for the id; otherwise
    ``loader: metadata-only`` plus a ``release_skip_reason`` the release gate accepts.
Domain, modalities and capabilities are keyword inferences from the spec text, and say so.

Usage: ``uv run python scripts/registry_fold_specs.py [--dry-run]``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from marinedata.licence_class import SPEC_ALIASES, coerce_class

ROOT = Path(__file__).resolve().parents[1]
REG, SPECS = ROOT / "registry" / "sources", ROOT / "registry" / "ingest-specs"
TREES = ROOT / "registry" / "inventory" / "bucket-trees.csv"
TODAY, ENDPOINT = "2026-10-08", "hel1.your-objectstorage.com"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]*$")
PER_RE = re.compile(r"per[\s-](child|source|campaign|item|row|series|survey|file)")
LICENCES = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc by 4.0": "CC-BY-4.0",
    "cc-by 4.0": "CC-BY-4.0",
    "cc-by-3.0": "CC-BY-3.0",
    "cc0-1.0": "CC0-1.0",
    "cc0": "CC0-1.0",
    "mit": "MIT",
    "mit-license": "MIT",
    "mit license": "MIT",
    "apache-2.0": "Apache-2.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "gpl-3.0": "GPL-3.0",
}
UNESTABLISHED = ("NO-LICENCE-STATED", "PER-SOURCE-VARIES")
OPEN_IDS = {"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "US-GOV-PD", "CC-BY-SA-4.0"}
DOMAINS = (  # first match wins
    ("seagrass", r"seagrass|posidonia|zostera"),
    ("mangrove", r"mangrove"),
    ("plankton", r"plankton|ecotaxa"),
    ("fish", r"\bfish|shark|salmon|bruv|\bray\b"),
    ("deep-sea", r"deep[\s-]?sea|abyssal|hadal|bathyal|fathomnet|nodule|hydrothermal|\bccz\b"),
    ("coral", r"coral|reef|bleach|benthic|photoquadrat"),
    ("imaging", r"stereo|sonar|slam|\b3d\b|enhanc|depth|inertial|turbid|dehaz|colou?r"),
)
NO_STAGED = (
    "no-staged-images: registry lists imagery, nothing staged under sources/ in rs-storage-open "
    "(U7/queue follow-up)"
)
TIGHTENED = "access_class: internal-only  # P1.2: was open; licence not established"


def q(s: object) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def licence_of(raw: str, per_row: bool) -> tuple[str, str]:
    """``(registry licence id, legal_basis)`` for a spec ``license`` string. Never guesses."""
    key = " ".join(raw.lower().split())
    if per_row or PER_RE.search(key):
        return "PER-SOURCE-VARIES", "unknown"
    head = re.sub(r"\s*\(.*$", "", key).strip()
    if head in LICENCES:
        return LICENCES[head], "licence"
    if key.startswith("public domain (us government"):
        return "US-GOV-PD", "licence"
    return "NO-LICENCE-STATED", "unknown"


def infer_domain(sid: str, sp: dict) -> str:
    m = sp.get("measured") or {}
    parts = (sid, sp.get("citation"), sp.get("homepage"), sp.get("notes"))
    text = " ".join(str(x) for x in (*parts, m.get("label_type"), m.get("habitat_region")))
    return next((d for d, rx in DOMAINS if re.search(rx, text.lower())), "fauna")


def capabilities_of(domain: str, sp: dict, audio: bool) -> list[str]:
    label = str((sp.get("measured") or {}).get("label_type") or "").lower()
    detect, segment = re.search(r"box|bbox|detect", label), re.search(r"mask|segment", label)
    if audio:
        return ["bioacoustics"]
    if domain == "fish":
        return ["fish-detection" if detect else "fish-classification"]
    if domain == "coral":
        return ["benthic-segmentation" if segment else "benthic-classification"]
    return ["general-pretraining"]


def spec_count(sp: dict) -> tuple[int | None, str | None]:
    def clean(v: object) -> int | None:
        s = str(v).replace(",", "").replace("_", "").strip()
        return int(s) if s.isdigit() else None

    if clean(sp.get("expected_images")) is not None:
        return clean(sp["expected_images"]), "spec:expected_images"
    measured = sp.get("measured") or {}
    for key in ("images", "items"):
        if clean(measured.get(key)) is not None:
            return clean(measured[key]), f"spec:measured.{key}"
    return None, None


def load_trees() -> dict[str, dict]:
    """One tree per id: the open bucket's, else the private one; newest write wins."""
    best: dict[str, dict] = {}
    for row in csv.DictReader(TREES.open()):
        rank = (row["bucket"] == "rs-storage-open", row["last_modified_max"])
        cur = best.get(row["source_id"])
        if cur is None or rank > (cur["bucket"] == "rs-storage-open", cur["last_modified_max"]):
            best[row["source_id"]] = row
    return best


def access_lines(sid: str, sp: dict, tree: dict | None, fname: str) -> list[str]:
    home = str(sp.get("homepage") or "")
    if tree:
        bkt, ver = tree["bucket"], tree["version"]
        note = f"{bkt} copy of the upstream dataset (staged tree {ver}); upstream: {home}"
        return [
            "    access:",
            "      method: s3",
            f"      uri: s3://{bkt}/sources/{sid}/{ver}",
            f"      notes: {q(note)}",
            "      params:",
            f"        bucket: {bkt}",
            f"        endpoint: {ENDPOINT}",
            "        region: hel1",
            f"        prefix: sources/{sid}/{ver}/",
            f"      size_bytes: {int(tree['bytes'])}",
        ]
    adapter = str(sp.get("adapter") or "")
    method = {"hf": "huggingface", "zenodo": "zenodo"}.get(adapter)
    method = method or ("request" if adapter.startswith("needs:") else "http")
    lines = ["    access:", f"      method: {method}"]
    if home.startswith("http"):
        lines.append(f"      uri: {q(home)}")
    lines.append(
        f"      notes: {q(f'ingest adapter {adapter}; see registry/ingest-specs/{fname}')}"
    )
    return lines


def entry_text(sid: str, fname: str, sp: dict, tree: dict | None) -> tuple[str, str, bool]:
    """``(domain, yaml text, class tightened)`` for one spec."""
    raw = str(sp.get("license") or "").strip()
    per_row = bool(sp.get("licence_per_row"))
    lic, basis = licence_of(raw, per_row)
    spec_cls = coerce_class(sp.get("access_class"))
    tightened = spec_cls == "open" and lic in UNESTABLISHED
    cls = "internal-only" if tightened else spec_cls
    domain = infer_domain(sid, sp)
    audio = bool(re.search(r"hydrophone|acoustic|audio|soundscape", f"{sid} {sp.get('citation')}"))
    n, how = spec_count(sp)
    home = str(sp.get("homepage") or "")
    cite = " ".join(str(sp.get("citation") or "").split())
    redistribution = "ok" if (cls == "open" and lic in OPEN_IDS) else "unknown"
    version = tree["version"] if tree else str(sp.get("version") or "unversioned")
    by = f"registry/ingest-specs/{fname}: license field copied verbatim, not independently checked"
    out = [
        f"  - id: {sid}",
        f"    domain: {domain}",
        f"    access_class: {cls}",
        f"    name: {q(sid.replace('-', ' '))}",
        f"    version: {q(version)}",
        f"    description: {q((cite or f'{sid} (folded from its ingest spec)')[:300])}",
        f"    licence: {lic}",
        f"    legal_basis: {basis}",
        f"    redistribution: {redistribution}",
        "    provenance: public",
    ]
    if per_row:
        out.append("    licence_per_row: true")
    out += ["    verification:", f"      verified_on: {TODAY}", f"      verified_by: {q(by)}"]
    out += ["      method: secondary", *access_lines(sid, sp, tree, fname)]
    out.append(f"    modalities: [{'audio' if audio else 'image'}]")
    out.append(f"    capabilities: [{', '.join(capabilities_of(domain, sp, audio))}]")
    layout = "staged-tree" if (tree and not audio) else "metadata-only"
    out.append(f"    loader: {{ layout: {layout} }}")
    if n is not None and not audio:
        out += [f"    n_images: {n}", f"    n_images_method: {q(how)}"]
    elif n is not None:
        out += [f"    n_files: {n}", f"    counts_note: {q(f'{n} clips per {how}')}"]
    out.append("    coverage: { regions: [unknown] }")
    if home.startswith("http"):
        out.append(f"    homepage: {q(home)}")
    if cite:
        out.append(f"    citation: {q(cite[:300])}")
    note = (
        f"Folded from registry/ingest-specs/{fname} on {TODAY} (P1.2). Spec license string, "
        f"verbatim: {raw!r}. Domain, modalities and capabilities are keyword inferences from the "
        "spec text."
    )
    if tightened:
        note += (
            " The spec said access_class: open; tightened to internal-only because its licence "
            "string does not establish a licence (never guess). Loosen after verifying it."
        )
    out += [f"    notes: {q(note)}", "    tags: [spec-folded]"]
    reason = "non-image-modality: audio" if audio else (None if tree else NO_STAGED)
    if reason:
        out.append(f"    release_skip_reason: {q(reason)}")
    return domain, "\n".join(out) + "\n\n", tightened


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    have: set[str] = set(SPEC_ALIASES.values())
    for p in sorted(REG.rglob("*.yaml")):
        have |= {e["id"] for e in (yaml.safe_load(p.read_text()) or {}).get("sources", [])}
    trees, new, tight, by_domain = load_trees(), [], [], defaultdict(list)
    for p in sorted(SPECS.glob("*.yaml")):
        sp = yaml.safe_load(p.read_text()) or {}
        sid = sp.get("id")
        if not isinstance(sid, str) or sid in have:
            continue
        if not ID_RE.match(sid):
            raise SystemExit(f"{p.name}: id {sid!r} does not match the registry id pattern")
        domain, text, tightened = entry_text(sid, p.name, sp, trees.get(sid))
        by_domain[domain].append(text)
        new.append(sid)
        if tightened:
            tight.append(sid)
            if not args.dry_run:
                pat = r"(?m)^access_class:\s*[\"']?open[\"']?\s*$"
                p.write_text(re.sub(pat, TIGHTENED, p.read_text(), count=1))
    head = (
        "# Entries folded from registry/ingest-specs/ by scripts/registry_fold_specs.py\n"
        f"# (P1.2, {TODAY}). Every field is the spec's own claim; edit freely, the script never\n"
        "# rewrites an id it did not add.\n\n"
        "sources:\n"
    )
    for domain, texts in sorted(by_domain.items()):
        path = REG / f"spec-{domain}.yaml"
        if not args.dry_run:
            path.write_text((path.read_text() if path.exists() else head) + "".join(texts))
    counts = {d: len(t) for d, t in sorted(by_domain.items())}
    staged = sum(1 for i in new if i in trees)
    mode = "dry run" if args.dry_run else "written"
    print(f"folded {len(new)} ids ({mode}) {counts}; s3: {staged}")
    print(f"tightened open -> internal-only (licence not established): {tight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
