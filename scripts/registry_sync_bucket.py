#!/usr/bin/env python3
"""Write n_images / n_images_method / checksums into the registry from the staged trees (P1.2).

Two stages, so the network is read once and the registry edit is repeatable offline:

  --probe   anonymous single-key GETs (CHECKSUMS.sha256, INGEST.json, metadata.parquet footer; at
            most 3 per tree; never a listing) for each versioned tree in
            registry/inventory/bucket-trees.csv -> registry/inventory/bucket-probe-2026-10-08.json
  --apply   patch registry/sources/*.yaml from that JSON (idempotent, line-based, comments kept);
            --parts checksums|counts limits the edit to one of the two (default: both)
  (neither) print what --apply would do

root_digest is the sha256 of the tree's CHECKSUMS.sha256, as checksums.py defines it.
Checksums.version must equal Source.version, so a digest is written only when the entry carries
the tree's version, or carries the placeholder ``unversioned`` and has exactly one staged tree
(the entry then adopts the tree's version). Any other tree with CHECKSUMS is listed, not written.
n_images: INGEST.json, else the metadata.parquet footer row count, else the listing's image-key
count, else the ingest spec's own figure (``spec:<field>``). A figure already in the registry is
kept and labelled ``registry-prior``; an entry with nothing measurable is labelled
``unavailable:non-image`` or ``unavailable:no-count`` and gets no number.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import yaml

from marinedata.licence_class import SPEC_ALIASES
from marinedata.registry_probe import probe_tree

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "registry" / "sources"
SPECS = ROOT / "registry" / "ingest-specs"
TREES = ROOT / "registry" / "inventory" / "bucket-trees.csv"
EVIDENCE = ROOT / "registry" / "inventory" / "bucket-probe-2026-10-08.json"
PRIOR = "registry-prior"
DASH = re.compile(r"^( *)- id: ")
NON_IMAGE = frozenset(
    {
        "context-layer",
        "reef-audio",
        "bioacoustics",
        "environmental",
        "covariate",
        "tabular",
        "bathymetry",
        "occurrence",
        "extent",
        "basemap",
        "lidar",
        "3d",
        "metric-3d",
        "tracking",
        "tooling",
        "habitat",
        "genus",
    }
)


def probe_all(workers: int = 6) -> dict:
    rows = list(csv.DictReader(TREES.open()))

    def one(row: dict) -> dict:
        try:
            return probe_tree(row)
        except Exception as exc:  # recorded in the evidence, never fatal
            head = {"bucket": row["bucket"], "id": row["source_id"], "version": row["version"]}
            return {**head, "error": f"{type(exc).__name__}: {exc}"[:160]}

    with ThreadPoolExecutor(workers) as pool:
        trees = list(pool.map(one, rows))
    no_ck = [
        {
            "bucket": r["bucket"],
            "id": r["source_id"],
            "version": r["version"],
            "gb": round(int(r["bytes"]) / 1e9, 2),
        }
        for r in rows
        if r["has_checksums"].strip().lower() != "true"
    ]
    return {
        "probed_on": "2026-10-08",
        "method": "anonymous HTTPS GET of single keys, at most 3 per tree, no listing",
        "gets": sum(t.get("gets", 0) for t in trees),
        "bytes_read": sum(t.get("bytes_read", 0) for t in trees),
        "trees": trees,
        "without_checksums": no_ck,
    }


def load_entries() -> dict[str, tuple[Path, dict]]:
    out: dict[str, tuple[Path, dict]] = {}
    for p in sorted(REG.rglob("*.yaml")):
        for e in (yaml.safe_load(p.read_text()) or {}).get("sources", []):
            out[e["id"]] = (p, e)
    return out


def spec_count_of(sid: str) -> tuple[int | None, str | None]:
    path = SPECS / f"{SPEC_ALIASES.get(sid, sid)}.yaml"
    if not path.exists():
        return None, None
    fold = importlib.import_module("registry_fold_specs")
    return fold.spec_count(yaml.safe_load(path.read_text()) or {})


def _is_private(entry: dict) -> bool:
    acc = entry.get("access") or {}
    bucket = (acc.get("params") or {}).get("bucket") or ""
    return bucket == "rs-storage-private" or str(acc.get("uri", "")).startswith(
        "s3://rs-storage-private"
    )


def _count_edit(sid: str, e: dict, tree: dict | None) -> tuple[dict, str | None]:
    """The n_images part of an entry's edit, and the reason when nothing can be measured."""
    usable = bool(tree and tree.get("images"))
    if usable and (tree["method"] != "listing" or e.get("n_images") is None):
        return {"n_images": (tree["images"], tree["method"])}, None  # a listing never overwrites
    if e.get("n_images") is not None:
        return ({} if e.get("n_images_method") else {"method": PRIOR}), None
    n, how = spec_count_of(sid)
    if n is not None:
        return {"n_images": (n, how)}, None
    why = "non-image" if NON_IMAGE & {str(t) for t in e.get("tags") or []} else "no-count"
    return {"method": f"unavailable:{why}"}, why


def make_plan(ev: dict, by_id: dict) -> tuple[dict, dict]:
    inv = {v: k for k, v in SPEC_ALIASES.items()}
    unread = [t for t in ev["trees"] if t.get("skipped") or t.get("error")]
    trees_of: dict[str, list[dict]] = defaultdict(list)
    for t in ev["trees"]:
        if t not in unread:
            trees_of[inv.get(t["id"], t["id"])].append(t)
    rep: dict = {
        "unread": [(t["id"], t.get("skipped") or t.get("error")) for t in unread],
        "no_entry": sorted(set(trees_of) - set(by_id)),
        "adopted": [],
        "corrected": [],
        "gaps": [],
    }
    plans: dict[str, dict] = {}
    used: set[tuple[str, str]] = set()
    for sid, (path, e) in by_id.items():
        ver = str(e.get("version"))
        ts = trees_of.get(sid, [])
        exact = next((t for t in ts if t["version"] == ver), None)
        single = ts[0] if exact is None and len(ts) == 1 else None
        adopt = (
            single is not None
            and ver == "unversioned"
            and bool(single.get("root_digest"))
            and not _is_private(e)
        )
        ck_tree = exact or (single if adopt else None)
        edit: dict = {"path": path}
        if ck_tree is not None and ck_tree.get("root_digest"):
            used.add((sid, ck_tree["version"]))
            keys = ("version", "root_digest", "files", "size_bytes")
            edit["checksums"] = {k: ck_tree[k] for k in keys}
        if adopt:
            edit["new_version"] = single["version"]
            rep["adopted"].append((sid, single["version"]))
        part, gap = _count_edit(sid, e, exact or single)
        edit.update(part)
        if gap:
            rep["gaps"].append((sid, gap))
        if "n_images" in part and e.get("n_images") not in (None, part["n_images"][0]):
            rep["corrected"].append((sid, e["n_images"], part["n_images"][0]))
        plans[sid] = edit
    rep["extra"] = [
        (t["id"], t["version"])
        for ts in trees_of.values()
        for t in ts
        if t.get("root_digest") and (inv.get(t["id"], t["id"]), t["version"]) not in used
    ]
    return plans, rep


def updates_for(edit: dict, parts: set[str]) -> dict[str, list[str]]:
    up: dict[str, list[str]] = {}
    if "checksums" in parts:
        if "new_version" in edit:
            up["version"] = [f'    version: "{edit["new_version"]}"']
        if "checksums" in edit:
            c = edit["checksums"]
            up["checksums"] = [
                "    checksums:",
                f'      version: "{c["version"]}"',
                f"      root_digest: {c['root_digest']}",
                f"      files: {c['files']}",
                f"      size_bytes: {c['size_bytes']}",
            ]
    if "counts" in parts:
        if "n_images" in edit:
            n, how = edit["n_images"]
            up["n_images"] = [f"    n_images: {n}", f'    n_images_method: "{how}"']
            up["n_images_method"] = []
        elif "method" in edit:
            up["n_images_method"] = [f'    n_images_method: "{edit["method"]}"']
    return up


def _fit(lines: list[str], ind: int) -> list[str]:
    """Re-indent lines authored for entries dashed at column 2 to the file's own indent."""
    shift = ind - 2
    return [" " * shift + ln if shift >= 0 else ln[-shift:] for ln in lines]


def set_keys(blk: list[str], updates: dict[str, list[str]], ind: int = 2) -> list[str]:
    """Replace top-level keys of one entry in place (append the missing ones, comments kept)."""
    key_re = re.compile(" " * (ind + 2) + r"([A-Za-z_]+):")
    deeper = " " * (ind + 4)
    out: list[str] = []
    seen: set[str] = set()
    skipping = False
    for ln in blk:
        m = key_re.match(ln)
        if m and m.group(1) in updates:
            if m.group(1) not in seen:
                out += _fit(updates[m.group(1)], ind)
                seen.add(m.group(1))
            skipping = True
            continue
        if skipping and ln.startswith(deeper):
            continue
        skipping = False
        out.append(ln)
    tail: list[str] = []
    while out and (not out[-1].strip() or out[-1].lstrip().startswith("#")):
        tail.insert(0, out.pop())
    for key, lines in updates.items():
        if key not in seen:
            out += _fit(lines, ind)
    return out + tail


def patch_text(text: str, updates: dict[str, dict[str, list[str]]]) -> tuple[str, set[str]]:
    lines = text.split("\n")
    heads = [(i, DASH.match(ln)) for i, ln in enumerate(lines)]
    heads = [(i, m) for i, m in heads if m]
    if not heads:
        return text, set()
    ind = len(heads[0][1].group(1))
    starts = [i for i, m in heads if len(m.group(1)) == ind]
    out, hit = lines[: starts[0]], set()
    for a, b in pairwise([*starts, len(lines)]):
        blk = lines[a:b]
        sid = re.sub(r"\s+#.*$", "", blk[0].split(":", 1)[1]).strip().strip("\"'")
        if sid in updates:
            blk = set_keys(blk, updates[sid], ind)
            hit.add(sid)
        out += blk
    return "\n".join(out), hit


def apply_plans(plans: dict, parts: set[str]) -> tuple[int, list[str]]:
    by_file: dict[Path, dict[str, dict[str, list[str]]]] = defaultdict(dict)
    for sid, edit in plans.items():
        up = updates_for(edit, parts)
        if up:
            by_file[edit["path"]][sid] = up
    changed, missed = 0, []
    for path, ups in by_file.items():
        old = path.read_text()
        new, hit = patch_text(old, ups)
        missed += sorted(set(ups) - hit)
        if new != old:
            path.write_text(new)
            changed += 1
    return changed, missed


def show(ev: dict, plans: dict, rep: dict, by_id: dict) -> None:
    trees = ev["trees"]
    gets = max(t.get("gets", 0) for t in trees)
    print(
        f"trees {len(trees)}, unread {rep['unread']}; GETs {ev['gets']}, max {gets}/tree, "
        f"{ev['bytes_read'] / 1e6:.1f} MB read"
    )
    ck = sum(1 for p in plans.values() if "checksums" in p)
    print(
        f"digests to write {ck}; version adopted {len(rep['adopted'])}; CHECKSUMS trees left "
        f"without a digest {len(rep['extra'])}: {rep['extra']}"
    )
    print(f"trees with no registry entry: {rep['no_entry']}")
    print(f"counts corrected {len(rep['corrected'])}: {rep['corrected'][:16]}")
    by_method = Counter(p["n_images"][1].split(":")[0] for p in plans.values() if "n_images" in p)
    print(f"counts to write {dict(by_method)}; unavailable {Counter(w for _, w in rep['gaps'])}")
    nock = ev["without_checksums"]
    print(f"trees without CHECKSUMS: {len(nock)}, {sum(x['gb'] for x in nock):.1f} GB")
    reg = {s: e for s, (_, e) in by_id.items() if "spec-folded" not in (e.get("tags") or [])}
    has = sum(1 for e in reg.values() if e.get("n_images") is not None)
    will = sum(1 for s, e in reg.items() if e.get("n_images") is not None or "n_images" in plans[s])
    meth = sum(
        1
        for s, e in reg.items()
        if e.get("n_images_method") or "n_images" in plans[s] or "method" in plans[s]
    )
    print(f"registered {len(reg)}: n_images {has} -> {will}; method recorded {meth}")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--parts", default="checksums,counts")
    args = ap.parse_args()
    if args.probe:
        EVIDENCE.write_text(json.dumps(probe_all(), indent=1, sort_keys=True) + "\n")
    ev = json.loads(EVIDENCE.read_text())
    by_id = load_entries()
    plans, rep = make_plan(ev, by_id)
    show(ev, plans, rep, by_id)
    if args.apply:
        changed, unmatched = apply_plans(plans, set(args.parts.split(",")))
        print(f"applied: {changed} files changed; unmatched {len(unmatched)}: {unmatched[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
