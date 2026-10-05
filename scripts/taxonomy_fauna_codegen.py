"""WP-U8a: fauna crosswalks through the pinned WoRMS snapshot (codegen, never runtime).

Reads the vocab TSVs of one crosswalk (``registry/taxonomy/vocab/*.tsv`` whose ``# crosswalk:``
header names it), resolves every label that has no edge yet and writes

* new snapshot rows (existing rows are never rewritten) for names the snapshot lacks,
* new ``rs-taxa-v1`` nodes (WoRMS-backed ranks) for the accepted targets that are not nodes yet,
* the crosswalk: every existing edge is kept verbatim, new labels get new edges.

Order of resolution per label: hand table (non-taxa, common names), the exact scientific name
(snapshot first, then the cached WoRMS REST ``AphiaRecordsByNames`` at <= 2 req/s), then a
deterministic reduction of a morphospecies / cf. / complex name to the named genus-or-above,
resolved the same way and recorded as ``coarsened``. Anything else is ``unmappable`` with a
reason; nothing is guessed (no fuzzy ``AphiaRecordsByMatchNames`` match is ever applied).

Usage: ``uv run python scripts/taxonomy_fauna_codegen.py <cache_dir> <crosswalk-id> [--dry]``
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marinedata.taxonomy import load_meta, read_vocab, read_vocab_rows, taxonomy_dir
from marinedata.worms_snapshot import WormsClient, build_rows, write_parquet

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "registry"
TODAY = date(2026, 10, 5)
KEEP_RANKS = {"Superdomain", "Kingdom", "Phylum", "Subphylum", "Class", "Order", "Family", "Genus"}
KINGDOMS = ("Animalia", "Plantae", "Chromista")
FISH_NOTE = (
    "fish is paraphyletic and WoRMS Pisces is unaccepted, so fish land on the smallest accepted "
    "clade holding all of them; includes: [Actinopterygii, Elasmobranchii, Holocephali, Myxini, "
    "Petromyzonti]"
)

# label -> (WoRMS name | None = unmappable, fidelity, note). Hand decisions, applied before the rules.
HAND: dict[str, dict[str, tuple[str | None, str, str]]] = {
    "fathomnet-concepts": {},
    # BrackishMOT gt.txt class ids (paper arXiv:2302.10645): 1 fish, 2 crab, 3 shrimp, 4 starfish, 5 small fish
    "brackishmot-class-id": {
        "1": ("Vertebrata", "coarsened", FISH_NOTE),
        "2": ("Brachyura", "exact", "crab = true crabs (Brachyura)"),
        "3": ("Decapoda", "coarsened", "shrimp is paraphyletic (Caridea, Dendrobranchiata, Stenopodidea); the smallest accepted clade holding all of them is Decapoda"),
        "4": ("Asteroidea", "exact", "starfish = sea stars"),
        "5": ("Vertebrata", "coarsened", "small fish (a school-forming size class of fish): " + FISH_NOTE),
    },
    "roboflow-aquarium": {
        "fish": ("Vertebrata", "coarsened", FISH_NOTE),
        "jellyfish": ("Medusozoa", "coarsened", "jellyfish are medusozoans; ctenophores would be missed"),
        "penguin": ("Spheniscidae", "exact", "penguins are the family Spheniscidae"),
        "puffin": ("Fratercula", "exact", "puffins are the genus Fratercula (Atlantic, horned, tufted)"),
        "shark": ("Selachii", "exact", "sharks are the infraclass Selachii"),
        "starfish": ("Asteroidea", "exact", None),
        "stingray": ("Myliobatiformes", "coarsened", "stingrays are Dasyatidae/Urolophidae/Myliobatidae; smallest accepted clade holding them"),
    },
}  # fmt: skip

_SP = re.compile(
    r"^(?P<head>[A-Z][A-Za-z-]+(?: [a-z-]+)??)(?: (?:cf\.|aff\.))? ?(?:sp\.|spp\.|gen\.)(?: ?[A-Za-z0-9]+)?(?: \(.*\))?$"
)
_CF = re.compile(r"^(?P<head>[A-Z][A-Za-z-]+) (?:cf\.|aff\.) [a-z-]+(?: .*)?$")
_COMPLEX = re.compile(r"^(?P<head>[A-Z][a-z]+) [a-z-]+ [Cc]omplex$")


def reductions(label: str) -> tuple[str, str] | None:
    """The one deterministic reduction of a morphospecies-style label, or None."""
    for rx, why in (
        (_SP, "morphospecies -> named taxon"),
        (_CF, "cf./aff. identification -> genus"),
        (_COMPLEX, "species complex -> genus"),
    ):
        m = rx.match(label.strip())
        if m:
            head = m.group("head").strip()
            if head != label:
                return head, why
    return None


COMMON_TSV = "fathomnet-common-names.tsv"
COMMON_XW = "fathomnet-concepts"
_FIDELITY = {
    "exact": "exact",
    "broader": "coarsened",
    "related": "approximate",
    "unmapped": "unmappable",
}


def read_common(path: Path) -> list[dict[str, str]]:
    """Rows of the hand-confirmed common-name table (``concept`` .. ``note``); ``#`` lines skipped."""
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"), strict=False)) for ln in lines[1:]]


def verify_common(rows: list[dict[str, str]], client: WormsClient) -> dict[str, int]:
    """``scientific_name -> AphiaID`` once WoRMS confirms every row's id: the record carries that
    name and is accepted. Nothing is looked up by name here, so homonyms cannot pick the wrong one."""
    ids = {int(r["aphia_id"]) for r in rows if r["aphia_id"]}
    recs = client.records_by_ids(ids)
    out: dict[str, int] = {}
    for r in rows:
        if r["match_type"] not in _FIDELITY:
            raise SystemExit(f"{r['concept']}: unknown match_type {r['match_type']!r}")
        if r.get("target"):  # non-taxon axis: an NT_* node, no WoRMS id involved (checked in main)
            if r["match_type"] != "exact" or r["aphia_id"] or r["scientific_name"]:
                raise SystemExit(f"{r['concept']}: a target row is exact with no name or aphia_id")
            continue
        if not r["aphia_id"]:
            if r["match_type"] != "unmapped":
                raise SystemExit(f"{r['concept']}: {r['match_type']} row without an aphia_id")
            continue
        rec = recs.get(int(r["aphia_id"]))
        ok = (
            rec
            and rec["status"] == "accepted"
            and rec["scientificname"].lower() == r["scientific_name"].lower()
        )
        if not ok:
            raise SystemExit(
                f"{r['concept']}: WoRMS does not confirm {r['scientific_name']} = {r['aphia_id']}"
            )
        out[r["scientific_name"]] = int(r["aphia_id"])
    return out


class Snapshot:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = pq.read_table(path).to_pylist()
        self.reindex()

    def reindex(self) -> None:
        self.by_id = {int(r["aphia_id"]): r for r in self.rows}
        self.by_name: dict[str, list[dict]] = {}
        for r in self.rows:
            self.by_name.setdefault(r["scientific_name"].lower(), []).append(r)

    def lookup(self, name: str) -> int | None:
        """The accepted AphiaID for a name the snapshot already holds (unique rows only)."""
        hits = self.by_name.get(name.lower(), [])
        return int(hits[0]["accepted_aphia_id"]) if len(hits) == 1 else None


def choose(name: str, records: list[dict]) -> tuple[int | None, str]:
    """Pick a record for an exact name; ambiguity is reported, never resolved by guessing."""
    exact = [r for r in records if (r.get("scientificname") or "").lower() == name.lower()]
    if not exact:
        return None, "no exact WoRMS record"
    acc = [r for r in exact if r.get("status") == "accepted"]
    if len(acc) > 1:
        marine = [r for r in acc if r.get("isMarine") == 1]
        pref = [r for r in (marine or acc) if r.get("kingdom") in KINGDOMS]
        if len(pref) == 1:
            return int(pref[0]["AphiaID"]), f"{len(acc)} homonyms; the only marine one"
        return None, f"ambiguous: {len(acc)} accepted homonyms"
    if acc:
        return int(acc[0]["AphiaID"]), "accepted exact"
    valid = {int(r["valid_AphiaID"]) for r in exact if r.get("valid_AphiaID")}
    if len(valid) == 1:
        return int(next(iter(valid))), "unaccepted exact; follows valid_AphiaID"
    return None, "unaccepted exact without a single valid_AphiaID"


def resolve(
    names: set[str], snap: Snapshot, client: WormsClient
) -> dict[str, tuple[int | None, str]]:
    out: dict[str, tuple[int | None, str]] = {}
    miss: list[str] = []
    for n in sorted(names):
        hit = snap.lookup(n)
        if hit is not None:
            out[n] = (hit, "snapshot")
        else:
            miss.append(n)
    for n, recs in client.records_by_names(miss).items():
        out[n] = choose(n, recs)
    return out


def main(cache: Path, xw: str, dry: bool) -> None:
    meta = load_meta(ROOT)
    snap = Snapshot(taxonomy_dir(ROOT) / meta["snapshot"])
    client = WormsClient(cache)
    labels: dict[str, int] = {}
    srcs = []
    for p in sorted((taxonomy_dir(ROOT) / "vocab").glob("*.tsv")):
        head, _rows = read_vocab_rows(p)
        if head.get("crosswalk") == xw:
            _head, counts = read_vocab(p)
            srcs.append(p.stem)
            for lab, c in counts.items():
                labels[lab] = labels.get(lab, 0) + (c or 0)
    xw_path = ROOT / f"crosswalks/{xw}.yaml"
    doc = yaml.safe_load(xw_path.read_text()) if xw_path.exists() else None
    old = {e["source_label"]: e for e in (doc["crosswalks"][0]["edges"] if doc else [])}
    hand = dict(HAND.get(xw, {}))
    verified: dict[str, int] = {}
    nt: dict[str, tuple[str, str | None]] = {}  # label -> (NT_* node, note), the non-taxon axis
    common_path = taxonomy_dir(ROOT) / "vocab" / COMMON_TSV
    if xw == COMMON_XW and common_path.exists():
        common = read_common(common_path)
        verified = verify_common(common, client)
        for r in common:
            if r["concept"] not in labels:
                raise SystemExit(f"{r['concept']}: not a fathomnet label")
            if r.get("target"):
                nt[r["concept"]] = (r["target"], r["note"] or None)
                continue
            hand[r["concept"]] = (
                r["scientific_name"] or None,
                _FIDELITY[r["match_type"]],
                r["note"] or None,
            )
        # the table is authoritative for its labels: their generated edges are rebuilt from it
        old = {k: v for k, v in old.items() if k not in {r["concept"] for r in common}}
    todo = sorted(set(labels) - set(old))
    # pass 1: exact names (hand table first); pass 2: the reductions of what pass 1 missed
    first = {n for n in todo if n not in hand and n not in nt}
    got = resolve(
        first | {h[0] for h in hand.values() if h[0] and h[0] not in verified}, snap, client
    )
    got.update({n: (a, "common-name table, WoRMS-confirmed id") for n, a in verified.items()})
    red = {n: reductions(n) for n in first if got[n][0] is None}
    got.update(resolve({r[0] for r in red.values() if r}, snap, client))

    plan: dict[str, tuple[int | None, str, str | None]] = {}  # label -> (aphia, fidelity, note)
    for n in todo:
        if n in nt:
            plan[n] = (None, "exact", nt[n][1])
        elif n in hand:
            tgt, fid, note = hand[n]
            aph, why = got[tgt] if tgt else (None, "hand: no taxon")
            plan[n] = (aph, fid, note) if aph else (None, "unmappable", note or f"{tgt}: {why}")
        elif got[n][0] is not None:
            plan[n] = (got[n][0], "exact", None)
        elif red.get(n) and got[red[n][0]][0] is not None:
            plan[n] = (got[red[n][0]][0], "coarsened", f"{red[n][1]} ({red[n][0]})")
        else:
            why = got[n][1] + (
                f"; reduced name {red[n][0]!r}: {got[red[n][0]][1]}" if red.get(n) else ""
            )
            plan[n] = (None, "unmappable", f"no WoRMS-backed taxon for this label: {why}")

    # snapshot rows for the chosen AphiaIDs (existing rows are never rewritten)
    ids = {a for a, _f, _n in plan.values() if a}
    fresh = sorted(ids - set(snap.by_id))
    new_rows, _ = build_rows(client, ids=fresh, names=[], retrieved_at=TODAY)
    known = {int(r["aphia_id"]) for r in snap.rows}
    added = [r for r in new_rows if int(r["aphia_id"]) not in known]
    chosen_by = {a: n for n, (a, _f, _x) in plan.items() if a and n in first}
    for r in added:
        if int(r["aphia_id"]) in chosen_by and r["origin"] == "id":
            r["query"], r["origin"] = chosen_by[int(r["aphia_id"])], "name"
    merged = sorted(snap.rows + added, key=lambda r: int(r["aphia_id"]))
    snap.rows = merged
    snap.reindex()

    def acc(a: int) -> int:
        return int(snap.by_id[a]["accepted_aphia_id"])

    def lineage(a: int) -> list[tuple[str, str, int]]:
        return [tuple(x) for x in json.loads(snap.by_id[acc(a)]["lineage"])]

    def ok(a: int) -> bool:
        r = snap.by_id.get(a)
        return bool(r) and r["status"] == "accepted" and int(r["accepted_aphia_id"]) == a

    # rs-taxa-v1 nodes
    tx_path = ROOT / "schemas/rs-taxa-v1.yaml"
    tx = yaml.safe_load(tx_path.read_text())
    nodes = tx["schemas"][0]["nodes"]
    have = {n["id"] for n in nodes}
    bad_nt = {t for t, _n in nt.values()} - {n["id"] for n in nodes if n.get("non_taxon")}
    if bad_nt:
        raise SystemExit(f"not non-taxon nodes in rs-taxa-v1: {sorted(bad_nt)}")
    existing_ids = {int(n["worms_aphia_id"]) for n in nodes if n.get("worms_aphia_id")}
    # an ancestor whose insertion would re-parent an existing node is not added (MINOR stays MINOR)
    under = {lid for a in existing_ids for _r, _n, lid in lineage(a) if lid != a}
    targets = sorted({acc(a) for a in ids})
    wanted: dict[int, None] = {}
    problems = []
    for a in targets:
        if not ok(a):
            problems.append(f"unaccepted target {a}")
            continue
        wanted[a] = None
        for rank, _n, lid in lineage(a)[:-1]:
            if rank in KEEP_RANKS and ok(lid) and lid not in under:
                wanted[lid] = None
    keep = {a for a in wanted if f"A{a}" not in have}
    present = existing_ids | keep
    new_nodes = []
    for a in sorted(keep, key=lambda x: (len(lineage(x)), x)):
        parent = next(
            (f"A{lid}" for _r, _n, lid in reversed(lineage(a)[:-1]) if lid in present), None
        )
        r = snap.by_id[a]
        new_nodes.append({
            "id": f"A{a}", "name": r["accepted_name"], **({"parent": parent} if parent else {}),
            "axis": "taxon", "worms_aphia_id": a, "worms_scientificname": r["accepted_name"],
            "worms_rank": r["rank"], "worms_status": "accepted", "worms_checked_on": TODAY,
        })  # fmt: skip
    # edges
    edges = [e for e in (doc["crosswalks"][0]["edges"] if doc else []) if e["source_label"] in old]
    for n in todo:
        a, fid, note = plan[n]
        e: dict = {"source_label": n}
        if n in nt:
            e["targets"] = {"taxon": nt[n][0]}
        if a:
            e["targets"] = {"taxon": f"A{acc(a)}"}
        e["fidelity"] = fid
        if a and not note and acc(a) != a:
            note = f"WoRMS: {n} -> accepted {snap.by_id[acc(a)]['accepted_name']} ({acc(a)})"
        if a and not note and snap.by_id[acc(a)]["accepted_name"] != n and plan[n][1] == "exact":
            note = f"WoRMS: {n} -> accepted {snap.by_id[acc(a)]['accepted_name']} ({acc(a)})"
        if note:
            e["note"] = note
        edges.append(e)
    edges.sort(key=lambda e: e["source_label"])
    nodes_out = sorted(
        [*nodes, *new_nodes],
        key=lambda n: (n.get("worms_aphia_id") is None, len(lineage(n["worms_aphia_id"])) if n.get("worms_aphia_id") else 0, n.get("worms_aphia_id") or 0),
    )  # fmt: skip
    fid_count = {
        k: sum(e["fidelity"] == k for e in edges)
        for k in ("exact", "coarsened", "approximate", "unmappable")
    }
    print(
        f"labels {len(labels)} old edges {len(old)} new {len(todo)}; new snapshot rows {len(added)}; new nodes {len(new_nodes)}"
    )
    print(f"edges now {len(edges)} {fid_count}; problems {problems}")
    print("worms cache files:", len(list(cache.glob("*.json"))))
    if dry:
        return
    write_parquet(merged, snap.path)
    hdr = "# GENERATED by scripts/taxonomy_codegen.py - do not edit by hand; rerun the codegen.\n"
    tx["schemas"][0]["nodes"] = nodes_out
    tx_path.write_text(
        hdr + yaml.safe_dump(tx, sort_keys=False, width=200, default_flow_style=None)
    )
    cw = doc["crosswalks"][0] if doc else {
        "id": xw, "source_schema": "dataset-native", "target_schema": "rs-taxa-v1",
    }  # fmt: skip
    cw = {
        **cw,
        "description": f"Crosswalk for {', '.join(sorted(srcs))}; observed vocabulary in "
        "registry/taxonomy/vocab/. Generated by scripts/taxonomy_codegen.py (WP-7) and "
        "scripts/taxonomy_fauna_codegen.py (WP-U8a).",
        "edges": edges,
    }
    cw = {k: cw[k] for k in ("id", "source_schema", "target_schema", "description", "edges")}
    xw_path.write_text(
        hdr
        + yaml.safe_dump({"crosswalks": [cw]}, sort_keys=False, width=200, default_flow_style=None)
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], "--dry" in sys.argv)
