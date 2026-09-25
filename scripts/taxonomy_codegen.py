"""WP-7 step 2: pinned WoRMS snapshot + rs-taxa-v1 backbone + crosswalks (codegen).

Reads registry/taxonomy/vocab/*.tsv (step 1) and the registry, resolves every AphiaID
and scientific name against WoRMS (cached, <= 2 req/s), then writes
registry/taxonomy/worms-<day>.parquet, registry/schemas/rs-taxa-v1.yaml,
registry/crosswalks/<vocab>.yaml and the WP-7 additions to rs-benthic-v1.
Usage: python scripts/taxonomy_codegen.py <worms_cache_dir> [--offline]
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marinedata.registry import Registry
from marinedata.worms_snapshot import WormsClient, build_rows, write_parquet

DAY = date(2026, 9, 25)
ROOT = Path(__file__).resolve().parents[1] / "registry"
VOCAB = ROOT / "taxonomy/vocab"
KEEP_RANKS = {"Superdomain", "Kingdom", "Phylum", "Subphylum", "Class", "Order", "Family", "Genus"}

# ── rs-taxa-v1 (detection / open-vocabulary sources) ──────────────────────────────
NT = {  # non-taxon nodes of the backbone
    "NT_DIVER": ("Diver / human", "human: people are outside WoRMS scope"),
    "NT_EQUIPMENT": ("Equipment (ROV, robot, gear)", "equipment: man-made instrument"),
    "NT_DEBRIS": ("Marine debris / trash", "debris: anthropogenic litter"),
    "NT_WRECK": ("Wreck / ruin", "habitat: artificial structure"),
    "NT_REEF": ("Reef structure", "habitat: reef framework mixes biota and substrate"),
    "NT_SEAFLOOR": ("Sea floor", "substrate: unconsolidated or bare bottom"),
    "NT_UNKNOWN": ("Unknown", "unknown: annotator could not identify"),
}
FISH_CLASSES = ("Actinopterygii", "Elasmobranchii", "Holocephali", "Myxini", "Petromyzonti")
FISH_NOTE = (
    "fish is paraphyletic and WoRMS Pisces is unaccepted, so fish land on the smallest ac"
    "cepted clade holding all of them; includes: [Actinopterygii, Elasmobranchii, "
    "Holocephali, Myxini, Petromyzonti]"
)

COMMON = {  # label -> (WoRMS name | NT id | None=unmappable, fidelity, note)
    "fish": ("Vertebrata", "coarsened", FISH_NOTE),
    "animal_fish": ("Vertebrata", "coarsened", FISH_NOTE),
    "turtle": ("Testudines", "exact", None),
    "jellyfish": (
        "Medusozoa",
        "coarsened",
        "jellyfish are medusozoans; ctenophores would be missed",
    ),
    "holothurian": ("Holothuroidea", "exact", None),
    "starfish": ("Asteroidea", "exact", None),
    "animal_starfish": ("Asteroidea", "exact", None),
    "echinus": ("Echinoidea", "exact", "RUOD 'echinus' = sea urchins"),
    "scallop": ("Pectinidae", "exact", None),
    "cuttlefish": ("Sepiida", "exact", None),
    "corals": (
        "Cnidaria",
        "coarsened",
        "coral is polyphyletic (Anthozoa + hydrozoan fire corals and stylasterids); smallest "
        "accepted clade",
    ),
    "coral": (
        "Cnidaria",
        "coarsened",
        "coral is polyphyletic (Anthozoa + hydrozoan fire corals and stylasterids); smallest "
        "accepted clade",
    ),
    "sponge": ("Porifera", "exact", None),
    "crinoid": ("Crinoidea", "exact", None),
    "aquatic plants": ("Plantae", "approximate", "includes macroalgae; brown algae are Chromista"),
    "plant": ("Plantae", "approximate", "TrashCan 'plant' = marine vegetation incl. algae"),
    "animal_shells": ("Mollusca", "approximate", "shells = bivalves and gastropods mostly"),
    "animal_crab": ("Brachyura", "exact", None),
    "animal_eel": ("Anguilliformes", "exact", None),
    "animal_etc": ("Animalia", "coarsened", "any other animal"),
    "bio": ("Biota", "coarsened", "Trash-ICRA19 'bio' = any biological material"),
    "diver": ("NT_DIVER", "exact", None),
    "Diver": ("NT_DIVER", "exact", None),
    "human divers": ("NT_DIVER", "exact", None),
    "rov": ("NT_EQUIPMENT", "exact", None),
    "robots": ("NT_EQUIPMENT", "exact", None),
    "reefs": ("NT_REEF", "exact", None),
    "sea-floor": ("NT_SEAFLOOR", "exact", None),
    "wrecks/ruins": ("NT_WRECK", "exact", None),
    "plastic": ("NT_DEBRIS", "coarsened", "material detail kept only in the source label"),
    "foreground": (None, "unmappable", "class-agnostic mask set; carries no class"),
    "garbage": ("NT_DEBRIS", "exact", None),
    "human": ("NT_DIVER", "exact", None),
    "mollusk": ("Mollusca", "exact", None),
    "plants": ("Plantae", "approximate", "includes macroalgae"),
    "reptiles": ("Reptilia", "exact", None),
    "ruins": ("NT_WRECK", "exact", None),
    "trash_branch": ("NT_DEBRIS", "approximate", "natural wood debris, not litter"),
    # FathomNet FGVC23 concepts that are not bare WoRMS names
    "yellow ruffled sponge": ("Porifera", "coarsened", "common name only -> phylum"),
    "Aeolidiidae sp. 1": ("Aeolidiidae", "coarsened", "morphospecies -> family"),
    "Bivalve": ("Bivalvia", "exact", None),
    "Bryozoan": ("Bryozoa", "exact", None),
    "Farrea truncata complex": ("Farrea", "coarsened", "species complex -> genus"),
    "Funiculina-Halipteris complex": ("Pennatulacea", "coarsened", "two sea-pen genera -> order"),
    "Hydrocoral": ("Stylasteridae", "approximate", "deep-sea hydrocorals are stylasterids"),
    "Lyssacinosida sp. 1": ("Lyssacinosida", "coarsened", "morphospecies -> order"),
    "Neptunea-Buccinum Complex": ("Buccinidae", "coarsened", "two genera -> family"),
    "Oneirophanta mutabilis complex": ("Oneirophanta", "coarsened", "species complex -> genus"),
    "Pandalus ampla": ("Pandalus amplus", "exact", "spelling: WoRMS epithet is amplus"),
    "Paralomis cf. papillata": ("Paralomis", "coarsened", "cf. identification -> genus"),
}
EXACT_BY_CONSTRUCTION = {
    "obsea-fish": "the labels are WoRMS scientific names (21 species, 1 family, 1 of them an "
    "unaccepted spelling: Oblada melanura -> Oblada melanurus) plus Diver -> NT_DIVER; each "
    "resolves to its own accepted AphiaID, so no edge loses or invents anything",
}
NODE_NOTES = {"Vertebrata": "Target of paraphyletic 'fish' labels (coarsened). " + FISH_NOTE}
"""Notes carried on generated rs-taxa-v1 nodes so downstream users can narrow."""
PREFER = {"Halophila": ("Plantae",), "Turbinaria": ("Animalia",)}
"""Homonyms: the kingdom each vocabulary means (seagrass Halophila, coral Turbinaria)."""
# ── rs-benthic-v1 (point / patch cover sources) ───────────────────────────────────
BENTHIC = {  # lowercased label or CoralNet name -> (taxon, form, fidelity, note)
    "turf": ("TA", None, "exact", None),
    "turf growing on hard substrate": ("TA", None, "exact", "substrate detail dropped"),
    "turf growing on rubble": ("TA", None, "exact", "substrate detail dropped"),
    "cca": ("CCA", None, "exact", None),
    "cca growing on hard substrate": ("CCA", None, "exact", None),
    "cca growing on rubble": ("CCA", None, "exact", None),
    "coralline alga (crustose)": ("CCA", None, "exact", None),
    "macro": ("SW", None, "exact", None),
    "macroalgae": ("SW", None, "exact", None),
    "macroalga": ("SW", None, "exact", None),
    "red macroalgae": ("SW", None, "coarsened", "red/brown/green macroalgae roll up to SW"),
    "brown macroalgae": ("SW", None, "coarsened", None),
    "green macroalgae": ("SW", None, "coarsened", None),
    "upright macroalgae": ("SW", None, "coarsened", None),
    "encrusting macroalgae": ("SW", None, "coarsened", None),
    "blue-green macroalga": ("@Cyanobacteria", None, "exact", "cyanobacterial mats, not algae"),
    "sand": ("SD", None, "exact", None),
    "fine sediment": ("SI", None, "exact", None),
    "hard substrate": ("RK", None, "exact", None),
    "rubble substrate": ("RB", None, "exact", None),
    "bare substrate": ("ABIOTIC", None, "coarsened", "bare = uncolonised, rock or sand"),
    "sediment / sand / rubble": ("ABIOTIC", None, "coarsened", "SD+RB+SI combined"),
    "shadow": ("TWS", None, "exact", None),
    "wand": ("TWS", None, "exact", None),
    "tape": ("TWS", None, "exact", None),
    "tape / wand": ("TWS", None, "exact", None),
    "transect hardware": ("TL", None, "exact", None),
    "unclassified/unknown": ("UNKNOWN", None, "exact", None),
    "unclassified / unknown": ("UNKNOWN", None, "exact", None),
    "unclear": ("UNKNOWN", None, "exact", None),
    "mobile fauna": (
        "OTHER_FAUNA",
        None,
        "approximate",
        "mobile animals; OTHER_FAUNA excludes cnidarians/sponges",
    ),
    "sessile invertebrate (non-coral)": (
        "BIOTIC",
        None,
        "coarsened",
        "sponges, tunicates, zoanthids ... no single node",
    ),
    "octocoral": ("SC", None, "exact", None),
    "soft coral": ("SC", None, "exact", None),
    "soft": ("SC", None, "exact", None),
    "sponge": ("SP", None, "exact", None),
    "sponges": ("SP", None, "exact", None),
    "zoanthid": ("ZO", None, "exact", None),
    "corallimorph": ("@Corallimorpharia", None, "exact", None),
    "tunicate": ("@Ascidiacea", None, "exact", None),
    "hard/stony coral": ("HC", None, "exact", None),
    "coral": ("HC", None, "exact", None),
    "other scleractinians": (
        "HC",
        None,
        "coarsened",
        "scleractinians outside the source's genus list",
    ),
    "encrusting hard coral": ("HC", "CE", "exact", None),
    "foliose hard coral": ("HC", "CF", "exact", None),
    "massive hard coral": ("HC", "CM", "exact", None),
    "free-living hard coral": ("HC", "CMR", "approximate", "free-living ~ mushroom form"),
    "dictyopteris/dictyota spp": ("@Dictyotaceae", None, "coarsened", "two genera, one family"),
    "goniopora/alveopora spp": ("@Poritidae", None, "coarsened", "two genera, one family"),
    "seagrass": ("SG", None, "exact", None),
    "bivalve": ("CLAM", None, "exact", "CLAM is Bivalvia"),
    "bryozoan": ("@Bryozoa", None, "exact", None),
    "hydrocoral": (
        "@Hydrozoa",
        None,
        "coarsened",
        "hydrocorals (Millepora, stylasterids) -> class",
    ),
    "all other": (
        None,
        None,
        "unmappable",
        "residual 'all other' bucket: biotic or abiotic, no taxonomic content",
    ),
    "acrop": ("@Acropora", None, "exact", None),
    "pavon": ("@Pavona", None, "exact", None),
    "monti": ("@Montipora", None, "exact", None),
    "pocill": ("@Pocillopora", None, "exact", None),
    "porit": ("@Porites", None, "exact", None),
    "strappy": (
        "SG",
        "SG_STRAP",
        "coarsened",
        "strap-like: Zostera/Halodule/Cymodocea/Syringodium",
    ),
    "ferny": ("@Halophila", "SG_FERN", "coarsened", "fern-like = Halophila spinulosa"),
    "rounded": ("@Halophila", "SG_ROUND", "coarsened", "rounded = Halophila ovalis group"),
    "background": (
        "ABIOTIC",
        None,
        "approximate",
        "negative class: substrate and/or water, no seagrass",
    ),
    "substrate": ("ABIOTIC", None, "coarsened", None),
    "water": ("WC", None, "exact", None),
}
MORPH = {
    "branching": "CB",
    "encrusting": "CE",
    "foliose": "CF",
    "massive": "CM",
    "submassive": "CS",
    "tabulate": "CT",
    "digitate": "CD",
    "mushroom": "CMR",
}
NON_TAXON_BENTHIC = {
    "ABIOTIC": "grouping: all non-living substrate",
    "TRANSITION": "state: dead-coral transition states",
    "NON_BENTHIC": "grouping: imaging artefacts, equipment and non-benthic content",
    "ALGAE": "functional-group: algae are polyphyletic (Rhodophyta, Chlorophyta, Ochrophyta)",
    "OTHER_FAUNA": "grouping: residual animals outside Cnidaria and Porifera; roll up via children",
    "TA": "functional-group: turf is a multi-phylum algal assemblage",
    "SW": "functional-group: macroalgae span three phyla",
    "NIA": "functional-group: nutrient-indicator algae",
    "OT": "unknown: other biota not identified further",
    "RB": "substrate: rubble",
    "SD": "substrate: sand",
    "SI": "substrate: silt / mud",
    "RK": "substrate: rock / pavement",
    "NL": "substrate: other non-living",
    "TRASH": "debris: anthropogenic litter",
    "DC": "state: recently dead coral skeleton",
    "DCA": "state: dead coral colonised by algae",
    "TWS": "equipment: tape, wand or shadow",
    "TL": "equipment: transect line",
    "DIV": "human: divers are outside WoRMS scope",
    "WC": "water: water column / background",
    "DRK": "unknown: dark or unreadable",
    "SCL": "equipment: scale reference",
    "UNKNOWN": "unknown: annotator could not identify",
}


def read_vocab(path: Path) -> tuple[dict, list[tuple[str, int | None, str]]]:
    head, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            k, _, v = line[2:].partition(": ")
            head[k] = v
        elif line and not line.startswith("label\t"):
            label, count, desc = ([*line.split("\t"), "", ""])[:3]
            rows.append((label, int(count) if count else None, desc))
    return head, rows


def coralnet_name(label: str, desc: str) -> str:
    if desc.startswith("CRED-"):
        return desc.split(" | ")[0].removeprefix("CRED-")
    return desc or label


def benthic_rule(label: str, desc: str) -> tuple:
    name = coralnet_name(label, desc)
    for key in (name.lower(), label.lower()):
        if key in BENTHIC:
            return BENTHIC[key]
    m = re.match(r"^([A-Z][a-z]+)(?: spp| sp\.?)?(?:_(\w+))?$", name)
    if m:
        form = MORPH.get(m.group(2) or "")
        return (f"@{m.group(1)}", form, "exact", None)
    return (None, None, "unmappable", f"NO RULE for {name!r}")


def main(cache: Path, offline: bool) -> None:
    reg = Registry.load(ROOT)
    vocabs = {p.stem: read_vocab(p) for p in sorted(VOCAB.glob("*.tsv"))}
    benthic_xw = {"coralnet-noaa-pifsc", "plc-beijbom2015", "mlc-moorea", "deepseagrass"}
    names: set[str] = set()
    plans: dict[str, dict[str, tuple]] = {}
    for _sid, (head, rows) in vocabs.items():
        xw = head["crosswalk"]
        for label, _count, desc in rows:
            if xw in benthic_xw:
                plan = benthic_rule(label, desc)
            elif label.startswith("trash_") and label not in COMMON:
                plan = (
                    "NT_DEBRIS",
                    "coarsened",
                    "item/material detail kept only in the source label",
                )
            else:
                plan = COMMON.get(label, (label, "exact", None))
            plans.setdefault(xw, {})[label] = plan
            tgt = plan[0]
            if tgt and tgt.startswith("@"):
                names.add(tgt[1:])
            elif tgt and xw not in benthic_xw and not tgt.startswith("NT_"):
                names.add(tgt)
    plans["usis-uiis"]["foreground"] = COMMON["foreground"]
    names |= {"Halophila", "Biota", *FISH_CLASSES}
    hand = [s for s in reg.schemas if s.id != "rs-taxa-v1"]  # never seed from our own output
    ids = {n.worms_aphia_id for s in hand for n in s.nodes if n.worms_aphia_id}
    ids |= {n.worms_aphia_id_asserted for s in hand for n in s.nodes if n.worms_aphia_id_asserted}
    client = WormsClient(cache)
    rows, choices = build_rows(client, ids=ids, names=names, retrieved_at=DAY, prefer=PREFER)
    by_id = {r["aphia_id"]: r for r in rows}
    name_to_acc = {
        c.name: by_id[c.aphia_id]["accepted_aphia_id"] for c in choices if c.aphia_id in by_id
    }
    problems = [f"name {c.name!r}: {c.reason}" for c in choices if c.aphia_id not in by_id]
    problems += [f"ambiguous {c.name!r}: {c.reason}" for c in choices if "homonym" in c.reason]
    for s in reg.schemas:
        for n in s.nodes:
            r = by_id.get(n.worms_aphia_id or -1)
            if n.worms_aphia_id and (
                r is None
                or r["status"] != "accepted"
                or r["scientific_name"] != n.worms_scientificname
                or r["rank"] != n.worms_rank
            ):
                problems.append(
                    f"drift {s.id}:{n.id} {n.worms_aphia_id} "
                    f"{n.worms_scientificname}/{n.worms_rank} -> "
                    f"{r and (r['scientific_name'], r['rank'])} "
                    f"{r and (r['status'], r['accepted_aphia_id'])}"
                )
    snap = ROOT / f"taxonomy/worms-{DAY.isoformat()}.parquet"
    write_parquet(rows, snap)
    acc = lambda a: by_id[a]["accepted_aphia_id"]  # noqa: E731
    lineage = lambda a: [tuple(x) for x in json.loads(by_id[acc(a)]["lineage"])]  # noqa: E731

    # rs-taxa-v1: every canonical AphiaID + every open-vocab target + kept lineage ranks
    wanted = {acc(a) for a in ids if a in by_id} | {
        name_to_acc[n] for n in names if n in name_to_acc
    }

    def ok(a: int) -> bool:
        """Only accepted WoRMS records become nodes (test_harmonize holds this strictly)."""
        return by_id[a]["status"] == "accepted" and by_id[a]["accepted_aphia_id"] == a

    problems += [
        f"unaccepted target {by_id[a]['scientific_name']} ({a})" for a in wanted if not ok(a)
    ]
    keep = {a for a in wanted if ok(a)}
    for a in wanted:
        keep |= {
            lid for rank, _n, lid in lineage(a) if rank in KEEP_RANKS and lid in by_id and ok(lid)
        }
    nodes = []
    for a in sorted(keep, key=lambda x: (len(lineage(x)), x)):
        chain = [lid for _r, _n, lid in lineage(a)][:-1]
        parent = next((f"A{lid}" for lid in reversed(chain) if lid in keep), None)
        r = by_id[a]
        nodes.append(
            {
                "id": f"A{a}",
                "name": r["accepted_name"],
                **({"parent": parent} if parent else {}),
                "axis": "taxon",
                "worms_aphia_id": a,
                "worms_scientificname": r["accepted_name"],
                "worms_rank": r["rank"],
                "worms_status": "accepted",
                "worms_checked_on": DAY,
                **(
                    {"notes": NODE_NOTES[r["accepted_name"]]}
                    if r["accepted_name"] in NODE_NOTES
                    else {}
                ),
            }
        )
    nodes += [
        {"id": k, "name": v[0], "axis": "taxon", "non_taxon": True, "non_taxon_reason": v[1]}
        for k, v in NT.items()
    ]
    doc = {
        "schemas": [
            {
                "id": "rs-taxa-v1",
                "name": "RS-Taxa v1 (WoRMS backbone)",
                "canonical": True,
                "description": "Generated by scripts/taxonomy_codegen.py from the pinned "
                "WoRMS snapshot "
                f"worms-{DAY.isoformat()}.parquet. Node id A<AphiaID>; parents follow WoRMS "
                "classification through the kept ranks. Every AphiaID in any canonical schema "
                "is here, "
                "so rs-benthic-v1 and rs-fauna-v1 join this tree by AphiaID.",
                "axes": ["taxon"],
                "nodes": nodes,
            }
        ]
    }
    hdr = "# GENERATED by scripts/taxonomy_codegen.py - do not edit by hand; rerun the codegen.\n"
    (ROOT / "schemas/rs-taxa-v1.yaml").write_text(
        hdr + yaml.safe_dump(doc, sort_keys=False, width=200, default_flow_style=None)
    )

    # rs-benthic-v1 additions: genus/family nodes, UNKNOWN root, seagrass forms, non_taxon flags
    benthic = reg.label_schema("rs-benthic-v1")
    have = {n.worms_aphia_id: n.id for n in benthic.nodes if n.worms_aphia_id}
    text = (ROOT / "schemas/rs-benthic-v1.yaml").read_text()
    add, resolved = [], {}
    lineage_ids = lambda a: {lid for _r, _n, lid in lineage(a)}  # noqa: E731
    for n in sorted(
        {
            p[0][1:]
            for xw in benthic_xw
            for p in plans.get(xw, {}).values()
            if p[0] and p[0].startswith("@")
        }
    ):
        if n not in name_to_acc:
            problems.append(f"benthic name unresolved {n!r}")
            continue
        a = name_to_acc[n]
        if a in have:
            resolved[n] = have[a]
            continue
        li, r = lineage_ids(a), by_id[a]
        if 1363 in li:
            parent, pre = "HC", "HC_"
        elif 1341 in li:
            parent, pre = "SC", "SC_"
        elif 1267 in li:
            parent, pre = "CNIDARIA", "CN_"
        elif 2 in li:
            parent, pre = "OTHER_FAUNA", "INV_"
        elif r["accepted_name"] == "Cyanobacteria" or 146537 in li:
            parent, pre = "BIOTIC", ""
        elif 182757 in li:
            parent, pre = "SG", "SG_"
        elif r["accepted_name"] == "Peyssonnelia":
            parent, pre = "ALGAE", "ALG_"
        else:
            parent, pre = "SW", "ALG_"
        nid = pre + re.sub(r"[^A-Z]", "", r["accepted_name"].upper())
        resolved[n] = nid
        have[a] = nid
        add.append(
            f"      - {{ id: {nid}, name: {r['accepted_name']}, parent: {parent}, axis: "
            f"taxon, worms_aphia_id: {a}, "
            f"worms_scientificname: {r['accepted_name']}, worms_rank: {r['rank']}, "
            f"worms_status: accepted, "
            f"worms_checked_on: {DAY.isoformat()} }}"
        )
    if 1 in by_id and "worms_aphia_id: 1," not in text:
        text = text.replace(
            "- { id: BIOTIC,       name: Biotic,        axis: taxon }",
            "- { id: BIOTIC,       name: Biotic,        axis: taxon, worms_aphia_id: 1, "
            "worms_scientificname: Biota, "
            f"worms_rank: {by_id[1]['rank']}, worms_status: accepted, worms_checked_on: "
            f"{DAY.isoformat()} }}",
        )
    for nid, reason in NON_TAXON_BENTHIC.items():
        pat = re.compile(r"(- \{ id: " + re.escape(nid) + r",[^}]*?)(\s*\})")
        if nid != "UNKNOWN" and "non_taxon" not in (
            pat.search(text).group(1) if pat.search(text) else "x"
        ):
            text, k = pat.subn(
                lambda m, r=reason: (
                    f'{m.group(1)}, non_taxon: true, non_taxon_reason: "{r}"{m.group(2)}'
                ),
                text,
                count=1,
            )
            if not k:
                problems.append(f"non_taxon flag not applied to {nid} (multi-line node?)")
    if "id: UNKNOWN," not in text:
        add.append(
            f"      - {{ id: UNKNOWN, name: Unknown / unclassified, axis: taxon, non_taxon: "
            f'true, non_taxon_reason: "{NON_TAXON_BENTHIC["UNKNOWN"]}" }}'
        )
        for fid, fname in [
            ("SG_STRAP", "Strap-like leaves"),
            ("SG_FERN", "Fern-like leaves"),
            ("SG_ROUND", "Rounded leaves"),
        ]:
            add.append(
                f'      - {{ id: {fid}, name: {fname}, axis: form, notes: "Seagrass leaf '
                f'morphology (DeepSeagrass superclasses)" }}'
            )
    if add:
        text = (
            text.rstrip("\n")
            + "\n\n      # ── WP-7 additions (scripts/taxonomy_codegen.py, WoRMS snapshot "
            + DAY.isoformat()
            + ") ──\n"
            + "\n".join(add)
            + "\n"
        )
    (ROOT / "schemas/rs-benthic-v1.yaml").write_text(text)

    # crosswalks
    for xw, plan in sorted(plans.items()):
        target = "rs-benthic-v1" if xw in benthic_xw else "rs-taxa-v1"
        edges = []
        for label, p in sorted(plan.items()):
            if xw in benthic_xw:
                tax, form, fid, note = p
                if tax and tax.startswith("@"):
                    tax = resolved.get(tax[1:])
                    if tax is None:
                        fid, note = "unmappable", f"WoRMS could not resolve {p[0][1:]!r}"
                tg = (
                    {k: v for k, v in (("taxon", tax), ("form", form)) if v}
                    if fid != "unmappable"
                    else {}
                )
            else:
                name, fid, note = p
                if name is None:
                    tg = {}
                elif name.startswith("NT_"):
                    tg = {"taxon": name}
                elif name in name_to_acc:
                    a = name_to_acc[name]
                    tg = {"taxon": f"A{a}"}
                    if by_id[a]["accepted_name"] != name and not note:
                        note = f"WoRMS: {name} -> accepted {by_id[a]['accepted_name']} ({a})"
                else:
                    tg, fid, note = {}, "unmappable", f"no WoRMS record for {name!r}"
            if fid != "unmappable" and not tg:
                fid = "unmappable"
            e = {"source_label": label, "targets": tg, "fidelity": fid}
            if not tg:
                e.pop("targets")
            if note:
                e["note"] = note
            if fid == "unmappable" and "note" not in e:
                problems.append(f"unmappable without reason {xw}:{label}")
            edges.append(e)
        srcs = sorted(s for s, (h, _r) in vocabs.items() if h["crosswalk"] == xw)
        doc = {
            "crosswalks": [
                {
                    "id": xw,
                    "source_schema": "dataset-native",
                    "target_schema": target,
                    "description": f"WP-7 crosswalk for {', '.join(srcs)}; observed vocabulary in "
                    "registry/taxonomy/vocab/. Generated by scripts/taxonomy_codegen.py.",
                    **(
                        {"exact_by_construction": EXACT_BY_CONSTRUCTION[xw]}
                        if xw in EXACT_BY_CONSTRUCTION
                        else {}
                    ),
                    "edges": edges,
                }
            ]
        }
        (ROOT / f"crosswalks/{xw}.yaml").write_text(
            hdr + yaml.safe_dump(doc, sort_keys=False, width=200, default_flow_style=None)
        )
    print(
        f"snapshot rows {len(rows)}; rs-taxa nodes {len(nodes)}; benthic additions "
        f"{len(add)}; problems {len(problems)}"
    )
    print("\n".join(problems))


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--offline" in sys.argv)
