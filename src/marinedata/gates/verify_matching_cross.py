"""Cross-source checks 4-7 of the label-matching verifier, and the report assembly.

4 agreement   which (source, native label) pairs feed each rs-benthic-v1 node; exact taxon edges
              whose native label shares no word with the node are listed for review
5 legacy      Coralscapes-95 against its 39-class legacy column: equal or refined targets, equal
              conditions (review list)
6 condition   which sources carry the condition axis, and whether a health word in a label agrees
              with the condition node it maps to (review list)
7 ignore      the ignore ids and every unmappable label of the published sources map to 255 in every
              training scheme (hard)

Checks 1-3 are in :mod:`verify_matching`. :func:`run` returns one JSON-ready report.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from ..schema import Axis, Fidelity, LabelSchema
from . import verify_matching as vm
from .verify_matching import TARGET, Context, UnitResult, capped, note_of

RELEASE = "1.2.0"
NEW_NODES = ("CN_ANTIPATHARIA", "SC_ALCYONIIDAE", "SC_NEPHTHEIDAE")
LEGACY = {"coralscapes-95": "coralscapes-39"}  # V2 crosswalk -> the 39-class crosswalk it extends
LEGACY_ALIASES = {"hard substrate": "unknown hard substrate"}  # V2 name -> V1 name
LEGACY_TAG = re.compile(r"^\[legacy:\s*([^\]]+)\]")
HEALTH = {
    "alive": "healthy", "healthy": "healthy", "bleached": "bleached", "bleaching": "bleached",
    "pale": "pale", "paling": "pale", "diseased": "diseased", "disease": "diseased",
    "dead": "dead", "unhealthy": "unhealthy",
}  # fmt: skip
STOP = frozenset({"a", "an", "and", "the", "of", "with", "on", "in", "to", "or", "sp", "spp"})
FIDELITY_KEYS = ("exact", "coarsened", "approximate", "narrower", "ignored")


def tokens(text: str | None) -> frozenset[str]:
    """Lower-case words of a label, with singular forms, minus stop words."""
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if word in STOP:
            continue
        out.add(word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            out.add(word[:-1])
            if word.endswith("es"):
                out.add(word[:-2])
    return frozenset(out)


def code_like(label: str) -> bool:
    """A bare code or numeric id: no run of four letters, so it cannot be compared with a name."""
    return re.search(r"[A-Za-z]{4,}", label) is None


def health_words(text: str | None) -> frozenset[str]:
    return frozenset(HEALTH[t] for t in tokens(text) if t in HEALTH)


def families(schema: LabelSchema) -> dict[str, str]:
    """Node id -> coverage-matrix family: the child of the single taxon root, else the root."""
    by_id = {n.id: n for n in schema.nodes}
    roots = [n for n in schema.nodes if n.parent is None and n.axis is Axis.TAXON]
    out: dict[str, str] = {}
    for node in schema.nodes:
        if node.axis is Axis.FORM:
            out[node.id] = "Growth form"
        elif node.axis is Axis.CONDITION:
            out[node.id] = "Condition"
        else:
            chain = lineage(node.id, by_id)
            out[node.id] = by_id[
                chain[-2] if len(roots) == 1 and len(chain) > 1 else chain[-1]
            ].name
    return out


def lineage(node_id: str, by_id: dict[str, Any]) -> list[str]:
    chain = [node_id]
    while chain[-1] in by_id and by_id[chain[-1]].parent:
        chain.append(by_id[chain[-1]].parent)
    return chain


# ── check 4 ──────────────────────────────────────────────────────────────────────


def check_agreement(results: list[UnitResult], ctx: Context) -> dict[str, Any]:
    target = ctx.schemas[TARGET]
    fam = families(target)
    by_id = {n.id: n for n in target.nodes}
    words = {
        n.id: tokens(
            " ".join([n.name, n.id.replace("_", " "), *n.aliases, n.worms_scientificname or ""])
        )
        for n in target.nodes
    }
    feeds: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    flagged: list[str] = []
    per_source: Counter[str] = Counter()
    coded: Counter[str] = Counter()
    for res in results:
        sid = res.record["id"]
        if res.record["target_schema"] != TARGET:
            continue
        for label, edge in res.pairs:
            for axis, ident in edge.targets.items():
                feeds[ident][sid].add(label)
                exact_taxon = edge.fidelity is Fidelity.EXACT and axis is Axis.TAXON
                if exact_taxon and ident in words and not tokens(label) & words[ident]:
                    if code_like(label):
                        coded[sid] += 1
                    else:
                        flagged.append(f"{sid}: {label} -> {ident}")
                        per_source[sid] += 1
    nodes = {
        nid: {
            "axis": by_id[nid].axis.value,
            "family": fam[nid],
            "sources": {s: len(v) for s, v in sorted(feeds.get(nid, {}).items())},
        }
        for nid in sorted(by_id)
    }
    rows: dict[str, dict[str, Any]] = {}
    for node in nodes.values():
        row = rows.setdefault(node["family"], {"nodes": 0, "fed": 0, "labels": 0, "sources": set()})
        row["nodes"] += 1
        row["fed"] += bool(node["sources"])
        row["labels"] += sum(node["sources"].values())
        row["sources"].update(node["sources"])
    return {
        "nodes": nodes,
        "families": {f: {**r, "sources": sorted(r["sources"])} for f, r in sorted(rows.items())},
        "unfed_nodes": sorted(n for n, v in nodes.items() if not v["sources"]),
        "single_source_nodes": sorted(n for n, v in nodes.items() if len(v["sources"]) == 1),
        "review": {
            "no_shared_word": capped(flagged),
            "by_source": dict(sorted(per_source.items())),
            "code_like_labels": sum(coded.values()),
        },
    }


# ── check 5 ──────────────────────────────────────────────────────────────────────


def check_legacy(ctx: Context) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for new_id, old_id in LEGACY.items():
        new, old = ctx.crosswalks.get(new_id), ctx.crosswalks.get(old_id)
        if new is None or old is None:
            continue
        by_id = {n.id: n for n in ctx.schemas[new.target_schema].nodes}
        old_by_name = {e.source_label.lower(): e for e in old.edges}
        rows: list[dict[str, Any]] = []
        untagged: list[str] = []
        unpaired: list[str] = []
        for edge in new.edges:
            tag = LEGACY_TAG.match(note_of(edge))
            if tag is None:
                untagged.append(edge.source_label)
                continue
            legacy = tag.group(1).strip()
            alias = LEGACY_ALIASES.get(legacy.lower(), "")
            other = old_by_name.get(legacy.lower()) or old_by_name.get(alias)
            if other is None:
                unpaired.append(f"{edge.source_label} -> {legacy}")
                continue
            n, o = edge.targets.get(Axis.TAXON), other.targets.get(Axis.TAXON)
            if n == o:
                kind = "same"
            elif n is None or o is None:
                kind = "ignore-vs-target"
            elif o in lineage(n, by_id):
                kind = "refines"
            elif n in lineage(o, by_id):
                kind = "coarser"
            else:
                kind = "retarget"
            rows.append({
                "v2": edge.source_label,
                "v2_id": edge.source_label_id,
                "legacy": other.source_label,
                "kind": kind, "v2_target": n, "legacy_target": o,
                "condition_equal": (
                    edge.targets.get(Axis.CONDITION) == other.targets.get(Axis.CONDITION)
                ),
                "form_equal": edge.targets.get(Axis.FORM) == other.targets.get(Axis.FORM),
            })  # fmt: skip
        out[new_id] = {
            "legacy_crosswalk": old_id, "edges": len(new.edges), "paired": len(rows),
            "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
            "refinements": [r for r in rows if r["kind"] != "same"],
            "condition_differences": [r for r in rows if not r["condition_equal"]],
            "form_differences": [r for r in rows if not r["form_equal"]],
            "untagged": sorted(untagged), "unpaired": sorted(unpaired),
        }  # fmt: skip
    return out


# ── check 6 ──────────────────────────────────────────────────────────────────────


def check_condition(results: list[UnitResult], ctx: Context) -> dict[str, Any]:
    target = ctx.schemas[TARGET]
    node_words = {
        n.id: health_words(" ".join([n.name, n.id.replace("_", " "), *n.aliases]))
        for n in target.nodes
        if n.axis is Axis.CONDITION
    }
    carriers: dict[str, set[str]] = defaultdict(set)
    with_axis: set[str] = set()
    without_axis: set[str] = set()
    health = 0
    missing: list[str] = []
    other: list[str] = []
    for res in results:
        sid = res.record["id"]
        if res.record["target_schema"] != TARGET:
            continue
        has = False
        for label, edge in res.pairs:
            node = edge.targets.get(Axis.CONDITION)
            if node:
                carriers[node].add(sid)
                has = True
            terms = health_words(label)
            if not terms:
                continue
            health += 1
            if node is None:
                missing.append(f"{sid}: {label}")
            elif not terms & node_words.get(node, frozenset()):
                other.append(f"{sid}: {label} -> {node}")
        (with_axis if has else without_axis).add(sid)
    return {
        "nodes": sorted(node_words),
        "carriers": {k: sorted(v) for k, v in sorted(carriers.items())},
        "sources_with_condition": sorted(with_axis),
        "sources_without_condition": sorted(without_axis),
        "health_labels": health,
        "review": {
            "health_label_without_condition": capped(missing),
            "health_label_other_condition": capped(other),
        },
    }


# ── check 7 ──────────────────────────────────────────────────────────────────────


def _class_index(spec: Any, value: Any) -> int:
    """Class index of a scheme result: None is the ignore index, a name is looked up."""
    from .. import label_schemes as ls

    if value is None:
        return ls.IGNORE
    return spec.classes.index(value) if isinstance(value, str) else int(value)


def check_ignore() -> dict[str, Any]:
    """Undeclared pixel values and unmappable labels reach 255, unless a scheme overrides one."""
    from .. import label_schemes as ls
    from .. import labels

    published, schemes = ls.sources(), ls.schemes()
    failures: list[str] = []
    if not (ls.IGNORE == labels.IGNORE_INDEX == 255):
        failures.append(f"ignore index is {ls.IGNORE}/{labels.IGNORE_INDEX}, expected 255")
    crosswalks = {c.id: c for c in vm.Registry.load().crosswalks}
    out: dict[str, Any] = {}
    overrides: list[dict[str, str]] = []
    for sid, src in sorted(published.items()):
        cw = crosswalks.get(src.crosswalk)
        dropped = {
            e.source_label for e in (cw.edges if cw else ()) if e.fidelity is Fidelity.UNMAPPABLE
        }
        unmappable = sorted(dropped & set(src.native_labels))
        rec: dict[str, Any] = {
            "kind": src.kind,
            "crosswalk": src.crosswalk,
            "schemes": [],
            "checks": 0,
            "unmappable_labels": unmappable,
        }
        if ls.IGNORE in src.ids:
            failures.append(f"{sid}: 255 is declared as a class id")
        for scheme_id, spec in sorted(schemes.items()):
            if sid not in spec.sources:
                continue
            rec["schemes"].append(scheme_id)
            override = spec.source_overrides.get(sid, {})
            wanted = {name: _class_index(spec, override.get(name)) for name in unmappable}
            overrides += [
                {"source": sid, "scheme": scheme_id, "label": name, "class": spec.classes[idx]}
                for name, idx in wanted.items()
                if idx != ls.IGNORE
            ]
            try:
                if src.kind == "semantic":
                    table = labels.lut(sid, scheme_id)
                    stray = [
                        p for p in range(256) if p not in src.ids and int(table[p]) != ls.IGNORE
                    ]
                    rec["checks"] += 256 - len(src.ids)
                    if stray:
                        failures.append(
                            f"{sid}/{scheme_id}: undeclared pixel values {stray[:5]} are not 255"
                        )
                    got = {
                        name: int(table[pixel]) for pixel, name in src.ids.items() if name in wanted
                    }
                else:
                    got = {
                        name: _class_index(spec, labels.map_label(sid, name, scheme_id))
                        for name in src.labels
                        if name in wanted
                    }
            except Exception as exc:
                failures.append(f"{sid}/{scheme_id}: {type(exc).__name__}: {exc}")
                continue
            rec["checks"] += len(got)
            failures += [
                f"{sid}/{scheme_id}: unmappable {name!r} gives {value}, expected {wanted[name]}"
                for name, value in sorted(got.items())
                if value != wanted[name]
            ]
        out[sid] = rec
    return {
        "sources": out,
        "checks": sum(r["checks"] for r in out.values()),
        "overrides": overrides,
        "failures": failures,
    }


# ── new nodes ────────────────────────────────────────────────────────────────────


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def check_new_nodes(ctx: Context) -> dict[str, Any]:
    """The nodes added for Coralscapes V2 resolve, and no node of the previous release is gone."""
    by_id = {n.id: n for n in ctx.schemas[TARGET].nodes}
    releases = ctx.root / "taxonomy" / "releases"
    version = str(ctx.meta["version"])

    def keys(v: str) -> set[str] | None:
        path = releases / f"{v}.json"
        return set(json.loads(path.read_text())["nodes"]) if path.exists() else None

    older = [p.stem for p in releases.glob("*.json") if _version(p.stem) < _version(version)]
    previous = max(older, key=_version, default=None)
    current, before = keys(version), keys(previous) if previous else None
    used_by: dict[str, set[str]] = defaultdict(set)
    for cw in ctx.crosswalks.values():
        for edge in cw.edges:
            for ident in edge.targets.values():
                if ident in NEW_NODES:
                    used_by[ident].add(cw.id)
    failures: list[str] = []
    nodes: dict[str, Any] = {}
    for nid in NEW_NODES:
        node = by_id.get(nid)
        rec = {
            "present": node is not None,
            "parent": node.parent if node else None,
            "parent_present": bool(node and node.parent in by_id),
            "in_release": None if current is None else f"{TARGET}:{nid}" in current,
            "used_by": sorted(used_by.get(nid, ())),
        }
        nodes[nid] = rec
        if node is None:
            failures.append(f"{nid} is not a node of {TARGET}")
        elif not rec["parent_present"]:
            failures.append(f"{nid} has no parent node {node.parent!r}")
        if rec["in_release"] is False:
            failures.append(f"{nid} is missing from taxonomy release {version}")
    removed = sorted(
        k.split(":", 1)[1]
        for k in (before or ())
        if k.startswith(f"{TARGET}:") and k.split(":", 1)[1] not in by_id
    )
    if removed:
        failures.append(f"nodes of release {previous} no longer in {TARGET}: {removed[:5]}")
    return {
        "taxonomy_version": version, "previous_release": previous, "nodes": nodes,
        "removed_since_previous": removed, "failures": failures,
    }  # fmt: skip


# ── report ───────────────────────────────────────────────────────────────────────


def failures_of(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for rec in report["sources"]:
        out += [f"source {rec['id']}: {x}" for x in rec["failures"]]
    for rec in report["crosswalks"]:
        out += [f"crosswalk {rec['id']}: {x}" for x in rec["failures"]]
    out += [f"ignore: {x}" for x in report["ignore"]["failures"]]
    out += [f"new nodes: {x}" for x in report["new_nodes"]["failures"]]
    return out


def summarize(records: list[dict[str, Any]], cws: list[dict[str, Any]]) -> dict[str, Any]:
    used = [r for r in records if r["crosswalk"]]
    count = lambda key: sum(r["counts"][key] for r in used)  # noqa: E731
    return {
        "units": len(records),
        "by_status": dict(sorted(Counter(r["status"] for r in records).items())),
        "by_origin": dict(sorted(Counter(r["origin"] for r in records).items())),
        "crosswalks": len(cws),
        "crosswalk_edges": sum(c["edges"] for c in cws),
        "edges_by_fidelity": {k: sum(c[k] for c in cws) for k in FIDELITY_KEYS},
        "unmappable_crosswalks": {c["id"]: c["ignored"] for c in cws if c["ignored"]},
        "labels_verified": count("labels"),
        "unresolved": count("unresolved"), "ambiguous": count("ambiguous"),
        "missing": count("missing"), "duplicate": count("duplicate"),
        "declared_unmapped": count("declared_unmapped"),
        "lossy_without_note": {
            c["id"]: c["lossy_without_note"] for c in cws if c["lossy_without_note"]
        },
        "no_crosswalk": sorted(r["id"] for r in records if r["status"] == "no-crosswalk"),
        "no_crosswalk_undocumented": sorted(
            r["id"] for r in records if r["status"] == "no-crosswalk" and not r["documented"]
        ),
    }  # fmt: skip


def run(registry: vm.Registry | None = None, root: Any = None) -> dict[str, Any]:
    ctx = vm.make_context(registry, root)
    cws = [
        vm.evaluate_crosswalk(cw, ctx) for cw in sorted(ctx.registry.crosswalks, key=lambda c: c.id)
    ]
    cw_records = {c["id"]: c for c in cws}
    units = vm.collect_units(ctx)
    results = [vm.evaluate_unit(u, ctx, cw_records) for u in units]
    records = [r.record for r in results]
    users: dict[str, list[str]] = defaultdict(list)
    for u in units:
        if u.crosswalk_id and u.origin != "crosswalk":
            users[u.crosswalk_id].append(u.id)
    unmappable = [
        {
            "crosswalk": c["id"],
            "label": e["label"],
            "note": e["note"],
            "sources": sorted(users[c["id"]]),
        }
        for c in cws
        for e in c["unmappable"]
    ]
    report = {
        "format": 1, "release": RELEASE, "target_schema": TARGET,
        "taxonomy_version": str(ctx.meta["version"]),
        "summary": summarize(records, cws), "sources": records, "crosswalks": cws,
        "unmappable": unmappable, "agreement": check_agreement(results, ctx),
        "legacy": check_legacy(ctx), "condition": check_condition(results, ctx),
        "ignore": check_ignore(), "new_nodes": check_new_nodes(ctx),
    }  # fmt: skip
    report["failures"] = failures_of(report)
    return report


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
