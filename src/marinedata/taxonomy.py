"""The unified taxonomy: pinned WoRMS snapshot, semver, per-release diff, label gate.

Runtime reads ONLY committed files — ``registry/taxonomy/taxonomy.yaml`` (the version
and the snapshot it pins), the snapshot parquet, the schemas, the crosswalks and the
observed vocabularies in ``registry/taxonomy/vocab/``. Network access lives in
:mod:`marinedata.worms_snapshot` (codegen) and nowhere else.

Three checks, all offline:

* **nodes** — every taxon-axis node of a canonical schema carries an accepted AphiaID
  that the snapshot confirms (name, rank, status), or ``non_taxon: true`` with a reason.
* **vocabularies** — every observed source label has a crosswalk edge. A label with no
  edge is a *silent drop* and fails; an ``unmappable`` edge is a listed, reasoned drop
  and counts against the mapped share (instance-weighted where counts exist).
* **coverage** — every labelled image source has a crosswalk or a documented exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .registry import Registry
from .schema import Axis, Fidelity

MIN_MAPPED = 0.95
LABELLED_KINDS = frozenset(
    {"point-label", "dense-mask", "instance-mask", "bbox", "image-label", "track"}
)


class TaxonomyGateError(Exception):
    """The release label gate failed."""


def taxonomy_dir(registry_root: str | Path) -> Path:
    return Path(registry_root) / "taxonomy"


def load_meta(registry_root: str | Path) -> dict:
    return yaml.safe_load((taxonomy_dir(registry_root) / "taxonomy.yaml").read_text())


def load_snapshot(registry_root: str | Path) -> dict[int, dict]:
    """AphiaID → snapshot row, from the pinned parquet named in taxonomy.yaml."""
    import pyarrow.parquet as pq

    meta = load_meta(registry_root)
    table = pq.read_table(taxonomy_dir(registry_root) / meta["snapshot"])
    return {int(r["aphia_id"]): r for r in table.to_pylist()}


# ── nodes ────────────────────────────────────────────────────────────────────────


def check_nodes(registry: Registry, snapshot: dict[int, dict]) -> list[str]:
    """Problems with canonical taxon nodes; empty means 100% anchored or declared.

    Strict: the snapshot must say ``accepted`` and resolve the id to itself. Informal
    WoRMS names (``Pisces``, ``... incertae sedis``) never anchor a canonical node.
    """
    out: list[str] = []
    for schema in registry.schemas:
        if not schema.canonical:
            continue
        for node in schema.nodes:
            if node.axis is not Axis.TAXON:
                continue
            where = f"{schema.id}:{node.id}"
            if node.worms_aphia_id is None:
                if not node.non_taxon:
                    out.append(f"{where}: no AphiaID and not declared non_taxon")
                continue
            row = snapshot.get(node.worms_aphia_id)
            if row is None:
                out.append(f"{where}: AphiaID {node.worms_aphia_id} absent from the snapshot")
            elif row["status"] != "accepted" or row["accepted_aphia_id"] != node.worms_aphia_id:
                out.append(
                    f"{where}: AphiaID {node.worms_aphia_id} is {row['status']} "
                    f"(accepted {row['accepted_aphia_id']} {row['accepted_name']})"
                )
            elif (row["scientific_name"], row["rank"]) != (
                node.worms_scientificname,
                node.worms_rank,
            ):
                out.append(
                    f"{where}: frozen {node.worms_scientificname}/{node.worms_rank} but "
                    f"snapshot says {row['scientific_name']}/{row['rank']}"
                )
    return out


def node_counts(registry: Registry) -> dict[str, int]:
    nodes = [n for s in registry.schemas if s.canonical for n in s.nodes if n.axis is Axis.TAXON]
    return {
        "nodes": len(nodes),
        "aphia": sum(n.worms_aphia_id is not None for n in nodes),
        "non_taxon": sum(n.non_taxon for n in nodes),
    }


# ── vocabularies ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VocabAudit:
    source_id: str
    crosswalk_id: str
    weighted: bool
    counts: dict[str, int | None]
    mapped: tuple[str, ...]
    unmappable: dict[str, str] = field(default_factory=dict)
    silent_drops: tuple[str, ...] = ()
    dead_targets: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        """Share of annotations (or label types when no counts exist) with a target."""
        if self.weighted:
            total = sum(v or 0 for v in self.counts.values())
            hit = sum(self.counts[k] or 0 for k in self.mapped)
        else:
            total, hit = len(self.counts), len(self.mapped)
        return hit / total if total else 0.0

    def line(self) -> str:
        flag = "✗" if self.silent_drops or self.dead_targets else "✓"
        kind = "annotations" if self.weighted else "types"
        return (
            f"{flag} {self.source_id:<22} {self.crosswalk_id:<22} {self.coverage:>7.2%} of "
            f"{kind}  {len(self.counts):>3} labels  {len(self.unmappable)} unmappable  "
            f"{len(self.silent_drops)} silent"
        )


def read_vocab(path: str | Path) -> tuple[dict[str, str], dict[str, int | None]]:
    head: dict[str, str] = {}
    counts: dict[str, int | None] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            head[key] = value
        elif line and not line.startswith("label\t"):
            label, count = ([*line.split("\t"), ""])[:2]
            counts[label] = int(count) if count else None
    return head, counts


def audit_vocab(registry: Registry, path: str | Path) -> VocabAudit:
    head, counts = read_vocab(path)
    crosswalk = registry.crosswalk(head["crosswalk"])
    target = registry.label_schema(crosswalk.target_schema)
    ids = {n.id for n in target.nodes}
    mapped, unmappable, silent, dead = [], {}, [], []
    for label in counts:
        edge = crosswalk.edge(label)
        if edge is None:
            silent.append(label)
        elif edge.fidelity is Fidelity.UNMAPPABLE:
            unmappable[label] = edge.note or ""
        else:
            mapped.append(label)
            dead += [f"{label}->{t}" for t in edge.targets.values() if t not in ids]
    return VocabAudit(
        source_id=head["source"],
        crosswalk_id=crosswalk.id,
        weighted=head.get("counts") == "annotations",
        counts=counts,
        mapped=tuple(mapped),
        unmappable=unmappable,
        silent_drops=tuple(silent),
        dead_targets=tuple(dead),
    )


def audit_all(registry: Registry, registry_root: str | Path) -> list[VocabAudit]:
    vocab = taxonomy_dir(registry_root) / "vocab"
    return [audit_vocab(registry, p) for p in sorted(vocab.glob("*.tsv"))]


# ── gate ─────────────────────────────────────────────────────────────────────────


def labelled_sources(registry: Registry) -> list[str]:
    """Image/video sources whose annotations carry classes."""
    out = []
    for s in registry.sources:
        mods = {getattr(m, "value", str(m)) for m in s.modalities}
        if not any("image" in m or "video" in m for m in mods):
            continue
        if any(a.kind.value in LABELLED_KINDS for a in s.annotations):
            out.append(s.id)
    return out


def gate(
    registry: Registry,
    registry_root: str | Path,
    *,
    source_ids: list[str] | None = None,
    min_mapped: float = MIN_MAPPED,
) -> list[str]:
    """Every failure the release label gate sees; an empty list passes.

    ``source_ids`` narrows the vocabulary and coverage checks to one release's sources.
    """
    meta = load_meta(registry_root)
    exceptions: dict[str, str] = meta.get("coverage_exceptions") or {}
    no_crosswalk: dict[str, str] = meta.get("no_crosswalk_yet") or {}
    fails = check_nodes(registry, load_snapshot(registry_root))
    audits = audit_all(registry, registry_root)
    scope = set(source_ids) if source_ids is not None else None
    for a in audits:
        if scope is not None and a.source_id not in scope:
            continue
        fails += [f"{a.source_id}: SILENT DROP {label!r}" for label in a.silent_drops]
        fails += [f"{a.source_id}: dead target {t}" for t in a.dead_targets]
        fails += [
            f"{a.source_id}: unmappable {k!r} has no reason"
            for k, v in a.unmappable.items()
            if not v
        ]
        if a.coverage < min_mapped and a.source_id not in exceptions:
            fails.append(
                f"{a.source_id}: {a.coverage:.2%} mapped < {min_mapped:.0%} and no exception"
            )
    covered = {a.source_id for a in audits}
    for sid in labelled_sources(registry):
        if scope is not None and sid not in scope:
            continue
        spec = registry.source(sid).loader
        if sid in covered or (spec and spec.crosswalk_id) or sid in no_crosswalk:
            continue
        fails.append(f"{sid}: labelled source with no crosswalk and no documented exception")
    return fails


def assert_release_gate(
    registry: Registry, registry_root: str | Path, source_ids: list[str]
) -> dict[str, str]:
    """Raise :class:`TaxonomyGateError` on any failure; return the release's taxonomy stamp."""
    fails = gate(registry, registry_root, source_ids=source_ids)
    if fails:
        raise TaxonomyGateError("label gate failed:\n  " + "\n  ".join(fails[:40]))
    meta = load_meta(registry_root)
    return {"taxonomy_version": str(meta["version"]), "taxonomy_snapshot": meta["snapshot"]}


# ── semver + diff ────────────────────────────────────────────────────────────────


def manifest(registry: Registry, registry_root: str | Path) -> dict:
    """Machine-readable taxonomy state: canonical nodes + crosswalk edges."""
    meta = load_meta(registry_root)
    nodes = {
        f"{s.id}:{n.id}": {
            "parent": n.parent,
            "axis": n.axis.value,
            "aphia": n.worms_aphia_id,
            "name": n.worms_scientificname or n.name,
            "rank": n.worms_rank,
            "non_taxon": n.non_taxon,
        }
        for s in registry.schemas
        if s.canonical
        for n in s.nodes
    }
    edges = {
        f"{c.id}:{e.source_label}": {
            "targets": {k.value: v for k, v in e.targets.items()},
            "fidelity": e.fidelity.value,
        }
        for c in registry.crosswalks
        for e in c.edges
    }
    return {
        "taxonomy_version": str(meta["version"]),
        "snapshot": meta["snapshot"],
        "nodes": nodes,
        "edges": edges,
    }


def diff(a: dict, b: dict) -> dict:
    """Added / removed / changed nodes and edges between two manifests."""
    out: dict = {"from": a.get("taxonomy_version"), "to": b.get("taxonomy_version")}
    for key in ("nodes", "edges"):
        old, new = a.get(key, {}), b.get(key, {})
        out[key] = {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "changed": {
                k: {"from": old[k], "to": new[k]}
                for k in sorted(set(old) & set(new))
                if old[k] != new[k]
            },
        }
    return out


def load_manifest(ref: str, registry_root: str | Path) -> dict:
    """A manifest by version (``registry/taxonomy/releases/<v>.json``) or by path."""
    path = Path(ref)
    if not path.exists():
        path = taxonomy_dir(registry_root) / "releases" / f"{ref}.json"
    return json.loads(path.read_text())
