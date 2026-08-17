# Label library specification — taxonomy backbone and per-dataset dictionaries

> Produced 2026-08-17 by a 6-agent workflow: authority/standards research, benthic and
> fish vocabulary extraction, then two adversarial critiques (taxonomist and engineer).
> **Status: specification, not yet implemented.** The DAG design originally proposed was
> REJECTED by both critiques; see §1.1. Four validation holes it identified in
> `schema.py` were verified as live bugs and are already fixed.

---

# FINAL SPECIFICATION — `marinedata` Label Library

Status: implementable today. Every design element below is a decision, not an option. Where the two critiques diverged, the choice and its reason are stated inline.

---

## 1. THE DATA MODEL

### 1.1 Decision: NOT one DAG. Three artifacts.

The DAG is rejected. Multi-valued `is_a` breaks `ancestors()` (its `seen` set conflates diamonds with cycles — the Acropora→HARD_CORAL diamond raises `ValueError('cycle')` on the exact motivating example), breaks `leaves()` silently (`{n.parent for ...}` becomes a set of tuples, so every node is a leaf), and makes `Fidelity.COARSENED` unverifiable in principle because "ancestor" stops naming one relation. Both critiques converged here from different directions; they are right.

Instead:

| Artifact | Owner | Shape | Storage |
|---|---|---|---|
| **Backbone** | generated from pinned WoRMS snapshot | strict single-parent **tree**, node id = `wid:<AphiaID>` | Parquet, not in git |
| **Concept layer** | humans, reviewed in diffs | flat nodes: functional groups, morphotypes, substrates, conditions, sentinels, equipment | YAML in git |
| **Membership** | humans + derivation rules | table: (concept, taxon, scheme, provenance, validity) | YAML in git |

Functional grouping is a **table lookup**, never a graph traversal. This is critique 2's "tag set" with critique 1's scheme-scoping and provenance bolted on — necessary because AGRRA and CoralNet *contradict each other* about Millepora and Cliona, and a single global edge set cannot hold "AGRRA says X, CoralNet says not-X".

### 1.2 Models

```python
# src/marinedata/ontology/model.py


class NodeKind(str, Enum):
    TAXON = "taxon"  # a nomenclatural concept; taxon_rank_id REQUIRED
    FUNCTIONAL = "functional"  # HARD_CORAL, HERBIVORE_SCRAPER — membership, not ancestry
    MORPHOTYPE = "morphotype"  # branching, tabulate
    SUBSTRATE = "substrate"  # sand, rubble, pavement
    CONDITION = "condition"  # bleached, recently dead
    EQUIPMENT = "equipment"  # transect line, scale bar
    SENTINEL = "sentinel"  # not-scorable, out-of-frame  (never a taxon)


# Axis (existing) = which HEAD predicts it. NodeKind = what the node IS.
# They are not duplicates; the legal pairs are declared once:
KIND_AXES: dict[NodeKind, frozenset[Axis]] = {
    NodeKind.TAXON: frozenset({Axis.TAXON}),
    NodeKind.FUNCTIONAL: frozenset({Axis.TAXON}),
    NodeKind.SUBSTRATE: frozenset({Axis.TAXON}),
    NodeKind.MORPHOTYPE: frozenset({Axis.FORM}),
    NodeKind.CONDITION: frozenset({Axis.CONDITION}),
    NodeKind.EQUIPMENT: frozenset({Axis.TAXON}),
    NodeKind.SENTINEL: frozenset(),  # sentinels carry NO axis label at all
}


class AliasKind(str, Enum):
    VERNACULAR = "vernacular"
    SYNONYM = "synonym"
    BASIONYM = "basionym"
    MISSPELLING = "misspelling"
    DATASET_LABEL = "dataset_label"


class Alias(_Frozen):
    name: str
    kind: AliasKind = AliasKind.DATASET_LABEL
    scheme: str | None = None  # "seamapd21-130" for dataset labels
    aphia_id: int | None = None  # for SYNONYM/BASIONYM: the id the name resolved to
    status: str | None = None  # WoRMS status string at resolution time


class LabelNode(_Frozen):
    id: str
    name: str
    parent: str | None = None  # SINGLE-VALUED. taxonomic only when kind=TAXON.
    axis: Axis = Axis.TAXON
    kind: NodeKind = NodeKind.TAXON
    taxon_rank_id: int | None = None  # WoRMS taxonRankID; REQUIRED iff kind=TAXON
    covers: tuple[int, ...] = ()  # AphiaIDs a merged/ambiguous concept spans
    worms_aphia_id: int | None = None  # accepted id
    worms_aphia_id_asserted: int | None = None  # id we were originally handed
    worms_status: str | None = None
    fishbase_spec_code: int | None = None
    fishbase_snapshot: str | None = None  # "v19.04" — SpecCode is stable per snapshot
    gbif_col_xr_id: str | None = None  # str, NOT the frozen 2023 int backbone key
    aliases: tuple[Alias, ...] = ()
    groups: tuple[str, ...] = ()  # DENORMALISED from membership.yaml at build time
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):  # v1 YAML compatibility, see §7
        if isinstance(v, dict) and isinstance(v.get("aliases"), (list, tuple)):
            v = {**v, "aliases": [{"name": a} if isinstance(a, str) else a for a in v["aliases"]]}
        return v

    @model_validator(mode="after")
    def _kind_invariants(self):
        legal = KIND_AXES[self.kind]
        if self.kind is NodeKind.SENTINEL and self.axis is not None and legal == frozenset():
            pass  # sentinels are excluded from every AxisIndex; see §6
        elif self.axis not in legal:
            raise ValueError(
                f"node {self.id}: kind={self.kind.value} illegal on axis={self.axis.value}"
            )
        if self.kind is NodeKind.TAXON and self.taxon_rank_id is None:
            raise ValueError(f"node {self.id}: kind=taxon requires taxon_rank_id")
        if self.kind is not NodeKind.TAXON and self.taxon_rank_id is not None:
            raise ValueError(f"node {self.id}: taxon_rank_id only legal on kind=taxon")
        return self
```

`LabelSchema` gains four validators that currently pass garbage (all four verified broken by critique 2):

```python
@model_validator(mode="after")
def _validate(self) -> LabelSchema:
    by_id = {n.id: n for n in self.nodes}
    if len(by_id) != len(self.nodes):
        raise ValueError("duplicate node id")
    for n in self.nodes:
        if n.axis not in self.axes and n.kind is not NodeKind.SENTINEL:
            raise ValueError(f"{n.id}: axis {n.axis.value} not declared by schema")
        if n.parent is None:
            continue
        p = by_id.get(n.parent)
        if p is None:
            raise ValueError(f"{n.id}: unknown parent {n.parent}")
        if p.axis is not n.axis:
            raise ValueError(f"{n.id}: cross-axis parent ({n.axis.value}->{p.axis.value})")
        if n.kind is NodeKind.TAXON and p.kind is not NodeKind.TAXON:
            raise ValueError(f"{n.id}: taxon may only descend from taxon (use membership.yaml)")
        if n.taxon_rank_id and p.taxon_rank_id and n.taxon_rank_id <= p.taxon_rank_id:
            raise ValueError(
                f"{n.id}: rank {n.taxon_rank_id} not finer than parent {p.taxon_rank_id}"
            )
    _assert_acyclic(self.nodes)  # iterative DFS, separate stack-set and visited-set
    return self
```

`_assert_acyclic` runs **at construction**, not lazily inside `ancestors()`. Today `LabelNode(id='A', parent='A')` loads clean through the whole registry and `leaves()` returns `()` silently.

### 1.3 Membership

```python
class Membership(_Frozen):
    concept: str  # FUNCTIONAL node id, e.g. "HARD_CORAL"
    taxon: str  # backbone or concept node id
    scheme: str = "rs"  # "rs" | "agrra-benthic" | "coralnet-labelset" | "ncrmp"
    provenance: Literal["asserted", "derived"] = "asserted"
    rule_id: str | None = None  # e.g. "R_HARD_CORAL_v1" — required if derived
    authority_version: str | None = None  # "worms-2026-08"
    valid_from: date
    superseded_by: str | None = None
    citation: str | None = None
```

`HARD_CORAL` is derivable, not hand-waved: rule `R_HARD_CORAL_v1` = WoRMS traits `Structure = Solid` ∧ `Composition ⊇ Calcareous` ∧ obligate-photosymbiotic, inherited from Scleractinia and Milleporidae respectively. It must **not** key on skeleton type (Scleractinia = endoskeleton, Milleporidae = exoskeleton — that row is exactly why no tree unifies them).

### 1.4 Failing examples, resolved

| Case | Resolution |
|---|---|
`resolve('SD', GENUS)` | `lift()` accepts only `kind=TAXON`; `SD` is `kind=SUBSTRATE` → raises `TypeError` at the call site, not silently returns.
`resolve(x, MORPHOTYPE)` | unrepresentable: `to_rank` is an `int` from the WoRMS `taxonRankID` vocabulary. FUNCTIONAL is not a rank; use `groups()`.
Acropora chain omits Refertina | rank vocabulary carries SUBPHYLUM/SUBCLASS/SUBORDER (§2); backbone is generated from WoRMS, so the real chain is stored and hand-editing cannot corrupt it.
`Goniopora/Alveopora`, `Funiculina_Balticina complex` | one `kind=TAXON` node with `covers=(205476, 206945)` and `taxon_rank_id` set to the **coarsest rank at which it is single-valued** (Refertina, 110). `lift()` stays single-valued. Merged concepts are NOT N parents.
Diamond → false cycle | no diamonds exist: `parent` is single-valued and taxonomic-only.
AGRRA vs CoralNet on Millepora | different `scheme` rows in `membership.yaml`; `groups(node, scheme=...)`.
AGRRA `FMA-CMA` 0.5/0.5, DCA | **in scope, changed:** `CrosswalkEdge.targets: dict[Axis, tuple[WeightedTarget, ...]]`, plus `substratum: str | None` on the edge. DCA becomes `surface=TA, substratum=DC` — the invented `TRANSITION` taxon branch is deprecated in v2.
`DRK`, `WC`, shadow | moved off the taxon axis to `Sample.validity: Validity` (`VALID / OCCLUDED / UNDEREXPOSED / OUT_OF_FRAME / NOT_SCORABLE / NOT_OF_INTEREST`), following CATAMI, which keeps its four sentinels out of its 287-node tree. `TL/TWS/DIV/SCL` become `kind=EQUIPMENT`.
CoralNet 4-level bleaching ordinal | **out of scope for v2.** Condition stays nominal; the crosswalk maps `1 - slight pale` and `2 - very pale` both to `PALE` at `fidelity: coarsened` with a note. An ordinal condition axis is a separate RFC; do not block the library on it.
DeepFish habitat, MFT25 identity, ACA geomorphic | **not label nodes.** `Sample.meta` keys `habitat`, `track_id`, `geomorphic_zone`. Adding `Axis.HABITAT` would allocate a training head nobody wants.
WoRMS/COL-XR rank disagreement | `lift()` is authority-scoped: `Ontology` carries `authority_version`, and `lift` results are only comparable within one snapshot. Documented, not papered over.

---

## 2. RANK VOCABULARY

No `Rank` enum with SUBSTRATE/CONDITION/EQUIPMENT in it — that is the category error. Rank is an **integer**, WoRMS `taxonRankID`, and it exists only on `kind=TAXON` nodes. Named constants for ergonomics:

```python
class Rank(IntEnum):
    KINGDOM = 10
    PHYLUM = 30
    SUBPHYLUM = 40
    SUPERCLASS = 50
    CLASS = 60
    SUBCLASS = 70
    ORDER = 100
    SUBORDER = 110
    INFRAORDER = 120
    SUPERFAMILY = 130
    FAMILY = 140
    SUBFAMILY = 150
    GENUS = 180
    SUBGENUS = 190
    SPECIES = 220
    SUBSPECIES = 230
```

Monotonic and comparable, which is the whole point. Non-taxonomic nodes carry `taxon_rank_id=None` and are reached by `kind`/`groups`, never by rank. Kingdom-dependent rank *strings* ("Phylum (Division)" in Plantae) are never stored.

---

## 3. PUBLIC API

Five functions, all on a prebuilt non-pydantic `Ontology` index (dicts, not `next(...)` scans — measured `ancestors()` at 687 µs/call on 17k nodes is 687 s/epoch inside a DataLoader).

```python
def ontology(version: str | None = None) -> Ontology: ...          # module-level, LRU-cached

class Ontology:
    authority_version: str                      # "worms-2026-08"
    def lift(self, node_id: str, to_rank: Rank) -> str | None: ...
    def groups(self, node_id: str, *, scheme: str = "rs") -> frozenset[str]: ...
    def resolve_name(self, text: str, *, scheme: str | None = None) -> Match | None: ...
    def explain(self, node_id: str) -> str: ...

@classmethod
def LabelIndex.at_rank(cls, samples, ont, rank, *, axes=None, min_count=1) -> LabelIndex: ...
```

Common case, one line:

```python
idx = LabelIndex.at_rank(samples, ontology(), Rank.GENUS)  # + idx.lift_report()
```

Real call sites:

```python
# loader: OzFish ships family,genus,species columns — no rank inference needed
node = ont.resolve_name(row["species"] or row["genus"] or row["family"])

# gate: what fraction of the corpus is hard coral, across two subphyla
frac = mean("HARD_CORAL" in ont.groups(s.labels[Axis.TAXON].node_id) for s in samples)

# CLI: marinedata explain wid:205902
$ Millepora  rank=Genus(180)  aphia=205902
  is_a: Milleporidae < Capitata < Anthoathecata < Hydroidolina < Hydrozoa < Medusozoa < Cnidaria
  groups(rs): HARD_CORAL [derived R_HARD_CORAL_v1 @ worms-2026-08]
  groups(coralnet-labelset): HARD_CORAL, OTHER_INVERTEBRATES, HARD_SUBSTRATE  ⚠ contradictory
```

`resolve` is deliberately not a name. `lift` navigates rank and returns `None` on failure rather than guessing; `resolve_name` does string→node and returns the matched alias kind plus authority version.

---

## 4. STORAGE AND SCALE

**Threshold: 1,000 nodes or 200 KB per YAML file.** Measured: 1,000 nodes ≈ 137 KB / 1,004 lines / 0.28 s parse — already past what a human reviews in a PR. 17,357 nodes = 2.44 MB / 6.00 s under `yaml.safe_load` (the code does not use `CSafeLoader`), paid on **every** CLI subcommand because `Registry.load()` is eager. Same rows as Parquet: 0.02 s.

```
registry/schemas/*.yaml            hand-authored, ≤1000 nodes, in git, reviewed
registry/membership.yaml           hand-authored, in git
registry/backbone/manifest.yaml    in git: {snapshot, sha256, row_count, rank_histogram}
registry/backbone/*.parquet        NOT in git — fetched by existing mirror/fetch machinery
```

Rules: (a) `_read_yaml` switches to `CSafeLoader`; (b) backbone loads **lazily** — `marinedata list` never touches it; (c) the backbone is regenerated, never merge-conflicted; (d) CI verifies `sha256` against the manifest.

FishNet's 17,357 species: build the fish backbone from FishNet's own `species/SpecCode/Genus/Subfamily/Family/Order/Class/SuperClass` columns plus WoRMS resolution → one Parquet, ~854 KB. `SpecCode` is stored *with* `fishbase_snapshot` because SpecCode is stable per snapshot, not per live site. No database, ever.

---

## 5. SYNONYMS AND REASSIGNMENTS

Mechanism: identity is the node id, never the name. Store `worms_aphia_id_asserted` **and** `worms_aphia_id` (accepted) **and** `worms_status`. Crosswalk rows are append-only with `valid_from` / `superseded_by`. A scheduled job walks `AphiaRecordsByDate` (50/page), intersects the change set against the registry's AphiaID set, and opens a review ticket per hit — it does not re-resolve silently.

**Montastraea → Orbicella, end to end:**

1. A 2011 AGRRA sheet says `MANN` = *Montastraea annularis*.
2. `AphiaRecordsByMatchNames` → 207479, `status="superseded combination"`, `valid_AphiaID=758260`.
3. Node written: `id: wid:758260, name: Orbicella annularis, taxon_rank_id: 220, worms_aphia_id: 758260, worms_aphia_id_asserted: 207479, worms_status: "superseded combination", aliases: [{name: "Montastraea annularis", kind: synonym, aphia_id: 207479, status: "superseded combination"}, {name: "Madrepora annularis", kind: basionym}, {name: MANN, kind: dataset_label, scheme: agrra-benthic}]`. Family under it is **Merulinidae** (from the accepted record), not the stale Montastraeidae the unaccepted record reports.
4. The hard case: a *genus-only* 2010 label `"Montastraea"` now spans two accepted genera in two families (*Montastraea* 204717 / Montastraeidae for *M. cavernosa*; *Orbicella* 758259 / Merulinidae). This is not a synonym — it is a **split**. Encode it as one node `AMB_MONTASTRAEA_2010`, `kind=TAXON`, `covers=(204717, 758259)`, `taxon_rank_id=Rank.ORDER` (1363 Scleractinia — the coarsest single-valued ancestor). `lift(…, GENUS)` returns `None`; `lift(…, ORDER)` returns Scleractinia. A 2010 genus label honestly degrades to order rank today. The v1 crosswalk row is not edited; a new row is appended with `valid_from: 2026-08-17` and the old one gets `superseded_by`.
5. **A model trained under the old name still works.** Node ids never change, and the frozen label index is a checked-in manifest keyed by `(schema_id, schema_version)` — not `tuple(sorted(set(classes)))`, which was verified to shift 6 of 7 indices when one node is added. Loading a checkpoint whose `index_manifest_sha256` differs from the current manifest **raises**. To score an old checkpoint on new ground truth: run the published `v1 → v2` crosswalk with `unmappable → abstain`, exactly as `docs/ARCHITECTURE.md:93` already prescribes for eval-v1.

---

## 6. HEAD WIDTH AND RANK

**Fixed rank per head. No hierarchical softmax.** A model has N heads (one per axis, optionally several taxon heads at declared ranks sharing a trunk); each head has one declared rank and one frozen vocabulary. "Both" is implemented as "several fixed heads", not as one structured output.

Rank rollup happens **before** the vocabulary is frozen:

```python
LabelIndex.at_rank(samples, ont, Rank.GENUS)
# for each sample: node -> ont.lift(node, rank); count only successes;
# returns idx with idx.lift_report() -> LiftReport(rate, by_reason={
#   "no_ancestor_at_rank": n, "not_taxon": n, "absent_from_ontology": n})
```

`lift_failure_rate` is a **gated metric**, defaulting to fail above 0.05. Today the path is silent: `AxisIndex.index_of` returns `IGNORE_INDEX` for any OOV node, indistinguishable at the loss from legitimate unsupervision — train a genus head on FishNet species ids and all 17,357 return −100, the loss curve looks fine, and the head sees zero gradient.

Sentinels (`kind=SENTINEL`) and `Validity != VALID` positions are excluded from every `AxisIndex` and emit `IGNORE_INDEX` by construction. Eval keeps `min_count=0` on a frozen union vocabulary and reports `oov_gt_pixel_fraction`.

Two more fixes required in the same PR, because they are load-bearing and currently dead:
- `Registry.harmonizer_for` must pass `supervised_axes=` (verified `None` today, so the central safety guard never runs);
- `supervises` moves onto `CrosswalkEdge` as a per-edge `frozenset[Axis]`, validated ⊆ the source-level declaration. CoralNet has no correct source-level value — `0 - no bleaching` carries condition only, `Acropora` taxon only, `Submassive Goniopora (Bleached)` all three.
- `_validate_targets` additionally asserts target `axis` == edge axis key, and that `COARSENED` targets are true `parent`-chain ancestors.

---

## 7. MIGRATION

`rs-benthic-v1` is **frozen forever**, byte-compatible, and stays the target of the live Coralscapes crosswalk and every minted release.

Purely additive, safe today:
- New fields are nullable with defaults, so all 78 v1 nodes and the 5 native schemas still validate. `extra='forbid'` means the field additions and any YAML using them must land in one commit.
- `aliases: ["sand"]` still parses via the `mode='before'` coercer → `Alias(name="sand", kind=DATASET_LABEL)`.
- `parent` keeps its name. `is_a` is not introduced; the backbone tree uses `parent` too.
- **AphiaID corrections are metadata patches applied to v1 in place**, because they are simply wrong and node ids and ancestor closures do not move: `MIL` 196197 (a red-alga family) → **205902**; `HC_ORBICELLA` 1400001 (a bryozoan) → **758259**; `HC_PORITES` 206970 (*Seriatopora stricta*) → **206485**. `label_digest` hashes the frozen index and per-sample labels, not node metadata, so existing releases still verify. A CI test pins v1's full ancestor closure to catch any accidental structural edit.
- New CI validator: every `worms_aphia_id` in every registry YAML is resolved against WoRMS and checked for existence, accepted status (or a recorded `valid_AphiaID`), rank agreement with `taxon_rank_id`, and name agreement modulo aliases. Never use `AphiaIDByName` (returns `-999` / HTTP 206 on homonyms — *Acropora* is both Cnidaria 205469 and Bryozoa 578625); use `AphiaRecordsByName` and disambiguate on `phylum` + `authority`.

Everything structural goes to **`rs-benthic-v2`**, published with an ordinary `v1 → v2` Crosswalk carrying fidelity — the machinery exists. v2 does: `HC` split into `HARD_CORAL` (`kind=FUNCTIONAL`, membership from both Scleractinia and Milleporidae — v1 currently asserts fire coral is *not* hard coral, the exact CoralNet inconsistency the project cites) and `wid:1363` Scleractinia; `MariMap CoralLabelCode` strings retained as `DATASET_LABEL` aliases so the shipped enum round-trips; `TRANSITION` deprecated in favour of surface+substratum; `DRK`/`WC` moved to `Validity`; form axis gains two levels (Branching → Corymbose/Tabulate/Digitate/Caespitose/Arborescent, per ReefNet and CATAMI — all 14 form nodes have `parent: null` today, so `ancestors()` returns `()` and a model abstaining from tabulate to branching has no edge to walk).

Also fix before building on top: `coralscapes-39.yaml`'s `other coral alive/bleached/dead → HC` at `fidelity: coarsened` violates the `COARSENED` contract — the source's own definition says the class **includes soft corals**, so `HC` is not an ancestor. Retarget to a coral-unspecified concept at `approximate`. `unknown hard substrate → RK` becomes `unmappable` (its definition includes pier columns, buoys, anchors and coral nursery tables). Fix the file header too: soft coral is documented as absorbed into "other coral", not into substrate.

---

## 8. STARTER BACKBONE

Two files. AphiaIDs marked `# ✓` were verified live 2026-08-17; `# ?` must be resolved by the WoRMS validator before merge (the validator fails the build on any `# ?` left at HEAD).

```yaml
# registry/schemas/backbone-reef-seed.yaml  — hand-seeded; superseded per-branch by
# the generated worms Parquet as it lands. ~120 nodes, well under the 1000 threshold.
schemas:
  - id: backbone-reef-seed
    name: Reef backbone seed
    authority_version: worms-2026-08
    axes: [taxon]
    nodes:
      # ── Cnidaria: the two lineages HARD_CORAL spans ──────────────────
      - {id: wid:1267,    name: Cnidaria,       taxon_rank_id: 30,  worms_aphia_id: 1267}    # ?
      - {id: wid:1292,    name: Anthozoa,       parent: wid:1267, taxon_rank_id: 40,  worms_aphia_id: 1292}    # ✓ SUBPHYLUM
      - {id: wid:1340,    name: Hexacorallia,   parent: wid:1292, taxon_rank_id: 60,  worms_aphia_id: 1340}    # ✓ CLASS
      - {id: wid:1363,    name: Scleractinia,   parent: wid:1340, taxon_rank_id: 100, worms_aphia_id: 1363}    # ✓
      - {id: wid:1517258, name: Refertina,      parent: wid:1363, taxon_rank_id: 110, worms_aphia_id: 1517258} # ✓ suborder, erected 2021+
      - {id: wid:196095,  name: Acroporidae,    parent: wid:1517258, taxon_rank_id: 140, worms_aphia_id: 196095} # ✓
      - {id: wid:205469,  name: Acropora,       parent: wid:196095, taxon_rank_id: 180, worms_aphia_id: 205469,
          aliases: [{name: Acropora, kind: dataset_label, scheme: coralscapes-39},
                    {name: "Acropora Reuss, 1869", kind: misspelling, aphia_id: 578625,
                     status: "unaccepted — BRYOZOAN HOMONYM, do not merge"}]}            # ✓
      - {id: wid:730685,  name: Isopora,        parent: wid:196095, taxon_rank_id: 180, worms_aphia_id: 730685} # ✓
      - {id: wid:203834,  name: Montipora,      parent: wid:196095, taxon_rank_id: 180, worms_aphia_id: 203834} # ✓
      - {id: wid:204599,  name: Astreopora,     parent: wid:196095, taxon_rank_id: 180, worms_aphia_id: 204599} # ✓
      # Medusozoa — the other side of the functional group
      - {id: wid:1337,    name: Medusozoa,      parent: wid:1267, taxon_rank_id: 40,  worms_aphia_id: 1337}    # ?
      - {id: wid:1336,    name: Hydrozoa,       parent: wid:1337, taxon_rank_id: 60,  worms_aphia_id: 1336}    # ?
      - {id: wid:19494,   name: Hydroidolina,   parent: wid:1336, taxon_rank_id: 70,  worms_aphia_id: 19494}   # ?
      - {id: wid:1348,    name: Anthoathecata,  parent: wid:19494, taxon_rank_id: 100, worms_aphia_id: 1348}   # ?
      - {id: wid:196235,  name: Milleporidae,   parent: wid:1348, taxon_rank_id: 140, worms_aphia_id: 196235}  # ✓
      - {id: wid:205902,  name: Millepora,      parent: wid:196235, taxon_rank_id: 180, worms_aphia_id: 205902,
          aliases: [{name: millepora, kind: dataset_label, scheme: coralscapes-39},
                    {name: "fire coral", kind: vernacular}]}                             # ✓

      # ── Caribbean scleractinian genera that matter (Merulinidae etc.) ──
      - {id: wid:196108,  name: Merulinidae,    parent: wid:1517258, taxon_rank_id: 140, worms_aphia_id: 196108} # ?
      - {id: wid:758259,  name: Orbicella,      parent: wid:196108, taxon_rank_id: 180, worms_aphia_id: 758259}  # ✓
      - {id: wid:758260,  name: Orbicella annularis, parent: wid:758259, taxon_rank_id: 220,
          worms_aphia_id: 758260, worms_aphia_id_asserted: 207479,
          worms_status: "accepted (was 207479, superseded combination)",
          aliases: [{name: "Montastraea annularis", kind: synonym, aphia_id: 207479},
                    {name: "Madrepora annularis",   kind: basionym},
                    {name: MANN, kind: dataset_label, scheme: agrra-benthic}]}            # ✓
      - {id: wid:718718,  name: Pseudodiploria, parent: wid:196108, taxon_rank_id: 180, worms_aphia_id: 718718} # ✓
      - {id: wid:267392,  name: Diploria,       parent: wid:196108, taxon_rank_id: 180, worms_aphia_id: 267392} # ✓
      - {id: wid:718746,  name: Dipsastraea,    parent: wid:196108, taxon_rank_id: 180, worms_aphia_id: 718746} # ✓
      - {id: wid:289696,  name: Colpophyllia,   parent: wid:196108, taxon_rank_id: 180, worms_aphia_id: 289696} # ✓
      - {id: wid:204717,  name: Montastraea,    parent: wid:1363,  taxon_rank_id: 180, worms_aphia_id: 204717,
          notes: "STILL ACCEPTED for M. cavernosa. Not a synonym of Orbicella."}          # ✓
      - {id: AMB_MONTASTRAEA_PRE2016, name: "Montastraea sensu lato (pre-split)",
          parent: wid:1363, taxon_rank_id: 100, covers: [204717, 758259],
          notes: "Genus-only labels predating the Orbicella split. lift(GENUS)=None by design."}
      - {id: wid:206485,  name: Porites,        parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 206485}  # ✓
      - {id: wid:204291,  name: Siderastrea,    parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 204291}  # ✓
      - {id: wid:204464,  name: Agaricia,       parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 204464}  # ✓
      - {id: wid:289232,  name: Meandrina,      parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 289232}  # ✓
      - {id: wid:418865,  name: Dendrogyra,     parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 418865}  # ✓
      - {id: wid:289807,  name: Dichocoenia,    parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 289807}  # ✓
      - {id: wid:289939,  name: Eusmilia,       parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 289939}  # ✓
      - {id: wid:290426,  name: Mycetophyllia,  parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 290426}  # ✓
      - {id: wid:290327,  name: Manicina,       parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 290327}  # ✓
      - {id: wid:135125,  name: Madracis,       parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 135125}  # ✓
      - {id: wid:291119,  name: Stephanocoenia, parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 291119}  # ✓
      - {id: wid:291054,  name: Solenastrea,    parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 291054}  # ✓
      - {id: wid:216135,  name: Mussa,          parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 216135}  # ✓
      - {id: wid:216134,  name: Isophyllia,     parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 216134}  # ✓
      - {id: wid:204384,  name: Scolymia,       parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 204384}  # ✓
      - {id: wid:290422,  name: Mussismilia,    parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 290422}  # ✓
      - {id: wid:206938,  name: Pocillopora,    parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 206938}  # ✓
      - {id: wid:204068,  name: Stylophora,     parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 204068}  # ✓
      - {id: wid:206641,  name: "Turbinaria (coral)", parent: wid:1363, taxon_rank_id: 180,
          worms_aphia_id: 206641,
          notes: "HOMONYM. The macroalga Turbinaria is 206630 (Ochrophyta/Sargassaceae) and is
                  what Coralscapes class 10 'algae covered substrate' explicitly includes.
                  NEVER alias the bare string 'turbinaria'."}                              # ✓
      - {id: wid:267930,  name: Tubastraea,     parent: wid:1363, taxon_rank_id: 180, worms_aphia_id: 267930}  # ✓
      # Fungiidae kept at FAMILY as well as genus — ReefNet's deliberate exception,
      # its genera are not visually separable.
      - {id: wid:196113,  name: Fungiidae,      parent: wid:1363, taxon_rank_id: 140, worms_aphia_id: 196113}  # ?

      # ── Octocorals / zoanthids — the branch no public scheme has ──────
      - {id: wid:1341,   name: Octocorallia,    parent: wid:1292, taxon_rank_id: 70, worms_aphia_id: 1341}     # ✓
      - {id: wid:125277, name: Gorgoniidae,     parent: wid:1341, taxon_rank_id: 140, worms_aphia_id: 125277}  # ?
      - {id: wid:125312, name: Gorgonia,        parent: wid:125277, taxon_rank_id: 180, worms_aphia_id: 125312}# ?
      - {id: wid:125285, name: Plexauridae,     parent: wid:1341, taxon_rank_id: 140, worms_aphia_id: 125285}  # ?
      - {id: wid:267197, name: Zoantharia,      parent: wid:1340, taxon_rank_id: 100, worms_aphia_id: 267197}  # ?
      - {id: wid:100974, name: Palythoa,        parent: wid:267197, taxon_rank_id: 180, worms_aphia_id: 100974}# ?
      - {id: wid:100979, name: Zoanthus,        parent: wid:267197, taxon_rank_id: 180, worms_aphia_id: 100979}# ?
      - {id: wid:1359,   name: Actiniaria,      parent: wid:1340, taxon_rank_id: 100, worms_aphia_id: 1359}    # ?

      # ── Porifera, algae ───────────────────────────────────────────────
      - {id: wid:558,    name: Porifera,        taxon_rank_id: 30, worms_aphia_id: 558}                        # ✓
      - {id: wid:164811, name: Demospongiae,    parent: wid:558, taxon_rank_id: 60, worms_aphia_id: 164811}    # ?
      - {id: wid:132021, name: Chondrilla,      parent: wid:164811, taxon_rank_id: 180, worms_aphia_id: 132021,
          notes: "HOMONYM: Chondrilla 1077021 is a plant (Compositae)."}                   # ✓
      - {id: wid:132026, name: Cliona,          parent: wid:164811, taxon_rank_id: 180, worms_aphia_id: 132026}# ✓
      - {id: wid:1806,   name: Rhodophyta,      taxon_rank_id: 30, worms_aphia_id: 1806}                       # ?
      - {id: wid:143993, name: Corallinales,    parent: wid:1806, taxon_rank_id: 100, worms_aphia_id: 143993}  # ?
      - {id: wid:801,    name: Chlorophyta,     taxon_rank_id: 30, worms_aphia_id: 801}                        # ?
      - {id: wid:144188, name: Halimeda,        parent: wid:801, taxon_rank_id: 180, worms_aphia_id: 144188}   # ?
      - {id: wid:206630, name: "Turbinaria (macroalga)", taxon_rank_id: 180, worms_aphia_id: 206630}           # ✓

      # ── Fish skeleton: class / order / family, extended from FishNet ──
      - {id: wid:1821,   name: Chordata,        taxon_rank_id: 30,  worms_aphia_id: 1821}                      # ?
      - {id: wid:10194,  name: Actinopterygii,  parent: wid:1821, taxon_rank_id: 60, worms_aphia_id: 10194}    # ?
      - {id: wid:10193,  name: Elasmobranchii,  parent: wid:1821, taxon_rank_id: 70, worms_aphia_id: 10193}    # ?
      - {id: wid:154712, name: Perciformes,     parent: wid:10194, taxon_rank_id: 100, worms_aphia_id: 154712}  # ?
      - {id: wid:125537, name: Labridae,        parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125537} # ?
      - {id: wid:125563, name: Scaridae,        parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125563,
          aliases: [{name: Scaridae, kind: dataset_label, scheme: f4k-23}]}                 # ?
      - {id: wid:125555, name: Acanthuridae,    parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125555} # ?
      - {id: wid:125564, name: Serranidae,      parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125564} # ?
      - {id: wid:125469, name: Lutjanidae,      parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125469} # ?
      - {id: wid:125556, name: Haemulidae,      parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125556} # ?
      - {id: wid:125561, name: Pomacentridae,   parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125561} # ?
      - {id: wid:125559, name: Chaetodontidae,  parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125559} # ?
      - {id: wid:125518, name: Balistidae,      parent: wid:10194,  taxon_rank_id: 140, worms_aphia_id: 125518} # ?
      - {id: wid:125531, name: Carangidae,      parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125531} # ?
      - {id: wid:125565, name: Sphyraenidae,    parent: wid:154712, taxon_rank_id: 140, worms_aphia_id: 125565} # ?
      - {id: wid:126175, name: Sebastes,        parent: wid:10194,  taxon_rank_id: 180, worms_aphia_id: 126175} # ?
```

```yaml
# registry/schemas/concepts.yaml  — functional groups, morphotypes, sentinels
nodes:
  - {id: HARD_CORAL,   name: Hard coral,   kind: functional, axis: taxon}
  - {id: SOFT_CORAL,   name: Soft coral,   kind: functional, axis: taxon}
  - {id: CORAL_UNSPEC, name: "Coral, unspecified (hard or soft)", kind: functional, axis: taxon,
      notes: "Target for Coralscapes 'other coral *', which its own definition says includes soft coral."}
  - {id: OTHER_INVERT, name: Other invertebrate, kind: functional, axis: taxon}
  - {id: AGGRESSIVE_INVERT, name: Aggressive invertebrate, kind: functional, axis: taxon,
      notes: "AGRRA AINV. Behavioural, cuts INSIDE Demospongiae — not derivable from traits."}
  - {id: TAXON_UNIDENTIFIED, name: "Unidentified organism", kind: functional, axis: taxon,
      notes: "OzFish's literal 'fish', MOUSS, LFW. An explicit abstention, not a species."}
  # trophic functional groups — fish; membership derivable from FishBase Troph/FeedingPath
  - {id: HERB_SCRAPER, name: Herbivore — scraper/excavator, kind: functional, axis: taxon}
  - {id: HERB_BROWSER, name: Herbivore — browser,           kind: functional, axis: taxon}
  - {id: HERB_GRAZER,  name: Herbivore/detritivore — grazer, kind: functional, axis: taxon}
  - {id: PLANKTIVORE,  name: Planktivore,   kind: functional, axis: taxon}
  - {id: INVERTIVORE,  name: Invertivore,   kind: functional, axis: taxon}
  - {id: PISCIVORE,    name: Piscivore,     kind: functional, axis: taxon}
  - {id: CORALLIVORE,  name: Corallivore,   kind: functional, axis: taxon}
  # substrate
  - {id: SD, name: Sand,  kind: substrate, axis: taxon}
  - {id: RB, name: Rubble, kind: substrate, axis: taxon}
  - {id: RK, name: "Rock / pavement (natural)", kind: substrate, axis: taxon}
  - {id: ARTIFICIAL, name: "Artificial hard structure", kind: substrate, axis: taxon,
      notes: "Pier columns, buoys, anchors, coral nursery tables. Split out of RK: merging
              restoration infrastructure into natural pavement is not acceptable."}
  # sentinels — carry NO axis label, excluded from every AxisIndex
  - {id: NOT_SCORABLE,   name: Not scorable,    kind: sentinel}
  - {id: NOT_OF_INTEREST, name: Not of interest, kind: sentinel}
```

```yaml
# registry/membership.yaml
memberships:
  - {concept: HARD_CORAL, taxon: wid:1363,   scheme: rs, provenance: derived,
     rule_id: R_HARD_CORAL_v1, authority_version: worms-2026-08, valid_from: 2026-08-17}
  - {concept: HARD_CORAL, taxon: wid:196235, scheme: rs, provenance: derived,
     rule_id: R_HARD_CORAL_v1, authority_version: worms-2026-08, valid_from: 2026-08-17,
     citation: "WoRMS traits: Structure=Solid, Composition=Calcareous, obligate photosymbiotic"}
  - {concept: SOFT_CORAL,  taxon: wid:1341,   scheme: rs, valid_from: 2026-08-17}
  - {concept: HARD_CORAL,  taxon: wid:205902, scheme: coralnet-labelset, valid_from: 2026-08-17,
     citation: "27 CoralNet labels"}
  - {concept: OTHER_INVERT, taxon: wid:205902, scheme: coralnet-labelset, valid_from: 2026-08-17,
     citation: "31 CoralNet labels — CONTRADICTS the row above. Both recorded deliberately."}
  - {concept: AGGRESSIVE_INVERT, taxon: wid:132026, scheme: agrra-benthic, valid_from: 2026-08-17}
```

---

## 9. PER-DATASET DICTIONARIES

Every dataset gets one file, `registry/dictionaries/<schema_id>.yaml`, and the pattern is fixed:

```yaml
schema:
  id: seamapd21-130
  name: SEAMAPD21 130-class
  axes: [taxon]
  provenance:
    source_of_truth: "Table A1, Sensors 22:8268 (PMC9658540)"
    confidence: verified          # verified | likely | unverified
    retrieved: 2026-08-17
    completeness: partial         # 82 of 130 classes enumerable
    open_vocabulary: false        # true for agrra-benthic — loaders MUST accept unknowns
  nodes:
    - {id: "EPINEPHELUSNIGRITUS", name: "Epinephelus nigritus", kind: taxon, taxon_rank_id: 220,
       aliases: [{name: "Hyporthodus nigritus", kind: synonym}]}
    - {id: "ANOMURA", name: Anomura, kind: taxon, taxon_rank_id: 120,
       notes: "Infraorder of CRUSTACEANS. Not a fish. Registry text '130 species' is wrong."}
    - {id: "MYCTEROPERCAINTERSTIALIS", name: "Mycteroperca interstitialis", kind: taxon,
       taxon_rank_id: 220, aliases: [{name: MYCTEROPERCAINTERSTIALIS, kind: misspelling}],
       notes: "Duplicate class — coexists with the correctly-spelled class."}
```

Non-negotiable rules: `confidence: unverified` ⇒ `nodes: []` (never invent plausible genera); every node carries `taxon_rank_id` or a non-TAXON `kind`; `open_vocabulary: true` sources route unknown codes to `TAXON_UNIDENTIFIED`, never error; dataset labels enter the backbone as `Alias(kind=DATASET_LABEL, scheme=<schema_id>)`.

**Priority order for the 59** — 54 currently declare no schema, so ordering is by supervision value per hour, not alphabetically:

1. **Tier 0 — fix what is wrong (days).** Correct the 3 bad AphiaIDs; register `deolho-21` (referenced by `coral-benthic.yaml`, absent from `native.yaml`, so the id does not resolve); split `worms-genus` into three real schemas (ReefNet's 85 shipped hard-coral labels / its unenumerated 39-class benchmark / the unrelated 39-class RSG test vocabulary); fix `catami-1.4` counts (287 nodes: 251 biota / 19 substrate / 8 bedforms) and `coralnet-labelset` (12,263 live labels, not 1,275 — that was an FC-layer width).
2. **Tier 1 — Caribbean supervision the project actually lacks (week 1–2).** `deolho-21` (21 nodes; supplies 6 verified soft-bodied/zoanthid labels for the SC branch and 5 shared Caribbean taxa Coralscapes cannot express), `agrra-benthic` (9 closed + BL-/ND- prefix grammar parsed, not enumerated), `reef-support-labelbox`, `coralscapes-39` (fidelity repairs).
3. **Tier 2 — free structure (week 2–3).** `reefnet` (adopt `ReefNet_labelmapping.xlsx` wholesale as the CoralNet crosswalk seed: 999 source labels → 125 harmonised, with AphiaIDs and rank already in it), `catami-1.4` (parse the xlsx; assert single-root reachability and record the two shipped defects — the `80000901` orphan parent and the `SUPBR` CPC collision; join on CAAB, never CPC), `coralnet-labelset` (one HTML scrape, 12,263 labels, short codes verified globally unique).
4. **Tier 3 — fish (week 3–4).** `fishnet-2023` (17,357 species, pre-ranked, ships `SpecCode` → direct join to `taxonomy.fish_length_weight`), `ozfish` (70/200/507 pre-ranked + the only mm-accurate lengths in the registry), `seamapd21-130`, `f4k-23`.
5. **Tier 4 — coarse context (later).** `aca-reef-habitat-v2` (benthic band only; geomorphic band → `Sample.meta`), `ncrmp-tier-1-2-3` (blocked on a primary PDF; stays `confidence: likely`), `deepsea-mot-vars`, `fathomnet-worms`.
6. **Tier 5 — declare and stop.** `mouss-fish-1`, `lfw-fish-1`, `mft25-identity`, `brackish-6`, `deepfish-habitat`, `urchin-roboflow-9`: one-line dictionaries, `supervises` corrected (MFT25 supervises identity, not taxon), no node authoring.

---

## 10. WHAT NOT TO BUILD

- **No multi-parent DAG, no diamond traversal, no `member_of` graph edges.** Membership is a table.
- **No hierarchical softmax, no 17k-logit head, no tree-structured loss.** Fixed rank per head plus an honest `lift_failure_rate`.
- **No database.** YAML + Parquet + a sha256 manifest.
- **No `Rank` enum containing SUBSTRATE / CONDITION / EQUIPMENT / FUNCTIONAL.** That is the original category error.
- **No `Axis.HABITAT`, `Axis.ZONE`, `Axis.IDENTITY`.** `Sample.meta`.
- **No hand-maintained taxonomic `is_a` in YAML beyond the ~120-node seed.** WoRMS erected Refertina and Vacatina recently; hand-maintained lineage rots. Generate it.
- **No GBIF integer backbone keys.** Frozen since 2023. `gbif_col_xr_id: str` or nothing.
- **No FishBase REST client.** The ropensci API is dead (connection failure, not 404). Parquet snapshots only.
- **No `AphiaIDByName` anywhere in the codebase.** It returns `-999`/HTTP 206 on homonyms.
- **No ordinal condition axis, no partial-extent qualifier, no annotation-certainty field in this release.** Real gaps, separate RFCs; they must not block the backbone.
- **No in-place crosswalk edits.** Append with `valid_from` / `superseded_by`. A 2011 sheet meant the 2011 concept.
- **No re-derivation of the whole registry on WoRMS change.** Change feed → review ticket → explicit migration.
- **No "ontology redesign" that consumes the Caribbean annotation schedule.** This spec makes soft coral *expressible*; it supplies zero Caribbean sand, rubble, pavement, turf or CCA examples. Those still have to be annotated.

🛠 Tools Used: None (analysis of supplied research + direct reads of `src/marinedata/{schema,labelindex,sample,harmonize,registry}.py`, `registry/schemas/{rs-benthic-v1,native}.yaml`, `registry/crosswalks/coralscapes-39.yaml`, `docs/ARCHITECTURE.md`)