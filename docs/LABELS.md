# The label system

How 60 datasets speaking a dozen incompatible vocabularies become one training set.

---

## The shape of it

Three layers. Each dataset declares what it speaks; a crosswalk maps that onto a
canonical schema; the loader emits canonical labels plus a supervision mask.

```
  dataset                native vocabulary        crosswalk            canonical
  ─────────────────────  ───────────────────────  ──────────────────   ─────────────
  coralscapes        →   coralscapes-39 (39)   →  39 edges         →   rs-benthic-v1
  benthicnet-1m      →   catami-1.4 (50)       →  50 edges         →   rs-benthic-v1
  reef-support-own   →   reef-support-labelbox →   5 edges         →   rs-benthic-v1
  mermaid / heron    →   coralnet-labelset (8) →   8 edges         →   rs-benthic-v1
  (AGRRA protocol)   →   agrra-benthic (21)    →  21 edges         →   rs-benthic-v1
  reefnet-species    →   worms-genus (19)      →  19 edges         →   rs-benthic-v1
  ozfish, fathomnet  →   worms-species (open)  →  resolve via OBIS →   rs-fauna-v1
  reefset, marrs     →   sonotype (open)       →  none, by design  →   —
```

**11 schemas · 6 crosswalks · 142 edges · 249 label nodes · 74 anchored to WoRMS.**

### Canonical schemas

**`rs-benthic-v1`** — 78 nodes on three axes (`taxon` / `form` / `condition`). L2 is
deliberately MariMap's shipped `CoralLabelCode`, so model output lands in the existing
PIT/LIT data model with no translation layer.

**`rs-fauna-v1`** — 29 nodes on `taxon` / `trophic` / `size`. Taxonomic spine pulled from
OBIS; trophic role is a **separate axis** because it cross-cuts phylogeny (parrotfish and
surgeonfish are both herbivores in different families; groupers and morays both piscivores
across different orders).

### Why axes instead of one flat class list

Coralscapes-39 is a flattened, incomplete cross-product of taxon × form × condition —
27 of its 39 classes decompose into two or three axes, and **nobody ever wrote a
soft-coral row**. On Caribbean reefs, where octocorals are ~24% of our annotations, the
model has no valid label and dumps ~50% of every frame into "unknown hard substrate".

Axes keep the questions separable, let a source supervise only what it annotates, and let
a model answer coarsely (`HC`) instead of guessing a leaf.

### Three schemas are deliberately empty

Registered with descriptions saying why; a test enforces the justification.

| Schema | Why it cannot be enumerated |
|---|---|
| `worms-species` | ~240k accepted marine species. Any snapshot is wrong within a month; resolve through the authority. |
| `sonotype` | Sound classes, **not taxa**. Which organism makes a given reef sound is largely unsolved — a sonotype→taxon crosswalk would invent the finding the field is chasing. |
| `dataset-native` | Vocabulary examined, crosswalk not yet authored. Distinct from a source nobody has looked at. |

### Fidelity is recorded, not assumed

Every edge carries how faithfully it maps. Across all 142:

```
exact 84   coarsened 41   approximate 16   unmappable 1
```

`unmappable` is the important one: CoralNet's "Other" has no canonical equivalent, so the
axis becomes **unsupervised** rather than being forced into a wrong class. Forcing
unmappable labels into a nearest neighbour is exactly how a Red-Sea schema produces "50%
unknown hard substrate" on a Caribbean reef.

### Taxonomy is anchored to OBIS

Node ids stay opaque project strings (`HC_ORBICELLA`, `SD`, `TL`); the AphiaID is an
**attribute**. Most of this library is substrate, morphology, condition and equipment,
none of which has one — keying on AphiaID would orphan the majority of nodes.

OBIS's `taxonID` **is** the WoRMS AphiaID, so one integer joins us to OBIS occurrences,
WoRMS classification and every other WoRMS-backed dataset, at every rank.

Each anchored node freezes five fields at resolve time:

```yaml
- { id: HC_ORBICELLA, name: Orbicella, parent: HC, axis: taxon,
    worms_aphia_id: 758259, worms_scientificname: Orbicella,
    worms_rank: Genus, worms_status: accepted, worms_checked_on: 2026-08-17 }
```

Freezing the name, rank and status is not decoration. **Three of the first eight AphiaIDs
entered here by hand were wrong** — one pointed at an unaccepted red-alga family, one at a
bryozoan. A bare integer looks identical whether it is right or not. With the fields
frozen, an error is visible in review and detectable by `marinedata taxa --drift`.

Synonyms need no machinery of our own: WoRMS answers them. *Montastraea annularis*
(207479) resolves to *Orbicella annularis* (758260), so two datasets labelled a decade
apart unify automatically. We record both ids and keep the trail.

---

## Should this live in Parquet?

**No. YAML in git, and the measurements say so with ~5× headroom.**

Measured on this machine, synthetic schemas with full WoRMS fields:

| nodes | YAML | parse | validate | total |
|--:|--:|--:|--:|--:|
| 250 *(today)* | 51 KiB | 11 ms | 1 ms | **12 ms** |
| 1,100 *(projected ceiling)* | 227 KiB | 50 ms | 5 ms | **55 ms** |
| 5,000 | 1.0 MiB | 430 ms | 27 ms | 457 ms |
| 20,000 | 4.2 MiB | 1.9 s | 92 ms | 2.0 s |

The full registry loads in **~165 ms** today.

**The binding constraint is human review, not performance.** GitHub collapses diffs over
~3,000 lines and drops rich diff over 1 MiB — both of which bite at roughly 5,000 nodes,
*before* load time becomes annoying. Since the realistic ceiling is ~1,100 nodes, YAML
wins on the axis that actually matters: a licence tier or a taxonomic correction is
reviewable as a diff by a human who is not a programmer.

### Why the ceiling really is ~1,100

The "17,357 species forces Parquet" framing was wrong twice. FishNet is not reef fish
(8 taxonomic classes, freshwater included). And **dataset class counts are crosswalk
inputs, never nodes** — a source's 1,275 labels enter as `source_label` strings on edges,
not as vocabulary.

The real ceilings: 114 zooxanthellate scleractinian genera worldwide (~25–30 Caribbean);
ReefNet's 39 classes is the published state of the art for coral from imagery; GCRMN
Level 2 scores **four** fish families; ~80–100 Caribbean reef fish are identifiable from
transect video.

### When Parquet would be right, and why we do not need it

If we ever materialised the full WoRMS backbone (~240k species) or FishNet's 17,357, YAML
would be wrong. We do not, because those are **live lookups and crosswalk inputs, not
nodes**. `marinedata taxa resolve` returns the AphiaID and full rank ladder on demand, so
a species label rolls up to genus, family or order without us storing the intermediates.

Storing 240k rows to serve the ~1,100 we use would be a mirror of WoRMS that is
immediately stale, offline-hostile, and unreviewable — three losses for no gain.

### The offline guarantee

The committed YAML **is** the pin. Nothing in the load path touches the network:
`taxa.py` is codegen, run by a maintainer, and its output is reviewed as a diff.
`tests/test_registry.py::test_registry_loads_with_the_network_hard_down` monkeypatches
`urlopen` and `socket` to raise, then loads the registry and harmonises — so a network
call cannot creep into loading, harmonisation or gating without a test failing.

---

## Using it

```python
import marinedata as md

reg  = md.Registry.load()
harm = reg.harmonizer_for("coralscapes")

harm.map_label("massive/meandering bleached")
# taxon=HC, form=CMM, condition=BLEACHED   (fidelity: coarsened)

print(harm.coverage_report())   # how lossy this crosswalk is, before you trust it
```

```bash
marinedata taxa --drift        # re-resolve every AphiaID, fail on any change
marinedata taxa resolve Orbicella
```
