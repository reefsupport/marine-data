# Labels

This page has two parts. [Training label mapping](#training-label-mapping) explains how the masks, instances and boxes
published on Hugging Face become training labels with fixed class ids. [The label system](#the-label-system) describes the
registry design behind it: schemas, crosswalks and how fidelity is recorded.

## Training label mapping

Every published source keeps its own native ids, and those ids collide across sources: pixel value 1 is `seagrass` in
Coralscapes and `Hard Coral` in the Reef Support benthic masks. The `marinedata.labels` module maps each source onto one
of three **schemes**. A scheme is a fixed class list with fixed ids, shared by every source that can be mapped onto it,
and 255 is the ignore value in all of them.

The mapping is data, not code. It is derived from `registry/label-schemes/` and the registry crosswalks, so a crosswalk
correction moves the tables on this page. The module needs numpy only.

| Scheme | Use |
|---|---|
| `benthic-coarse` | Benthic cover in six classes: hard coral, fire coral, soft coral, algae, abiotic and other fauna. It reuses the registry task of the same name. |
| `coral-binary` | Coral against everything else. Combines every coral-annotated source, including the Reef Support masks. |
| `scene` | The eight SUIM classes: background waterbody, divers, plants, wrecks, robots, reefs and invertebrates, fish and vertebrates, sea-floor. Combines SUIM with the instance and box sources. |

```python
from marinedata import labels

# Class names; the id of a class is its position.
labels.class_names("benthic-coarse")  # ('HC', 'MIL', 'SC', 'ALGAE', 'ABIOTIC', 'OTHER_FAUNA')

# A (256,) uint8 table from native pixel value to scheme id; 255 = ignore.
table = labels.lut("coralscapes", "benthic-coarse")

# The same for a whole mask (any integer array of native ids).
labels.remap_mask(mask, "coralscapes", "benthic-coarse")

labels.map_label("uiis10k", "reptiles", "scene")  # 6, the id of FV; None means ignored
labels.supervised_classes("reef-support-benthic-own", "benthic-coarse")  # ('HC', 'SC')
```

For a row of the published `coral-masks` or `scene-masks` configs, `labels.remap_row(row, scheme)` returns the row with a
`label` array added. It also checks the row's `class_map` against the registry and raises on a mismatch. See the
[README](../README.md#label-mapping) for a complete example.

### Class lists

Ids are the position in the class list. The ids below apply to the published masks.

<!-- BEGIN GENERATED: label-classes -->

#### `benthic-coarse`

| Id | Class | Meaning |
|---|---|---|
| 0 | `HC` | Hard coral |
| 1 | `MIL` | Fire coral (Millepora) |
| 2 | `SC` | Soft coral |
| 3 | `ALGAE` | Algae |
| 4 | `ABIOTIC` | Abiotic |
| 5 | `OTHER_FAUNA` | Other fauna |
| 255 | ignore | Not a class: excluded from the loss and from metrics |

#### `coral-binary`

| Id | Class | Meaning |
|---|---|---|
| 0 | `NOT_CORAL` | Every other mapped biotic, abiotic or transition label |
| 1 | `CORAL` | Hard coral, soft coral and fire coral (dead and bleached included by default) |
| 255 | ignore | Not a class: excluded from the loss and from metrics |

#### `scene`

| Id | Class | Meaning |
|---|---|---|
| 0 | `BW` | Background waterbody |
| 1 | `HD` | Human divers |
| 2 | `PF` | Plants and sea-grass |
| 3 | `WR` | Wrecks and ruins |
| 4 | `RO` | Robots and instruments |
| 5 | `RI` | Reefs and invertebrates |
| 6 | `FV` | Fish and vertebrates |
| 7 | `SR` | Sea-floor and rocks |
| 255 | ignore | Not a class: excluded from the loss and from metrics |

<!-- END GENERATED: label-classes -->

### Sources per scheme

Config is the Hugging Face config that holds the source. Annotation is `dense` when every pixel of a mask carries a class
(255 then marks only the source's own ignore region) and `partial` otherwise. Supervised classes are the classes a source
can ever label. Mapped and Ignored count the source's native labels. Sources that are not part of the public release are
listed by id only.

<!-- BEGIN GENERATED: label-sources -->

#### `benthic-coarse`

| Source | Config | Annotation | Supervised classes | Mapped | Ignored |
|---|---|---|---|---|---|
| `coralscapes` | `coral-masks` | dense | `HC`, `MIL`, `ABIOTIC`, `OTHER_FAUNA` | 30 | 9 |
| `reef-support-benthic-own` | `coral-masks` | partial | `HC`, `SC` | 2 | 0 |
| `reef-support-seaview-labels` | `coral-masks` | partial | `HC`, `SC` | 2 | 0 |
| `coralseg-ucsd-mosaics` | not public | dense | `HC`, `SC` | 2 | 1 |

#### `coral-binary`

| Source | Config | Annotation | Supervised classes | Mapped | Ignored |
|---|---|---|---|---|---|
| `coralscapes` | `coral-masks` | dense | `NOT_CORAL`, `CORAL` | 34 | 5 |
| `reef-support-benthic-own` | `coral-masks` | partial | `CORAL` | 2 | 0 |
| `reef-support-seaview-labels` | `coral-masks` | partial | `CORAL` | 2 | 0 |
| `coralseg-ucsd-mosaics` | not public | dense | `NOT_CORAL`, `CORAL` | 3 | 0 |
| `coralscop-masks-rs` | not public | partial | `CORAL` | 1 | 0 |

#### `scene`

| Source | Config | Annotation | Supervised classes | Mapped | Ignored |
|---|---|---|---|---|---|
| `suim` | `scene-masks` | dense | `BW`, `HD`, `PF`, `WR`, `RO`, `RI`, `FV`, `SR` | 8 | 0 |
| `uiis` | `instance-masks` | partial | `HD`, `PF`, `WR`, `RO`, `RI`, `FV`, `SR` | 7 | 0 |
| `uiis10k` | `instance-masks` | partial | `HD`, `PF`, `WR`, `RO`, `RI`, `FV` | 9 | 1 |
| `usis10k` | `instance-masks` | partial | `HD`, `PF`, `WR`, `RO`, `RI`, `FV`, `SR` | 7 | 0 |
| `roboflow-aquarium` | `fish-boxes` | partial | `RI`, `FV` | 7 | 0 |

<!-- END GENERATED: label-sources -->

### Native label to class

<!-- BEGIN GENERATED: label-mapping -->

#### `benthic-coarse`

| Source | Class (id) | Native labels |
|---|---|---|
| `coralscapes` | `HC` (0) | `other coral dead`, `other coral bleached`, `other coral alive`, `massive/meandering bleached`, `massive/meandering alive`, `branching bleached`, `branching dead`, `branching alive`, `massive/meandering dead`, `acropora alive`, `turbinaria`, `table acropora alive`, `pocillopora alive`, `table acropora dead`, `meandering bleached`, `stylophora alive`, `meandering alive`, `meandering dead` |
| `coralscapes` | `MIL` (1) | `millepora` |
| `coralscapes` | `ABIOTIC` (4) | `trash`, `sand`, `unknown hard substrate`, `rubble` |
| `coralscapes` | `OTHER_FAUNA` (5) | `fish`, `other animal`, `clam`, `sea cucumber`, `sea urchin`, `crown of thorn`, `dead clam` |
| `coralscapes` | 255 (ignored) | `seagrass`, `human`, `transect tools`, `algae covered substrate`, `background`, `dark`, `transect line`, `sponge`, `anemone` |
| `reef-support-benthic-own` | `HC` (0) | `Hard Coral` |
| `reef-support-benthic-own` | `SC` (2) | `Soft Coral` |
| `reef-support-seaview-labels` | `HC` (0) | `Hard Coral` |
| `reef-support-seaview-labels` | `SC` (2) | `Soft Coral` |
| `coralseg-ucsd-mosaics` | `HC` (0) | `Hard Coral` |
| `coralseg-ucsd-mosaics` | `SC` (2) | `Soft Coral` |
| `coralseg-ucsd-mosaics` | 255 (ignored) | `Other` |

#### `coral-binary`

| Source | Class (id) | Native labels |
|---|---|---|
| `coralscapes` | `NOT_CORAL` (0) | `seagrass`, `trash`, `sand`, `fish`, `algae covered substrate`, `other animal`, `unknown hard substrate`, `rubble`, `clam`, `sea cucumber`, `sponge`, `anemone`, `sea urchin`, `crown of thorn`, `dead clam` |
| `coralscapes` | `CORAL` (1) | `other coral dead`, `other coral bleached`, `other coral alive`, `massive/meandering bleached`, `massive/meandering alive`, `branching bleached`, `branching dead`, `millepora`, `branching alive`, `massive/meandering dead`, `acropora alive`, `turbinaria`, `table acropora alive`, `pocillopora alive`, `table acropora dead`, `meandering bleached`, `stylophora alive`, `meandering alive`, `meandering dead` |
| `coralscapes` | 255 (ignored) | `human`, `transect tools`, `background`, `dark`, `transect line` |
| `reef-support-benthic-own` | `CORAL` (1) | `Hard Coral`, `Soft Coral` |
| `reef-support-seaview-labels` | `CORAL` (1) | `Hard Coral`, `Soft Coral` |
| `coralseg-ucsd-mosaics` | `NOT_CORAL` (0) | `Other` |
| `coralseg-ucsd-mosaics` | `CORAL` (1) | `Hard Coral`, `Soft Coral` |
| `coralscop-masks-rs` | `CORAL` (1) | `coral` |

#### `scene`

| Source | Class (id) | Native labels |
|---|---|---|
| `suim` | `BW` (0) | `BW` |
| `suim` | `HD` (1) | `HD` |
| `suim` | `PF` (2) | `PF` |
| `suim` | `WR` (3) | `WR` |
| `suim` | `RO` (4) | `RO` |
| `suim` | `RI` (5) | `RI` |
| `suim` | `FV` (6) | `FV` |
| `suim` | `SR` (7) | `SR` |
| `uiis` | `HD` (1) | `human divers` |
| `uiis` | `PF` (2) | `aquatic plants` |
| `uiis` | `WR` (3) | `wrecks/ruins` |
| `uiis` | `RO` (4) | `robots` |
| `uiis` | `RI` (5) | `reefs` |
| `uiis` | `FV` (6) | `fish` |
| `uiis` | `SR` (7) | `sea-floor` |
| `uiis10k` | `HD` (1) | `human` |
| `uiis10k` | `PF` (2) | `plants` |
| `uiis10k` | `WR` (3) | `ruins` |
| `uiis10k` | `RO` (4) | `robots` |
| `uiis10k` | `RI` (5) | `arthropoda`, `corals`, `mollusk` |
| `uiis10k` | `FV` (6) | `fish`, `reptiles` |
| `uiis10k` | 255 (ignored) | `garbage` |
| `usis10k` | `HD` (1) | `human divers` |
| `usis10k` | `PF` (2) | `aquatic plants` |
| `usis10k` | `WR` (3) | `wrecks/ruins` |
| `usis10k` | `RO` (4) | `robots` |
| `usis10k` | `RI` (5) | `reefs` |
| `usis10k` | `FV` (6) | `fish` |
| `usis10k` | `SR` (7) | `sea-floor` |
| `roboflow-aquarium` | `RI` (5) | `jellyfish`, `starfish` |
| `roboflow-aquarium` | `FV` (6) | `fish`, `penguin`, `puffin`, `shark`, `stingray` |

<!-- END GENERATED: label-mapping -->

### Rules to know

**255 is ignore.** Train with an ignore index of 255 (`CrossEntropyLoss(ignore_index=255)`), exclude it from metrics, and
never treat it as a class.

**For partial sources, 255 means "not annotated", not background.** This holds for `reef-support-benthic-own`,
`reef-support-seaview-labels` and `coralscop-masks-rs`: they annotate a subset of the classes, and everything else in the
image, including real hard coral, fire coral or other organisms, is 255. Mask the loss to
`labels.supervised_classes(source, scheme)` for these sources, and do not train 255 as a negative. In `coral-binary` they
supervise `CORAL` only, so they provide positives and no `NOT_CORAL` examples. The CoralSCOP masks are produced by a
model: `1` is a detected coral instance and 255 is "not detected", which is not a verified negative. The same applies to
the instance and box sources (`uiis`, `uiis10k`, `usis10k`, `roboflow-aquarium`): pixels outside the annotated objects
are unannotated, not background. `labels.is_dense(source)` returns the flag.

**Defaults follow the registry.** Dead and bleached coral stay hard coral (`HC` in `benthic-coarse`, `CORAL` in
`coral-binary`), because the registry records them as hard coral with a condition. To send them to ignore instead, pass
`exclude_conditions=("dead", "bleached")`. Other aliases are `pale`, `diseased`, `healthy` and `unhealthy`, and any
condition node id of the `rs-benthic-v1` schema is accepted. The option applies to every label whose crosswalk entry has
that condition, not only to corals. With `exclude_conditions=("dead", "bleached")` the labels below go to ignore, and they
include `dead clam` in Coralscapes:

<!-- BEGIN GENERATED: label-conditions -->

| Scheme | Source | Labels | Sent to 255 |
|---|---|---|---|
| `benthic-coarse` | `coralscapes` | 10 | `other coral dead`, `other coral bleached`, `massive/meandering bleached`, `branching bleached`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering bleached`, `meandering dead`, `dead clam` |
| `coral-binary` | `coralscapes` | 10 | `other coral dead`, `other coral bleached`, `massive/meandering bleached`, `branching bleached`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering bleached`, `meandering dead`, `dead clam` |

<!-- END GENERATED: label-conditions -->

`ignore=` adds labels to the ignore set. It takes native label names or `rs-benthic-v1` taxon node ids, and a node
carries its subtree:

```python
labels.lut(
    "coralscapes", "benthic-coarse", exclude_conditions=("dead", "bleached"), ignore=("sand",)
)
```

Unknown schemes, sources, labels and options raise instead of being skipped.

**Coralscapes has no soft-coral class.** Octocorals and other soft corals are inside its `unknown hard substrate` class,
which maps to `ABIOTIC` in `benthic-coarse` and to `NOT_CORAL` in `coral-binary`. For joint training this means that
Coralscapes supplies no `SC` examples, and that its `ABIOTIC` or `NOT_CORAL` regions can contain soft coral that the
Reef Support sources label as `SC` or `CORAL`. To keep that label noise out, send the class to ignore with
`ignore=("unknown hard substrate",)`; the cost is that genuinely abiotic hard substrate in Coralscapes is ignored too.

**The scene scheme follows SUIM's definitions.** Ids are SUIM's pixel values, so SUIM masks pass through unchanged. Other
sources are mapped by label name: `arthropoda`, `mollusk`, `jellyfish`, `starfish` and `corals` go to reefs and
invertebrates (`RI`); `reptiles`, `shark`, `stingray`, `penguin` and `puffin` go to fish and vertebrates (`FV`); and
`garbage` in `uiis10k` goes to ignore, because debris is none of the eight classes.

**Which id space applies.** Published masks use the ids of `marinedata.labels`, which are the registry order shown above.
`LabelIndex.for_task` builds a different, alphabetical id space for sample-level labels and is not used for masks.

---

## The label system

How 60 datasets speaking a dozen incompatible vocabularies become one training set.

---

### The shape of it

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

#### Canonical schemas

**`rs-benthic-v1`** — 78 nodes on three axes (`taxon` / `form` / `condition`). L2 is
deliberately MariMap's shipped `CoralLabelCode`, so model output lands in the existing
PIT/LIT data model with no translation layer.

**`rs-fauna-v1`** — 29 nodes on `taxon` / `trophic` / `size`. Taxonomic spine pulled from
OBIS; trophic role is a **separate axis** because it cross-cuts phylogeny (parrotfish and
surgeonfish are both herbivores in different families; groupers and morays both piscivores
across different orders).

#### Why axes instead of one flat class list

Coralscapes-39 is a flattened, incomplete cross-product of taxon × form × condition —
27 of its 39 classes decompose into two or three axes, and **nobody ever wrote a
soft-coral row**. On Caribbean reefs, where octocorals are ~24% of our annotations, the
model has no valid label and dumps ~50% of every frame into "unknown hard substrate".

Axes keep the questions separable, let a source supervise only what it annotates, and let
a model answer coarsely (`HC`) instead of guessing a leaf.

#### Three schemas are deliberately empty

Registered with descriptions saying why; a test enforces the justification.

| Schema | Why it cannot be enumerated |
|---|---|
| `worms-species` | ~240k accepted marine species. Any snapshot is wrong within a month; resolve through the authority. |
| `sonotype` | Sound classes, **not taxa**. Which organism makes a given reef sound is largely unsolved — a sonotype→taxon crosswalk would invent the finding the field is chasing. |
| `dataset-native` | Vocabulary examined, crosswalk not yet authored. Distinct from a source nobody has looked at. |

#### Fidelity is recorded, not assumed

Every edge carries how faithfully it maps. Across all 142:

```
exact 84   coarsened 41   approximate 16   unmappable 1
```

`unmappable` is the important one: CoralNet's "Other" has no canonical equivalent, so the
axis becomes **unsupervised** rather than being forced into a wrong class. Forcing
unmappable labels into a nearest neighbour is exactly how a Red-Sea schema produces "50%
unknown hard substrate" on a Caribbean reef.

#### Taxonomy is anchored to OBIS

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

### Should this live in Parquet?

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

#### Why the ceiling really is ~1,100

The "17,357 species forces Parquet" framing was wrong twice. FishNet is not reef fish
(8 taxonomic classes, freshwater included). And **dataset class counts are crosswalk
inputs, never nodes** — a source's 1,275 labels enter as `source_label` strings on edges,
not as vocabulary.

The real ceilings: 114 zooxanthellate scleractinian genera worldwide (~25–30 Caribbean);
ReefNet's 39 classes is the published state of the art for coral from imagery; GCRMN
Level 2 scores **four** fish families; ~80–100 Caribbean reef fish are identifiable from
transect video.

#### When Parquet would be right, and why we do not need it

If we ever materialised the full WoRMS backbone (~240k species) or FishNet's 17,357, YAML
would be wrong. We do not, because those are **live lookups and crosswalk inputs, not
nodes**. `marinedata taxa resolve` returns the AphiaID and full rank ladder on demand, so
a species label rolls up to genus, family or order without us storing the intermediates.

Storing 240k rows to serve the ~1,100 we use would be a mirror of WoRMS that is
immediately stale, offline-hostile, and unreviewable — three losses for no gain.

#### The offline guarantee

The committed YAML **is** the pin. Nothing in the load path touches the network:
`taxa.py` is codegen, run by a maintainer, and its output is reviewed as a diff.
`tests/test_registry.py::test_registry_loads_with_the_network_hard_down` monkeypatches
`urlopen` and `socket` to raise, then loads the registry and harmonises — so a network
call cannot creep into loading, harmonisation or gating without a test failing.

---

### Using it

```python
import marinedata as md

reg = md.Registry.load()
harm = reg.harmonizer_for("coralscapes")

harm.map_label("massive/meandering bleached")
# taxon=HC, form=CMM, condition=BLEACHED   (fidelity: coarsened)

print(harm.coverage_report())  # how lossy this crosswalk is, before you trust it
```

```bash
marinedata taxa --drift        # re-resolve every AphiaID, fail on any change
marinedata taxa resolve Orbicella
```
