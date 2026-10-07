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
correction moves the tables on this page. The module needs numpy only. The defaults are set for joint training; the
[label policy](#label-policy) below lists where they differ from the registry.

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
| 1 | `CORAL` | Hard coral, soft coral and fire coral (bleached included, dead ignored by default) |
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
can ever label. Mapped and Ignored count the source's native labels after the [label policy](#label-policy) is applied.
Sources that are not part of the public release are listed by id only.

<!-- BEGIN GENERATED: label-sources -->

#### `benthic-coarse`

| Source | Config | Annotation | Supervised classes | Mapped | Ignored |
|---|---|---|---|---|---|
| `coralscapes` | `coral-masks` | dense | `HC`, `MIL`, `ALGAE`, `ABIOTIC`, `OTHER_FAUNA` | 26 | 13 |
| `reef-support-benthic-own` | `coral-masks` | partial | `HC`, `SC` | 2 | 0 |
| `reef-support-seaview-labels` | `coral-masks` | partial | `HC`, `SC` | 2 | 0 |
| `coralseg-ucsd-mosaics` | not public | dense | `HC`, `SC` | 2 | 1 |

#### `coral-binary`

| Source | Config | Annotation | Supervised classes | Mapped | Ignored |
|---|---|---|---|---|---|
| `coralscapes` | `coral-masks` | dense | `NOT_CORAL`, `CORAL` | 27 | 12 |
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
| `coralscapes` | `HC` (0) | `other coral bleached`, `other coral alive`, `massive/meandering bleached`, `massive/meandering alive`, `branching bleached`, `branching alive`, `acropora alive`, `turbinaria`, `table acropora alive`, `pocillopora alive`, `meandering bleached`, `stylophora alive`, `meandering alive` |
| `coralscapes` | `MIL` (1) | `millepora` |
| `coralscapes` | `ALGAE` (3) | `algae covered substrate` |
| `coralscapes` | `ABIOTIC` (4) | `trash`, `sand`, `rubble` |
| `coralscapes` | `OTHER_FAUNA` (5) | `fish`, `other animal`, `clam`, `sea cucumber`, `sponge`, `anemone`, `sea urchin`, `crown of thorn` |
| `coralscapes` | 255 (ignored) | `seagrass`, `other coral dead`, `human`, `transect tools`, `unknown hard substrate`, `background`, `dark`, `transect line`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering dead`, `dead clam` |
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
| `coralscapes` | `NOT_CORAL` (0) | `seagrass`, `trash`, `sand`, `fish`, `algae covered substrate`, `other animal`, `rubble`, `clam`, `sea cucumber`, `sponge`, `anemone`, `sea urchin`, `crown of thorn` |
| `coralscapes` | `CORAL` (1) | `other coral bleached`, `other coral alive`, `massive/meandering bleached`, `massive/meandering alive`, `branching bleached`, `millepora`, `branching alive`, `acropora alive`, `turbinaria`, `table acropora alive`, `pocillopora alive`, `meandering bleached`, `stylophora alive`, `meandering alive` |
| `coralscapes` | 255 (ignored) | `other coral dead`, `human`, `transect tools`, `unknown hard substrate`, `background`, `dark`, `transect line`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering dead`, `dead clam` |
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

### Label policy

The schemes are set up for training on several sources at once. They differ from the registry's published `coarse`
column in exactly two ways: one default exclusion (dead coral) and a short list of overrides. Everything else follows
the registry.

1. **Dead coral is ignored.** In `benthic-coarse` and `coral-binary`, a label whose crosswalk condition is
   `RECENTLY_DEAD` or `OLD_DEAD` (the `dead` alias) goes to 255. This covers the five dead coral classes of Coralscapes
   and `dead clam`; a dead label that is not coral follows the same rule. Coralscapes separates alive, dead and bleached
   coral, while the Reef Support and Seaview masks record no condition, so ignoring dead pixels avoids contradictory
   targets in joint training. Bleached coral is alive and counts as live cover in monitoring, so it stays `HC` /
   `CORAL`.
2. **`unknown hard substrate` is ignored in every scheme.** In Coralscapes this class mixes rock with octocorals, so it
   is neither `ABIOTIC` nor `NOT_CORAL`.
3. **Clear gaps are filled.** In `benthic-coarse`, `sponge` and `anemone` map to `OTHER_FAUNA`, and
   `algae covered substrate` maps to `ALGAE` (turf on dead substrate). The crosswalk places the first two at nodes
   coarser than every class, and the third on a transition node. `seagrass` stays 255 because no class matches it, and so do the non-benthic labels:
   water, dark, divers, transect tools and the transect line.
4. **Reef Support and Seaview masks keep `HC` and `SC` as drawn.** They record no coral condition, so their `HC` can
   include recently dead colonies. Because of rule 1, no source labels dead coral as anything else, so the sources do not
   contradict each other.

The overrides (rules 2 and 3, and two older ones for sources whose crosswalk has no taxon edge) are set in
`registry/label-schemes/schemes.yaml`; the registry tasks and crosswalks are unchanged, so the published `coarse` column
still follows the crosswalks. `Crosswalk alone` below is what the registry gives:

<!-- BEGIN GENERATED: label-overrides -->

| Scheme | Source | Native label | Crosswalk alone | Scheme |
|---|---|---|---|---|
| `benthic-coarse` | `coralscapes` | `unknown hard substrate` | `ABIOTIC` | ignored (255) |
| `benthic-coarse` | `coralscapes` | `sponge` | ignored (255) | `OTHER_FAUNA` |
| `benthic-coarse` | `coralscapes` | `anemone` | ignored (255) | `OTHER_FAUNA` |
| `benthic-coarse` | `coralscapes` | `algae covered substrate` | ignored (255) | `ALGAE` |
| `coral-binary` | `coralscapes` | `unknown hard substrate` | `NOT_CORAL` | ignored (255) |
| `coral-binary` | `coralseg-ucsd-mosaics` | `Other` | ignored (255) | `NOT_CORAL` |
| `coral-binary` | `coralscop-masks-rs` | `coral` | ignored (255) | `CORAL` |

<!-- END GENERATED: label-overrides -->

The condition exclusions, per condition alias. `Default` marks the ones that apply without any option; the others are
opt-in. Only Coralscapes carries a coral condition today.

<!-- BEGIN GENERATED: label-conditions -->

| Scheme | Source | Condition | Default | Labels sent to 255 |
|---|---|---|---|---|
| `benthic-coarse` | `coralscapes` | `dead` | yes | `other coral dead`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering dead`, `dead clam` |
| `benthic-coarse` | `coralscapes` | `bleached` | no | `other coral bleached`, `massive/meandering bleached`, `branching bleached`, `meandering bleached` |
| `coral-binary` | `coralscapes` | `dead` | yes | `other coral dead`, `branching dead`, `massive/meandering dead`, `table acropora dead`, `meandering dead`, `dead clam` |
| `coral-binary` | `coralscapes` | `bleached` | no | `other coral bleached`, `massive/meandering bleached`, `branching bleached`, `meandering bleached` |

<!-- END GENERATED: label-conditions -->

To opt out, pass `exclude_conditions`. `None` (the default) means the scheme default, and any explicit value replaces
it, so `()` restores the registry and `("bleached",)` ignores bleached coral but keeps dead coral:

```python
labels.lut("coralscapes", "benthic-coarse", exclude_conditions=())  # dead coral is HC again
labels.lut(
    "coralscapes", "benthic-coarse", exclude_conditions=("dead", "bleached")
)  # live coral only
```

The option has the same meaning in `remap_mask`, `remap_row`, `map_label`, `supervised_classes` and
`to_torch_dataset(scheme_options={"exclude_conditions": ()})`. The overrides have no switch; to undo one, edit the table
that `lut` returns (it is a copy), for example `table[12] = 4` sends `unknown hard substrate` to `ABIOTIC`.
`remap_row` checks a row's `class_map` against the registry, not against the scheme, so the overrides and the default
exclusions never raise.

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

**Condition options.** Besides `dead` and `bleached`, the aliases are `pale`, `diseased`, `healthy` and `unhealthy`, and any
condition node id of the `rs-benthic-v1` schema is accepted. The option applies to every label whose crosswalk entry has that condition, not only
to corals; `dead clam` in Coralscapes is the one non-coral example today.

`ignore=` adds labels to the ignore set. It takes native label names or `rs-benthic-v1` taxon node ids, and a node
carries its subtree:

```python
labels.lut(
    "coralscapes", "benthic-coarse", exclude_conditions=("dead", "bleached"), ignore=("sand",)
)
```

Unknown schemes, sources, labels and options raise instead of being skipped.

**Coralscapes has no soft-coral class.** Octocorals and other soft corals are inside its `unknown hard substrate` class.
That class is ignored by default (see the [label policy](#label-policy)), so Coralscapes supplies no `SC` examples and
its `ABIOTIC` and `NOT_CORAL` regions do not carry soft coral that the Reef Support sources label as `SC` or `CORAL`. The
cost is that genuinely abiotic hard substrate in Coralscapes is ignored too.

**Seaview masks include whole-frame hard coral.** About 8 percent of the Seaview rows label the whole frame as hard coral.
These frames are close-ups of dense coral, so their masks carry little boundary information. Weight or sample the
source accordingly when you mix it with sources that outline individual colonies.

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
