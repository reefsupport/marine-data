# marine-data

**A licence-aware registry and dataloaders for marine and coral-reef datasets.**

Marine computer vision has an abundance of data and no way to tell what you are allowed
to do with it. Roughly 160 relevant datasets exist across a dozen incompatible label
schemes and at least six licence regimes. Some are public domain. Some forbid commercial
use. Some forbid *derivative works entirely* — meaning you cannot legally generate a
mask, a crop, or an augmentation from them. At least one redistributes stock photography
under a licence its publisher does not have the rights to grant.

Every team re-derives this from scratch, usually incompletely, usually after training.

`marine-data` makes licence tier a **first-class dimension of the data layer**, so that
*"we had the right to train on every one of these labels"* is a mechanical, provable
property rather than a remembered one.

```python
import marinedata as md

# What can we legally train a shippable benthic segmenter on?
result = md.find(task="benthic-segmentation", profile="ship-commercial")
print(result.summary())

# And what did the gate remove — with the reason?
for decision in result.excluded:
    print(decision.reason)
```

> ⚠️ **This is metadata, not legal advice.** Tier assignments record our reading of
> primary sources, cited per entry. They are a starting point for your own review, not a
> substitute for counsel. See [`docs/LEGAL.md`](docs/LEGAL.md).

### Relationship to `rs-ai`

[`reefsupport/rs-ai`](https://github.com/reefsupport/rs-ai) is the inference side —
model workers, orchestration, and content-hashed MRV manifests. `marine-data` is the
training side: what may lawfully go *into* a model, and what each source can and cannot
express.

The two meet at provenance. `rs-ai` emits a manifest describing how a measurement was
produced; `marine-data` emits a `LINEAGE.json` describing what the model was trained on
and under which licence. An MRV claim needs both halves, and neither is convincing
alone — a reproducible pipeline over data you had no right to use is not defensible,
and nor is clean data behind an unauditable model.

---

## Why a registry, not a data lake

**We deliberately do not host or redistribute any data.** For non-commercial and
no-derivatives sources, redistribution would itself be a violation — a data lake of this
material could not be built lawfully by anyone. The registry points at canonical sources,
records what we verified and how, and caches locally on fetch.

That constraint turns out to be a feature: it keeps the project small, keeps liability
where it belongs, and means a contributor adds value by *verifying a licence*, not by
uploading terabytes.

## The model

Three orthogonal facets, all queryable:

| Facet | Question it answers |
|---|---|
| **Licence tier + flags** | What are we permitted to do? |
| **Capability** | What is this useful for — benthic segmentation, fish ID, 3D, pretraining…? |
| **Coverage + domain shift** | Will it actually work where we deploy? |

### Licence tiers

| Tier | Meaning |
|---|---|
| `T0_OWN` | Rights held outright |
| `T1_PERMISSIVE` | CC0, CC-BY, Apache-2.0, MIT, US-Gov public domain |
| `T2_COPYLEFT` | CC-BY-SA, GPL — shippable **iff** the derivative is shared alike |
| `T3_NONCOMMERCIAL` | CC-BY-NC and variants — research and internal use only |
| `T4_TDM_ONLY` | No grant; lawful access; relies on a statutory TDM exception |
| `TX_PROHIBITED` | Provenance-defective or contract-blocked — never usable |

Tier alone is not enough, so licences also carry **flags**:

- `no_derivatives` — ND blocks masks, crops and augmentations. **Untrainable, not merely
  non-commercial.** This is the most commonly missed restriction in the field.
- `contract_gated` — access required assent to terms, which can override the EU
  commercial TDM exception (DSM Art. 7 protects Arts. 3/5/6 but not Art. 4).
- `provenance_defective` — the licensor appears to grant rights it does not hold. No
  downstream permission cures this.
- `share_alike`, `attribution_required`.

### Profiles — the gate

A profile declares what a build is *for*. **The gate raises; it does not warn.** A
warning that scrolls past in a training log is how tainted data reaches shipped weights.

| Profile | Admits | For |
|---|---|---|
| `ship-commercial` | T0, T1 | Closed weights in a paid product |
| `ship-open` | T0, T1, T2 | Weights released under share-alike |
| `research` | T0, T1, T2, T3 | Papers and benchmarks; never shipped |
| `pretrain-eu` | + T4 | SSL pretraining under EU TDM; requires a counsel opinion reference |

`pretrain-eu` cannot be instantiated without `legal_opinion_ref`. The profile enforces
what a policy document only asks for.

### Domain shift, as data

Cross-region collapse is the most common deployment failure in marine CV, and no existing
catalog records it. Entries carry measured drops and — more usefully — **missing
classes**:

```yaml
domain_shift:
  trained_regions: [red-sea]
  missing_classes: ["soft coral", "gorgonian / sea fan", "octocoral (any)"]
  known_drops:
    - to_region: caribbean
      metric: "live-coral cover fidelity"
      delta: -30.0
```

That example is Coralscapes, the field's reference dense-segmentation dataset. Its 39
classes contain **no soft-coral class of any kind**. On Caribbean reefs, where octocorals
are roughly a quarter of all annotations, a model trained on it has no valid label for
what it is looking at and collapses into `unknown hard substrate`. A dataset can be
excellent and still be the wrong choice for your region — that belongs in metadata rather
than in folklore.

## Dataloaders

Sources **declare** an on-disk layout rather than shipping bespoke code:

```yaml
loader:
  layout: image-mask-pairs
  schema_id: coralscapes-39
  crosswalk_id: coralscapes-39
  params: { images_dir: images, masks_dir: masks }
```

```python
from marinedata.loaders import build_loader

loader = build_loader(registry.source("coralscapes"), "/data/coralscapes")
loader.bind_harmonizer(registry.harmonizer_for("coralscapes"))

for sample in loader:
    sample.image  # a path, not decoded bytes — lazy by default
    sample.labels  # {Axis.TAXON: HC, Axis.FORM: CMM, Axis.CONDITION: BLEACHED}
    sample.supervised  # which axes this source actually annotates
    sample.licence_tier  # rides along, so lineage reflects what was consumed
```

Eight layouts cover the field's conventions: `image-folder`, `image-mask-pairs`,
`coco-json`, `yolo-txt`, `csv-points`, `labelbox-ndjson`, `audio-clips`, `metadata-only`.

Adding a source is usually a YAML change, not code. Loaders never download, and an empty
result is always an error — silent emptiness is indistinguishable from a filter that
matched nothing.

### Verification — because a declared layout is only a hypothesis

Tests run at two levels, and they fail differently:

- **Synthetic fixtures** prove a *reader* works. A reader bug raises.
- **Live samples (~100 items)** prove a *declaration* is right. A wrong declaration
  quietly finds nothing, or the wrong thing.

```bash
marinedata fetch coralscapes --limit 100      # bounded sample into ~/.cache/marinedata
marinedata verify                             # run every declared layout against real data
marinedata verify --unverified-only           # what has never been checked
```

Sources that cannot be auto-fetched are not all the same problem — a missing sample URL,
a gated form and a 5 GB Zenodo monolith need different answers. Per-source routes are in
[`docs/ACCESS_PLANS.md`](docs/ACCESS_PLANS.md).

This is not theoretical. The first full sweep caught two wrong declarations:
**Coralscapes** was declared as `images/` + `masks/` directories and is in fact
HuggingFace parquet with `image`/`label` columns; **MOUSS** was declared `coco-json`
and its HuggingFace mirror ships images only, with the boxes in a separate release.
Synthetic fixtures would have passed on both, forever.

So `loader.verified_on` is first-class metadata, exactly like licence verification:

```yaml
loader:
  layout: flat-images
  verified_on: 2026-08-17
  verified_note: "30 live rows fetched from HF; images only — coco-json was a wrong guess"
```

Unset means the layout is a guess from documentation. `marinedata verify` sets it
honestly, and the backlog is visible rather than assumed away.

## Harmonisation

Datasets use a dozen incompatible schemes. Crosswalks map them onto canonical axes —
`taxon`, `form`, `condition` — and **record what each mapping loses**:

```
crosswalk coralscapes-39 → rs-benthic-v1  (39 edges)
  exact           27  (69%)
  coarsened        9  (23%)
  approximate      3  (8%)
```

The rule that matters: **a source label with no canonical equivalent leaves the axis
unsupervised rather than being coerced into the nearest class.** Coercion is exactly how
a Red-Sea schema with no soft-coral row produces "50% unknown hard substrate" on a
Caribbean reef, and nothing in a training log reveals it.

## Model development

```python
import marinedata as md

reg = md.Registry.load()
ds = md.DatasetBuilder(
    reg,
    profile="ship-commercial",
    roots={"coralscapes": "/data/coralscapes", "reef-support-benthic": "/data/rs"},
).build()

ds.split(by="site")  # group-wise — see below
print(ds.summary())

frame = ds.to_pandas()  # exploration, stratification, leakage checks
torch_ds = ds.to_torch(split="train")
tf_ds = ds.to_tf(split="train", batch_size=8)
```

**Splits are group-wise by default.** Consecutive transect frames overlap heavily — the
same colony appears in dozens of them — so a random split puts near-duplicates on both
sides and inflates every metric. `by="random"` exists, but you have to ask for it, and
`leakage_report(ds)` shows exactly what it costs.

**Unsupervised axes encode to `-100`**, PyTorch's `CrossEntropyLoss` default
`ignore_index`. So combining a 39-class dense set with our own 2-class masks needs no
custom masking:

```python
loss = sum(F.cross_entropy(logits[a], batch["labels"][a]) for a in heads)
```

Rows from a source that never annotated an axis receive exactly zero gradient on that
head — verified, not assumed (`tests/test_builder.py`). TensorFlow has no such
convention, so the TF adapter emits an explicit mask and ships
`masked_sparse_categorical_crossentropy` to match the torch behaviour rather than
approximate it.

**The builder refuses to mix vocabularies.** A source with no crosswalk into the target
schema would inject native labels into the index — a class list of `["HC", "SC", "18",
"47"]` is two vocabularies pretending to be one. That raises unless you pass
`allow_unmapped=True`.

Also available: `class_weights()` (reef data is severely long-tailed; unweighted training
optimises for sand), `class_counts()`, and `supervision_coverage()` — which answers
"why is my growth-form head weak?" far faster than a loss curve.

## Storage and sharding

**Should everything go into a data lake?** Yes for some of it, never for other parts, and
the licence tier decides which — because "upload to storage" is two different acts:

| | Private cache | Public mirror |
|---|---|---|
| Legally | internal copying | **redistribution** |
| Admits | T0, T1, T2, **and T3 non-commercial** | T0, T1, T2 only |

Non-commercial material may be cached for our own research but never republished.
Conflating those is the one mistake here with real legal consequence, so it is enforced in
code:

```bash
marinedata mirror --target private-cache     # what may we cache?
marinedata mirror --target public-mirror     # what may we publish?
```

Four bars apply to every target, checked before tier: `no_derivatives`,
`provenance_defective`, TDM-basis (retention is time-limited by statute), and unknown
basis. Sharding is itself a derivative act, so `write_shards()` goes through the same gate.

```python
from marinedata.shard import write_shards

write_shards(dataset, "s3-staging/shards", split="train")  # WebDataset tar shards
```

Raw objects in cloud storage are the real training bottleneck — one GET per image starves
a GPU. Shards are ~512 MB, deterministic (zeroed mtimes), and carry `SHARD_MANIFEST.json`
with class lists, head widths and full lineage, plus a generated `ATTRIBUTION.md`.

Full rationale, retention policy and where to physically put things:
[`docs/STORAGE.md`](docs/STORAGE.md).

## Lineage — the audit artifact

Every build emits a record of what contributed, under which tier and legal basis, what
was excluded and why, and the obligations carried forward:

```python
lineage = md.build_lineage(
    list(result), registry.profile("ship-commercial"), excluded=list(result.excluded)
)
open("LINEAGE.json", "w").write(lineage.to_json())
print(lineage.attribution_text())  # for the model card / NOTICE file
```

Attach it to your model card. It is the answer to *"prove you had the right to train on
this"* — and, because the registry is public, a claim a stranger can check.

## Install

```bash
pip install marinedata
```

```bash
git clone https://github.com/reefsupport/marine-data && cd marine-data
uv venv && uv pip install -e ".[dev]" && pytest
```

## Contributing

Adding a dataset is a pull request against `registry/sources/`. Two rules:

1. **`verified_by` must cite a primary source** — a licence file you opened, a dataset
   card, written permission. Not "the paper says". Our own prior catalog recorded
   MarineInst20M as "Mixed / Open" on a paper's phrasing; the `LICENSE.txt` in the
   repository said CC-BY-NC-SA, over imagery including Getty and Shutterstock.
2. **Tier changes need a second reviewer.**

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Status

**59 sources across 24 capabilities; 9 loader layouts; 208 unit + 10 live integration tests.**

**14 sources verified against live fetched data** — including our own Hetzner buckets,
Coralscapes, ReefNet species images, MARRS reef soundscapes and AIMS satellite coral
mapping. `marinedata verify` re-checks them; `--unverified-only` lists the backlog.

Coverage by domain: coral and benthic (deepest), fish and mobile fauna, 3D and
photogrammetry, coastal ecosystems (mangrove, seagrass, bathymetry, debris),
bioacoustics, enhancement and depth.

Honest gaps:

- **One crosswalk is complete** (Coralscapes-39 → RS-Benthic). CATAMI, CoralNet, NCRMP
  and AGRRA are registered as schemas but not yet mapped.
- **Many fish and enhancement entries carry `method: secondary`** — transcribed from an
  internal catalog rather than checked at source. They work on `research` and are
  blocked from shipping profiles until someone opens the primary source. That backlog is
  deliberate and visible.
- **No fetchers yet.** Loaders read what is already on disk.
- **No Croissant emission yet.**
- Plankton and megafauna are absent; the schema accommodates them.

`marinedata check --profile ship-commercial` prints exactly what is blocked and why.

## Licence

Apache-2.0 for the code. CC-BY-4.0 for the registry metadata. We want both copied widely.

Built by [Reef Support](https://reef.support).
