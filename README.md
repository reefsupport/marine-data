# marine-data

A licence-aware registry and set of dataloaders for marine and coral-reef datasets, and the open image datasets built from it.

Marine computer vision draws on many datasets, with incompatible label schemes and a wide range of licence terms.
Some are public domain, some forbid commercial use, and some forbid derivative works, which rules out masks, crops and
augmentations. `marine-data` records the licence of every source before the source is used, so that "every label in
this training set was permitted for this purpose" can be checked by code instead of remembered.

The project has two parts:

- **The registry and tooling** (`src/marinedata`, `registry/`): per-source licence tiers, access rules, label
  crosswalks, dataloaders, a dataset builder and a release pipeline. The registry lists 143 sources and defines 26
  capabilities.
- **The published datasets** on Hugging Face: [`reefsupport/marine-data`](https://huggingface.co/datasets/reefsupport/marine-data),
  58,803 images with masks or bounding boxes, each row carrying its own `licence` and `attribution` fields.

This is metadata, not legal advice. Tier assignments record Reef Support's reading of primary sources, cited per entry,
and are a starting point for your own review. See [`docs/LEGAL.md`](docs/LEGAL.md).

## Published datasets

Release v1.0 (October 2026) is a set of four Parquet configs with images and masks embedded. Splits follow the
upstream splits where a source defines them, and a deterministic 80/10/10 group hash otherwise.

| Config | Task | Images | Sources |
|---|---|---|---|
| [`coral-masks`](https://huggingface.co/datasets/reefsupport/marine-data/viewer/coral-masks) | Benthic semantic segmentation | 5,997 | Coralscapes, Reef Support benthic surveys, Seaview imagery with Reef Support masks |
| [`scene-masks`](https://huggingface.co/datasets/reefsupport/marine-data/viewer/scene-masks) | Underwater scene semantic segmentation | 1,598 | SUIM |
| [`instance-masks`](https://huggingface.co/datasets/reefsupport/marine-data/viewer/instance-masks) | Underwater instance segmentation | 25,296 | UIIS, UIIS10K, USIS10K |
| [`fish-boxes`](https://huggingface.co/datasets/reefsupport/marine-data/viewer/fish-boxes) | Object detection (fish and fauna boxes) | 25,912 | UIIS, UIIS10K, USIS10K, Roboflow Aquarium Dataset |

Load a config with the `datasets` library, in one of three ways.

Quick look in a notebook: stream the split and read the first row.

```python
from datasets import load_dataset

stream = load_dataset("reefsupport/marine-data", "coral-masks", split="train", streaming=True)
row = next(iter(stream))
print(row["licence"], row["attribution"])
```

Scripts and training: download the config once, cached afterwards. `coral-masks` is 12.6 GB (all splits are fetched);
to try the pipeline first, use `scene-masks` (0.2 GB) or the one-source form below.

```python
from datasets import load_dataset

ds = load_dataset("reefsupport/marine-data", "coral-masks", split="train")
```

One source only: download just that source's shards (file names are `<split>-<source>-<n>-of-<total>.parquet`).

```python
from datasets import load_dataset

ds = load_dataset(
    "reefsupport/marine-data",
    "coral-masks",
    split="train",
    data_files={"train": "data/coral-masks/train-reef-support-seaview-labels-*.parquet"},
)
```

In a plain Python script, stopping a stream after a few rows can keep the process from exiting (an upstream issue in
pyarrow's dataset scanner). Download the config or a single source instead.

Licences differ per source. Keep the `attribution` column when you share or publish results, and see
[Licensing](#licensing) below. The dataset card documents fields, splits, annotation process and known limitations.

## Install

Requires Python 3.10 or newer and git. The package is not on PyPI; install the v1.0.0 release from GitHub:

```bash
pip install "marinedata[hf] @ git+https://github.com/reefsupport/marine-data@v1.0.0"
```

To work on the code, clone the release and install it in editable mode:

```bash
git clone --branch v1.0.0 https://github.com/reefsupport/marine-data
cd marine-data
uv venv && uv pip install -e ".[dev]"
```

Optional extras pull in heavier dependencies only when needed: `pandas`, `torch`,
`tf`, `hf`, `croissant`, `geo`, `eval` and `quality`.

## Quick start

Ask the registry which sources may be used for a purpose, and what the licence gate excluded:

```python
import marinedata as md

# Which sources may train a closed-weights benthic segmenter?
result = md.find(task="benthic-segmentation", profile="ship-commercial")
print(result.summary())  # matched sources with tier and image count, then each exclusion
for source in result.sources:
    print(source.id, source.licence.id)

# What did the gate exclude, and why?
for decision in result.excluded:
    print(decision.reason)
```

The same check from the command line:

```bash
marinedata check --profile ship-commercial          # every source, admitted or refused, with the reason
marinedata list --profile research --task benthic-segmentation
marinedata show coralscapes
```

## Label mapping

The masks on Hugging Face keep each source's native ids, and those ids differ between sources. `marinedata.labels` maps
them onto three fixed schemes, `benthic-coarse`, `coral-binary` and `scene`, with 255 as the ignore value. The example
needs the `hf` extra (`uv pip install -e ".[hf]"`), which adds `datasets` and Pillow:

```python
from datasets import load_dataset
from marinedata import labels

stream = load_dataset("reefsupport/marine-data", "coral-masks", split="train", streaming=True)
row = labels.remap_row(next(iter(stream)), "benthic-coarse")  # adds `label`, an HxW uint8 array
names = labels.class_names("benthic-coarse")
present = sorted({int(v) for v in row["label"].ravel()} - {labels.IGNORE_INDEX})
print(row["source"], [names[i] for i in present])
```

In a script, download the config or one source as described under [Published datasets](#published-datasets) and pass its
rows to `remap_row` the same way:

```python
from datasets import load_dataset
from marinedata import labels

ds = load_dataset("reefsupport/marine-data", "scene-masks", split="train")
row = labels.remap_row(ds[0], "scene")
```

```python
from datasets import load_dataset
from marinedata import labels

ds = load_dataset(
    "reefsupport/marine-data",
    "coral-masks",
    split="train",
    data_files={"train": "data/coral-masks/train-reef-support-seaview-labels-*.parquet"},
)
row = labels.remap_row(ds[0], "benthic-coarse")
```

The other configs use the `scene` scheme. `scene-masks` rows work with `remap_row` as above:

```python
from datasets import load_dataset
from marinedata import labels

stream = load_dataset("reefsupport/marine-data", "scene-masks", split="train", streaming=True)
row = labels.remap_row(next(iter(stream)), "scene")
names = labels.class_names("scene")
present = sorted({int(v) for v in row["label"].ravel()} - {labels.IGNORE_INDEX})
print(row["source"], [names[i] for i in present])
```

`instance-masks` rows carry a list of `instances` instead of one semantic mask, so map each instance label with
`labels.map_label` (it returns `None` for a label the scheme does not cover):

```python
from datasets import load_dataset
from marinedata import labels

stream = load_dataset("reefsupport/marine-data", "instance-masks", split="train", streaming=True)
row = next(iter(stream))
ids = [labels.map_label(row["source"], inst["label_native"], "scene") for inst in row["instances"]]
print(row["source"], ids)
```

`fish-boxes` rows have `boxes` and name their source in `source_id`:

```python
from datasets import load_dataset
from marinedata import labels

stream = load_dataset("reefsupport/marine-data", "fish-boxes", split="train", streaming=True)
row = next(iter(stream))
ids = [labels.map_label(row["source_id"], box["label"], "scene") for box in row["boxes"]]
print(row["source_id"], ids)
```

For a dataset built with `marinedata.builder`,
`to_torch_dataset(dataset, decode_masks=True, scheme="benthic-coarse")` remaps each decoded mask to the scheme ids.

Some sources annotate only a subset of the classes, and for them 255 means "not annotated" rather than background.
[`docs/LABELS.md`](docs/LABELS.md) lists the class ids per scheme, how every source maps, and the rules for training with
partial sources, dead and bleached coral, and the Coralscapes classes. For joint training, `benthic-coarse` and
`coral-binary` send dead coral and the Coralscapes `unknown hard substrate` class to 255 by default and map a few
Coralscapes labels the registry leaves open; `exclude_conditions=()` restores the registry behaviour, and the
[label policy](docs/LABELS.md#label-policy) lists every difference.

## How the registry works

Each source is described along three facets that can be queried together.

| Facet | Question it answers |
|---|---|
| Licence tier and flags | What may be done with the data? |
| Capability | What is the source useful for: benthic segmentation, fish detection, 3D, pretraining? |
| Coverage and domain shift | Will it work in the region where it is deployed? |

### Licence tiers

| Tier | Meaning |
|---|---|
| `T0_OWN` | Rights held outright |
| `T1_PERMISSIVE` | CC0, CC BY, Apache-2.0, MIT, US-Government public domain |
| `T2_COPYLEFT` | CC BY-SA, GPL: usable if the derivative is shared alike |
| `T3_NONCOMMERCIAL` | CC BY-NC and variants: research and internal use |
| `T4_TDM_ONLY` | No grant; lawful access under a statutory text-and-data-mining exception |
| `TX_PROHIBITED` | Provenance-defective or contract-blocked: never usable |

Tier alone is not enough, so licences also carry flags. `no_derivatives` marks terms that forbid masks, crops and
augmentations, which makes a source unusable for training and not merely non-commercial. `contract_gated` marks access
that required assent to terms, `provenance_defective` marks a grant the licensor appears not to hold, and
`share_alike` and `attribution_required` record the remaining obligations.

### Profiles

A profile declares what a build is for. The gate raises an error when a source is not admitted; it does not warn.

| Profile | Admits | Intended for |
|---|---|---|
| `ship-commercial` | T0, T1 | Closed weights in a paid product |
| `ship-open` | T0, T1, T2 | Weights released under a share-alike licence |
| `ship-noncommercial` | T0, T1, T2, T3 | Public non-commercial releases |
| `research` | T0, T1, T2, T3 | Papers and benchmarks that are not shipped |
| `pretrain-eu` | T0 to T4 | Self-supervised pretraining under the EU text-and-data-mining exception; requires a `legal_opinion_ref` |

### Domain shift as data

Cross-region performance drops are a common deployment failure, and entries record them together with the classes a
source lacks:

```yaml
domain_shift:
  trained_regions: [red-sea]
  missing_classes: ["soft coral", "gorgonian / sea fan", "octocoral (any)"]
```

Coralscapes, for example, has no soft-coral class. A model trained only on it has no valid label for octocorals, which
make up a large share of annotations on Caribbean reefs.

## Dataloaders

Sources declare an on-disk layout instead of shipping bespoke code. Registered layouts include `image-folder`,
`image-mask-pairs`, `coco-json`, `yolo-txt`, `csv-points`, `labelbox-ndjson`, `audio-clips` and `metadata-only`.
Loaders never download data, and an empty result is always an error.

```python
import marinedata as md
from marinedata.loaders import build_loader

registry = md.Registry.load()
loader = build_loader(registry.source("suim"), "/data/suim")
for sample in loader:
    print(sample.licence_tier)  # provenance travels with every sample
```

A declared layout is only a hypothesis until it is run against real data. `marinedata fetch <source> --limit 100`
downloads a bounded sample, and `marinedata verify` runs each declared layout against it and records the result.

## Harmonisation

Label schemes are mapped onto canonical axes (`taxon`, `form`, `condition`) by crosswalks, and each mapping records what
it loses as `exact`, `coarsened` or `approximate`. A source label with no canonical equivalent leaves the axis
unsupervised instead of being coerced into the nearest class. The registry holds 41 crosswalks.

## Building a training set

```python
import marinedata as md

registry = md.Registry.load()
dataset = md.DatasetBuilder(
    registry,
    profile="ship-commercial",
    roots={"coralscapes": "/data/coralscapes"},
).build()

dataset.split(by="site")  # group-wise split, the default
print(dataset.summary())

frame = dataset.to_pandas()
torch_ds = dataset.to_torch(split="train")
```

- **Splits are group-wise by default.** Consecutive transect frames overlap heavily, so a random split places
  near-duplicates on both sides and inflates metrics. `by="random"` is available when requested explicitly.
- **Unsupervised axes encode as `-100`**, the default `ignore_index` of PyTorch's `CrossEntropyLoss`. Datasets with
  different label sets can be combined without custom masking. The TensorFlow adapter emits an explicit mask instead.
- **Vocabularies are not mixed.** A source without a crosswalk into the target schema raises an error unless
  `allow_unmapped=True` is passed.
- `class_weights()`, `class_counts()` and `supervision_coverage()` help with long-tailed class distributions.

## Lineage

Every build can emit a record of which sources contributed, under which tier and legal basis, what was excluded and
why, and which obligations carry forward:

```python
import marinedata as md

registry = md.Registry.load()
result = md.find(task="benthic-segmentation", profile="ship-commercial")
lineage = md.build_lineage(
    list(result.sources),
    registry.profile("ship-commercial"),
    excluded=list(result.excluded),
)
with open("LINEAGE.json", "w") as handle:
    handle.write(lineage.to_json())
print(lineage.attribution_text())  # for a model card or NOTICE file
```

## Command-line overview

`marinedata --help` lists every command. The main groups are:

| Purpose | Commands |
|---|---|
| Explore the registry | `list`, `show`, `check`, `doctor`, `taxa`, `labels`, `taxonomy` |
| Check sources against real data | `fetch`, `verify`, `registry verify` |
| Build datasets | `release`, `splits`, `dedup`, `decon`, `export-wds`, `manifest`, `verify-release` |
| Quality and evaluation | `quality`, `privacy-scan`, `labelquality`, `eval`, `bench`, `captions` |
| Provenance | `lineage`, `mirror` |
| Add or ingest sources | `add`, `ingest`, `ingest-source`, `ingest-batch`, `metadata` |

`marinedata mirror <source>` reports what may be copied into storage you control and what is refused, because copying
a source is a redistribution decision as well as a technical one.

## Licensing

**Code.** The code in this repository is released under the Apache License 2.0 ([`LICENSE`](LICENSE)).

**Data.** The code licence does not apply to data. Each source keeps its own licence, recorded in `registry/` and in
the `licence` and `attribution` columns of every published row. The sources in the published release are:

| Source | What is included | Licence | Required attribution |
|---|---|---|---|
| [Coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) | `coral-masks`: 2,075 images | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sauder et al. 2025; keep the licence text and note changes (Parquet re-encoding, class map) |
| Reef Support benthic surveys | `coral-masks`: 1,226 images | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Reef Support, https://reef.support |
| [Seaview survey photographs](https://doi.org/10.14264/UQL.2019.930) with Reef Support masks | `coral-masks`: 2,696 images | Images [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); masks CC BY 4.0 | Images: González-Rivero et al., University of Queensland, doi:10.14264/UQL.2019.930. Masks: Reef Support |
| [SUIM](https://github.com/IRVLab/SUIM) | `scene-masks`: 1,598 images | [MIT](https://opensource.org/license/mit) | Islam et al. 2020; Copyright (c) 2020 Md Jahidul Islam |
| UIIS, UIIS10K, USIS10K | `instance-masks`: 25,296 images; part of `fish-boxes` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), as distributed by the authors | Lian et al. 2023; Li et al. 2025; Lian et al. 2024 |
| [Roboflow Aquarium Dataset](https://public.roboflow.com/object-detection/aquarium) | `fish-boxes`: 637 images | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Roboflow, Aquarium Dataset, 2020 |

Images in the UIIS, UIIS10K and USIS10K configurations originate from datasets distributed by their authors under the
Apache License 2.0. The original publications state that the images were gathered from public sources and earlier
datasets, so copyright in individual images may rest with third parties. Rights holders can request removal by opening
an [issue](https://github.com/reefsupport/marine-data/issues) or through the [contact page](https://www.reef.support/contact).

Sources whose terms forbid redistribution are not part of the published datasets. The registry still records
them, so that their status is visible and gated.

## Limitations

- Labels follow each upstream vocabulary. Native labels are kept and mapped to shared nodes only where a crosswalk
  exists.
- 35 `coral-masks` images and 12 `instance-masks` rows were dropped because no class map was available.
- The mask configs have not been checked for near-duplicate images.
- Some frames may show divers or visitors.
- Registry coverage is uneven: some entries are verified from secondary sources and are blocked from shipping
  profiles until a primary source is checked.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). To cite the release and the software:

```bibtex
@misc{reefsupport2026marinedata,
  title        = {Reef Support Marine Data},
  author       = {{Reef Support}},
  year         = {2026},
  version      = {1.0},
  howpublished = {\url{https://huggingface.co/datasets/reefsupport/marine-data}},
  note         = {v1.0 (October 2026). Licences are set per source; see the dataset card.}
}

@software{reefsupport2026marinedatacode,
  title   = {marine-data: a licence-aware registry and dataloaders for marine datasets},
  author  = {{Reef Support}},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/reefsupport/marine-data},
  license = {Apache-2.0}
}
```

Please also cite the upstream sources you use.

```bibtex
@inproceedings{sauder2025coralscapesdatasetsemanticscene,
  title     = {The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs},
  author    = {Sauder, Jonathan and Domazetoski, Viktor and Banc-Prandi, Guilhem and Perna, Gabriela and Meibom, Anders and Tuia, Devis},
  booktitle = {Proceedings of the International Conference on Computer Vision Joint Workshop on Marine Vision},
  year      = {2025}
}

@misc{gonzalezrivero2019seaview,
  title     = {Seaview Survey Photo-quadrat and Image Classification Dataset},
  author    = {Gonz{\'a}lez-Rivero, Manuel and others},
  year      = {2019},
  publisher = {The University of Queensland},
  doi       = {10.14264/UQL.2019.930},
  note      = {CC BY 3.0}
}

@inproceedings{islam2020suim,
  title     = {Semantic Segmentation of Underwater Imagery: Dataset and Benchmark},
  author    = {Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2020}
}

@inproceedings{lian2023watermask,
  title     = {WaterMask: Instance Segmentation for Underwater Imagery},
  author    = {Lian, Shijie and Li, Hua and Cong, Runmin and Li, Suqi and Zhang, Wei and Kwong, Sam},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {1305--1315},
  year      = {2023}
}

@article{li2025uwsam,
  title   = {Advancing Marine Research: UWSAM Framework and UIIS10K Dataset for Precise Underwater Instance Segmentation},
  author  = {Li, Hua and Lian, Shijie and others},
  journal = {arXiv preprint arXiv:2505.15581},
  year    = {2025}
}

@inproceedings{lian2024usis10k,
  title     = {Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset},
  author    = {Lian, Shijie and Zhang, Ziyi and Li, Hua and Li, Wenjie and Yang, Laurence Tianruo and Kwong, Sam and Cong, Runmin},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {235},
  pages     = {29545--29559},
  year      = {2024}
}

@misc{roboflow2020aquarium,
  title        = {Aquarium Dataset},
  author       = {{Roboflow}},
  year         = {2020},
  howpublished = {\url{https://public.roboflow.com/object-detection/aquarium}},
  note         = {CC BY 4.0}
}
```

## Contributing

The most valuable contribution is a verified licence. Adding a source is a pull request that adds an entry to the
registry, under two rules:

1. `verified_by` must cite a primary source: a licence file, a dataset card, written permission, or a terms-of-use
   page with the date it was read. A paper's phrasing or a hosting platform's default is not enough.
2. Changes to a source's tier need a second reviewer.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## Contact

Questions, corrections to a licence entry and removal requests: open an
[issue](https://github.com/reefsupport/marine-data/issues) or use the [contact page](https://www.reef.support/contact).
Built by [Reef Support](https://reef.support).
