---
pretty_name: Reef Support Marine Data
license: cc0-1.0
language:
- en
task_categories: []
task_ids: []
annotations_creators:
- expert-generated
- found
source_datasets: []
size_categories:
- n<1K
tags:
- marine
- coral-reef
- underwater
- biology
- ecology
- conservation
- reef-support
configs:
- config_name: images
  data_files:
  - split: train
    path: data/images/train-*.parquet
  - split: validation
    path: data/images/validation-*.parquet
- config_name: pretrain-set
  data_files:
  - split: train
    path: data/pretrain-set/train-*.parquet
  - split: validation
    path: data/pretrain-set/validation-*.parquet
dataset_info:
- config_name: images
  features:
  - name: image_sha256
    dtype: string
  - name: image
    dtype: image
  - name: source_id
    dtype: string
  - name: source_ids
    dtype: string
  - name: split_group
    dtype: string
  splits:
  - name: train
    num_bytes: 35462
    num_examples: 22
  - name: validation
    num_bytes: 8904
    num_examples: 4
  download_size: 44366
  dataset_size: 44366
- config_name: pretrain-set
  features:
  - name: image_sha256
    dtype: string
  - name: source_id
    dtype: string
  - name: sample_key
    dtype: string
  splits:
  - name: train
    num_bytes: 3616
    num_examples: 22
  - name: validation
    num_bytes: 2267
    num_examples: 4
  download_size: 5883
  dataset_size: 5883
---

# Reef Support Marine Data

**Underwater imagery with segmentation masks and bounding boxes for coral-reef computer vision, with the licence and attribution of every row recorded.**

<p align="center">
  <a href="https://github.com/reefsupport/marine-data"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-reefsupport%2Fmarine--data-181717?logo=github"></a>
  <a href="https://reef.support"><img alt="reef.support" src="https://img.shields.io/badge/reef.support-0B6E99"></a>
  <a href="https://huggingface.co/reefsupport"><img alt="Hugging Face organisation" src="https://img.shields.io/badge/Hugging%20Face-reefsupport-FFD21E?logo=huggingface&logoColor=black"></a>
  <img alt="Licence: per source" src="https://img.shields.io/badge/licence-per%20source-blue">
  <img alt="Rows: 52" src="https://img.shields.io/badge/rows-52-2ea44f">
</p>

## Dataset summary

Reef Support Marine Data wp4-e2e collects two sources into two configurations. Every row carries its `licence` and `attribution`, so provenance stays attached to the pixels. The release is built with the [marine-data](https://github.com/reefsupport/marine-data) registry, which records the licence of every source before it is admitted.

The licences differ per source (CC0-1.0), so read [Licensing](#licensing) before you share or publish results.

| | |
|---|---|
| Configurations | 2: `images`, `pretrain-set` |
| Rows | 52 (train 44, validation 8) |
| Tasks | – |
| Format | Parquet shards with images and masks embedded, streamable |
| Size | 0.0 GB |
| Sources | 2: e2e-src-a, e2e-src-b |
| Version | wp4-e2e |
| Code | [github.com/reefsupport/marine-data](https://github.com/reefsupport/marine-data) |

## Supported tasks

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("reefsupport/marine-data", "images", split="train")
```

Configurations: `images`, `pretrain-set`. Each loads on its own.

## Dataset structure

### Configurations

| config | task | train | validation | total | GB |
|---|---|---|---|---|---|
| [`images`](#images) | Image pixels | 22 | 4 | 26 | 0.0 |
| [`pretrain-set`](#pretrain-set) | pretrain-set | 22 | 4 | 26 | 0.0 |

Splits are the upstream splits where the source provides them, otherwise a deterministic 80/10/10 hash of `split_group`.

Machine-readable metadata (Croissant) is served by the Hub at `https://huggingface.co/api/datasets/reefsupport/marine-data/croissant`. `marinedata release croissant` writes a richer file for a build, with per-field types and the sha256 of every shard.

### images

Every image once (pixels embedded), keyed by `image_sha256`. 26 images.

| source | images | licence |
|---|---|---|
| `e2e-src-a` | 13 | CC0-1.0 |
| `e2e-src-b` | 13 | CC0-1.0 |

### pretrain-set

`pretrain-set`: 26 rows.

| source | rows | train / val | licence |
|---|---|---|---|
| `e2e-src-a` | 13 | 11 / 2 | CC0-1.0 |
| `e2e-src-b` | 13 | 11 / 2 | CC0-1.0 |

### Data fields

Generated from the export layout of this build. A field with an en dash has no description yet.

`images`:

| field | type | description |
|---|---|---|
| `image_sha256` | string | SHA-256 of the image bytes. Stable id and join key across configs. |
| `image` | Image | RGB photo (embedded bytes). |
| `source_id` | string | Source id; see the licensing table. |
| `source_ids` | string | Comma-separated ids of every source that published these image bytes. |
| `split_group` | string | Group that is never divided across splits (site, station, transect or upstream group). |

`pretrain-set`:

| field | type | description |
|---|---|---|
| `image_sha256` | string | SHA-256 of the image bytes. Stable id and join key across configs. |
| `source_id` | string | Source id; see the licensing table. |
| `sample_key` | string | Key of one (image, source) sample. |

## Dataset creation

### Sources

Registry facts per source. An en dash means the registry does not record the field.

| Source | Version | Access class | Images (build / upstream) | Root digest | Checksums | Platform | Licence evidence | Per-row licence | Rights holder | Expected metadata | Other versions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `e2e-src-a` | v1 | open | – | – | pinned | – | – | no | – | date=none, depth=none, geo=none, site=none | – |
| `e2e-src-b` | v1 | open | – | – | pinned | – | – | no | – | date=none, depth=none, geo=none, site=none | – |

### Build and verification

No `GATES.json` accompanied this build, so no gate result is recorded here.

No `release.lock.yaml` accompanied this build.

Provenance: `provenance/RELEASE.json` (build facts, public-safe), `provenance/GATES.json` (gate
results) and `provenance/release.lock.yaml` (pinned inputs) ship with the data.

## Licensing

The data are not under one licence. Every row records its own `licence`, `annotation_licence` and `attribution`; the table lists each source with the licence its authors chose and the attribution it requires. Combine sources only in ways each licence allows. The table records the licences as Reef Support read them from the primary sources and is not legal advice.

| Source | In configs | Licence | Required attribution |
|---|---|---|---|
| `e2e-src-a` | `images`, `pretrain-set` | CC0-1.0 | e2e-src-a (CC0-1.0) |
| `e2e-src-b` | `images`, `pretrain-set` | CC0-1.0 | e2e-src-b (CC0-1.0) |

## Takedown requests and corrections

If you hold rights in an image, appear in one, or find a wrong label or licence, tell us:

- **Public:** open a [takedown request](https://github.com/reefsupport/marine-data/issues/new?template=takedown_request.yml),
  a [label error](https://github.com/reefsupport/marine-data/issues/new?template=label_error.yml) or a
  [licence question](https://github.com/reefsupport/marine-data/issues/new?template=licence_question.yml) on GitHub.
- **Private**, for example when the request involves personal data: write to hello@reef.support or use the
  [contact page](https://www.reef.support/contact).

Identify the rows by `image_sha256`, or by source and a description, and say what you ask for. We acknowledge a request
within a few working days. An accepted request removes the rows from the next revision and from later builds. The
[changelog](https://github.com/reefsupport/marine-data/blob/main/CHANGELOG.md) records the change without naming the
requester. The full procedure is in [SECURITY.md](https://github.com/reefsupport/marine-data/blob/main/SECURITY.md).

## Versions and changelog

| Version | Date | Notes |
|---|---|---|
| wp4-e2e | – | – |

Files of a released version are not edited. Corrections and removals, including accepted takedown requests, are published as a new version, and the changelog lists them.

Pin a version with `revision`:

```python
from datasets import load_dataset

ds = load_dataset("reefsupport/marine-data", "images", split="train", revision="wp4-e2e")
```

## Citation

```bibtex
@misc{reef_support_marine_data_2026,
  title = {marine-data},
  author = {Reef Support},
  year = {2026},
  version = {1.0.0},
  url = {https://huggingface.co/datasets/reefsupport/marine-data},
}
```

## Contact

Questions about the data: [hello@reef.support](mailto:hello@reef.support) or the [GitHub repository](https://github.com/reefsupport/marine-data).
