# Datasheet: Reef Support Marine Data (v1.0, October 2026)

This datasheet follows the seven sections of Gebru et al., *Datasheets for Datasets* (Communications of the ACM, 2021).
It describes the four configurations published as
[`reefsupport/marine-data`](https://huggingface.co/datasets/reefsupport/marine-data). The dataset card on Hugging Face
holds the field-level reference and the full licence and citation tables. Counts are those of the release built on
2026-10-06.

## 1. Motivation

**For what purpose was the dataset created?** To provide ready-to-train, licence-traceable data for marine computer
vision: benthic and underwater-scene semantic segmentation, instance segmentation, and fish detection. Marine imagery is
spread across many datasets with different label schemes and licence terms, and the release collects openly licensed
sources into one consistent format.

**Who created it?** Reef Support B.V. ([reef.support](https://reef.support)), a marine conservation technology company.

**Who funded it?** Reef Support B.V.

## 2. Composition

**What do the instances represent?** Each row is one underwater or reef photograph with its annotation: a segmentation
mask (`coral-masks`, `scene-masks`, `instance-masks`) or bounding boxes (`fish-boxes`).

**How many instances are there?** 58,803 rows in four configurations. `instance-masks` and `fish-boxes` are built from
the same UIIS, UIIS10K and USIS10K images and are split per configuration.

| Configuration | Task | Rows | Train / validation / test |
|---|---|---|---|
| `coral-masks` | Benthic semantic segmentation | 5,997 | 4,585 / 569 / 843 |
| `scene-masks` | Underwater scene semantic segmentation | 1,598 | 1,181 / 163 / 254 |
| `instance-masks` | Underwater instance segmentation | 25,296 | 19,409 / 2,284 / 3,603 |
| `fish-boxes` | Object detection | 25,912 | 18,967 / 3,473 / 3,472 |

**Is it a sample of a larger set?** The release contains the images that passed conversion from eight sources:
Coralscapes, SUIM, UIIS, UIIS10K, USIS10K, the Roboflow Aquarium Dataset, Reef Support's benthic surveys, and Seaview
imagery with Reef Support masks. 35 `coral-masks` images and 12 `instance-masks` rows were dropped because no class map
or valid mask could be built.

**What data does each instance consist of?** An embedded image, the annotation, a `licence` and `attribution`, a source
id, a `split_group`, and the source's native label alongside a shared taxon node and coarse group where a mapping exists.

**Are there labels?** Yes. Annotations are the sources' own, except for the masks Reef Support drew on Seaview images
and Reef Support's benthic set. Boxes for UIIS, UIIS10K and USIS10K are derived from the upstream instance annotations.

**Is any information missing?** Location fields (`lat`, `lon`) are null unless the source provides them. The shared
taxon mapping is partial and the coarse group is often null.

**Are there recommended splits?** Yes, the `train`, `validation` and `test` splits in the data. They follow the upstream
splits where the source defines them, and a deterministic 80/10/10 hash of `split_group` otherwise.

**Errors, noise and redundancy.** The mask configurations were not checked for near-duplicate images, so near-identical
frames can fall on both sides of a split. `fish-boxes` ran a deduplication gate: no image spans two splits, and no upstream
test image is in train. The Seaview `split_group` (site partition plus the first five characters of the image id) is
provisional.

**Does the dataset contain confidential or sensitive data?** No. See "Ethics" below for people and locations.

## 3. Collection Process

**How was the data acquired?** Each source was downloaded from its authors' distribution and checked against the
registry before it was admitted. Coralscapes, SUIM, UIIS, UIIS10K, USIS10K and the Roboflow Aquarium Dataset are public
research datasets. The Seaview imagery is the XL Catlin Seaview Survey photo-quadrat dataset of The University of
Queensland. The benthic set is Reef Support's own imagery.

**Who collected it and over what timeframe?** The original collections were made by the source authors, as described in
their publications. Reef Support collected its own benthic imagery.

**Was ethical review conducted?** The release adds no new data collection. Ethical review of the original collections is
described, where it exists, in the source publications.

## 4. Preprocessing / Cleaning / Labeling

**What preprocessing was done?** Images and masks were re-encoded into Parquet shards. Per-image `class_map` or
`instances` fields were added, and splits and `split_group` were assigned. Each configuration ends with a
`CHECKSUMS.sha256`.

**How were the labels created?** The sources' own annotators created their labels. Reef Support drew the masks for the
Seaview images, decoded from stitched RGB renders with a fixed five-colour palette (Hard Coral = 1, Soft Coral = 2,
`0` = unlabelled). Every `annotator_type` in the mask configurations is `human`.

**Is the raw data available?** Yes, from the upstream distributions. The registry records the upstream location and
version of every source.

**Is the software available?** Yes: [github.com/reefsupport/marine-data](https://github.com/reefsupport/marine-data),
Apache-2.0. The release builder is `src/marinedata/hf_export.py`.

## 5. Uses

**What has the dataset been used for?** Reef Support uses it to train and evaluate its own models.

**What else could it be used for?** Benchmarking segmentation and detection models and studying cross-dataset transfer
in marine imagery.

**What should be kept in mind?** Models trained here may not transfer to other regions or imaging conditions. When
training on one configuration and evaluating on another, join on `image_sha256` first, since `instance-masks` and
`fish-boxes` share images.

### Ethics

**Diver / face presence.** Some images show divers or aquarium visitors, and `fish-boxes` has `human` and
`human divers` labels. Depicted people and rights holders can request removal through
the contacts in section 7.

**Sensitive-species location policy.** The release carries no coordinates of its own. Location fields are null unless
the source provides them.

**Licence per source.** Each source keeps its own licence. Every row carries `licence` and `attribution`, and the dataset
card lists the licence, the included material and the attribution to reproduce. The UIIS, UIIS10K and USIS10K authors
distribute under the Apache License 2.0. Their publications state that the images were gathered from public sources and
earlier datasets, so copyright in individual images may rest with third parties.

**Label-quality residuals.** The limitations below are the known residuals. No independent expert audit of the labels
has been run.

### Limitations

| Limitation | Detail |
|---|---|
| Dropped rows | 35 `coral-masks` images (24 of 1,250 in `reef-support-benthic-own`, 11 of 2,707 in `reef-support-seaview-labels`) and 12 `instance-masks` rows (6 of 4,628 in `uiis`, 6 of 10,048 in `uiis10k`) |
| Near-duplicates in mask configurations | Not checked; near-identical frames can fall on both sides of a split |
| Provisional Seaview grouping | `split_group` uses the site partition plus the first 5 characters of the image id |
| Shared images across configurations | `instance-masks` and `fish-boxes` share UIIS-family images and are split per configuration |
| Vocabularies differ per source | `label_native` is the source's own label; `taxon_node` and `coarse` are partial |
| Image provenance | UIIS-family images were gathered by their authors from public sources and earlier datasets |
| Geographic coverage | Uneven and source-driven, for example Coralscapes images come from Red Sea dive sites |
| Fixed image sizes | `scene-masks` and `fish-boxes` images are 640x480 |

## 6. Distribution

**How is it distributed?** On Hugging Face at
[huggingface.co/datasets/reefsupport/marine-data](https://huggingface.co/datasets/reefsupport/marine-data), as Parquet
shards of up to about 0.5 GB, 15.5 GB in total. Code and the registry are on
[GitHub](https://github.com/reefsupport/marine-data).

**Under what licence?** Per source: Apache-2.0, MIT, CC BY 4.0 or CC BY 3.0, each with required attribution. Reef Support's
own annotations are CC BY 4.0. The code is Apache-2.0. See `README.md` and the dataset card for the table.

**Have third parties imposed restrictions?** Attribution is required for every source. Rights in some UIIS-family
images may rest with third parties, as described above.

## 7. Maintenance

**Who maintains it?** Reef Support.

**How can the maintainers be contacted?** Through the
[GitHub issue tracker](https://github.com/reefsupport/marine-data/issues) or the
[contact page](https://www.reef.support/contact).

**Will it be updated?** New versions are released under a new version label and recorded in `CHANGELOG.md`.

**How can removal be requested?** Rights holders and depicted people can write through the contacts above.

**How should it be cited?** Use the BibTeX in `README.md` or `CITATION.cff`, and cite the upstream source papers listed
there for the sources you use.
