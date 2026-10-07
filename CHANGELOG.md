# Changelog

Release notes for `marine-data`: the registry and tooling in this repository, and the datasets published as
[`reefsupport/marine-data`](https://huggingface.co/datasets/reefsupport/marine-data) on Hugging Face. Code history is in
git. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-07

First public release.

### Published datasets (v1.0, October 2026)

- Four Parquet configurations with images and masks embedded, 58,803 rows in total: `coral-masks` (5,997),
  `scene-masks` (1,598), `instance-masks` (25,296) and `fish-boxes` (25,912).
- Eight sources: Coralscapes, SUIM, UIIS, UIIS10K, USIS10K, Roboflow Aquarium Dataset, Reef Support's benthic
  surveys, and Seaview imagery with Reef Support masks.
- Every row carries its source `licence` and `attribution`. Splits follow the upstream splits where a source defines
  them, and a deterministic 80/10/10 group hash otherwise.
- Each configuration ships `README.md`, `LICENSE`, `NOTICE` and `CHECKSUMS.sha256`. The `fish-boxes` build runs a
  deduplication gate so that no image spans two splits.
- Seaview images are recorded as CC BY 3.0, the licence stated by the University of Queensland. Masks drawn by Reef
  Support on those images are CC BY 4.0.

### Registry and licence gate

- A registry of 143 sources with licence tier, legal basis, flags, access rules, coverage and domain-shift notes, each
  verified against a primary source.
- Release profiles, and a gate that admits or refuses every source for a profile and records the reason.
- Faceted query by capability, annotation kind, modality, region and provenance, with an audit trail of exclusions.
- Lineage records of contributing sources, excluded sources and carried obligations.

### Datasets, labels and loaders

- Dataset builder with group-wise splits, harmonised label schemas, crosswalks and PyTorch, TensorFlow and pandas
  adapters.
- Label and taxonomy tooling, including WoRMS and IUCN-linked taxon nodes and label-quality checks.

### Release pipeline

- Near-duplicate grouping, a frozen split map, Parquet export, per-file checksums, release verification and Croissant
  metadata (`docs/croissant-v1.json`).
- Ingestion tooling for adding a source from its upstream distribution, with resumable transfers and checksum markers.

### Documentation

- `README.md`, `docs/DATASHEET.md`, `docs/LEGAL.md`, `CITATION.cff` and `CONTRIBUTING.md`.

### Fixed

- `QueryResult.summary()` and the `marinedata list` and `marinedata show` commands read the source count fields
  (`n_images`, `n_files`) instead of a non-existent attribute.
- The Seaview image licence is CC BY 3.0, not CC BY 3.0 AU.

### Known limitations

See "Considerations and limitations" in the dataset card and section 5 of `docs/DATASHEET.md`. In brief: the mask
configurations were not checked for near-duplicates across splits, the Seaview `split_group` is provisional, label
vocabularies differ per source, and image provenance for the UIIS family is described in the licensing section.
