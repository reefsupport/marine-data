# Changelog

Release entries for the published dataset (`reefsupport/marine-data`), not for
the `marinedata` tool itself — code changes are tracked in git history. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v1] — 2026-09-24

First public release. See `docs/DATASHEET.md` for the full datasheet and
`RELEASE.json` in the release directory (`_release/2026-09-24-neardup/releases/v1/`,
outside this repo, produced by `marinedata release`) for the frozen manifest this build
was cut from.

### Added
- 9 imagery sources staged, sha256-verified and deduplicated: NOAA PIFSC bleaching,
  5 Roboflow Universe exports, Reef Support's own benthic and bleaching surveys, and
  CoralSCOP model-output pseudo-masks.
- 69,600 unique-sha images in the `images` config; 6 label-only task configs
  (`benthic-coarse`, `benthic-l2`, `bleaching-condition`, `coral-health-binary`,
  `masks`, `general-pretraining`) joined on `image_sha256`; `coralscop-pseudo-masks`
  as a separate, never-eval config.
- A frozen group split (train/validation/test), near-duplicate handling
  (sha256 + dHash-64, Hamming <=4 union before splitting, Hamming <=8 never-eval
  exclusion, chain guard), and a byte-identical split-map regeneration test.
- Licence, tier and citation recorded per source in the registry and in the dataset
  card's sources table.
- `docs/DATASHEET.md` (the 7 Gebru et al. sections), `CITATION.cff`, BibTeX in
  `README.md`, and Croissant 1.0 (+ RAI) metadata (`src/marinedata/croissant.py`,
  `docs/croissant-v1.json`).

### Known limitations (quantified in `docs/DATASHEET.md`)
- `coral-genus-caribbean` carries 0 image labels and is dropped from the HF configs.
- CoralSCOP alone is 54% of images; habitat is 100% shallow coral reef; no depth,
  region, platform, GPS or capture date is recorded per sample.
- 7,085 images are byte-distinct duplicates at Hamming distance 0; 14,265 near-dup
  pairs exist under the release's union rule; no whole-corpus confirm-hash pass has
  been run yet.
- The realized split is 85.7/7.2/7.1 against a 70/15/15 target.
- No label-quality expert audit has been run; residuals are listed, not faked.

### Not yet done
- No DOI (Zenodo or otherwise) has been minted for this release.
- No external domain-expert review of the datasheet or labels.
- The Hub repository has not been published (`marinedata.hf_upload` has never been run
  with `--execute`).
