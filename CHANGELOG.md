# Changelog

Release notes for `marine-data`: the registry and tooling in this repository, and the datasets published as
[`reefsupport/marine-data`](https://huggingface.co/datasets/reefsupport/marine-data) on Hugging Face. Code history is in
git. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Nothing below is in a release yet. It is on `main` or on the Phase A branches named in the headings.

### On main

- Docs: install requirements, standalone snippets, fish-boxes statistics.
- Repository: security policy, code of conduct, issue and pull request templates; broader `.gitignore`; README install pinned to
  v1.0.0; annotator e-mails in the Labelbox test fixtures replaced by placeholders; neutral wording and placeholder paths in docs and
  script comments; the open dataset card no longer references the gated repository.

### Phase A: `feat/card-croissant` (merged into `feat/phase-a`; includes `feat/join` and `feat/gates`)

- Release gates (P1.6): `marinedata release gate BUILD --all --require-all` runs the licence, dedup, splits, masks, privacy (EXIF GPS), load, card and croissant gates and merges them into `GATES.json`; `--masks DIR` and `--metadata DIR --gates FILE` keep working in the same command (docs/GATES.md). The dedup gate no longer fails EXIF-transposed rows on `image_sha256`: it checks that the bytes decode and match the row's width/height.
- Dataset card and Croissant (P1.9): `release card` and `release croissant` generate `README.md` and `croissant.json` from a build; `release_croissant.py` is the single Croissant emitter (the old `croissant.py` is gone) and the croissant gate validates the file it writes. Validation no longer alters the document or the shared context.
- Per-row metadata resolvers (P2.1): `marinedata metadata resolve --source ID --out DIR` resolves geo, date, site and quadrat per image for four sources.
- SampleRow v3, per-row licence layer and metadata gate (P2.2, P2.4): `release gate --metadata DIR` checks resolver coverage floors and per-row licences.
- Seaview benthic points table (P3.1, P2.5): `marinedata tasks build-points` writes it keyed by `image_sha256`; `--metadata-dir` joins the resolved lat/lon/date/licence onto it (P2.5).
- HF `points` config (P2.6): geo/date/licence columns joined by `image_sha256`; `metadata` and `points` agree on lat/lon/date/date_precision for every image and the resolver output wins over a staged value.

### Phase A: `feat/repo-hygiene` (merged into `feat/phase-a`)

#### Added

- Takedown path: a "Takedown requests" section in `SECURITY.md` with a public form and a private e-mail route, and four
  issue forms (new source, takedown request, label error, licence question). `.github/CODEOWNERS`.
- `CONTRIBUTING.md`: data-problem section, development setup, `make ci`, commit and pull request conventions.
- `README.md`: per-config loading table, a note on why label-index masks look black, takedown pointers.
  `CITATION.cff`: version 1.0.0, contact, licence message and references for the dataset and its source papers.

### Phase A: `feat/phase-a`

#### Added

- `marinedata release lock` and `release build --lock`: a committed `release.lock.yaml` pins every source by `root_digest`,
  and a build refuses a source that is not in the lock.
- Masks gate, `marinedata release gate --masks BUILD_DIR`: checks mask size, EXIF orientation, class ids, dtype, instance
  ids and the empty-mask rate per source, and merges its result into `GATES.json`. Mask export no longer resizes silently.
- One S3 credential resolver (`marinedata.s3_auth`) and one client factory (`marinedata.s3_client`); writers refuse to run
  anonymously. A test (`tests/test_s3_auth_boundary.py`) keeps every other module away from the rclone config and from
  boto3 client constructors.
- `docs/sources.md` and `docs/configs.md`, generated from the registry by `scripts/gen_sources_docs.py`
  (`make docs`; `make docs-check` runs in `make ci`). `make ci` also runs the gate tests and the masks gate on a committed
  2-image fixture.
- `.github/dependabot.yml`: weekly, grouped updates for pip and GitHub Actions.
- Release build image: `Dockerfile` (python:3.12-slim by digest, uv, ffmpeg, git, non-root) and `.dockerignore`; `make
  image` builds `marinedata:$(git describe)`, `make image-check` runs `--version` and `check --profile research` in
  the container and enforces the 1.5 GB cap. `marinedata --version` is new.
- `marinedata release publish --release-id ID --flavour open|nc [--go]`: the dry run prints the plan (repo, configs,
  files and bytes, card, lock sha, tag `v<id>`) and refuses without a passing `GATES.json`; `--go` needs `HF_TOKEN`,
  uploads one commit, tags it, and refuses an existing tag and the frozen v1.0 without `--allow-frozen`.
- `deploy/release-vm/`: `create.sh` (prints the `hcloud` commands; `--run` is opt-in), `bootstrap.sh` (cloud-init
  user-data: lock, build, export, gate, publish dry run; never `--go`) and a README with prerequisites, cost, teardown
  and what to copy back.

#### Changed

- `marinedata check` separates expected denials from unexpected ones and exits 0 when only expected denials remain.
  `marinedata lineage` exits 0 on denials.
- Fetching verifies integrity (hash-pinned manifest, checksum markers, retries on truncated transfers). `fetch.py` is now the
  `marinedata.fetch` package; behaviour is unchanged.
- Lineage describes the build that happened, not the intent, and verification claims in the registry are computed instead of
  typed by hand.
- Three sources that were never ingested (`noaa-coral-icra`, `plasticinwater`, `mouss-seg`) moved from open to unknown until
  their licence is stated.
- The old "Data issue" form is retired; the label error, licence question and takedown forms cover its cases.

#### Fixed

- The licence gate failed open on three paths; it now fails closed.
- Nightly `verify-layouts`: the job no longer swallows failures (a fetch outage skips, a layout mismatch fails). Its three
  failing tests are repaired: the face-detector test downloaded a Git LFS pointer instead of the YuNet weights, and the
  coralscapes tests assumed `items` counts images and a complete sample root.

### Phase A: briefs integrated into `feat/phase-a` (P1.13)

Every brief named here has a report (`reports/2026-10-08-PA-<id>.md`).

- Registry as the single truth (P1.2, P1.2b, P1.2c, P1.2d): every ingest-spec id has an entry, licences carry evidence, root digests and the reconstructed v1.0 lock are pinned; `marinedata check` reports 174 permitted / 79 expected / 0 unexpected.
- `marinedata check` separates expected from unexpected denials (P1.3); `release lock` and `release build --lock` (P1.5).
- Signed bucket reads and fetch integrity (P1.4); every S3 constructor goes through the resolver and `fetch.py` is the `fetch/` package (P1.4b).
- `release publish` dry run and the release build image (P1.7); one Croissant emitter and one `release gate` subparser (P1.9b); generated docs and CI gates (P1.10).
- Mask configs gain exif_transposed and EXIF-rotated JPEGs are re-encoded from v1.1; v1.0 files unchanged.
- `release build` refuses the frozen v1.0 (release id or lock) unless `--allow-frozen`; `tests/golden/v1.0-configs-schema.json` pins the 4 published configs as additive-only.
- `release gate --require-all` runs every gate (privacy skippable only without image bytes); `release publish` takes its verdict from `marinedata.gates`, uploads with `upload_large_folder` and sends only data parquet, README.md, croissant.json and `provenance/{RELEASE.json,GATES.json,release.lock.yaml}`.
- Integration (P1.13): `docs/PHASE_A.md`, `docs/cli.md` generated from argparse, `task_layers/configs/` split into modules with re-exports.

## [1.0.0] - 2026-10-07

First public release.

### Added

- Training label mapping, `marinedata.labels`: fixed class ids for every published source in three schemes
  (`benthic-coarse`, `coral-binary`, `scene`), with 255 as the ignore value. It provides `lut`, `remap_mask`,
  `remap_row`, `map_label` and `supervised_classes`, options to ignore dead or bleached coral and chosen labels, and an
  optional `scheme=` argument for `to_torch_dataset`. See `docs/LABELS.md`.
- Label policy defaults for joint training. In `benthic-coarse` and `coral-binary`, dead coral (the `RECENTLY_DEAD` and
  `OLD_DEAD` conditions, including `dead clam`) goes to 255 by default; bleached coral stays hard coral. Coralscapes
  `unknown hard substrate` is 255 in every scheme, and `sponge`, `anemone` and `algae covered substrate` map to
  `OTHER_FAUNA`, `OTHER_FAUNA` and `ALGAE` in `benthic-coarse`. `exclude_conditions=()` restores the registry behaviour.
  The registry tasks and crosswalks are unchanged. See "Label policy" in `docs/LABELS.md`.

### Published datasets (v1.0, October 2026)

- Four Parquet configurations with images and masks embedded, 58,803 rows and 15.5 GB in total: `coral-masks` (5,997),
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

[Unreleased]: https://github.com/reefsupport/marine-data/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/reefsupport/marine-data/releases/tag/v1.0.0
