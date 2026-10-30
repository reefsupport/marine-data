# Changelog

Release notes for `marine-data`: the registry and tooling in this repository, and the datasets published as
[`reefsupport/marine-data`](https://huggingface.co/datasets/reefsupport/marine-data) on Hugging Face. Code history is in
git. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Nothing below is in a release yet. It is on `main` or on the Phase A branches named in the headings.

### v1.2.0: Coralscapes V2 (in preparation)

- **Coralscapes V2 joins `semseg` and `instances` (additive).** V2 supersedes V1 for Coralscapes tasks; V1 configs are kept frozen for reproducibility. The new source id is `coralscapes-v2` ([josauder/coralscapesV2](https://huggingface.co/datasets/josauder/coralscapesV2), Apache-2.0, [arXiv:2609.12826](https://arxiv.org/abs/2609.12826), pinned revision `8a4567b2fbec14882e09aebff63a7a1ca5e24dca`): 2,433 frames of 1024 x 2048 pixels from 41 Red Sea dive sites, 95 classes including soft corals and black coral. The official split is kept: train 1,790, validation 170, test 473. No config is added or renamed, and no v1.0, v1.1.0 or v1.1.1 file, row, id or path changes. Downstream model pins are unchanged.
- **Encodings.** `semseg` holds the V2 masks as `uint8` PNGs with the upstream class ids 1 to 95; pixel `0` is unlabelled and is not counted. `instances` holds the `uint16` instance PNGs unchanged from upstream: a pixel below 2000 is the class id of a stuff class, a pixel of 2000 or more is `class_id * 2000 + instance_index`, and fish is class 25.
- **Label space: taxonomy 3.1.0 (additive).** Three `rs-benthic-v1` nodes, `CN_ANTIPATHARIA` (WoRMS AphiaID 22549), `SC_ALCYONIIDAE` (125269) and `SC_NEPHTHEIDAE` (146762), and 95 edges are added; taxonomy 3.0.0 is untouched. The crosswalk `coralscapes-95` maps all 95 V2 classes into `rs-benthic-v1`: 60 exact, 25 coarsened, 9 approximate and 1 unmappable (`hard substrate`; its pixels train as ignore, `255`). Eight V2 classes leave the hard-coral target of the V1 class they came from: `cirrhipathes` (black coral), `leather coral alive`, `nephtheidae alive`, `xeniidae alive` and the two `soft coral (malacalcyonacea)` classes (soft corals), and `millepora bleached` and `millepora dead` (fire coral). The `coral-binary` view counts black coral (`CN_ANTIPATHARIA`) as coral: the V1 class "other coral alive", which `cirrhipathes` came from, is hard coral, and counting it keeps a V1 frame and its V2 twin consistent.
- **Label matching verified for every source.** A registry-only check (`scripts/verify_label_matching.py`, `make labels-check`) resolves every native label of every crosswalked source into `rs-benthic-v1`: 3,670 labels and 3,258 edges in 44 crosswalks (2,251 exact, 777 coarsened, 71 approximate, 159 unmappable), with no unresolved, ambiguous, missing or duplicate label. The 159 unmappable edges (12 crosswalks, each with a note) train as ignore. Method and tables: [docs/label-matching.md](docs/label-matching.md); data: `results/label-matching-v1.2.0.json`.
- **V1 and V2 twins are linked by stem.** 2,074 V2 frames are V1 frames. V2 re-encoded the images and re-annotated the masks, so bytes and pixels differ and the sha256 keys that carry the v1.0 splits forward cannot link them. The split map and the dedup and splits gates link a V1 frame to its V2 twin by stem (`site<N>_<frame>_<timestamp>`) and count the pair as one group. For all 2,074 twins the release split equals the official V2 split, so no frame moves. V2 adds 359 frames (273 train, 4 validation, 82 test) and six sites (site36 to site41; V1 has 35 sites, V2 has 41). One V1 test frame has no V2 twin and stays in the frozen V1 configs. The 55 V2-only frames of site30 are V2 train frames while the frames of site30 shared with V1 are validation frames: the official split is kept and the gates re-check that boundary at build time.
- **One byte-identical test pair (upstream data, kept as published).** The V2 frames `site23_000097_017682` and `site23_000152_017682` have identical image bytes and different masks, and both are in the upstream `test` split. Neither is dropped, merged or moved. The benchmark manifest (`registry/benchmarks/manifests/coralscapes-v2.parquet`) has 643 rows (473 test, 170 validation) and 642 unique image digests. The card says so in its Coralscapes V2 section and the registry in the `coralscapes-v2` note.
- **Changed: the card and two registry texts.** The Hugging Face card gains a "Coralscapes V2" section and the V2 citation (`src/marinedata/card_v12_text.py`, rendered when the export carries `coralscapes-v2`), and its Coralscapes limitation names the V2 soft-coral classes. The V2 registry entry text and the `coralscapes-95` schema header no longer say that no crosswalk exists or that 2,075 V1 frames are carried forward.
- **The card lists every version and explains the labels.** The versions table of the v1.2.0 card lists v1.0.0, v1.1.0, v1.1.1 and v1.2.0, the v1.1.1 `condition-masks` note stays in the card, and a new section "Using labels across sources" gives the label-matching result of `results/label-matching-v1.2.0.json`, how the training views apply the crosswalks and the caveat for lossy edges. `docs/label-matching.md` gains a "How to use this" paragraph.

### On main

- **v1.1.1 `condition-masks` (patch, published 2026-10-24).** `hf_export --patch condition-masks` writes one new label-only config, the 634 Tayrona bleaching rasters (`reef-support-bleaching`, palette PNG; 255 ignore, 1 bleached, 2 non_bleached), plus the v1.1.0 card, Croissant, CITATION and lock with the config added. Every v1.1.0 file is unchanged; the frozen gate compares against the v1.1.0 Hub tree. Gates: `python -m marinedata.gates.condition_masks`. Hidden-test caveat: the 658 Tayrona images are in the hackathon hidden-test manifest (635 public since v1.0, 23 withheld), so models trained on v1.1.x must not be scored on it.

- Hidden-test drop at export and an id-free open card (#50). `hf_export --hidden-test <manifest> --hidden-test-exempt-v1
  <v1.0 export>` (open flavour only; `--flavour nc` refuses) removes every image the manifest lists that v1.0 does not
  already publish from every table before anything is counted or embedded; stderr and `v11.hidden_test_excluded` hold
  counts only, and the card states the count in Limitations. The open README, LICENSE, NOTICE and `GATES.json` name no
  non-open source id (benchmark ids in the decon entry are leak labels), and `release gate` re-scans the merged
  `GATES.json` for private names after the gates have written to it (`leak`).
- Card hand-finish for 1.1.0. The open README states the split-map exclusions (215 band-excluded and 51 pHash-excluded
  images, no withheld group), the 4,544 images whose v1.0 split a pin overrode, the per-stratum ratio check and the NOAA
  whole-site assignment, the 6 cross-source pairs the pHash-256 confirm kept, the privacy review result and the diver
  class, the 1,226 hidden-test images v1.0 already published, where the Seaview images are and the sixth Roboflow fork
  dropped as a duplicate (`card_v11_text.py`; the numbers sit behind the open-flavour switch, the nc card gets only the
  shared sentences). The decon entry of the open `GATES.json` drops its `*_sample` lists that carry image digests and
  keeps every count and `by_split`.
- Run-6 gates: dedup confirm-hash waiver and a split ratio per stratum. Dedup: a same-split pair between sources of
  different families at pHash-64 within `--phash-bits` is confirmed on pHash-256 (`dedup.features`) and grey entropy
  before it fails; it is waived only when the pHash-256 distance is above 32 and both entropies reach 4.0 bits
  (`neardup.UNION_CONFIRM_*`, the split-map rule), counted in `near_pairs_across_sources_unconfirmed` with a capped
  sample (split, distances, entropy, the two source ids, no shas). A failing pair now carries that evidence, and
  unreadable bytes fail closed. Splits: the ratio is judged per stratum (registry family, else source id) over the
  distinct images of the `images` config, because the split map allocates image groups stratified by source and
  co-assigns near-duplicate groups across a fork family; row shares and per-config image shares are reported in
  `ratio_by_config` and never fail. Pinned benchmark sources, images with an `upstream_split` label and strata under
  100 images (`ratio_small_strata`) are left out. `ratio_pins.sources` lists only the pinned sources that have rows in
  the build, plus an `absent` count, so GATES.json no longer names sources the open build leaves out. Tolerance (5
  points) and the pHash threshold are unchanged. Acceptance is a VM run: dedup failed 0 (unconfirmed 6), splits
  failures 0.
- `release split-map` confirms the gates' pHash-64 pairs before it unions them (split-map run 8). Run 8 unioned all
  53,835 pairs within 8 bits and the chain guard refused a 1,906-image component: 19 generic look-alikes (pHash-64 at 8
  bits, pHash-256 76 to 112 bits apart) chained NOAA site groups of up to 923 images. A pair is now unioned when rule B's
  confirm-hash accepts it or both images sit alone in their split groups, and never when both are v1.0 images. Every
  pair the assigned map still ships across splits loses one image (never a v1.0 one) to the new SPLIT_MAP.json
  `phash_excluded` (`{image_sha256: [distance, other sha256, other split]}`, schema 2, additive), which `release build`
  (RELEASE.json `phash_excluded`), `hf_export` and the v1.1.0 tables drop like `band_excluded`. On run 8's hashes the
  largest merged component is 62 images.
- Prior-release band is per image; `release split-map` also unions the gates' pHash rule. Run 5 withheld noaa's 26
  sites (5,793 images, 72 % of its bleached images) because a few images of each site sat 5 to 8 dHash bits from a
  v1.0 image, in a release noaa was never part of. SPLIT_MAP.json is now `schema_version` 2 and adds `band_excluded`
  (`{image_sha256: [distance, v1.0 sha256, v1.0 split]}`): a component with band hits on both sides keeps the side with
  more hit images (a tie keeps train) and only the other side's hit images are excluded; `withheld` no longer carries
  the band reason. `release build` drops those images from every config and writes RELEASE.json `prior_band_excluded
  {count, sha256}`; `hf_export` and the v1.1.0 tables drop them too (`V11_TABLES.json` `band_excluded_images`,
  `band_excluded_by_source`). Schema-1 maps still load. `generate_split_map` also unions DCT pHash-64 pairs within 8
  bits (the dedup and splits gates' rule; `NearDupConfig.phash_max`, set by `release split-map`) under the existing
  chain guard, so `near_dup_across_splits` and `near_dup_leak` cannot fail across splits (run 5: 70 pairs). Acceptance
  is a VM run (run 8): no group withheld for the band reason, noaa withheld at most 400 images, splits `near_dup_leak`
  0, dedup `across_splits` 0.
- Dedup gate discloses same-split near pairs between sources of one registry `family`, and the two Seaview sources now
  share one. A source may declare `family: <slug>` (`Source.family`, read by `licence_class.source_families`); a near
  pair (pHash within `--phash-bits`) between two sources of one family inside one split is counted as
  `near_pairs_across_sources_in_family`, with a sample capped at 50, and no longer fails the dedup gate. The four
  Roboflow Universe forks of one web-scraped set (`roboflow-coral-bleaching-final-v6i`,
  `roboflow-coral-bleaching-general-v1-yolov8s`, `roboflow-coral-classification-copy-changed-v13i`,
  `roboflow-coral-reef-classification-v3i`) declare `roboflow-coral-universe`; `seaview-survey-imagery` and
  `reef-support-seaview-labels` declare `seaview-survey`, because run 5 still failed 11 same-split pairs between them
  (10 at distance 0; 22 images with distinct sha256, re-encoded copies of the same Seaview photographs). The two
  families are separate: a Seaview source against a Roboflow fork still fails. A pair across splits, a same-split pair
  between sources that share no family and an exact duplicate still fail. The v1.1.0 card's "Build and gates"
  paragraph and `docs/GATES.md` say so.
- G7 `decon` gets `--hidden-test-published-exception` (default off, so the gate behaves as before until it is
  used). A hidden-test overlap whose `image_sha256` the frozen v1.0 tree (`--v1-dir`) already publishes is exempt
  and reported as `counts.hidden_test_already_published` (plus a sample of 20); every other hidden-test overlap
  still fails as `hidden_test_new` (`counts.hidden_test_new`). Benchmark overlaps are never exempt, and the flag
  without a v1.0 sha256 set fails the gate. Docs: `docs/GATES.md`.
- **gates:** `splits_v11` (G6) accepts the moves of benchmark pins off the v1.0 split (D-STRADDLE): a group the split map records in `pins.v1_continuity_overridden` and the build puts in the map's split counts in `pin_override_applied` (`_<source>`, `pin_override_mates`), a recorded group that stayed is a `pin_override_noop`; other v1.0 moves still fail `reassigned_v1`. See `docs/GATES.md`.
- Frozen v1.1 build places the rows of a source whose loader sets no `split_group`. `marinedata release build --tasks
  v2 --v11` aborted in `benthic-l2` with `no split_group for source(s) suim (1598 sample(s))`. `release split-map`
  groups every row through `source_release_entries` (the staged column, else the registry pattern, else the
  `metadata_norm` chain, which for suim ends at the image: `suim/sha:<sha256>`), but the build and `hf_export` read that
  enumeration only for layouts other than `staged-tree`, and the `staged-tree` loader leaves a null column null on a
  source with no registry pattern. Both now take the enumeration's group for every source whose loader can leave one
  unset (`release_group_index`: a layout other than `staged-tree`, or a `staged-tree` source with no pattern); a
  loader-set group still wins. In the v1.1.0 locks that is `suim` (open) and `synthetic-seabed-debris` (nc). The
  per-image key is spelled in one place (`sha_split_group`). Withheld groups are still dropped and counted, and a group
  in neither `assignments` nor `withheld` still raises the frozen error. The gap dates from the release-time groups of
  WP-R2d (2026-10-05), not from the withheld change; the p4 rehearsals built only `reef-support-bleaching`, whose
  staged column is set.
- Frozen v1.1 build accepts the split map's header `withheld` groups. `marinedata release build --tasks v2 --v11`
  against `releases/marine-data/v1.1.0/SPLIT_MAP.json` aborted with `... is frozen: group(s) not in the map: ...` on
  the 459 groups the map withholds. A withheld group is now known: `resolve_splits(frozen=True)` neither allocates nor
  appends it and returns no split for it (an `assignments` entry wins if a group is in both), `Dataset.split` and the
  streaming plan leave its samples out, and `build_v11_tables` drops its images from the metadata spine and every task
  table, so nothing of it reaches export, croissant, the v1.1 configs or the stats. `V11_TABLES.json` keeps
  `withheld_images` and adds `withheld_groups` and `withheld_by_source` (`{source: {images, groups}}`);
  `unsplit_images` stays 0 and `allow_unsplit` is unchanged, so G6 counts the images as dropped, never as unsplit. A
  group in neither `assignments` nor `withheld` still raises the frozen error. The card generator does not read
  `V11_TABLES.json`: the hand-finished v1.1.0 card must state the withheld counts.
- Split map pins each `route-to-our-test` benchmark's upstream eval and held-out images from
  `registry/benchmarks/manifests` (design eval-decontamination-split-v2, 2.2): the entry's eval split pins `test`,
  its second held-out split (`heldout_val`, else upstream `val` beside another eval split) pins `val`, matched by image
  sha256 or bytes_sha256. uiis's val-only eval, suim's `images` label, coralscapes `validation` and usis10k `val` no
  longer fall to the allocator. OOD claims and never-eval train still win over a pin; the `release split-map` summary counts
  the pins (`--no-benchmark-pins` opts out). Gate G7 now accepts a held-out manifest row in validation, test or ood-* (train still
  fails) and counts {eval, heldout} x {pinned_ok, failed} per manifest.

- Fix (split map, v1.1.0): a benchmark pin now wins over an inherited v1.0 split, the order design
  eval-decontamination-split-v2 3.1 rule 3 gives (pinned first, then v1 continuity). The first v1.1.0 run kept v1.0
  over 3 pinned images (uiis 2, uiis10k 1, all train -> test) and G7 failed on them. The pinned component now moves
  whole, its v1.0 groups included, and `SPLIT_MAP.json` records it as `pins.v1_continuity_overridden`
  (`images`, `groups`, `by_source`, `group_ids` listed up to 200) in place of `pins.v1_continuity_conflicts`. The
  `release split-map` summary prints `benchmark pins override v1 continuity for N image(s) in M group(s)`. OOD
  claims and never-eval train still win over a pin. Gate G6 `reassigned_v1` accepts exactly the listed groups and
  fails any other changed v1.0 group, or a listed group that did not change. `registry/SPLIT_MAP.json` (v1.0) and
  the v1.0 configs stay frozen: a v1.0 train image that moves to v1.1 test is a documented v1.0 -> v1.1 split
  change, and the v1.1.0 card states the count. `--no-benchmark-pins` is unchanged and keeps v1.0 continuity.

- Fix (v1.1 gates): G7 `decon` follows `registry/benchmarks.yaml`. Each `--benchmark` manifest resolves to its entry (`benchmark_id` column, else file stem); a `route-to-our-test` eval image in `test` or an OOD split is `pinned_ok`, in any other split it fails; `exclude`, hidden-test and unknown-benchmark (`unknown_benchmark`) manifests fail on any overlap. `GATES.json` lists `{benchmark_id, policy, manifest_shas, overlap, pinned_ok, failed}` per manifest and the first 20 failing images.
- Fix (v1.1 rehearsal): `release split-map` and `release build` resolve a staged row's image from its own `image_path` when that file exists, and the partition stem index now walks sub-directories, so a partition that nests its images (Seaview `labelled_data/segments/`) no longer fails with "not found under".
- Fix (v1.1 rehearsal): the export writes the Hub split name (`validation`, not the build's `probe`) and a resolved `licence_class` on every table row and records the per-config row counts in RELEASE.json; the load gate reads the card's `configs:` patterns, resolves them against the build and `--v1-dir` and loads each config with the parquet builder; the card and masks gates accept the frozen v1.0 configs under `--v1-dir`.
- Docs: install requirements, standalone snippets, fish-boxes statistics.
- Repository: security policy, code of conduct, issue and pull request templates; broader `.gitignore`; README install pinned to
  v1.0.0; annotator e-mails in the Labelbox test fixtures replaced by placeholders; neutral wording and placeholder paths in docs and
  script comments; the open dataset card no longer references the gated repository.

### v1.1.0: `feat/v1.1-gates` (PR-G, PR-G2)

- Release gates (PR-G): G1 join, G2 frozen v1.0 tree, G3 leak, G4 bytes once, G5 masks, G6 splits and G7 decontamination in `marinedata.gates`; `release publish` requires them on a v1.1.0 build (`V11_REQUIRED`) on top of the v1.0 gates, and a skipped gate never publishes. G8 packaging pins the YuNet model by sha256 (`verified_model_path`, never downloaded).
- v1.1.0 gate CLI (PR-G2): `marinedata release gate BUILD --v11` runs G1-G9 (join, frozen, leak, bytes_once, masks, splits_v11, decon, privacy, labelmap) into `GATES.json` and ends with the `release publish` verdict; `--rehearsal` only skips, never passes (docs/GATES.md). New G9 labelmap gate (`registry/gates.yaml` `labelmap`, now in `V11_REQUIRED`) and an images-only G8 privacy pass that loads YuNet through `verified_model_path`; the v1.0 gate path is unchanged.
- Release image on the classic Docker builder: the pinned YuNet model is fetched by `deploy/release-vm/fetch-pinned.py` (URL and sha256 pin; a mismatch fails the build) instead of `ADD --checksum`/`--chmod`, which only BuildKit has, so `docker build` works again on the release VM (no buildx). `tests/test_dockerfile_classic_builder.py` fails on BuildKit-only syntax in either Dockerfile.

### v1.1.0 registry items (PR-R: R4, R5, R6)

- Split ratio (R4): `registry/split_ratio.yaml` defaults to 70/15/15 for new configs and records the four v1.0 configs at their measured train/validation/test shares (coral-masks 76.5/9.5/14.1, instance-masks 76.7/9.0/14.2, scene-masks 73.9/10.2/15.9, fish-boxes 73.2/13.4/13.4). README, the v1.0 entry below and `docs/GATES.md` no longer say 80/10/10. Gate fixtures declare their own split map, and their synthetic mask config is now `fixture-masks`.
- Frozen exceptions (R5): `registry/flavours.yaml` lists the two v1.0 sources of the non-commercial flavour whose registry class is now internal-only. Their published files stay as they are; no source or new row is admitted. The licence gate raises `frozen_exception_new_row` for any row of those sources outside an allowlisted v1.0 shard (sha256), and RELEASE.json lists the exceptions.
- Attribution terms (R6): `registry/licences.yaml` gains `WORMS-CC-BY-4.0` and `MEOW-WORKING-GROUP`. MEOW polygon geometry is never published, only the derived realm, province and ecoregion strings with the citation.

### v1.1.0: `feat/v1.1-dedup-split` (D1, D2)

- Inherited split map (D1): `marinedata release split-map --inherit PRIOR_SPLIT_MAP.json` embeds the v1.0 `SPLIT_MAP.json` verbatim and
  allocates only the new groups around it. An image whose digest is in a v1.0 group keeps that split, and a new group that shares a
  component with it (digest, dHash within 4 bits, or a `dedup run` group) takes it; a component the v1.0 map had split apart withholds its
  new groups. New images 5 to 8 dHash bits from a v1.0 image are never-eval (near train) or never-train (near val/test). `--lock` (repeatable)
  scopes the enumeration to the sources the locks name. v1.0 assignments are never moved (an append-only check raises).
- `test_ood` (D2): the holdouts in `registry/splits/v2.yaml` route matching new groups, with their whole component, to a fourth split
  outside the 70/15/15 quota (`--ood-rules`, `--no-ood`, `--ood-metadata`); the map header records the rules sha256, each holdout's outcome
  and the matching rule per group. Per-image geo/depth/platform/time fields come from an optional Parquet until SampleRow v3 carries them.
- Docs: the exact flags for the VM run are in `docs/release-runbook.md` (step 2) and `docs/cli.md`.

### v1.1.0: `feat/v1.1-build` (PR-B, PR-B2)

- Build stage (PR-B): `release build --tasks v2 --v11` writes the v1.1.0 task tables (`metadata`, `points`, `bleaching`, `boxes`, `semseg`, `instances`) stamped with `split`, `taxonomy_version`, `fidelity` and `bytes_config`, plus `task_layers/V11_TABLES.json`.
- Split guard (PR-B2): the build now fails, naming the count and the first five `image_sha256`, when an admitted image has no split in the split map (`--allow-unsplit` keeps `split = null` for rehearsal builds and the report records it); `--v11 --generate-split-map` is refused in favour of `marinedata release split-map --inherit --lock <lock> --out <SPLIT_MAP_FILE>`.

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
- `release gate --require-all` runs every gate (an absent, skipped or failed gate refuses, privacy, card and croissant included); `release publish` takes its verdict from `marinedata.gates`, uploads with `upload_large_folder` and sends only data parquet, README.md, croissant.json and `provenance/{RELEASE.json,GATES.json,release.lock.yaml}`.
- `release publish` is fail-closed (P1.13 delta 1): privacy and croissant are required whatever GATES.json lists and a gate absent from it is a failure; `provenance/RELEASE.json` goes up as a public-safe projection (`marinedata.release_provenance`: counts, config names, ids, digests and the lock's sources only, no excluded or skipped sources, buckets, remotes or local paths); the card gains a Provenance line.
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
  them, and a deterministic group hash (70/15/15 target) otherwise. (Corrected in the v1.1.0 registry work: this entry
  first said 80/10/10, which never described v1.0; the measured shares are in the README.)
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
