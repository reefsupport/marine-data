# Datasheet: Reef Support Open Marine Imagery (v1)

Follows the seven sections of Gebru et al., *Datasheets for Datasets* (CACM 2021). Every
number below cites the file it was measured from — the release manifest
(`RELEASE.json`, in `_release/2026-09-24-neardup/releases/v1/`, outside this repo), the
built Hub export's card (`_hf/v1/README.md`, also outside this repo), the registry
(`registry/sources/*.yaml`), or the rating audit
(`2026-09-25-5star-rating-r0.md`, referenced here as **r0**). Nothing here is invented;
where a number is not yet measured, that is stated as a residual, not filled in.

## 1. Motivation

**For what purpose was the dataset created?** To give a coral-reef computer-vision model
a licence-traceable, deduplicated training corpus with condition (health/bleaching) and
segmentation labels, replacing the common practice of scraping public reef imagery
without recording what any of it may lawfully be used for.

**Who created it and on whose behalf?** Reef Support B.V. (`reef.support`), a marine
conservation technology company, for its own model training and for public release as a
research resource.

**Who funded it?** Reef Support B.V.

## 2. Composition

**What do the instances represent?** Each instance is one coral-reef photograph
(`images` config, keyed by `image_sha256`), joined by that key to zero or more label
rows in six task-specific configs and to zero or more machine-generated pseudo-mask rows.

**How many instances are there, and how do they relate?**
- 69,600 unique-sha images in `images` (59,732 train / 4,947 validation / 4,921 test).
  Source: r0 D1; `_hf/v1/README.md` per-config counts.
- 29,984 unique human-labeled images: 28,734 in `coral-health-binary` plus 1,250 with
  ground-truth masks. Only one task (`coral-health-binary`/`benthic-coarse`/
  `benthic-l2`, which share the same row count) clears 10,000 human-labeled images.
  Source: r0 D1.
- Task configs carry one row per *(image, source)*, so a task's row count (e.g. 73,915
  for `benthic-coarse`/`benthic-l2`/`coral-health-binary`, 69,359 for
  `bleaching-condition`) exceeds the 69,600 image count whenever a sha appears under
  more than one source. Source: `_hf/v1/data/*/*.parquet` schemas (read directly by
  `src/marinedata/croissant.py`).
- `masks`: 1,908 rows (658 Tayrona images carry both a benthic and a bleaching mask).
  `coralscop-pseudo-masks`: 37,273 rows, model output only, never ground truth. Source:
  `_hf/v1/README.md`; `registry/sources/reef-support-own.yaml` (`coralscop-masks-rs`).
- `coral-genus-caribbean` has **0** image labels in v1 (every label and
  `mask_class_map` value is null) and is dropped from the Hub configs per charter
  decision D-A. Source: r0 D4; the current `_hf/v1/data/` tree already has 8 configs,
  not 9 (verified by `croissant.scan_build_dir`).

**Splits.** One frozen group split serves every task (`split_map_sha256` in
`RELEASE.json`), stratified by source only. The realized split is 85.7/7.2/7.1 against a
70/15/15 target — there is no OOD holdout (by realm, depth or source). Source: r0 D7.

**Noise, redundancy, errors (quantified — see also the Limitations table below).**
- 14,265 near-duplicate pairs under the release's Hamming-distance-4 union rule; 7,085
  of those pairs are at Hamming distance **0** (visually identical) with a *different*
  sha256, meaning they are not caught by exact-hash dedup alone. Source: r0 D1, D6.
- 557 images are excluded from evaluation splits only (never dropped) by the
  Hamming-distance-8 "never-eval" chain guard. Source: `RELEASE.json`
  `never_eval_near_dup_excluded.count`.
- 4,556 rows are excluded from `bleaching-condition` because `roboflow-coral-reef-classification-v3i`
  abstains on them (`partial_abstain_excluded`). Source: `RELEASE.json`
  `partial_abstain_excluded[0]`.
- 399 multi-label rows abstain in `roboflow-coral-reef-bleach-detection-v2i`. Source: r0
  D5.
- 14 images that are byte-identical (same sha256) carry **conflicting** labels across
  sources — never resolved by majority vote or any other rule; both labels ship.
  Source: r0 D5.
- No confirm-hash (perceptual embedding) pass has been run over the whole corpus; the
  9-source dHash union is the only dedup that has run. Source: r0 D6.

**Is the dataset self-contained?** Yes for pixels (embedded once in `images`, joined by
sha256) and labels (frozen release TSVs). It is not self-contained for licence
provenance below the source level: there is no per-sample licence column yet (see
Limitations, D13).

**Does it contain data that might be considered sensitive?** Yes — see Ethics, below.
It contains no named individuals' identity, no medical data, and no per-sample GPS,
depth, date or camera metadata at all (r0 D8), so there is nothing to redact on that
axis in v1.

## 3. Collection Process

**How was the data collected?** Nine sources were staged from `rs-storage-open` (a
public, anonymously-fetchable bucket), each sha256-verified against the source's own
manifest, then deduplicated (sha256 exact + dHash-64 near-dup) before being frozen into
the v1 release. Source: `_hf/v1/…hf-layout` report (pixel provenance table);
`registry/sources/*.yaml` `verification` blocks per entry.

**Who was involved and over what timeframe?** Reef Support engineering staged and
verified all nine sources between 2026-08-17 and 2026-09-24 (per-source
`verification.verified_on` dates in `registry/sources/coral-benthic.yaml` and
`registry/sources/reef-support-own.yaml`); five sources are third-party Roboflow
Universe community exports, one is a US federal agency (NOAA PIFSC), one is a
model-generated derivative (CoralSCOP), and two are Reef Support's own field surveys.

**Was any ethical review conducted?** No external ethics board review has been
conducted (charter D-I: "no external experts for now"). The checks below are what has
been done in-house.

### Ethics

**Diver / face presence.** A subset of in-water photographs may show a diver's face,
hands or body. The original OpenCV Haar-cascade proxy (12,433/69,600 = 17.9% flagged,
mostly coral-texture false positives) was replaced (WP-5b/5d) by two real, open-weight
detectors — YuNet (faces) and a COCO person model — run over all 69,600 images, with
the results manually audited against a stratified, seed-fixed sample of 350 images
(150 face-flagged, 100 person-flagged, 100 negatives; `docs/privacy-audit-2026-09-25.tsv`).
Audited results: **face precision 12.0% (18/150, 95% CI 7.7–18.2%)**, **person precision
33.0% (33/100, 95% CI 24.6–42.7%)** — both detectors still fire mostly on coral/rock
texture and polyp clusters, not real people. Of the 69,600 images, the detector flags
3,387 as face-positive and 1,117 as person-positive (296 overlap). Extrapolating the
audit's identifiable-face rate per stratum gives an estimated **~348 images (0.50%,
95% CI 0.30–4.32%)** contain a face someone could actually recognise; the 100-image
negative sample found zero missed faces/persons (Wilson upper bound on the miss rate:
3.7%). Full method, per-detector precision, the occlusion-heuristic calibration and
raw counts are in [`docs/ETHICS_FACE_AUDIT.md`](ETHICS_FACE_AUDIT.md) and the audit TSV.
No face has been identified to a named individual; no consent-tracking field exists per
sample. Per `docs/PRIVACY.md`, the v2 build blurs every `face_identifiable` box in the
released pixels and ships a `face_identifiable` column in `metadata`.

**Sensitive-species location policy.** v1 carries **zero** per-sample geographic
coordinates — no `lat`/`lon` field exists anywhere in the current schema (r0 D8), so
there is nothing to redact in this release. This policy is written now, pre-emptively,
for the release that adds them (WP-2's `lat/lon` + `gps_precision_m` columns): before
publication, the coordinates of any sample whose source or crosswalked taxon is
CITES Appendix I/II or IUCN Red List Vulnerable/Endangered/Critically Endangered will be
rounded to 0.1 degrees (~11 km at the equator) in the public release; full-precision
coordinates are retained only in the private staging registry and are never
redistributed. No sample in v1 carries a species-level identification at all (ground
truth is condition/bleaching classes, not taxa — r0 D3), so this policy currently
constrains zero rows; it constrains future ones.

**Licence per source.** Per charter D-C, licence status is recorded, never a storage or
publication blocker. Per charter D-B, Reef Support's own imagery (`benthic-own`,
`rs-*`) is licensed CC-BY-4.0 by delegation, "set 2026-09-25 by delegation; confirm
before publish" — that confirmation has not yet happened.

| Source (registry id) | Images | Licence | Note |
|---|---|---|---|
| `coralscop-masks-rs` | 37,273 | CC-BY-NC-SA-4.0 | Inherited from the CoralSCOP model licence (HKUST VGD); model output, never ground truth. |
| `noaa-pifsc-bleaching` | 10,419 | US-GOV-PD | US federal agency (NOAA PIFSC), public domain. |
| `roboflow-coral-reef-bleach-detection-v2i` | 10,544 | CC-BY-4.0 | Roboflow Universe community export; sidecar-verified. |
| `roboflow-coral-reef-classification-v3i` | 4,541 | CC-BY-4.0 | Roboflow Universe community export. |
| `roboflow-coral-classification-copy-changed-v13i` | 2,785 | CC-BY-4.0 | Roboflow Universe community export; sidecar-verified. |
| `roboflow-coral-bleaching-final-v6i` | 2,550 | CC-BY-4.0 | Roboflow Universe community export. |
| `roboflow-coral-bleaching-general-v1-yolov8s` | 2,543 | CC-BY-4.0 | Roboflow Universe community export. |
| `reef-support-benthic-own` | 1,250 | CC-BY-4.0 | Reef Support's own survey (D-B delegation; unconfirmed). |
| `reef-support-bleaching` | 658 | CC-BY-4.0 | Reef Support's own survey (D-B delegation; unconfirmed). |

Source: image counts and licence column, `_hf/v1/README.md` sources table (current
build); the two Reef Support entries are recorded `PROPRIETARY-OWN` in
`registry/sources/reef-support-own.yaml` as of this writing — the card and this table
follow the D-B delegation, which supersedes the stale registry value pending WP-2's
formal update. No source in this table carries a `NO-LICENCE-STATED` or
`PROVENANCE-DEFECTIVE` flag (see `docs/LEGAL.md` for what those mean and why they would
block use).

**Label-quality residuals (D-I — human-only, listed, never faked).** No external
domain-expert review has been done. Specifically absent, and each requiring a human or
a not-yet-run pipeline to close:
- No cross-source label-agreement metric has been computed. The only measured
  disagreement evidence is the 14 identical-sha, conflicting-label images above (r0
  D5) — a count, not a rate, because the denominator (all cross-source overlaps) has
  never been enumerated.
- No confident-learning pass (e.g. an out-of-fold DINOv2 linear-probe scan for label
  noise) has been run on the health or bleaching tasks.
- No 500-sample expert-audit protocol has been executed; a protocol and audit sheet are
  planned (charter D-I) for Yohan's team to fill later, not yet produced by this
  worker.
- `partial_abstain_excluded` (4,556 rows) and the 399 `v2i` multi-label abstains are
  the only two *known, upstream-declared* ambiguity signals baked into the release; both
  are exclusions, not corrections.

## 4. Preprocessing / Cleaning / Labeling

**Deduplication.** sha256 exact-match plus dHash-64 (grayscale, LANCZOS-resized 9x8,
row-major MSB-first) near-duplicate detection, scoped to the 9 release sources only (not
the whole corpus). Union rule: Hamming <=4 merges before the split is computed (14,265
pairs). Never-eval rule: Hamming <=8 excludes an image from evaluation splits only (557
images) without dropping it. A chain-guard fraction of 0.01 bounds transitive-closure
drift. Source: `RELEASE.json` `near_dup` and `never_eval_near_dup_excluded`; r0 D6.

**Labeling.** Labels are taken as-shipped from each source's native class set
(`native_label`) and, where a crosswalk exists, mapped onto Reef Support's own
ontology (`label`). Crosswalk coverage is partial: 15 of 54 registered image sources
(28%) have a `crosswalk_id` at all. In `rs-benthic-v1`, 32 of 58 taxon-axis nodes (55%)
carry an accepted AphiaID; `rs-fauna-v1` is 21/21 (100%), but no fauna task ships in
v1. Source: r0 D4. CoralSCOP masks are explicitly tagged `PSEUDO_TAG` (model output)
in the HF export and are structurally separated into their own config so they cannot be
loaded as ground truth by accident.

**Is the raw, unprocessed data retained?** Yes — each source's staged tree is kept
under its own `root_digest` in the registry, independent of the frozen release, so a
different split or dedup policy can be re-derived without re-fetching.

**Software used.** `marinedata` (`src/marinedata/release.py`, `neardup.py`,
`splitmap.py`, `hf_export.py`) at commit `7ab29a6` on `feat/splitmap-group`.

## 5. Uses

**What has this dataset been used for?** Nothing yet — v1 has never been published
(the Hub upload has only run as a dry run; see Distribution).

**What tasks does it reasonably support?** Coral-reef image classification (health,
bleaching condition), weakly-supervised or pretraining use of CoralSCOP pseudo-masks,
and general reef-image pretraining (`general-pretraining` config). It does **not**
support detection, point-based annotation, captioning, VQA, tracking or
image-enhancement training — no ground truth for any of those exists in v1 (r0 D3).

**What should this dataset not be used for?**
- **Diver identification or any biometric use of the face-positive images** flagged
  in Ethics — this dataset was not collected with that use in mind and carries no
  consent record for it.
- **Population- or region-level ecological claims.** Provider diversity is about 3
  independent sources, habitat is 100% shallow coral reef, and region/depth/platform
  are unrecorded for the Roboflow-sourced 54% of images (r0 D2). A model trained here
  generalizes to "shallow tropical reef, similar camera distance," not to reefs, depths
  or regions outside that envelope.
- **Commercial redistribution of CoralSCOP-derived weights or masks** without
  separately clearing the CC-BY-NC-SA-4.0 restriction — it is non-commercial and
  share-alike, and derivation from it (a segmentation head trained on its masks) may
  inherit that restriction. See `docs/LEGAL.md` on why an ND/NC source is "untrainable,"
  not merely "non-shippable," when in doubt.
- **Any use requiring geolocation, capture date or depth** — none of these fields
  exist per sample in v1 (r0 D8).

## 6. Distribution

**Will this be distributed, and how?** Yes, as a public Hugging Face Hub dataset repo,
one commit set per the S55 layout plan (4 commits, 51-52 files including
`.gitattributes`, ~13.9 GB). As of this datasheet, the Hub upload has run only as a
`--execute`-less dry run; it has never actually published. Source:
`2026-09-25-wsd-S55-hf-layout.md` (commit plan section).

**Licence and terms of use.** Mixed per source — see the Ethics table above and the
generated card's sources table (`_hf/v1/README.md`) for the authoritative, regenerated
version. The repository-level `license:` tag is `other` with a `LICENSE` file, because
no single SPDX id covers a corpus mixing US-GOV-PD, CC-BY-4.0 and CC-BY-NC-SA-4.0.
`CITATION.cff` deliberately omits a single `license:` field for the same reason.

**IP and legal restrictions.** The CoralSCOP branch (masks and any model fine-tuned
predominantly on them) carries a non-commercial, share-alike restriction it cannot
shed by redistribution through us (`docs/LEGAL.md`, "Provenance defects cannot be
cured downstream" — the same principle that makes an NC-SA restriction persist,
applied to the reverse case of a permissive re-wrapper). No source in the v1 registry
selection is flagged `PROVENANCE-DEFECTIVE` or `NO-LICENCE-STATED`.

## 7. Maintenance

**Who maintains this dataset?** Reef Support B.V. (`reef.support`). Contact via issues
on `https://github.com/reefsupport/marine-data`.

**Erratum and update process.** Each release is content-addressed: every source's
`root_digest`, the split-map sha256, and per-shard sha256 are pinned in `RELEASE.json`
and this build's Croissant metadata (`docs/croissant-v1.json`, one `FileObject` per
Parquet shard with its own sha256). A correction ships as a new version (`v2`, …) with
its own entry in `CHANGELOG.md`; v1 is never silently mutated in place.

**Will older versions be supported?** v1's manifest and frozen split map remain
resolvable indefinitely via `RELEASE.json` and the registry's per-source
`root_digest`/`version` pins, independent of whatever v2+ changes.

**No DOI has been minted for this release.** Zenodo minting is a human action outside
this worker's scope (charter: "a Zenodo DOI" is a human-only blocker for some
dimensions, per the rating report's closing note).

## Limitations (quantified)

Every number below is repeated verbatim from a section above. The Source column names
a file **in this repository** so the number can be checked mechanically; where the
underlying evidence lives in operational task state outside the repo, the cited file is
`docs/RATING_EVIDENCE_R0.md`, which quotes that evidence verbatim (see its header for
why).

| Limitation | Number | Source |
|---|---|---|
| Single-source concentration | CoralSCOP is 54% of images | `docs/RATING_EVIDENCE_R0.md` |
| Provider diversity | ~3 independent providers: NOAA PIFSC (10,419 images), the Roboflow community across 5 registry ids (about 23k images), and Reef Support's own | `docs/RATING_EVIDENCE_R0.md` |
| Habitat/depth scope | Single habitat (coral reef) and single depth band (shallow) across all 69,600 images; not recorded per sample | `docs/RATING_EVIDENCE_R0.md` |
| NOAA image resolution | 10,419 images at 224×224 px | `docs/RATING_EVIDENCE_R0.md` |
| Roboflow image resolution | about 23k images resized to 640×640 px | `docs/RATING_EVIDENCE_R0.md` |
| Exact visual duplicates (different sha) | 7,085 images at Hamming distance 0 | `docs/RATING_EVIDENCE_R0.md` |
| Near-duplicate pairs | 14,265 pairs (Hamming <=4 union) | `docs/RATING_EVIDENCE_R0.md` |
| Never-eval exclusions | 557 images | `docs/RATING_EVIDENCE_R0.md` |
| No per-sample GPS/depth/date/camera/licence | None of the 69,600 images carries GPS, depth, capture date, camera model or licence metadata | `docs/RATING_EVIDENCE_R0.md` |
| Task-type coverage | Only 2 ground-truth task types across all 29,984 human-labeled images: classification and segmentation. No detection, points, captions, VQA, tracking or enhancement | `docs/RATING_EVIDENCE_R0.md` |
| Human-labeled images | 29,984 unique images (28,734 classification + 1,250 masks) | `docs/RATING_EVIDENCE_R0.md` |
| Partial-abstain exclusion | 4,556 rows dropped from bleaching-condition | `docs/RATING_EVIDENCE_R0.md` |
| Multi-label abstain | 399 rows abstain in v2i | `docs/RATING_EVIDENCE_R0.md` |
| Conflicting labels | 14 identical-sha images, conflicting labels | `docs/RATING_EVIDENCE_R0.md` |
| Ontology AphiaID coverage | 32/58 (55%) rs-benthic-v1 nodes | `docs/RATING_EVIDENCE_R0.md` |
| Crosswalk coverage | 15/54 (28%) image sources | `docs/RATING_EVIDENCE_R0.md` |
| Zero-label configs | benthic-coarse, benthic-l2: 0 image labels (1,908 mask_class_map rows only); coral-genus-caribbean: 0 labels, dropped | `docs/RATING_EVIDENCE_R0.md` |
| Split drift | realized 85.7/7.2/7.1 vs 70/15/15 target | `docs/RATING_EVIDENCE_R0.md` |
| Pseudo-labels | 37,273 CoralSCOP masks are model output | `docs/RATING_EVIDENCE_R0.md` |
| Diver/face presence | YuNet+COCO-person scan: 3,387 face-flagged / 1,117 person-flagged (296 overlap); audited precision 12.0% faces / 33.0% persons; ~348 (0.50%) images estimated to contain an identifiable face | `docs/ETHICS_FACE_AUDIT.md` |
