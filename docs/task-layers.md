# WP-8 task layers — points, VQA, semantic segmentation, benthic-coarse, benthic-cover

Status 2026-09-25 (WP-8b pass): **data-gap fetch attempts done and logged for every
source below (D-Y/D-Z binding); the release/HF config wiring (`points`, `vqa`, `semseg`,
`benthic-coarse`, `benthic-cover`) is still not built.** See `## WP-8b data-gap findings`
for what changed this pass, `## Not done` for why the wiring itself is still open, and
the resume plan at the end. Decisions relied on: charter D-D (staged-tree), D-U
(`label_status`), D-X (v1 frozen), D-Y (rollup rules), D-Z (benthic-coarse stays
single-label, `benthic-cover` is the new D-Y multi-label config). `registry/sources/*.yaml`
paths are `coral-benthic.yaml` and `reef-support-own.yaml` unless noted.

## WP-8b data-gap findings (this pass)

- **CoralVQA — train split found and fetched (text only).** `CoralReefData/CoralVQA` on
  HF has `CoralVQA_train.jsonl` (62,809,221 B, direct file not LFS) alongside the
  already-known `CoralVQA_test.jsonl` (7,074,574 B). Fetched
  `CoralVQA_train.jsonl` to `$SP/wp8b/coralvqa/` (byte size matches the HF API listing
  exactly): 226,883 Q/A pairs over 10,537 unique images. Re-fetching
  `CoralVQA_test.jsonl` hit HF's anonymous-IP rate limit twice ("create an account or
  pass HF_TOKEN") — not a licence/login wall, but the brief's Do-Not list forbids
  logging in or passing a token to work around it, so the test split was not
  re-downloaded this pass (it is on HF, same repo, same direct-file shape, no `lfs` key;
  a later pass can retry once the rate limit clears). Per D-Y: CoralVQA ships **train +
  test found in principle, train fetched this pass** — no longer test-only, but the
  `vqa` config build itself (JSONL → per-image parquet) is not implemented (see Not
  done).
- **SEAVIEW — confirmed images-only, no CSV route found.** Tried three routes for a
  point-label CSV/text export beyond the known-excluded pickle: (1)
  `https://espace.library.uq.edu.au/view/UQ:734799` — HTML shell only (client-rendered,
  no CSV/annotation links in the static markup); (2)
  `https://espace.library.uq.edu.au/view/view/UQ:734799/Seaview_Survey_photoquadrat_Data.pdf`
  — `403 Request blocked`; (3) `https://api.library.uq.edu.au/v1/records/UQ:734799` —
  `403 Request blocked`. Per the brief, a 401/403 is not to be worked around. SEAVIEW
  ships images-only, matching the registry's existing `annotations: [{kind: none}]` —
  no code or registry change needed.
- **Coralseg (UCSD) — masks are real, but not sha256-keyable without the images.**
  Anonymous `ListObjectsV2` on `rs-storage-open` for
  `benthic_datasets/mask_labels/Coralseg/` returned `AccessDenied` (bucket policy allows
  anonymous `GetObject` on known keys, not anonymous listing). Listing via boto3 +
  rclone-conf creds (per the creds rule) is not blocked. The real blocker is
  architectural, not access: `release.py`'s `build_release` keys every task row by
  `hashlib.file_digest(Path(sample.image))` — the sha256 of the **image file on disk**.
  Coralseg's images were never staged (D-D) and have no `images_from` link to an
  already-hashed Reef Support source (unlike rs_labelled, below), so there is no way to
  produce a valid sha256 key for a Coralseg mask row without opening the paired image —
  which D-Z and the Do-Not list both forbid ("never the images"). This is a real
  contradiction between "keyed by sha256" and "never touch Coralseg's images", not a
  fetch failure — flagged for the manager in `## Open` equivalent below (see report).
- **rs_labelled masks — real format is different from the registry's placeholder.**
  Listed `rs-storage-private/cache/rs-labelled-masks/2026-09-23-07d6247cea85/` via
  boto3 (creds rule). The registry says `loader: {layout: metadata-only}` with "Mask/label
  structure not verified this pass" — verified this pass: each site directory (e.g.
  `SEAFLOWER_BOLIVAR/`) holds one `export-result.ndjson` (a Labelbox-style annotation
  export, 1.4 MB for that site) plus an `images/` subfolder of full-resolution JPEGs.
  There are no pixel-mask PNGs at all — the "masks" are polygon/instance annotations in
  the ndjson, referencing image filenames. Per `images_from: [reef-support-benthic-own,
  reef-support-seaview-labels]`, those image filenames should already have a known
  sha256 in the corresponding staged source's `metadata.parquet` (both are `staged-tree`,
  D-D), so `rs_labelled` masks *can* in principle be sha256-keyed without touching its
  own `images/` copy — only the ndjson files need reading. Rasterising the ndjson
  polygons into per-image pixel-fraction counts (the D-Y rollup's input shape) was not
  implemented this pass — see Not done.
- **IBF — confirmed no labels, images-only as expected.** No new evidence found;
  matches WP-8's finding (verified twice previously). Listed in this doc as
  images-only/pretrain per the brief, no points row.

## Inventory (per source)

| Source | `id` | `loader.layout` | Staged (D-D)? | Local cache (`~/.cache/marinedata/`) | Label reality |
|---|---|---|---|---|---|
| SEAVIEW raw imagery | `seaview-survey-imagery` | `metadata-only` | No | empty (not fetched) | `annotations: [{kind: none}]`. The one label artifact, `labelled_data/labelled_segmentation_data.pickle`, is explicitly excluded from this entry (unsafe pickle format, "slated for deletion"). No CSV/parquet point-label export exists in the bucket prefix. |
| SEAVIEW masks (ours) | `reef-support-seaview-labels` | `labelbox-rgb` | No | empty (not fetched) | Real annotation (`kind: instance-mask`, 73,045 images), loadable via the existing `LabelboxRgbMaskLoader`, but it is **masks**, not points, and nothing is cached locally to read from this pass. |
| Reefolution | `reefolution` | `staged-tree` | **Yes** | populated: `images/default/*.JPG`, `labels/points.parquet`, `metadata.parquet` | Real: CoralNet-style export, 872 images × 50 points = 43,500 point rows, native vocabulary of 85 short codes (`schema_id: reefolution-coralnet-native`), crosswalk `reefolution` registered. |
| IBF | `ibf` | `flat-images` | No | empty (not fetched) | `annotations: [{kind: none}]` — "images only, grouped by site... No point labels in the bucket" (verified twice, 2026-08-17 and 2026-09-18). Zero rows possible for the `points` config from IBF; not a loader gap, a data gap. |
| CoralVQA | `coralvqa` | `flat-images` | No | images cached (`~/.cache/marinedata/coralvqa/*.jpg`) | The registry's own note: the real Q/A pairs live in `CoralVQA_train.jsonl`/`CoralVQA_test.jsonl`, "not fetched here" by the source's own `access` block. Found in-repo at `data/_stage/samples/benthic_datasets/point_labels/coralvqa/CoralVQA_test.jsonl` — **test split only**, no train split anywhere under the repo's data tree. |
| Coralscapes | `coralscapes` | `image-mask-pairs` | No (image-mask-pairs, not staged-tree) | populated: `images/`, `masks/`, `_shards/` | Real: 39-class schema (`coralscapes-39`), crosswalk registered, loadable today via the existing `ImageMaskPairLoader`. |
| Coralseg (UCSD) | `coralseg-ucsd-mosaics` | `coralseg-r-channel` | No | empty (not fetched) | Loader exists (`CoralsegRMaskLoader`) but nothing is cached locally to read; tagged `needs-permission` (D-C: licence is not a blocker for this build, but the data still is not on disk). |
| rs_labelled masks | `rs-labelled-masks` | `metadata-only` | No | empty (not fetched) | `annotations: [{kind: none}]`, and the registry entry itself says "Mask/label structure not verified this pass." There is no existing loader for its raw layout — `metadata-only` is a placeholder, not a decodable format. |

## The D-Y rollup (done)

`src/marinedata/task_layers/rollup.py` implements `rollup_counts()`: one pure function,
shared by points and masks (D-Y: "the same three fields"), taking one image's
per-coarse-class counts and returning `cover` (fractions of *known* points/pixels),
`dominant` (a class at >= 50%, else `"mixed"`, else `None` when withheld), `present`
(classes at >= 10%), and `excluded` (> 50% unknown). Unknown is excluded from every
denominator; an image with fewer than 10 known points/pixels gets `cover` only.

`tests/test_task_layers_rollup.py` — 10 tests, all passing (`uv run pytest -q
tests/test_task_layers_rollup.py`): the 50% dominant cutoff (inclusive) and its "mixed"
fallback, the 10% present cutoff (inclusive), unknown excluded from the denominator
without triggering exclusion at exactly 50/50, the > 50%-unknown image exclusion, the
< 10-known "cover only" case (and its boundary at exactly 10), the same rule applied at
pixel scale, and a negative-count rejection.

## Not done (and why)

D-Z resolved the `benthic-coarse` question (extend, don't replace; the D-Y multi-label
fields ship as a new `benthic-cover` config). That removes one blocker from WP-8, but
the five release/HF configs (`points`, `vqa`, `semseg`, `benthic-coarse`,
`benthic-cover`) are still not built this pass either. What each needs, given the
WP-8b data-gap findings above:

- **`points`** (Reefolution): loadable today, no remaining data gap. Needs the
  crosswalk-through-canonical-through-coarse wiring (`Harmonizer`/`schema.py`) plus the
  `release.py`/`hf_export.py`/`hf_card.py`/`loaders/` integration — not reached this
  pass; this is genuinely new plumbing (`release.py` has no `--tasks`/CLI flag at all
  today, v1 or v2 — confirmed by grep, so "behind `--tasks v2` only" is new surface, not
  a toggle on existing code).
- **`vqa`** (CoralVQA): train split now fetched (see above); building the JSONL →
  per-image parquet (dedupe 226,883 Q/A pairs to 10,537 images, keep q/a + question type)
  is not implemented.
- **`semseg`** (Coralscapes): loadable today via `ImageMaskPairLoader`, same crosswalk
  wiring gap as `points`.
- **`benthic-coarse`/`benthic-cover`**: same wiring gap, plus — for Coralseg and
  rs_labelled specifically — the sha256-keying architecture question raised above
  (Coralseg: no path to a valid key without opening an image D-Z forbids opening;
  rs_labelled: possible in principle via `images_from`, but the ndjson→pixel-fraction
  rasteriser is unwritten).

Consequently the loader tests per config, the human/model separation test, the real
`$SP/wp8/out/` build, the `--tasks v2` flag, and the `make ci`/full-suite run against new
wiring are still not done. No `src/` file changed this pass (only this doc), so v1
byte-identity and the existing suite are unaffected by construction — not re-run, since
nothing that could regress them was touched.

## Resume plan

1. **Manager decision needed**: Coralseg mask rows have no sha256-safe path under the
   current "never open the image" constraint (see WP-8b findings). Options: (a) accept
   opening the image only to hash it, never persisting or shipping its bytes, or (b) drop
   Coralseg from the sha256-keyed configs and note it as mask-only/unusable for v2.
2. Add a small per-source adapter that turns a loaded `Sample`'s raw label payload
   (Reefolution's `labels/points.parquet` rows; Coralscapes' mask + its 39-class map;
   rs_labelled's `export-result.ndjson` polygons matched by filename against the sha256
   already known from `reef-support-benthic-own`/`reef-support-seaview-labels`
   `metadata.parquet`) into per-image class counts, then calls `rollup_counts()`.
3. Build the `vqa` adapter from `CoralVQA_train.jsonl` (fetched, `$SP/wp8b/coralvqa/`)
   + `CoralVQA_test.jsonl` (re-fetch once HF's rate limit clears; no login/token).
4. Add the `--tasks {v1,v2}` flag and wire `points`/`vqa`/`semseg`/`benthic-coarse`/
   `benthic-cover` into `release.py`, `hf_export.py`, `loaders/` and `hf_card.py`'s table
   (all disclosed per D-M), then the loader tests, human/model separation test, real
   build, and `make ci`/full suite.
5. SEAVIEW and IBF stay at 0 point/vqa/mask rows (images-only, confirmed twice) —
   correct end state per D-Z ("IBF... gets no points row"), not a gap to chase further.
