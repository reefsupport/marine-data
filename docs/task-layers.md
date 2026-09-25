# WP-8 task layers — points, VQA, semantic segmentation, benthic-coarse

Status 2026-09-25: **inventory + the D-Y rollup done and tested; the four release/HF task
configs are not wired yet.** See `## Not done` for why, and the resume plan at the end.
Decisions relied on: charter D-D (staged-tree), D-U (`label_status`), D-X (v1 frozen),
D-Y (rollup rules). `registry/sources/*.yaml` paths are `coral-benthic.yaml` and
`reef-support-own.yaml` unless noted.

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

Building the four release/HF configs (`points`, `vqa`, `semseg`, `benthic-coarse`) needs,
per source above, either data that is not staged or cached locally (SEAVIEW, IBF,
Coralseg, rs_labelled, CoralVQA train split — none of which this brief permits fetching:
"Do not restage images"), or — for the two sources that ARE loadable today (Reefolution
points, Coralscapes masks) — wiring `rollup_counts()` through the *canonical* taxonomy
coarsening (crosswalk native id -> WP-7 canonical id -> `rs-benthic-v1`'s 6-class coarse
level) before it can run for real. That coarsening step, `release.py`'s
`enumerate_release_rows`/`TaskManifest`, `hf_export.py`'s parquet builder, and
`hf_card.py`'s per-config table were not reached this pass. `benthic-coarse` already
exists as a task id in `registry/tasks/benthic.yaml` (a single-label classification
task) — the new multi-label `benthic_cover`/`benthic_dominant`/`benthic_present` fields
this brief asks for are additive to that, not a replacement, and that relationship needs
a decision before the config is added (see Open).

Consequently the loader tests per config, the human/model separation test, the real
`$SP/wp8/out/` build, the `--tasks v2` flag, and the `make ci`/full-suite run against the
new wiring are not done. The rollup module and its tests are self-contained (no changes
to `release.py`, `hf_export.py`, `hf_card.py`, or `loaders/`), so they carry no risk to
v1 byte-identity or the existing suite.

## Resume plan

1. Decide whether `benthic-coarse` (D-Y fields) extends or replaces the existing
   `registry/tasks/benthic.yaml` `benthic-coarse` classification task (Open, below).
2. Add a small per-source adapter that turns a loaded `Sample`'s raw label payload
   (Reefolution's `labels/points.parquet` rows; Coralscapes' mask + its 39-class map)
   into per-image class counts, then calls `rollup_counts()`.
3. Wire `points`/`vqa`/`semseg`/`benthic-coarse` into `release.py` behind `--tasks v2`,
   with a single call/flag line into `hf_export.py` and the card table into `hf_card.py`
   (both disclosed per D-M).
4. IBF, SEAVIEW points, Coralseg, rs_labelled and the CoralVQA train split stay at 0 rows
   until their data is fetched/verified by a source-scoped worker — that is out of this
   brief's scope ("do not restage images").
