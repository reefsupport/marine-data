# Split v2 dry run (P3, 2026-09-25)

Real data, run through `marinedata.splitv2.{holdouts,allocate,mapfile}` (design
`docs/design/eval-decontamination-split-v2.md` §3). The working SPLIT_MAP v2 this run
produced stays under `$SP/p3/` (scratch), never in git — the integrator regenerates it
at the v2 build. `registry/SPLIT_MAP.json` (v1) was not read or written by this run
(see `tests/test_splitv2_allocate.py::test_v1_split_map_json_is_untouched_by_running_split_v2`).

## Data used, and its provenance

| pool | n | source of the counts |
|---|---|---|
| v1 eligible (7 sources) | 32,327 | **not re-derived** — carried from design §0/§3.6, itself measured on the v1 HF `metadata` config. No per-image v1 table with `source_id`/`split_group_id` is available in this worktree; `registry/SPLIT_MAP.json` has only 43,331 `split_group_id -> split` pairs, no group->source or group->image-count join. |
| `coralscop-masks-rs` (never-eval) | 37,273 | same as above; excluded from the ID ratio pool per §0 point 1 (train-only). |
| `coralscapes` 1.0 | 2,075 | **real**: `sources/coralscapes/1.0/metadata.parquet` pulled from `rs-storage-open` (rclone, `rs-hel1` remote, per the charter creds rule). `upstream_split` column gives the actual train/validation/test membership. |
| `mermaid-aws` 2026-09-19 | 18,561 | **real**: `sources/mermaid-aws/2026-09-19-bc53d5a2c0b6/metadata.parquet`, same remote. |

Neither staged metadata file carries `meow_realm`/`meow_province`/`depth_m`/`platform`/
`capture_datetime` yet (the D-O staged-tree layout's `metadata.parquet` only has
`stem, partition, upstream_path, upstream_split, width, height`; WP-2b's per-sample
enrichment pipeline has not been run over these two sources). Consequently **no OOD
holdout is constructible from this run's real data** — every one of the 6 holdouts in
`registry/splits/v2.yaml` reports `not constructible (0 images, 0 groups)` here. This
is a data-coverage gap, not a rule-logic gap: the any-member-OOD rule, the geo/platform/
depth fallbacks, and the size guards are all exercised and passing against synthetic
fixtures with the missing fields present (`tests/test_splitv2_holdouts.py`).

`split_group_id`: no WP-10 dedup group table (`sha256 -> split_group_id`) exists yet
for these two sources in this worktree either, so this run treats each image as its own
singleton group. That is a disclosed simplification for the dry run only — it cannot
hide a group-splitting bug (a singleton group can never span two splits by
construction). The actual any-member-OOD and multi-image-group packing behaviour is
verified on real multi-member groups by the unit tests, not by this run.

## Counts per pool x split

| pool | train | val | test | train-only |
|---|---|---|---|---|
| v1 eligible (carried) | 22,459 | 4,947 | 4,921 | — |
| coralscapes (real, allocator run) | 1,386 | 297 | 392 | — |
| mermaid-aws (real, allocator run) | 12,992 | 2,785 | 2,784 | — |
| coralscop-masks-rs (never-eval) | — | — | — | 37,273 |
| **ID pool total** | **36,837** | **8,029** | **8,097** | — |
| **all pools** | 36,837 | 8,029 | 8,097 | 37,273 |

Grand total: 52,963 (eligible ID) + 37,273 (train-only) = **90,236** rows.

`coralscapes`'s allocator run used the real upstream pins from `registry/benchmarks.yaml`
(`eval_split: test` = 392, `heldout_val: validation` = 166) and reproduced design §3.6's
worked example **exactly**: train 1,386 / val 297 / test 392 (test-overfull stratum:
392/2,075 = 18.9% ≥ 15%, so `train:val` = 82.35:17.65 of the remaining 1,683, with the
166 pinned-val images counted toward val before the free 1,517 upstream-train images
fill the rest). `mermaid-aws` has no benchmark pin (absent from `registry/benchmarks.yaml`)
and was allocated as a single free stratum.

## Achieved vs. target ratios (eligible ID pool, 52,963 images)

| split | target | achieved | delta |
|---|---|---|---|
| train | 70.0% | 69.54% | −0.46 pt |
| val | 15.0% | 15.16% | +0.16 pt |
| test | 15.0% | 15.29% | +0.29 pt |

Within design §3.3's overall tolerance (±1.5 pt). Per-pool:

| pool | train | val | test |
|---|---|---|---|
| coralscapes (n=2,075, test-overfull, exempt & reported per §3.3.2) | 66.8% | 14.3% | 18.9% |
| mermaid-aws (n=18,561) | 69.99% | 15.00% | 15.00% |

## Split groups found in more than one split

**0.** (Singleton groups in this run make this trivially true — see the provenance
note above. The load-bearing check is
`tests/test_splitv2_holdouts.py`/`tests/test_splitv2_allocate.py`, which use real
multi-member groups and assert the any-member-OOD and no-group-spans-splits
invariants directly.)

## What a real v2 build still needs before this dry run becomes the release numbers

1. A WP-10 `split_group_id` table for `coralscapes` and `mermaid-aws` (dedup/near-dup
   grouping), so groups — not single images — are the allocation unit.
2. WP-2b geography/depth/platform/time enrichment run over these two sources, so the 6
   OOD holdouts have real candidates to evaluate (currently 0/6 constructible here).
3. The v1 69,600-row per-image table (`source_id`, `split_group_id`, pins) joined
   against `registry/SPLIT_MAP.json`, so the "v1 eligible" row above is measured by
   this package rather than carried from the design doc's own §0 measurement.
