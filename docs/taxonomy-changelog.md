# Taxonomy changelog

## 1.1.0 (2026-09-25, WP-7b)

MINOR bump from WP-7's 1.0.0 — nodes and crosswalks added, nothing removed or retargeted.

- **1 node added:** `rs-benthic-v1:SG_BG` ("Background (no seagrass)"), `non_taxon`
  category `background` (new category — see `schema.NON_TAXON_CATEGORIES`).
- **1 edge changed:** `deepseagrass:Background` now targets `SG_BG` (`exact`) instead of
  `ABIOTIC` (`approximate`). The manager's correction: Background is not "non-living
  substrate" (ABIOTIC's own reason) — the frame may show water, and substrate is
  unspecified either way.
- **43 edges added** across 4 new crosswalks, closing 4 of the 23 `no_crosswalk_yet`
  sources: `reefolution` (39 edges, real CoralNet-native points data, ~62% coverage —
  see the `coverage_exceptions` entry for the other 24 codes), `coralscop-masks-rs` (1
  edge, class-agnostic model masks), `labeled-fishes-in-the-wild` and `mouss-detection`
  (1 edge each, single implicit "fish" class per the registry's own loader notes).
- **Gate flipped:** `no_crosswalk_yet` is no longer a release-gate exemption
  (`taxonomy.gate`). `marinedata taxonomy check` (the full-registry check, no
  `source_ids`) now FAILS on the 19 sources still listed in `no_crosswalk_yet` — see
  `## Open` in the WP-7b report. Per-release gates scoped to a release's actual
  `source_ids` (e.g. `assert_release_gate(..., ["ruod", "noaa-benthic-t1"])`) are
  unaffected and still pass, because releases only ever name sources that are staged.
- **19 sources remain in `no_crosswalk_yet`,** unresolved: only 2 of the original 23
  (`reefolution`, `coralscop-masks-rs`) turned out to be staged in `rs-storage-open` —
  confirmed by a `list_objects_v2` scan of `sources/` 2026-09-25. The rest are
  prospective registry entries with no ingested label data (many blocked by a Kaggle,
  Roboflow or Google Drive login wall per D-E, or genuinely open-vocabulary/text —
  `coralvqa`, `marineinst20m`, the `coralnet` catch-all). Real crosswalks for those need
  ingestion (or, for `coralnet`/`fathomnet`, a per-source-at-ingest step by design —
  their entries already say so) — writing edges now would mean inventing label names.
