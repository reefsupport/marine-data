# Taxonomy changelog

## 2.2.1 — 2026-10-05

PATCH-sized cut for already-landed work: crosswalk edges added after 2.2.0 was frozen. The
frozen 2.2.0 manifest is untouched; this release freezes the working tree as it stands.

- **WP-U3:** `seaview-point-labels` and `ibf-cpce-codes` crosswalks (+ vocab TSVs), no
  nodes added and no existing edge retargeted.

## 2.2.0 — 2026-09-30

MINOR: new crosswalk + vocab, no nodes/edges retargeted.

- **`rf100-coral-lwptl` crosswalk** (14 growth-form classes onto rs-benthic-v1's form/taxon
  axes) closes the source's `no_crosswalk_yet` gap; `vocab/rf100-coral-lwptl.tsv` (6,483
  bbox annotations, 594 label files) measures 100% mapped.
- **Fixed `registry/sources/coral-benthic.yaml` `classes: 16` → `14`**: the pinned rev's
  `data.yaml` (byte-identical from bucket and HF mirror) lists 14 classes; 16 was stale.

## 2.0.0 — 2026-09-25 (WP-7d)

MAJOR: one edge retargeted (`coralscop-masks-rs` `coral`: HC -> CNIDARIA). v2 is unreleased,
so nothing downstream pins the old target. Everything else in this release is additive.

- **Retarget (manager decision 1).** CoralSCOP masks are class-agnostic coral (hard and
  soft), so HC (Scleractinia) asserted a taxon the source never distinguished. The edge now
  targets the lowest node holding both HC and SC. rs-benthic-v1 has no Anthozoa/"coral"
  node, so that node is CNIDARIA. Task fits that need HC vs SC now see CoralSCOP as coarser.
- **Fidelity (manager decision 2).** `noaa-pifsc-bleaching-condition` `CORAL -> HEALTHY` is
  `coarsened`, not `exact`: a binary bleached/not-bleached split only asserts "not
  bleached". NOAA CORAL, Roboflow Healthy (hb, hu) and RS non_bleached share one rule,
  and a test asserts it.
- **The 9 unmeasured staged sources now have vocab TSVs** (D-S1): noaa-pifsc-bleaching,
  5 Roboflow bleaching sets, reef-support-benthic-own, reef-support-bleaching (mask pixel
  values streamed, not stored) and coralscop-masks-rs (one annotation per mask file). All 9
  are at 100% mapped. v13i `Non-Corals` (a presence tag) -> UNKNOWN, approximate.
- **Scoped gate:** a staged or released source with a crosswalk but no vocab TSV FAILS.
- **CoralNet label table** `taxonomy/coralnet-labels-2026-09-25.parquet`: all 12,644 public
  labels (id, name, short code, functional group, verified/duplicate/calcification flags)
  from `/label/list/`; descriptions for 2,013 of them (per-label pages,
  <= 2 req/s, cached, no login; a later brief refreshes the rest).
- **`coralnet-label-id` crosswalk** (generated, 160 curated ids: 64 Reefolution + 96 NOAA
  tier-3, 0 conflicts) + `CoralNetLabelIdResolver`, which falls back through the label's
  functional group via `coralnet-labelset` (+2 group edges: Hard Substrate -> HS,
  Soft Substrate -> ABIOTIC). Reefolution re-derived through it: 43,500/43,500 points, same
  targets as WP-7c. 6 NOAA "NO RULE" ids get their coarsest true node here.
- D-I audit +20 (2 fixed, 3 uncertain); see `docs/taxonomy-audit-2026-09-25.tsv`.

## 1.2.0 — 2026-09-25 (WP-7c)

MINOR: nodes, edges, snapshot rows and one vocabulary added; five edges retargeted to finer
nodes that already existed (no node removed or re-parented).

- **Reefolution 62.23% -> 100.00% of 43,500 points** (64 observed codes, 0 unmappable). The
  codes were resolved from the source's `labelset.csv` (CoralNet label id -> code;
  rs-storage-private) joined to the public CoralNet label pages. Its `coverage_exceptions`
  entry is removed. New vocabulary `vocab/reefolution.tsv` (point-weighted), so the gate now
  measures it. Before, a crosswalked source without a vocab TSV was never measured.
- 16 rs-benthic-v1 nodes: HC_DIPSASTRAEA, HC_FUNGIIDAE, HC_GARDINEROSERIS, HC_GONIOPORA,
  HC_MYCEDIUM, SC_XENIIDAE, SC_RHYTISMA, SC_TUBIPORA, CN_AGLAOPHENIA, INV_DIDEMNIDAE,
  ALG_RHODOPHYTA, ALG_CHLOROPHYTA, ALG_OCHROPHYTA, ALG_SARGASSUM, plus non_taxon HS
  (hard substrate, cover unspecified) and BIOFILM.
- Snapshot +21 rows (12 names + ancestors), retrieved_at 2026-09-25, via
  `scripts/taxonomy_snapshot_add.py` (snapshot first, cached WoRMS REST at <= 2 req/s).
- Fixed 5 Reefolution edges (Mil, Hal, Urchins, Anemone, GA); see the D-I audit (+30 rows).
- Scoped gate (D-Q): `taxonomy check` fails only for staged sources (rs-storage-open
  `sources/`) and sources named in a `--release RELEASE.json`; `--strict` lists the rest.
  `source_crosswalks` declares coralvqa and marineinst20m `crosswalk: open_vocabulary`
  (removed from `no_crosswalk_yet`, 19 -> 17). `make taxonomy-check`.

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
