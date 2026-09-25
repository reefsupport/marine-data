# v2 integration — `feat/v2-int` (INT-core, 2026-09-25)

Base: `feat/splitmap-group` @ `7ab29a6`. Every branch merged with `--no-ff`, in the order below
(S59 → WP-4 → WP-1 → WP-5 → WP-5b → WP-6 → WP-7 → WP-10 → the rest). S61 (`s61-stage-local`) and the
ingest stack are out of scope for this integration (see "Still to merge").

## Merge log

| # | Branch | Merged sha | Merge commit | Conflicts | Resolution |
|---|--------|-----------|--------------|-----------|------------|
| 1 | S59 `feat/hf-v1-pack` | `b1218ec` | `97c5b1f` | none | — |
| 2 | WP-4 `feat/wp4-ci` | `ad12f52` | `e19ce9c` | none | — |
| 3 | WP-1 `feat/wp1-quality` | `139e99b` | `5e250a2` | `pyproject.toml` | Pillow `11.0.0` → `12.3.0` in `ingest`, `quality` and `dev`; `quality` extra kept. `tests/test_quality.py` 23/23 under 12.3.0, so no fixture values changed. |
| 4 | WP-5 `feat/wp5-datasheet` | `aebb212` | `71ebf4a` | `pyproject.toml` | Union: `croissant` extra and `mlcroissant`/`cffconvert` added to `dev`, Pillow kept at 12.3.0. |
| 5 | WP-5b `feat/wp5b-privacy` | `df05dcc` | `45ff7ac` | `cli.py` | Both subcommands are registered (`privacy-scan` and `quality`). |
| 6 | WP-6 `feat/wp6-ingest` (pinned) | `8150c8f` | `30d792a` | `cli.py` | Both registered (`ingest-source` is added). Pinned because release code imports it. WP-6b is excluded. |
| 7 | WP-7/7b/7c `feat/wp7-taxonomy` | `ee81554` | `d55a13d` | `Makefile` (add/add) | Kept WP-4's Makefile. Added the `taxonomy-check` target and wired it into `ci: lint check taxonomy-check test e2e`. |
| 8 | WP-10 `feat/wp10-dedup` | `66af021` | `bc21cfa` | `cli.py` | Both registered (`dedup` is added). `--dedup-v2` defaults to off. |
| 9 | WP-2/2b `feat/wp2-metadata` | `29bedbb` | `5717da6` | `pyproject.toml` | Union: `geo` extra added (shapely), Pillow kept at 12.3.0. WP-2's `sample_schema.py` fields are kept: WP-2 already contains `8150c8f`, so the schema did not diverge. |
| 10 | WP-3 `feat/wp3-verify` | `2f3369b` | `03ee2a1` | `pyproject.toml`, `cli.py` | Union: `verify` extra added, and `boto3`/`moto`/`datasets`/`webdataset` added to `dev`. Both registered (`verify-release`, `manifest`, `export-wds`). |
| 11 | WP-11/12 design `feat/wp11-12-design` | `705db87` | `f27f114` | none | — |
| 12 | P4 `feat/p4-eval-core` | `2f37011` | `bd34f2f` | `pyproject.toml`, `cli.py` | Union: `eval` extra added (dropped P4's `pillow==11.0.0` ingest line). Both registered (`eval`). |

Fix-ups on `feat/v2-int`:
- `a49f179`: `uv lock`. WP-1 pins `numpy==2.4.6`, which needs Python ≥ 3.11, but the project declares
  `>=3.10`, so the lock had no solution for the 3.10 split. **Design call (conservative):** the pin now
  applies to Python ≥ 3.11 (CI runs 3.12), and 3.10 falls back to `numpy>=1.24`. `requires-python` is
  unchanged. The lock resolves 155 packages, including Pillow 12.3.0.
- `ab669d3`: `ruff format` on 7 files from branches that predate WP-4's `ruff format --check`
  (`quality.py`, `cli_quality.py`, `test_quality.py`, `croissant.py`, `iucn_worms.py`,
  `test_metadata_release.py`, `test_eval_predictions.py`). The change is formatting only.

## Verification

- Targeted tests passed after every merge (the touched test modules plus `test_e2e_release.py`).
- `make ci` (lint → check → taxonomy-check → test → e2e) exits 0. The full suite gives **945 passed**,
  4 skipped (WoRMS/OBIS integration) and 10 deselected (`integration`). The run used uv 0.11.4,
  Python 3.12.13, ruff 0.16.3 and Pillow 12.3.0. The `slow` marker is not deselected by `addopts`, so
  `make ci` already runs the `slow` tests (WP-3 note).
- The e2e golden hashes are unchanged: no merge changed the e2e fixture outputs. The fixture registry
  has no taxonomy root, so WP-7's stamp is empty there.
- **v1 byte-identity** (`--dedup-v2` off; `--decon` does not exist in any merged branch yet):
  - Rebuild against the frozen `registry/SPLIT_MAP.json` (same sha as the published map, `bb2e73e5`):
    `SPLIT_MAP.json` and all 6 task TSVs are byte-identical to
    `_release/2026-09-24-neardup/releases/v1/`.
  - `RELEASE.json` is the one intended diff. It has two keys appended, from WP-7's release label gate
    (`release_label_gate`): `"taxonomy_snapshot": "worms-2026-09-25.parquet"` and
    `"taxonomy_version": "1.2.0"`. Remove those two keys and the file is identical. The published S3
    `CHECKSUMS.sha256` pins the old `RELEASE.json` sha `24f0a41c…`, so a v1 rebuilt from this branch
    will not match it.
  - `CHECKSUMS.sha256` exists only in the reference. It is S59's packaging artefact, not a
    `release build` output.
  - Regenerating the map with `--generate-split-map` changes only `generated_at` (wall clock) and the
    `split_map_sha256` that follows from it. The assignments, `near_dup` and all 6 TSVs are identical.

## INT-core2 merge log (2026-09-25)

Continues the log above from `fa46154`. All 8 branches were merged at the brief's fixed commits
(never a tip, since the branches were still moving).

| # | Branch | Merged sha | Merge commit | Conflicts | Resolution |
|---|--------|-----------|--------------|-----------|------------|
| 1 | WP-10c `feat/wp10-dedup` | `8c26170` | `03648b6` | none | — |
| 2 | WP-9c `feat/wp9-label-quality` | `0975973` | `3d41952` | `cli.py` | Both registered (`labelquality` import + subparser added alongside the existing ones). |
| 3 | P3 `feat/p3-splitv2` (incl. P1 `d376e95`) | `fed2608` | `17f7fb6` | `cli.py` | Both registered (`cli_bench`/`cli_splits` imports and `add_bench_subparser`/`add_splits_subparser` calls added alongside `cli_verify`/`eval.cli`/`cli_labelquality`). `registry/benchmarks.yaml` union auto-merged, no manual resolution needed. |
| 4 | P2 `feat/p2-decon` (merges WP-10c `8c26170`) | `6bb90cf` | `ad61374` | `cli.py` | `add_decon_subparser(sub)` added alongside `add_splits_subparser`. `cli_release.py`/`release.py` auto-merged (the `decon` hook, default off, D-X). |
| 5 | S61 `feat/s61-stage-local` | `9b3fcb9` | `ebc0054` | none | — |
| 6 | WP-2c `feat/wp2-metadata` | `7b5ab23` | `6478862` | none | — |
| 7 | WP-5d `feat/wp5b-privacy` | `2398fc4` | `ee93336` | none | — |
| 8 | WP-8c `feat/wp8-tasks` | `c39807c` | `a048ef6` | `cli_release.py`, `release.py` | `build_release(..., decon=args.decon, tasks=args.tasks)` — both kwargs kept as independent parameters. `hf_card.py`/`hf_export.py` auto-merged, untouched beyond that (per the brief). |

### Bug found and fixed in this round

`feat/wp8-tasks`'s `build_release()` declared `tasks: str = "v1"` as a parameter, then a few lines
into the function body shadowed it with a same-named local `tasks: list[TaskManifest] = []` — so the
later `if tasks == "v2":` always compared a list to a string and never fired. `--tasks v2` silently
built nothing extra, both on the branch and after merge. No test called `build_release(tasks="v2")`
end to end, so nothing caught it. Fixed by binding `tasks_mode = tasks` before the shadow and checking
`tasks_mode == "v2"`; `tests/test_task_layers_wp8c.py` and `tests/test_e2e_release.py` still pass.

### `--split-v2`, `--dedup-crop` and the `v2=True` preset (new in this round)

Not present on any merged branch; added directly to `build_release()`/`cli_release.py`:

- `--dedup-crop` (WP-10c) threads into `run_decon_gate(..., dedup_crop=dedup_crop)`; it only changes
  output together with `--decon`. Default off.
- `--split-v2` (WP-11/12 P3) is a config-only smoke check: it loads `registry/splits/v2.yaml` and
  `registry/benchmarks.yaml` and records their hashes under `RELEASE.json["split_v2"]`. It does
  **not** run the full per-sample OOD/allocator gate — `metadata.parquet` has no `meow_realm`/
  `depth_m`/`platform`/`capture_datetime` columns yet (`docs/split-v2-dry-run.md` already says this
  gate needs WP-2's per-sample geo enrichment) — so the real gate is deferred to the v2 build. Default
  off.
- `v2=True` is one preset that sets `decon`, `dedup_crop`, `split_v2` and `tasks="v2"` together. It
  does **not** enable `--dedup-v2` (the WP-10 split-leak gate), which needs an explicit
  `groups.parquet` path and cannot be turned on by a bare bool. All defaults stay off, so an unflagged
  build is byte-identical to before this preset existed (D-X).

## INT-core2 verification (2026-09-25)

- Targeted tests passed after every merge (the touched test modules, plus `test_e2e_release.py`,
  `test_decon.py` and `test_dedup_v2.py` where relevant).
- `ruff check .`: all checks passed.
- `make ci` (lint → check → taxonomy-check → test → e2e): see the report for the exit code and count.
- Full suite `pytest -q tests/`: see the report for the pass/skip/deselect counts.
- v1 identity (D-X) proved at **manifest level, with no image bytes** (INT-core2b,
  `src/marinedata/manifest_identity.py`). The INT-core round's two-pass `release build` rebuild was
  NOT re-run: `_resolve_roots` fetches every non-`--local` staged tree, images included, and on this
  Mac it re-downloaded mermaid-aws + coralscapes (+14 GB). Instead: rows = each admitted staged-tree
  source's local `metadata.parquet` + its `CHECKSUMS.sha256` (anonymous HTTPS GET, digest-pinned to
  `checksums.root_digest`; the only network I/O), split from `registry/SPLIT_MAP.json`, and every
  metadata column rebuilt by the v1 code path itself (`metadata_release.build_rows`, fed the
  CHECKSUMS key as the file path). Compared against the published HF tree's `data/metadata/*.parquet`
  (`data/_hf/v1/`). `~/.cache/marinedata` 51621 → 51624 MB. Result (2026-09-25):
  - Row ids / sha256: 69,600 published rows, 0 missing from the rebuild; every published
    `(sha256, source_id)` pair is reconstructed. 557 extra `coralscop-masks-rs` shas are admitted rows
    that sit in no v1 task (the local v1 release's task TSVs hold exactly the 69,600 published shas and
    none of the 557) — the row manifest does not replicate per-task inclusion rules.
  - Splits: 69,600 / 69,600 identical (`val` ↔ HF `validation`).
  - Columns: 33 compared. 25 identical on every row (licence/provenance, quality `min_side`/`q_*`,
    `quality_flags`, `habitat`, `capture_datetime`, …). `upstream_id` differs on 933 rows, all shas
    that occur under more than one stem in the SAME source (byte-identical files); the published
    stem is among the rebuilt candidates — a tie-break of which duplicate `hf_export` embedded, not a
    content change. `lat`/`lon`/`geo_precision`/`geo_source` differ on the 10,186 `noaa-pifsc-bleaching`
    rows only: WP-2c's site-level geo backfill (`67b141e`) post-dates the published v1 metadata (with
    an empty backfill root those rows match). `meow_*` (1,250 `reef-support-benthic-own` rows) could
    not be checked — no MEOW polygon file is on this machine.
  - The release-build path (task TSVs) is not rebuilt byte-for-byte by this check; no existing test
    fetched images (network tests are already `-m 'not integration'`), so no test was re-marked.

## Still to merge (deltas)

- Excluded from this round by the brief (INT-core3 takes them): WP-7d, WP-8d, WP-5e, P5, WP-13, and
  the ingest branches — `feat/ingest-int` (WP-6b `b70ec81`, WP-6c, spec-w0/w2a/w2b, w1a) and
  `feat/spec-w3` (PARTIAL, per `$T/reports/2026-09-25-5star-integration-notes.md`).
- Carried follow-ups from the notes, not done here:
  - Regenerate `_hf/v1/MANIFEST.tsv` and re-run `verify-release` after WP-2's metadata config (WP-3).
  - Dedupe `s3_client.client_from_rclone` against `s3_upload.client_from_rclone`. The WP-6 copy adds
    `request/response_checksum_*="when_required"`. This is deferred until WP-6b/6c reconcile `s3_upload`.
  - Recompute `_quality/v1/quality.parquet` under Pillow 12.3.0 (WP-1).
  - Regenerate the datasheet and Croissant at the v2 build (WP-5).
  - Run the full split-v2 per-sample gate at the v2 build, once WP-2's per-sample geo columns land
    (this round only wires a config-only smoke check — see above).
  - D-T2's 50-random-crop-merge hand audit (P ≥ 90%) is still owed before `--dedup-crop` ships on.

## INT-core3 merge log (2026-09-25)

| Commit | What | Notes |
|---|---|---|
| `b40dac4` | merge WP-8e `feat/wp8e-tasklabels` @ `a0269e3` | S3-keyed tasklabels producers |
| `d776ecc` | merge WP-5e/5f face blur `feat/wp5b-privacy` @ `30b92b4` | |
| `ed0bf4d` | merge P5 probes `feat/p5-baselines` @ `bf554e5` | its 8 unformatted files were `ruff format`ted in INT-core3c |
| `1b60abf` | merge WP-13 captions `feat/wp13-captions` @ `2bfae35` | |
| `8140176` | fix: D-X2a explicit per-column null sentinels | `manifest_identity.NULL_SENTINELS` |
| `97a127f` | fix: coralvqa `vqa.parquet` keyed by staged image sha256 | tracked file marked `invalid` |
| `2d5e689` | feat: `marinedata captions` wired into the CLI | off in the `--v2` preset |

**`cli.py` conflict resolution:** kept HEAD's superset subcommand list; `captions` is
registered behind `try/except ImportError` (pandas is imported at module top), with a
stub that exits 2 when pandas is absent.

**D-X2 result (manifest level, no image bytes):** 69,600 rows; 6/6 task TSVs
byte-identical to v1; 7 metadata columns enriched null→value on 10,186 rows; the
`upstream_id` duplicate tie-break reproduces v1 with 0 diffs, so no allowlist is needed
(this corrects INT-core2b's "allowlist of 933"); the MEOW region map is vendored and the 3
MEOW columns verified (corrects INT-core2b's "MEOW unverified"). D-X2a: sentinels are an
explicit per-column table; published v1 has 0 `""` and 0 NaN in every column, so the
live result cannot move.

**coralvqa:** the tracked `data/_tasklabels/coralvqa/vqa.parquet` had `sha256 = null` on
all 254,867 rows — the WP-8d producer wrote `None` literally because the images (one
26.7 GB zip) were never staged. It stays `invalid` in `data/_tasklabels/MANIFEST.json`,
so `task_layers.configs` skips it. INT-core3c found no staged
`sources/coralvqa/**/CHECKSUMS.sha256` in `rs-storage-open`: anonymous HEAD → 403. The
source is queued for the ingest line; regenerate once it is staged.

**Tasklabels root (INT-core3c):** `build_release(tasks="v2")` read `Path(".")/_tasklabels`,
and a missing parquet reads as `[]`, so a build run from the repo root produced 6 empty
configs. Now `task_layers.configs.resolve_tasklabels_root` takes the explicit
`--tasklabels-root`, or else derives `<repo>/data` from the registry's location; it
fails closed if no `_tasklabels/` exists there. `tests/test_tasklabels_root.py` checks
that the configs are identical and non-empty when built from two different cwds.

**Taxonomy:** frozen at 2.1.0 (charter D-AD), pinned in `tests/test_taxonomy.py`.
