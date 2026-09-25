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

## Still to merge (deltas)

- `feat/wp10-dedup`: now at `fe4923f`, one commit (WP-10b) past `66af021`.
- `feat/wp2-metadata` (WP-2c) and `feat/wp7-taxonomy` (WP-7d): merged at `29bedbb` and `ee81554`,
  and still receiving commits.
- The ingest stack, via INT-ingest (`feat/ingest-int`): WP-6b `b70ec81`, WP-6c, spec-w0/w2a/w2b and
  w1a. Also S61 `feat/s61-stage-local`.
- Carried follow-ups from the notes, not done here:
  - Regenerate `_hf/v1/MANIFEST.tsv` and re-run `verify-release` after WP-2's metadata config (WP-3).
  - Dedupe `s3_client.client_from_rclone` against `s3_upload.client_from_rclone`. The WP-6 copy adds
    `request/response_checksum_*="when_required"`. This is deferred until WP-6b/6c reconcile `s3_upload`.
  - Recompute `_quality/v1/quality.parquet` under Pillow 12.3.0 (WP-1).
  - Regenerate the datasheet and Croissant at the v2 build (WP-5).
  - Flip `--dedup-v2` together with `--decon` at the v2 build (WP-11/12).
