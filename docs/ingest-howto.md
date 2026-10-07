# Ingest one open source to `<open-bucket>` (recipe)

Use this for any catalog row. Design and failure modes: `docs/design/ingestion.md`.

## 0. Preconditions (stop if any fails)

1. The source is **open**: open HF (not gated), Zenodo `access_right: open`, anonymous
   bucket, public GitHub. If you would need a login, terms click-through, or a gated HF repo,
   **do not ingest**. Add it to the "needs decision" list with the exact URL and what has to be
   accepted (D-E).
2. You know its licence (SPDX, e.g. `CC-BY-4.0`) and attribution from the upstream card or
   README. A missing licence gets `NOASSERTION`; never guess one.
3. Run `df -g /System/Volumes/Data`: at least 40 GiB must be free (D-F). If not, pause
   and report.

## 1. Write the spec (`$SP/<n>/<id>.yaml`)

```yaml
id: rf100-coral-lwptl              # kebab-case; becomes sources/<id>/<version>/
adapter: hf                        # hf | zenodo | http | bucket | github
params:
  repo: LibreYOLO/coral-lwptl
  revision: 83f0a33679b0c845a8370cc2541d745157820fe1   # pin it
  label_patterns: ["*/labels/*.txt", "data.yaml"]      # loose label files to keep
license: CC-BY-4.0
attribution: "Nikita Manolis (Roboflow Universe), via RF100 coral-lwptl"
citation: "Ciaglia et al. 2022, arXiv:2211.13523"
homepage: https://universe.roboflow.com/roboflow-100/coral-lwptl
```

`params` per adapter:
- **hf**: `repo`, `revision`, `include` (globs; the default prefers parquet),
  `label_columns`, `columns` (schema field -> column), `image_column`.
- **zenodo**: `zenodo_record`, `include`, `label_patterns`.
- **http**: `urls` plus a top-level `version:` (required).
- **bucket**: `endpoint`, `bucket`, `prefix`.
- **github**: `repo` plus either `release: <tag>|latest` or `ref` (+ optional `raw` paths).

Optional top-level keys:
- `defaults:`: only fields the upstream **states** for every image (e.g. `platform: auv`).
- `expected_images:`: set it when above 200k, so the run picks `shards`.
- `max_images:`: for a trial run.
- `naive_datetime_is_utc: true`: only if the card says so.

Never invent lat/lon/depth.

## 2. Dry run (no bytes move)

```bash
cd ~/dev/.wt/marine-data/<your-worktree>
PYTHONPATH=src .venv/bin/python -m marinedata.cli ingest-source hf $SP/<n>/<id>.yaml \
  --dry-run --work $SP/<n>/work
```

Check the output:
- `plan.items` and `plan.declared_bytes` are what you expect.
- `plan.target` is `s3://<open-bucket>/sources/<id>/<version>/`.
- `layout` is `objects` at 200k images or fewer, `shards` above that.

Exit code 3 plus a `NEEDS-YOHAN` line means the source is gated: record it and stop.

## 3. Real run (under nohup, bounded wait)

```bash
PYTHONPATH=src nohup .venv/bin/python -m marinedata.cli ingest-source hf $SP/<n>/<id>.yaml \
  --work $SP/<n>/work > $SP/<n>/<id>.log 2>&1 &
PID=$!; perl -e 'alarm 1800; exec @ARGV' sh -c "while kill -0 $PID 2>/dev/null; do sleep 15; done"
tail -30 $SP/<n>/<id>.log
```

- Temp stays at or below `temp_cap_gb` (default 6). Local copies of uploaded files are
  deleted after a verified upload.
- **Killed or failed?** Re-run the exact same command. Finished files are skipped and
  multipart uploads resume.

### 3a. Bounded concurrency (`--jobs`, WP-6b)

Fetch and PUT are network-bound, so both are pooled behind a bounded thread pool.
Output is byte-identical to `--jobs 1` regardless of `N`: only fetch (network) and PUT
(network) are parallel; decode and write stay single-threaded and strictly in
enumeration order, so shard layout, `metadata.parquet`/`index.parquet` row order, and
`CHECKSUMS.sha256` — and therefore `root_digest` — never depend on `N`.

```bash
PYTHONPATH=src nohup .venv/bin/python -m marinedata.cli ingest-source hf $SP/<n>/<id>.yaml \
  --work $SP/<n>/work --jobs 8 > $SP/<n>/<id>.log 2>&1 &
```

- `--jobs N` (default 8, max 32): in-flight fetch prefetch depth and PUT pool size.
  `--jobs 1` takes the original sequential code path unchanged — no pool is created.
- `--max-per-host N` (default 4): per-host semaphore, independent of `--jobs` — one slow
  or rate-limited host never starves the others.
- `--part-jobs N` (default 4): parallel part upload for one multipart object; only
  matters for files large enough to need more than one part.
- 429/5xx/connection-reset errors (HTTP or S3) retry with capped exponential backoff
  + jitter; a 4xx auth/validation error, or a real interrupt (Ctrl-C/kill), never retries.
- The D-L temp-disk cap (`temp_cap_gb`) stays authoritative under concurrency: fetches
  that spool to disk block on the cap via a bounded wait, they don't bypass it. Loose
  in-memory items (the common case `--jobs` speeds up) never touch disk during fetch,
  so they don't count against the cap at all.
- Killed mid-run at any `--jobs N`? Re-run the exact same command — already-uploaded
  objects are skipped by ETag/sha and in-progress multipart uploads resume from their
  checkpoint, same as `--jobs 1`.

**`--fetch-only` (network-light throughput probe).** Measures fetch files/s at a given
`--jobs` with no staging and no S3 — useful before committing to a full run:

```bash
PYTHONPATH=src .venv/bin/python -m marinedata.cli ingest-source hf $SP/<n>/<id>.yaml \
  --fetch-only --jobs 8 --limit 500
```

| Check | N=1 | N=8 | N=16 |
|---|---|---|---|
| `rf100-coral-lwptl` smoke re-run (0 B expected, skip-on-ETag) | — | not yet measured live | — |
| `fathomnet-vme` COCO, 500 image URLs, fetch-only | not yet measured live | not yet measured live | not yet measured live |

The table above is intentionally unfilled: the runner this change landed on had less
than the D-F 40 GiB free-disk floor at verification time (`DiskGuard` correctly refused
non-dry-run work), so the real-network smoke re-run and the fathomnet-vme throughput
probe were not run live. `--fetch-only` and `--jobs` are implemented and covered by
`tests/test_ingest_source.py` (`test_jobs_n_matches_jobs_1_root_digest`,
`test_temp_cap_never_exceeded_with_jobs`, `test_kill_mid_run_then_resume_uploads_only_missing`)
plus `tests/test_s3_upload.py` (429 retry, parallel part upload); fill this table the
next time free disk is back above the floor.

## 4. Check it is done

- The log's final JSON must show all of these:
  - `verified == files`
  - `root_digest` is non-empty
  - `peak_temp_bytes` is below the cap
- `CHECKSUMS.sha256` exists under the prefix. It is written last, so its presence means the
  version is complete.
- Report `id`, `version`, `images`, `files`, `root_digest`, and the log path.
- Hand `work/registry-stub-<id>.yaml` to the integrator. It still needs `description`,
  `capabilities`, and coverage before it goes into `registry/sources/`.

## Never

- Delete bucket objects, even a half-run prefix: re-run instead (maintainers run deletions).
- Print credentials. The client reads the `[rs-hel1]` section of `~/.config/rclone/rclone.conf` itself.
- Push to HF.
- Edit `hf_card.py`, `hf_export.py`, or `hf_parquet.py` (D-M).
- `rm` files you did not create.
