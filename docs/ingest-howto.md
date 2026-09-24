# Ingest one open source to `rs-storage-open` (recipe)

Use this for any catalog row. Design and failure modes: `docs/design/ingestion.md`.

## 0. Preconditions (stop if any fails)

1. The source is **open**: open HF (not gated), Zenodo `access_right: open`, anonymous
   bucket, public GitHub. If you would need a login, terms click-through, or a gated HF repo,
   **do not ingest**. Add it to the "needs Yohan" list with the exact URL and what has to be
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
- `plan.target` is `s3://rs-storage-open/sources/<id>/<version>/`.
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

- Delete bucket objects, even a half-run prefix: re-run instead (Yohan runs deletions).
- Print credentials. The client reads the `[rs-hel1]` section of `~/.config/rclone/rclone.conf` itself.
- Push to HF.
- Edit `hf_card.py`, `hf_export.py`, or `hf_parquet.py` (D-M).
- `rm` files you did not create.
