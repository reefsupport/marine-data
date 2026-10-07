# QUEUE-local: draining the ok ingest queue (2026-09-25)

One background runner drains the 54 remaining `ok` rows of
`registry/ingest-specs/_queue.tsv` (the 55th, `usis10k`, runs separately —
see below). It streams per item (`fetched.path.unlink()` after each item's
decode, `images/`/`labels/` files removed only after their S3 upload
verifies), so local temp never holds more than one flush window
(`temp_cap_gb/2`, default 3 GB) regardless of source size.

## Relaunch command

If the runner dies or needs restarting, it is safe to just re-run — every
source is resumable via its `CHECKSUMS.sha256` marker (skip-done) and every
partial upload resumes via its multipart checkpoint:

```sh
SP=/path/to/scratch
nohup sh "$SP/queue/run.sh" > "$SP/queue/run.log" 2>&1 &
echo $! > "$SP/queue/pid"
```

`run.sh` runs two sequential `ingest-batch` calls (wave-w1a first, per
manager priority, then the rest), each `--jobs 3` (the Do-NOT-exceed
concurrency cap). Regenerate the id lists if the queue changes:

```sh
cd ~/dev/.wt/marine-data/ingest-int
awk -F'\t' 'NR==1{next} !/^#/ && $8=="ok" && $1!="usis10k"{print $10"\t"$1}' \
  registry/ingest-specs/_queue.tsv | sort -t$'\t' -k1,1nr | cut -f2
```

## Disk floor

Every spec this run touches got `disk_floor_gib: 36.0` added (charter D-F's
40 GiB default paused every source at the ~38 GiB free this Mac had at
launch). `DiskGuard.check()` raises `DiskFloorError` when free space drops
below the floor; `ingest-batch` catches that as a per-source `error` (not a
batch abort — the JSONL line is written and the next source still runs).
**No staged data is ever deleted by this**: `_Uploader.push` only unlinks a
local file after `upload_file`'s own verify (etag/size match), so an
interrupted source just resumes into its own checkpoint on the next
`run.sh` pass.

## Runner 2 (INT-ingest2, 2026-09-25)

A second, independent background runner drains the 17 rows the WP-6d-A/B
merge newly flipped to `ok` (`aqualoc, caddy, duo, eilat-rsmas, fathomnet,
hicrd, lsui, ntnu-arl-uw, oceaninstruct, pangaea-ccz-gsr, pangaea-ofos-msm77,
salmon-cage, seamapd21, seathru-nerf, underwater-images-2542305, usod10k,
viame-public`) plus the 4 rows that hit an HF 429 in runner 1's `run.log`
(`ruod, marineeval, uiis10k, uiis`) — 21 ids total, `--jobs 2`.

```sh
SP=/path/to/scratch
nohup sh "$SP/queue2/run.sh" > "$SP/queue2/run.log" 2>"$SP/queue2/run.err" &
echo $! > "$SP/queue2/pid"
```

- **PID:** `70215` (recorded at `$SP/queue2/pid`; the first launch, `55020`,
  already ran its one pass and exited — see the note below)
- **Log:** `$SP/queue2/run.log` (JSONL, one line per finished source; stderr
  separately at `$SP/queue2/run.err`)
- **Ids:** `$SP/queue2/ids.txt` (21 ids, comma-separated, same file as passed
  to `--only`)
- **Disk floor:** `disk_floor_gib: 34.0` set on all 21 specs (down from the
  default 40 GiB / runner 1's 36 GiB — free disk was 33 GiB at launch, so
  this runner **starts paused**: every source raises `DiskFloorError`,
  caught as a per-source `error` line, until free space rises above 34 GiB.
  That is expected; do not lower the floor further. It shares this Mac's
  disk with runner 1 (36 GiB floor) and `usis10k`, so it will start doing
  real work once either of those frees enough space or completes.
- **Unlike runner 1, this process does not stay alive while paused** —
  `DiskFloorError` is raised (and caught as `error`) within seconds per
  source, so a fully-paused pass finishes and the `sh` process exits after
  ~1-2 minutes (`QUEUE2_RUN_DONE` in the log), instead of blocking. Re-launch
  the same command above whenever free space might have risen; every source
  is resumable (same `CHECKSUMS.sha256`-marker skip-done as runner 1), so
  re-running is always safe.
- **D-AA (HF 429 backoff):** `ingest-batch` now cools an HF source down
  (>= 15 min) and retries it in place on a 429 rather than failing the
  batch, keeps HF concurrency at 1 regardless of `--jobs`, and gives up as
  `needs-yohan: hf-rate-limit` after 3 cool-downs — see
  `_run_one_hf_aware` in `src/marinedata/cli_ingest_batch.py`. This runner
  is the first to exercise it live (`ruod`, `marineeval`, `uiis10k`, `uiis`
  are all HF sources that 429'd on runner 1).
- **Relaunched under a pass-loop wrapper (INT-ingest3, 2026-09-25):** the
  `70215` process above had already run its one pass and exited (fully
  paused, `disk_floor_gib: 34.0`, free disk 8-12 GiB at the time). Same 21
  ids, same floor, same `--jobs 2` — nothing about the runner's scope
  changed, only that it no longer needs a manual re-launch. New PID
  **`17011`**, started via `$SP/queue2/wrapper.sh` (pass → grep the pass's
  own stdout for `DiskFloorError` → if found, `sleep 900` and re-run the
  pass; exit the loop on a pass with no match), bounded by
  `perl -e 'alarm 259200; exec …'` sh wrapper.sh` (72 h hard cap). Log is
  the same `$SP/queue2/run.log` (now prefixed with `=== PASS N ===`
  markers per pass); stderr still `$SP/queue2/run.err`.

## Runner 3 (INT-ingest3, 2026-09-25)

Drains the 7 W3-wave `ok` rows (`elliott-bay-benthic, nes-plankton-2022,
noaa-oceaneyes, plankton-interaction-videos, seattle-aquarium,
seaturtleid2022, uwbench` — ~124 GB declared). `planktonzilla` (also W3,
also `dry_run: ok`) is deliberately **excluded**: per D-AB it must be
fetched as a subset only (759,694 of 17.4M rows via
`stratify:[dataset, proposed_label]`), and the `hf` adapter has no
per-row subset filter yet, so its queue row was flipped to
`needs_adapter:subset-filter` instead of being fetched in full.

```sh
SP=/path/to/scratch
nohup perl -e 'alarm 259200; exec @ARGV' sh "$SP/queue3/wrapper.sh" \
  > "$SP/queue3/wrapper.out" 2>"$SP/queue3/wrapper.err" &
echo $! > "$SP/queue3/pid"
```

- **PID:** `16843` (recorded at `$SP/queue3/pid`)
- **Log:** `$SP/queue3/run.log` (JSONL per source, `=== PASS N ===`
  markers between passes); stderr `$SP/queue3/run.err`; wrapper's own
  stdout/stderr at `$SP/queue3/wrapper.out` / `.err`
- **Ids:** `$SP/queue3/ids.txt` (7 ids, comma-separated, matches `--only`)
- **Jobs:** `--jobs 1` (per the brief, lower concurrency than runner 2)
- **Disk floor:** `disk_floor_gib: 34.0` set on all 7 specs. Free disk was
  8-12 GiB at launch (below both runner 1's 36 GiB and runner 2/3's 34 GiB
  floors), so this runner **starts paused on every source** — expected;
  do not lower the floor.
- **Pass-loop wrapper:** `$SP/queue3/wrapper.sh` runs `$SP/queue3/run.sh`
  (one pass over all 7 ids), greps that pass's stdout for `DiskFloorError`;
  if found, appends the pass to `run.log`, sleeps 900s, and re-runs; if a
  pass has zero `DiskFloorError` matches, it appends `QUEUE3_WRAPPER_DONE`
  and exits. The whole wrapper is exec-chained under
  `perl -e 'alarm 259200; exec …'` (72 h hard cap) so the same PID (`16843`)
  is valid start to finish (`exec` replaces the process image, doesn't
  fork). Every source is resumable (same `CHECKSUMS.sha256`-marker
  skip-done as runners 1/2), so a `sleep 900`-triggered re-run is always
  safe.
- **Do not touch `16843` or `17011`** while alive, same rule as runners 1/2.

## Checking progress

```sh
SP=/path/to/scratch
tail -5 "$SP/queue/run.log"                      # latest JSONL results
.venv/bin/python "$SP/queue/ledger_from_log.py" "$SP/queue/run.log" "$SP/queue/ledger.tsv" \
  && column -t -s$'\t' "$SP/queue/ledger.tsv"    # rebuild + view the ledger
ps -p "$(cat "$SP/queue/pid")"                   # confirm the runner is alive
```

## Stopping

```sh
SP=/path/to/scratch
kill "$(cat "$SP/queue/pid")"   # kills run.sh; its current ingest-batch child exits with it
```

Killing mid-source is safe (resumable, see above) — never `kill -9` a source
mid multipart-complete, since that could leave an orphaned incomplete
upload (harmless, but `S3_ENDPOINT` storage costs accrue until a maintainer runs a
multipart-abort sweep).

## `usis10k` (separate process, do not touch)

`usis10k` runs as its own `ingest-source` process (PID noted in the QUEUE-local
report), started before this batch, with `--work $SP/w1a/work`. It is
**not** part of `run.sh`. Once `ps -p <that pid>` shows it has exited, verify
its S3 prefix (size+ETag against `$SP/w1a/work/uploaded.json`) the same way
`verify_source.py` does below, then add its row to `ledger.tsv` by hand (it
never appears in `run.log`, since it never ran through `ingest-batch`).

## Verification helpers used to prove `floating-marine-debris`

`$SP/queue/list_sources.py` — flat `list_objects_v2` + `Delimiter=/` listing
of `sources/` (top-level staged ids). `$SP/queue/verify_source.py <id>
<version>` — lists one source's full prefix, confirms object count, and
confirms `CHECKSUMS.sha256` has the latest `LastModified` (uploaded last, the
version-complete marker). Anonymous GETs are
`https://<open-bucket>.hel1.your-objectstorage.com/<key>` (virtual-hosted
style — the path-style `hel1.../<open-bucket>/<key>` 403s).

## Runner 4 (INT-ingest4, 2026-09-25)

Drains the 3 SPEC-w4 `ok` rows: `noaa-pifsc-esa-coral-icra, noaa-pifsc-bleaching, aqua20` (~6.1 GB dry-run).
Deep-sea-first ordering was requested, but no w4 deep-sea id is `ok` and not already owned by another runner
(`deepsea-mot` is in runner 1). Same shape as runner 3: `--jobs 1`, `disk_floor_gib: 34.0` on all 3 specs,
D-AA via `ingest-batch` (plus `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, token env unset in `run.sh`), pass-loop
wrapper (`sleep 900` after any `DiskFloorError` pass) under `perl -e 'alarm 259200; exec …'`.

- **Wrapper PID:** `71389` (`$SP/queue4/pid`). **Runner:** spawned per pass as `sh run.sh` → `.venv/bin/python -m
  marinedata.cli ingest-batch`; transient (pass 1 exited in ~1 s on the floor). Between passes the child is
  `sleep 900` (`71537` at launch).
- **Log:** `$SP/queue4/run.log`; stderr `run.err`; ids `$SP/queue4/ids.txt`.
- Launched at 10 GiB free: paused on the floor by design until the disk cleanup runs. Do not lower the floor.
