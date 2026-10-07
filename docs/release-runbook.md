# Release runbook: the command list (per flavour)

Group near-duplicates first, then generate ONE global split map from those groups, then build each flavour from it (`open` ->
`reefsupport/marine-data`). Where several flavours are built, an image or
near-dup group must have the same split in every flavour. Dry runs always pass `--local-only` (no
pinned-tree fetch). Nothing here uploads.

1. `marinedata dedup run --staged LABEL=DIR [...] --out dedup/` (once per release)
   Hashes and groups the staged corpus; writes `dedup/groups.parquet` (`sha256`, `split_group_id`,
   `group_upstream_splits`). Near-duplicates never straddle splits, so this runs BEFORE the split map.
2. `marinedata release split-map --release <id> --split-map SPLIT_MAP.json --near-dup dedup/groups.parquet
   --local-only [--local=ID=PATH ...]` (once per release; refuses to overwrite)
   Enumerates every open and restricted-licence source (never-released sources excluded), merges shared images
   and the dedup near-dup groups into one component, honours an upstream test set, and writes the frozen
   map. It FAILS without a near-dup input; `--no-near-dup` is the explicit opt-out (recorded in the map
   header). Pass the `--local` trees of BOTH flavours. Every source with >= 3 groups gets at least one
   group in each of train/val/test (val is taken from train, never from the upstream test). A source
   with no registry split rule takes its group from the `metadata_norm` chain at release time; a source
   with >= 100 rows in < 10 groups logs a warning and groups per image.
3. `marinedata release build --flavour {open|nc} --release <id> --split-map SPLIT_MAP.json
   --out rel --local-only [--local=ID=PATH ...]` (run for each flavour, same `SPLIT_MAP.json`)
   Writes `rel/releases/<id>/<flavour>/` (`tasks/*.tsv`, `RELEASE.json`). A source that cannot be
   grouped (no `split_group` from the loader or the enumeration) fails the build with an error naming
   the source. A source without a crosswalk into a task's taxonomy is excluded from that task only; the
   reason is logged under `task_source_exclusions` in `RELEASE.json`. A per-row-licence source
   (fathomnet, inat-marine, planktonzilla, qut-fish) is admitted to both flavours; each row ships in the
   flavour its own licence (the `metadata_norm` per-row normaliser) allows. `--generate-split-map` still
   works for a single flavour (like `split-map` it fails without `--near-dup` or `--no-near-dup`).
   Pass `--decon` to run the decon gate inside the build and record it, with its exemptions, in `RELEASE.json`
   (`decon` block; the card's Limitations section reads it there). The standalone `decon check` of step 5 writes
   only `decon/overlap.*`, never `RELEASE.json`.
4. `marinedata dedup gate rel/releases/<id>/<flavour> --groups dedup/groups.parquet` -> 0 spanning
   groups, 0 upstream-test images in train.
5. `marinedata decon check rel/releases/<id>/<flavour> --manifests-root registry ...` (prerequisite: a
   benchmark manifest parquet under `registry/benchmarks/manifests/<benchmark_id>.parquet` for every
   benchmark whose status is staged/w1/w2, with >= `gate_manifest_min_coverage` of its eval images;
   otherwise the gate fails with "manifest coverage below gate_manifest_min_coverage". A manifest holds
   per-image `sha256`, `pixel_sha256`, `dhash`, `phash`, so it needs the image bytes, not only
   `CHECKSUMS.sha256`: build each with `marinedata bench manifest <benchmark_id> --source
   {bucket|upstream}`, see `marinedata bench manifests` for the pending list.)
6. `python -m marinedata.hf_export --release-dir rel/releases/<id>/<flavour> --out hf-<flavour>
   --summary hf-<flavour>.json [--quality quality.parquet]`
   Writes the `images`, `masks` and task configs **and the `metadata` config** (one row per exported
   image: `image_sha256`, `source_id`, `source_version`, `license`, `licence_class`, `attribution`,
   `lat`/`lon`, `capture_datetime`, `camera`, `split_group`, `split`, ...). `split` is the frozen-map
   split, so "upstream test honoured" can be checked from the export. The metadata rows are flavour
   filtered per row, so every row has a non-null `licence_class`. `--no-metadata` skips the config.
7. (optional refresh) `python -m marinedata.metadata_release --release-dir ... --stage-root ...
   --quality ... --out hf-<flavour> --summary ... --coverage ...` rewrites the same `metadata`
   shards with the staged-version lookup and the coverage report. Its `--summary` holds only the
   metadata config: use a file of its own, never hf_export's summary (the card step reads that one).
8. `marinedata hf-card` / upload stages per `docs/formats.md`; check that every config in the card has
   its `data_files` and that `metadata` is listed.

## Privacy policy (WP-R9)

After `hf_export` (first pass) run `marinedata privacy-scan` on the shards, then re-run
`python -m marinedata.hf_export ... --privacy <privacy.parquet>` and the metadata refresh with the
same `--privacy`. A face score >= 0.85 excludes the image from every config and lists it in
`RELEASE.json` (`privacy_exclusions`: image id, source, score, detector + version); 0.60-0.85 keeps
it with `privacy_flag` = `possible_face` and `face_score` in `metadata`. The thresholds are the two
constants in `src/marinedata/privacy/policy.py`; the dataset card prints them. Without `--privacy`
nothing is excluded or flagged.
