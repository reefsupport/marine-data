# Release runbook: the D5 command list (per flavour)

Generate ONE global split map first, then build each flavour from it (`open` ->
`reefsupport/marine-data`, `nc` -> `reefsupport/marine-data-nc`). "Full" = open + nc, so an image or
near-dup group must have the same split in both repos. Dry runs always pass `--local-only` (no
pinned-tree fetch). Nothing here uploads.

0. `marinedata release split-map --release <id> --split-map SPLIT_MAP.json [--local-only]
   [--local=ID=PATH ...]` (once per release; refuses to overwrite)
   Enumerates every open + restricted-nc source (never-released sources excluded), merges shared and
   near-dup images into one component, honours an upstream test set, and writes the frozen map. Pass
   the `--local` trees of BOTH flavours. A source with no registry split rule takes its group from the
   `metadata_norm` chain at release time; a source with >= 100 rows in < 10 groups logs a warning and
   groups per image.
1. `marinedata release build --flavour {open|nc} --release <id> --split-map SPLIT_MAP.json
   --out rel [--local-only] [--local=ID=PATH ...]` (run for each flavour, same `SPLIT_MAP.json`)
   Writes `rel/releases/<id>/<flavour>/` (manifests, `RELEASE.json`). A source the map cannot cover
   (non-`staged-tree` layout, e.g. coralvqa, atlantis) is not rooted and is listed under
   `skipped_sources` in `RELEASE.json`. A per-row-licence source (fathomnet, inat-marine,
   planktonzilla, qut-fish) is admitted to both flavours; each row ships in the flavour its own licence
   (the `metadata_norm` per-row normaliser) allows. `--generate-split-map` still works for a single
   flavour and also generates the global map.
2. `python -m marinedata.hf_export --release-dir rel/releases/<id>/<flavour> --out hf-<flavour>
   --summary hf-<flavour>.json [--quality quality.parquet]`
   Writes the `images`, `masks` and task configs **and the `metadata` config** (one row per exported
   image: `image_sha256`, `source_id`, `source_version`, `license`, `licence_class`, `attribution`,
   `lat`/`lon`, `capture_datetime`, `camera`, `split_group`, ...). The metadata rows are flavour
   filtered per row, so every row has a non-null `licence_class`. `--no-metadata` skips the config.
3. (optional refresh) `python -m marinedata.metadata_release --release-dir ... --stage-root ...
   --quality ... --out hf-<flavour> --summary ... --coverage ...` rewrites the same `metadata`
   shards with the staged-version lookup and the coverage report.
4. `marinedata hf-card` / upload stages per `docs/formats.md`; check that every config in the card has
   its `data_files` and that `metadata` is listed.
