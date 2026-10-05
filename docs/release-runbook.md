# Release runbook: the D5 command list (per flavour)

Run once per flavour (`open` -> `reefsupport/marine-data`, `nc` -> `reefsupport/marine-data-nc`).
Dry runs always pass `--local-only` (no pinned-tree fetch). Nothing here uploads.

1. `marinedata release build --flavour {open|nc} --release <id> --split-map SPLIT_MAP.json
   [--generate-split-map] --out rel [--local-only] [--local=ID=PATH ...]`
   Writes `rel/releases/<id>/<flavour>/` (manifests, `RELEASE.json`). A per-row-licence source
   (fathomnet, inat-marine, planktonzilla, qut-fish) is admitted to both flavours; each row ships
   in the flavour its own licence (the `metadata_norm` per-row normaliser) allows.
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
