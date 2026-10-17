# `marinedata` command reference

Generated from the argparse tree by `scripts/gen_cli_docs.py` (run `make docs`); do not edit by hand. `marinedata <command> --help` is the full reference for one command.

`release gate --all --require-all` and `release publish` both refuse a build whose privacy, card or croissant gate is absent, skipped or failed.

| Command | Kind | What it does | Options |
|---|---|---|---|
| `marinedata add` | command | Probe a HuggingFace dataset and print a draft registry entry for review | `--output` |
| `marinedata auth` | group | S3 credential diagnostics (read-only) |  |
| `marinedata auth check` | command | Show the identity bucket reads use; one head_bucket + one head_object, no listing |  |
| `marinedata bench` | group | Benchmark registry: validate, hash, manifest coverage |  |
| `marinedata bench check` | command | Validate the registry and report coverage | `--path`, `--strict` |
| `marinedata bench hash` | command | Print benchmarks_sha256 only | `--path` |
| `marinedata bench manifest` | command | Build one benchmark's eval-image manifest parquet | `--bucket`, `--layout`, `--max-bytes`, `--path`, `--remote`, `--source`, `--specs-dir` |
| `marinedata bench manifests` | command | List which benchmarks have a manifest vs. are pending | `--output`, `--path` |
| `marinedata captions` | group | WP-13 caption generation and checks |  |
| `marinedata captions check` | command | Automated consistency check + schema validation | `--in`, `--out` |
| `marinedata captions pilot` | command | Tier B: stratified VLM captioning pilot | `--images-root`, `--metadata`, `--n`, `--out`, `--seed` |
| `marinedata captions template` | command | Tier A: deterministic captions for all rows | `--benthic`, `--bleaching`, `--metadata`, `--min-facts`, `--out` |
| `marinedata check` | command | Evaluate every source against a profile | `--legal-opinion-ref`, `--profile` |
| `marinedata decon` | group | Benchmark decontamination gate (S0-S5) |  |
| `marinedata decon check` | command | Run the gate against a built release | `--benchmark-root`, `--benchmarks-yaml`, `--dedup-crop`, `--embed-cache`, `--embed-weights`, `--hf-images`, `--manifests-root`, `--out`, `--staged`, `--workers` |
| `marinedata dedup` | group | Cross-source dedup v2 and the split-leak gate |  |
| `marinedata dedup audit` | command | Stratified precision-audit sheets, or --score a filled TSV | `--cache`, `--dedup-crop`, `--hf-images`, `--n`, `--no-dedup-crop`, `--out`, `--score`, `--staged`, `--weights` |
| `marinedata dedup bench` | command | MIH scaling benchmark | `--n` |
| `marinedata dedup eval` | command | Synthetic recall + v1 low-texture replay | `--cache`, `--dedup-crop`, `--hf-images`, `--no-dedup-crop`, `--out`, `--per-family`, `--staged`, `--weights` |
| `marinedata dedup gate` | command | Fail if a split group spans splits or upstream test is in train | `--allow-ungrouped`, `--groups`, `--write` |
| `marinedata dedup run` | command | Hash, embed, dedup and group a corpus | `--cache`, `--dedup-crop`, `--hf-images`, `--no-dedup-crop`, `--no-embed-knn`, `--out`, `--staged`, `--weights`, `--workers` |
| `marinedata doctor` | command | Per-source completeness — layout, licence, crosswalk, fetchable. No network. | `--incomplete-only` |
| `marinedata eval` | group | Score model predictions against ground truth |  |
| `marinedata eval reproduce` | command | Extract cached features, fit a frozen probe and score it (design §5 P5 check) | `--backbone`, `--cache`, `--config`, `--fixture`, `--out`, `--task` |
| `marinedata eval score` | command | Compute metrics + cluster-bootstrap CIs for one task/split | `--boot`, `--gt`, `--metric`, `--out`, `--pred`, `--release`, `--seed`, `--split`, `--task`, `--task-type` |
| `marinedata eval table` | command | Render the design §4.5 card results table | `--metrics`, `--out` |
| `marinedata export-wds` | command | WebDataset tars from the images config + label joins | `--limit-shards` |
| `marinedata fetch` | command | Fetch a bounded verification sample | `--force`, `--limit` |
| `marinedata ingest` | command | Stage a source version into sources/<id>/<version>/ | `--out`, `--profile`, `--slice-cap-bytes`, `--workers` |
| `marinedata ingest-batch` | command | Run ingest-source over every spec in a directory, resumably (WP-6c server Job) | `--jobs`, `--only`, `--remote`, `--stream`, `--verify-only`, `--work` |
| `marinedata ingest-source` | command | Adapter-driven ingest of an open source to sources/<id>/<version>/ | `--dry-run`, `--fetch-only`, `--jobs`, `--limit`, `--max-images`, `--max-per-host`, `--part-jobs`, `--part-size-mib`, `--stream`, `--work` |
| `marinedata labelquality` | group | Label-quality audit (agreement, confident learning) |  |
| `marinedata labelquality agreement` | command | Cross-source agreement on identical / dHash-0 images | `--dhash-db`, `--hf`, `--out` |
| `marinedata labelquality confident` | command | Out-of-fold probe + confident learning | `--features`, `--groups`, `--hf`, `--out` |
| `marinedata labelquality features` | command | DINOv2 feature cache over an HF build's images | `--device`, `--hf`, `--out`, `--revision` |
| `marinedata labels` | command | Audit crosswalk labels against the labels real data contains | `--audit-cache`, `--bucket`, `--json`, `--limit`, `--release-sources`, `--strict`, `--tsv`, `--verbose` |
| `marinedata lineage` | command | Emit a LINEAGE.json for a build | `--legal-opinion-ref`, `--output`, `--profile`, `--task` |
| `marinedata list` | command | List sources matching facets under a profile | `--annotation`, `--legal-opinion-ref`, `--modality`, `--profile`, `--region`, `--task` |
| `marinedata manifest` | command | Write MANIFEST.tsv (path, size, sha256, s3_etag, parquet rows) for a build | `--out` |
| `marinedata metadata` | group | Normalise staged metadata (local output only) |  |
| `marinedata metadata normalise` | command | Write a SampleRow-shaped normalised parquet locally | `--events`, `--limit`, `--meow`, `--out`, `--source`, `--version` |
| `marinedata metadata resolve` | command | Resolve per-row geo/date/site/quadrat metadata (local output only) | `--cache`, `--out`, `--source`, `--version` |
| `marinedata mirror` | command | Plan a copy into our own storage; reports what is refused and why | `--attribution`, `--target` |
| `marinedata privacy-scan` | command | Scan HF-build image shards for identifiable faces / people (WP-5b) | `--batch-size`, `--face-weights`, `--flush-every`, `--input`, `--limit`, `--output`, `--person-weights` |
| `marinedata quality` | command | score per-sample image quality (D9) | `--out`, `--workers` |
| `marinedata registry` | group | Registry pointer checks against the bucket |  |
| `marinedata registry verify` | command | Every s3 pointer must hold a CHECKSUMS.sha256 (offline snapshot or --live) | `--all`, `--listing`, `--live`, `--report-only`, `--sample` |
| `marinedata release` | group | Build a release from admitted sources |  |
| `marinedata release build` | command | Build every registry task against a frozen SPLIT_MAP.json | `--allow-frozen`, `--allow-skip`, `--allow-unsplit`, `--decon`, `--dedup-crop`, `--dedup-v2`, `--flavour`, `--generate-split-map`, `--local`, `--local-only`, `--lock`, `--manifest-only`, `--min-groups`, `--near-dup`, `--no-honour-upstream-test`, `--no-near-dup`, `--out`, `--profile`, `--ratios`, `--release`, `--seed`, `--sources-from`, `--split-map`, `--split-v2`, `--tasklabels-root`, `--tasks`, `--v1-dir`, `--v11`, `--v2` |
| `marinedata release card` | command | Generate the Hugging Face dataset card (README.md) of a built release | `--citation`, `--facts`, `--flavour`, `--out`, `--pretty-name`, `--release-dir`, `--repo-id`, `--template` |
| `marinedata release croissant` | command | Write Croissant 1.0 JSON-LD for a built release | `--citation`, `--date-published`, `--name`, `--no-validate`, `--out`, `--release-dir`, `--repo-id` |
| `marinedata release gate` | command | Run release gates on a build directory and merge them into GATES.json | `--all`, `--benchmark`, `--card`, `--croissant`, `--dedup`, `--detector`, `--empty-threshold`, `--flavour`, `--gates`, `--hidden-test`, `--licence`, `--load`, `--masks`, `--metadata`, `--nc-id`, `--only`, `--phash-bits`, `--privacy`, `--privacy-review`, `--privacy-threshold`, `--rehearsal`, `--release-dir`, `--require-all`, `--sample-rows`, `--split-tolerance`, `--splits`, `--taxonomy-version`, `--v1-dir`, `--v11` |
| `marinedata release lock` | command | Write release.lock.yaml from the registry pins (no silent gaps) | `--code-sha`, `--config`, `--flavour`, `--out`, `--reconstructed-at-sha`, `--release-id`, `--source` |
| `marinedata release publish` | command | Upload a gated release to its Hub repo and tag it (dry run unless --go), all at once or one --part per call. Fail-closed: refuses unless GATES.json passes all eight gates | `--allow-frozen`, `--dir`, `--flavour`, `--go`, `--list-parts`, `--lock`, `--out`, `--part`, `--release-dir`, `--release-id`, `--v1-dir` |
| `marinedata release split-map` | command | Generate the one global SPLIT_MAP.json (open + nc sources) both flavours build from | `--inherit`, `--local`, `--local-only`, `--lock`, `--min-groups`, `--near-dup`, `--no-benchmark-pins`, `--no-honour-upstream-test`, `--no-near-dup`, `--no-ood`, `--ood-metadata`, `--ood-rules`, `--out`, `--ratios`, `--release`, `--seed`, `--split-map` |
| `marinedata show` | command | Show one source in detail |  |
| `marinedata splitmap` | group | Generate and inspect SPLIT_MAP.json |  |
| `marinedata splitmap generate` | command | Write a fresh, deterministic group-keyed SPLIT_MAP.json | `--in`, `--min-groups`, `--now`, `--out`, `--preseed`, `--ratios`, `--release`, `--seed`, `--stratify` |
| `marinedata splits` | group | split v2 (design §3): pools, OOD holdouts, allocator |  |
| `marinedata splits check` | command | run split v2 and gate on ratio tolerance + group invariant | `--rules`, `--samples` |
| `marinedata tasks` | group | Build task-layer tables from staged trees (local output) |  |
| `marinedata tasks build-points` | command | Benthic points table for a staged source (Seaview) | `--annotations-root`, `--cache`, `--master-sample-bytes`, `--metadata-dir`, `--no-worms`, `--out`, `--source`, `--version`, `--worms-cache` |
| `marinedata taxa` | command | Re-resolve anchored nodes against WoRMS/OBIS and report drift | `--show-unanchored` |
| `marinedata taxonomy` | command | Check the taxonomy gate, export a manifest, or diff two versions | `--release`, `--strict` |
| `marinedata verify` | command | Check declared layouts against real fetched samples | `--limit`, `--strict`, `--unverified-only` |
| `marinedata verify-release` | command | Check a build (local dir or s3://bucket/prefix) against its MANIFEST.tsv | `--deep`, `--manifest`, `--scratch`, `--strict` |
