# Release gates

`marinedata release gate BUILD_DIR --all --require-all` runs every gate below on a build directory
(an `hf_export` output: `data/<config>/<split>-NNNNN-of-NNNNN.parquet`, `README.md`, optional
`croissant.json`) and merges each result into `BUILD_DIR/GATES.json`. `release publish` (P1.8) refuses
on that file. `RELEASE.json` is read from `--release-dir` (default: `BUILD_DIR`); `hf_export` adds a
`configs: {config: {split: rows}}` block to it after a full export, and `release build` adds
`split_ratio.target_pct`. Gates read parquet only; no image bytes come from the bucket.

```
marinedata release gate BUILD --all | --licence --dedup --splits --masks --privacy --load --card --croissant
    [--flavour open|nc] [--release-dir DIR] [--phash-bits 8] [--split-tolerance 5] [--empty-threshold 0.05]
    [--detector MODULE:ATTR] [--privacy-threshold 0.85] [--sample-rows 50] [--require-all]
```

`GATES.json` is a dict keyed by gate name (`passed`, `counts`, `failures` capped at 50, `skipped` reason
when it checked nothing) plus `summary{passed, failed, skipped, required_for_publish[]}`. A skipped gate
never counts as passed. Exit codes: 1 when a selected gate failed, or with `--require-all` when a required
gate is not `passed`; 2 for usage errors. `release gate --masks DIR` (the P1.5 form) still works. The
metadata gate (`--metadata`) lives on the join branch. A gate that raises is reported as `gate_error`.

**Required for publish:** licence, dedup, splits, masks, load, card. privacy and croissant only when they ran.

## licence (required)
Checks: each lock source's `access_class` ships in the flavour (a per-row-licence source is admitted
unless internal-only or unknown); rows of every config with `license`/`licence_class`/`attribution`
columns: licence not null and not stricter than the flavour, class equals the flavour's, attribution
not empty, source is in the lock. Flags: `--flavour` (default: RELEASE.json). Fix: drop or relabel the
source or rows, or build the right flavour.
`metadata/test 92d07e13ade4 row_licence_outside_flavour: license 'CC-BY-NC-4.0' is restricted-nc`

## dedup (required)
Checks: sha256 of the bytes equals `image_sha256`; no exact duplicate across splits, across sources or
inside a split; no 64-bit DCT pHash within `--phash-bits` (default 8) across splits or sources; every
image decodes; `DEDUP_GATE.json` from `release build`, when present, is ok. Near pairs inside one
split and one source (burst frames) are counted only. Writes `BUILD/phash.parquet`
(`image_sha256, config, split, source_id, phash`) for later reuse. An image-free build is `skipped`.
Fix: remove the duplicates from the source selection and rebuild.
`images/test 886a59f87818 near_dup_across_splits: hamming 0 <= 8 to images/train 839186194c6d (src-a vs src-a)`

## splits (required)
Checks: no `split_group` in two splits; each config within `--split-tolerance` (5 points) of its target;
no validation/test image within `--phash-bits` of a train image; some config carries `split_group`.
Target, first match: `registry/split_ratio.yaml` `configs:` entry, the RELEASE.json
`split_ratio.target_pct` (ratios of the frozen split map the build used), the registry `default` (80/10/10
for new configs; the v1.0 code default 70/15/15 is unchanged and recorded per build). Configs without
train, validation and test are only ratio-checked when the registry names them. Fix: regenerate the
split map, or set the target in the registry.
`images/test ratio_off_target: 10.0% of 20 rows, target 25% +/- 5 (release:split_ratio)`

## masks (required)
Checks (P1.5): mask size equals image size, no leftover EXIF orientation, uint8 semantic and uint16
instance masks, ids inside the class map, 255 never a class, instance ids 1..N, empty masks per source
under `--empty-threshold` (5%). A build without mask rows fails. Fix: re-export the mask config.
`data/coral-masks/train-00000-of-00001.parquet[3] 3f2a9c1b4d5e size_mismatch: mask 64x32 != image 64x64`

## privacy (required whenever it ran)
Always on: an embedded image whose EXIF carries a GPSInfo IFD (tag 34853) fails its row, unless its
source is `true` under `privacy.allow_exif_gps` in `registry/gates.yaml`. An image whose EXIF cannot
be read fails as `exif_unreadable`. Only the EXIF GPSInfo IFD is checked, not GPS written to XMP.
Counts rows whose labels name a person (human, diver, snorkeler, operator, ...) from `label`,
`native_label`, `class_map`, `mask_class_map` and `instances`; they never fail. With
`--detector MODULE:ATTR` (an object or factory with `name`, `version`, `detect(bytes) -> [Detection(kind,
score)]`) every embedded image is also scanned and a face or plate scoring at least
`--privacy-threshold` (0.85) fails. `skipped` only when the build holds no image bytes.
Fix: strip the GPS tags before export (for example `exiftool -gps:all=`), or allow the source in
`registry/gates.yaml` (`privacy: {allow_exif_gps: {<source_id>: true}}`) after its licence and privacy
review; exclude or blur a detected face or plate.
`images/train-00000-of-00001.parquet[1] image_sha256=<64 hex> source=src-b exif_gps_present: GPSInfo (tag 34853)`
`images/test 92d07e13ade4 face_detected: score 0.90 >= 0.85 (fake 1)`

## load (required)
Checks: RELEASE.json `configs` and the shards list the same configs and row counts; every shard's
schema equals the card's `dataset_info.features` for the config (the first shard until the card has
one); `datasets.load_dataset(BUILD, config, streaming=True)` loads and the first `--sample-rows` (50)
rows per split decode. Fix: re-export, or correct the card.
`images/train count_mismatch: 16 rows in the shards, RELEASE.json says 99`

## card (required)
Checks: `README.md` parses (`huggingface_hub.DatasetCard`, `datasets` metadata configs); every config
has a `configs:` entry with the build's splits and no config is extra; `dataset_info` split
`num_examples` equal the release rows and `download_size` the shard bytes; `size_categories` matches the
row counts. The card is generated (P1.9); the gate only verifies. Fix: regenerate the card.
`` coral-masks missing_config_section: README.md `configs:` has no entry for it ``

## croissant (required only when it ran)
`croissant.json` must be JSON with `@context`, `@type`, `name` and validate with `mlcroissant`
(`uv sync --extra croissant`). `skipped` when the file is absent or `mlcroissant` is not installed.
`croissant.json unparseable_json: Expecting property name enclosed in double quotes`
