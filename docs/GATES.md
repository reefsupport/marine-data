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
never counts as passed. Exit codes: 1 when a selected gate failed, or with `--require-all` when any of the
eight gates is not `passed` (absent, skipped or failed: privacy, card and croissant included;
`release publish` applies the same rule); 2 for usage errors. `release gate --masks DIR` (the P1.5 form) still works. The
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
`split_ratio.target_pct` (ratios of the frozen split map the build used), the registry `default` (70/15/15
for new configs, as in `registry/splits/v2.yaml`; the four v1.0 configs sit under `configs:` at their measured
shares, because upstream splits dominate them and no nominal ratio describes v1.0). Configs without
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
one); every card `configs:` entry is resolved against the build (a pattern that matches nothing there
falls back to `--v1-dir`) and loaded with `datasets.load_dataset("parquet", data_files=..., streaming=True)`,
and the first `--sample-rows` (50) rows per split decode. `datasets` infers the builder of a folder from
the card's FIRST config, so that config must resolve in the build or `--v1-dir` (`first_config_unresolved`);
a pattern that matches in neither is `config_files_missing`. Without `--v1-dir` a frozen v1.0 config absent
from the build (coral-masks, scene-masks, instance-masks, fish-boxes) is counted `skipped_frozen`; every
v1.1 config must still load. Fix: re-export, correct the card, or pass `--v1-dir`.
`images/train count_mismatch: 16 rows in the shards, RELEASE.json says 99`
`coral-masks/train config_files_missing: pattern 'data/coral-masks/train-*.parquet' matches no file in the build or --v1-dir`

## card (required)
Checks: `README.md` parses (`huggingface_hub.DatasetCard`, `datasets` metadata configs); every config
has a `configs:` entry with the build's splits and no config is extra (one under `--v1-dir` is not); `dataset_info` split
`num_examples` equal the release rows and `download_size` the shard bytes; `size_categories` matches the
row counts. The card is generated (P1.9); the gate only verifies. Fix: regenerate the card.
`` coral-masks missing_config_section: README.md `configs:` has no entry for it ``

## croissant (required only when it ran)
`croissant.json` must be JSON with `@context`, `@type`, `name` and validate with `mlcroissant`
(`uv sync --extra croissant`). `skipped` when the file is absent or `mlcroissant` is not installed.
`croissant.json unparseable_json: Expecting property name enclosed in double quotes`

## Metadata gate and build gates in one command (PA-P2.4 + P1.6)

`marinedata release gate --metadata DIR [--gates FILE]` runs the metadata gate (coverage floors and licence checks over the
resolver outputs `DIR/<source>/<version>/coverage.json`) and merges it into `--gates` (default `BUILD_DIR/GATES.json`, else
`./GATES.json`). With `BUILD_DIR --all --require-all` it runs next to the build gates; once it ran it is required for publish
like any other extra gate. `GATES.json` is one dict keyed by gate name plus `summary`.

## Dedup gate and EXIF-transposed rows

`image_sha256` is the identity of the upstream bytes. A row with `exif_transposed` true stores bytes re-encoded after the EXIF rotation, so the dedup gate does not compare their sha256 with `image_sha256`; it requires that the bytes decode and that their width/height equal the row's (`size_mismatch`). Rows with `exif_transposed` false keep the `sha256_mismatch` check.

## v1.1.0 gates (library level)

`publish_verdict` also requires `V11_REQUIRED` for a v1.1.0 release (`task_layers/V11_TABLES.json` or
`data/images` in the release directory, or any of these in GATES.json), next to `masks`, `croissant`
and `privacy`. A gate that is absent or only `skipped` blocks `release publish`; a `--rehearsal` run
can only skip, never pass.

| Gate | Module | Fails when |
|---|---|---|
| G1 `join` | `gates/join.py` | a table row has no `metadata` row; an open row joins a non-open image |
| G2 `frozen` | `gates/frozen.py` | a v1.0 file other than README.md differs by one byte; a v1.0 `configs:` entry changes |
| G3 `leak` | `gates/leak.py` | an nc id, the nc repo name or a private bucket appears in the open set (never echoed) |
| G4 `bytes_once` | `gates/bytes_once.py` | image bytes twice, a v1.0 image re-uploaded, a bytes column outside `images` |
| G5 `masks` | `gates/masks_palette.py` | a `semseg` mask is not mode P with its declared palette |
| G6 `splits_v11` | `gates/splits_v11.py` | unsplit / `--allow-unsplit` build, v1.0 reassignment, straddling group, OOD guards, +/-5 pt ratio |
| G7 `decon` | `gates/decon_gate.py` | a release sha256 is in a hidden-test or benchmark manifest; either input is missing |

Not wired yet: the `release gate` CLI entry for these, G9 `labelmap`, and the G8 privacy pass scoped
to `images`. The pinned YuNet model (`gates/yunet_model.py`) is baked into the image by the Dockerfile.


## v1.1.0 gates (G1-G9)

`marinedata release gate BUILD_DIR --v11` runs the nine gates below and merges each into
`BUILD_DIR/GATES.json`; `release publish` reads that file (staged as `provenance/GATES.json`).
`--v11` is implied by `--only`, `--rehearsal`, `--hidden-test`, `--benchmark`, `--nc-id`,
`--taxonomy-version` and, when no v1.0 gate flag is given, `--v1-dir` (with one, `--v1-dir` feeds the
load, card and masks gates: configs and mask shards of the frozen v1.0 tree count as present), and by a build with `task_layers/V11_TABLES.json` when no v1.0
gate flag is given; the v1.0 flags and their behaviour are unchanged. `--only GATE` (repeatable)
runs a subset.

| Gate | GATES.json key | What it checks (first sentence of the module docstring) |
|---|---|---|
| G1 | `join` | G1 join coverage: every row of every v1.1.0 table joins to ``metadata`` on ``image_sha256``. |
| G2 | `frozen` | G2 frozen v1.0: the v1.1.0 build never rewrites the v1.0 tree. |
| G3 | `leak` | G3 leak: the open upload set names no nc source id, no nc repo and no private bucket. |
| G4 | `bytes_once` | G4 bytes once: an image's bytes are embedded once, in ``images``, and referenced elsewhere. |
| G5 | `masks` | Masks gate (Phase A, P1.5): refuse a build whose masks do not line up with their images. |
| G6 | `splits_v11` | G6 splits (v1.1.0): the split columns keep the frozen v1.0 assignment and the split policy. The dHash bands are not recomputed: the v1.0 `dedup` gate covers cross-split near-duplicates. |
| G7 | `decon` | G7 decontamination: no release image is a hackathon hidden-test or benchmark image. |
| G8 | `privacy` | G8 privacy (v1.1.0 gate side): EXIF GPS and faces over the ``images`` config only (every new image lives there once; the v1.0 configs were checked in v1.0 and are frozen). |
| G9 | `labelmap` | G9 labelmap (v1.1.0): every class id the exported label tables carry resolves in the pinned taxonomy and in the per-source crosswalk, and every fidelity word is one of the four. |

Exit code: 1 when a gate failed or, outside `--rehearsal`, was skipped (a gate that did not look has
not passed); 2 on a usage error. `--rehearsal` reaches only `splits_v11` (an unsplit-only failure)
and `decon` (missing manifests): they become `skipped`, never `passed`, and `release publish` still
refuses a build with a skipped gate.

The publish verdict also needs the v1.0 gates (`licence`, `dedup`, `splits`, `load`, `card`,
`croissant`): run `marinedata release gate BUILD_DIR --all` for them first, then `--v11` (a later run
overwrites only the entries it ran). The command ends with `publish verdict: OK|BLOCKED`, the check
`release publish` makes.

G8 scans the `images` config only (every new image lives there once; the v1.0 configs are frozen).
The face detector is the pinned YuNet model, loaded through `verified_model_path` (sha256 checked,
never downloaded; `MARINEDATA_YUNET_MODEL` overrides the path); `--detector MODULE:ATTR` replaces
it. A model that cannot be verified, an unreadable image or a detector error FAILS the gate, and a
detection lists the full `image_sha256`.

G9 reads the `labelmap` section of `registry/gates.yaml`: `taxonomy_version` (null = the current
registry taxonomy) and `fidelities` (exact, coarsened, approximate, unmappable).
