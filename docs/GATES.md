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
split and one source (burst frames) are counted only. Near pairs inside one split between two
sources that declare the same registry `family` (`registry/sources/*.yaml`: `roboflow-coral-universe`,
the four Roboflow Universe forks; `seaview-survey`, the Seaview survey imagery and our labels on it)
are disclosed, not failed: `counts.near_pairs_across_sources_in_family`,
with a capped sample in `counts.near_pairs_across_sources_in_family_sample`; across splits they still
fail. Writes `BUILD/phash.parquet`
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

The ratio is taken over the rows of the **non-pinned** sources. A source the split map pins to its
upstream test manifest (a key of `pins.benchmarks` in `SPLIT_MAP.json`, read from `--release-dir`, else
the build dir) follows that manifest, so its test share is structural: its rows leave the ratio and never
fail it. Rows are matched by the table's `source_id` column (else `source`); the benchmark label is the
`source_id` of its rows. `counts.ratio_pinned_sources` holds the pinned sources' train/validation/test rows
per table, `counts.ratio_excluding_pinned` the ratio the rule judged (`rows` and a percent per split) and
`counts.ratio_pins` the map used (`release_dir`, `build_dir`, `unreadable` or none) and the pinned ids.
Tolerance, targets and finding kinds are unchanged. A finding that left rows out says so:
`images/test ratio_off_target: 5.0% of 20 non-pinned rows (excl. 20 rows of pinned src-p), target 15% +/- 5 (registry:default)`
No map, one that does not parse, or one without `pins.benchmarks` pins nothing: every row counts, as
before. A table without a source column keeps every row (`counts.ratio_unsourced_configs`). A table whose
rows are all pinned (`ratio_all_pinned_configs`), or whose non-pinned rows lack a split, is not
ratio-checked, like a table that never had that split (`ratio_skipped_configs`).

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
review; exclude or blur a detected face or plate. This v1.0 gate cannot record a human review of a
false positive; the v1.1.0 gate G8 (`--v11`) reads one, see "v1.1.0 gates" below.
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
| G6 `splits_v11` | `gates/splits_v11.py` | unsplit / `--allow-unsplit` build (images of split-map `withheld` groups are dropped from the tables, so they never count as unsplit), v1.0 reassignment (other than the groups a benchmark pin moved), straddling group, OOD guards, +/-5 pt ratio |
| G7 `decon` | `gates/decon_gate.py` | a release sha256 is in a hidden-test manifest (unless `--hidden-test-published-exception` exempts an image the v1.0 tree already publishes), an `exclude` or unknown benchmark manifest, a `route-to-our-test` eval image outside `test`/`ood-*`, or a held-out image outside `validation`/`test`/`ood-*`; either input is missing |

G7 reads `registry/benchmarks.yaml`: each `--benchmark` manifest resolves to its entry by the `benchmark_id` column (else the file stem). A `route-to-our-test` eval image in `test`, `test_ood` or `ood-*` counts as `pinned_ok`, and so does a held-out image (the entry's `heldout_val`, else the upstream `val`) in `validation` too (counts per manifest: {eval, heldout} x {pinned_ok, failed}); in any other split it fails, as does any overlap with an `exclude` benchmark, a hidden-test manifest or a manifest with no registry entry (`unknown_benchmark`, fail closed). `GATES.json` `decon.counts.manifests` lists `{benchmark_id, policy, manifest_shas, overlap, pinned_ok, failed}` per manifest and `failing` the first 20 failing images. The split map applies the pins first and a pin outranks an inherited v1.0 split: the moved groups are recorded in the SPLIT_MAP header (`pins.v1_continuity_overridden`: `images`, `groups`, `by_source`, `group_ids`), the v1.0 configs stay frozen (a documented v1.0 -> v1.1 split change), and G6 accepts exactly those groups as the only v1.0 reassignment.

G7 hidden-test published exception (`--hidden-test-published-exception`, default off: any hidden-test overlap fails, as above). With the flag and the frozen v1.0 tree (`--v1-dir`; the `image_sha256` set G4 reads as `v1_reupload`), a hidden-test overlap whose digest is an `image_sha256` in that tree is exempt: the image is public already, so shipping it again publishes nothing new. The exempt overlaps are counted in `counts.hidden_test_already_published` (`hidden_test_already_published_sample` lists the first 20, and the manifest record's `already_published` and `overlap` count them; `failed` does not). Every other hidden-test overlap keeps failing: the failure line reads `hidden_test_new: N of M overlapping image(s) are not in the v1.0 tree`, `counts.hidden_test_new` holds N and the `failing` entries carry `kind: hidden_test_new`. Benchmark manifests (`route-to-our-test`, `exclude`, unknown) are judged exactly as before. The flag without a v1.0 sha256 set (no `--v1-dir`, or a tree with no `image_sha256` under `data/*/*.parquet`) fails the gate. On the release VM add the flag to `GATE_V11_ARGS`; only the `--v11` call receives it.

### G6 and benchmark pins (D-STRADDLE)

A benchmark pin wins over v1.0 split continuity (decision D-STRADDLE): the release's
`SPLIT_MAP.json` moves whole near-duplicate components to the split a benchmark manifest asks for
and records the groups it moved in `pins.v1_continuity_overridden` (`groups`, `images`,
`by_source`, and `group_ids` up to 200 groups). Gate G6 (`splits_v11`) therefore accepts a v1.0
image in a split v1.0 did not give it when

- the map records an override and covers the image's group: it is listed in `group_ids` when the
  record lists them, otherwise only the count is known; and
- the build's split for the group equals the map's assignment for it (the map's `val` is
  `validation` in the tables).

These groups count in `pin_override_applied` (per source `pin_override_applied_<source>`); groups
of a source outside `pins.benchmarks` are component-mates, which a pin moves with its component,
and also count in `pin_override_mates`. A recorded group that did not leave its v1.0 split is a
no-op when the map keeps it there (`pin_override_noop`; with a count-only record this is `groups`
minus the groups seen moving) and passes. The gate fails `override_unchanged` when a pin was not
applied (the map assigns a listed group, or a benchmark-source group, a split the build does not
have) or when more groups moved than the record counts. Every other v1.0 image whose split
changed still fails `reassigned_v1`, with the reason; so does every move when there is no map,
the map is unreadable or it records no override.

Caveat: a model trained on the v1.0 configs must not be evaluated on the v1.1.0 validation/test
splits of the pinned sources (coralscapes, coralvqa, suim, uiis, uiis10k, usis10k): images it saw
in its v1.0 training split may now sit in those splits.

Not wired yet: the `release gate` CLI entry for these, G9 `labelmap`, and the G8 privacy pass scoped
to `images`. The pinned YuNet model (`gates/yunet_model.py`) is baked into the image by the Dockerfile.


## v1.1.0 gates (G1-G9)

`marinedata release gate BUILD_DIR --v11` runs the nine gates below and merges each into
`BUILD_DIR/GATES.json`; `release publish` reads that file (staged as `provenance/GATES.json`).
`--v11` is implied by `--only`, `--rehearsal`, `--hidden-test`, `--benchmark`, `--nc-id`,
`--hidden-test-published-exception`, `--privacy-review`, `--taxonomy-version` and, when no v1.0 gate flag is given, `--v1-dir` (with one,
`--v1-dir` feeds the load, card and masks gates: configs and mask shards of the frozen v1.0 tree
count as present), and by a build with `task_layers/V11_TABLES.json` when no v1.0
gate flag is given; the v1.0 flags and their behaviour are unchanged. `--only GATE` (repeatable)
runs a subset.

| Gate | GATES.json key | What it checks (first sentence of the module docstring) |
|---|---|---|
| G1 | `join` | G1 join coverage: every row of every v1.1.0 table joins to ``metadata`` on ``image_sha256``. |
| G2 | `frozen` | G2 frozen v1.0: the v1.1.0 build never rewrites the v1.0 tree. |
| G3 | `leak` | G3 leak: the open upload set names no nc source id, no nc repo and no private bucket. |
| G4 | `bytes_once` | G4 bytes once: an image's bytes are embedded once, in ``images``, and referenced elsewhere. |
| G5 | `masks` | Masks gate (Phase A, P1.5): refuse a build whose masks do not line up with their images. |
| G6 | `splits_v11` | G6 splits (v1.1.0): the split columns keep the frozen v1.0 assignment and the split policy. The one exception is a benchmark-pin override: a v1.0 group that `SPLIT_MAP.json` `pins.v1_continuity_overridden.group_ids` names may differ (a listed group that did not change fails too); any other reassignment fails `reassigned_v1`. The dHash bands are not recomputed: the v1.0 `dedup` gate covers cross-split near-duplicates. |
| G7 | `decon` | G7 decontamination: no release image is a hackathon hidden-test or benchmark image. |
| G8 | `privacy` | G8 privacy (v1.1.0 gate side): EXIF GPS and faces over the ``images`` config only (every new image lives there once; the v1.0 configs were checked in v1.0 and are frozen). |
| G9 | `labelmap` | G9 labelmap (v1.1.0): every class id the exported label tables carry resolves in the pinned taxonomy (a bare id in any of its namespaces) and in the per-source crosswalk, and every fidelity word is one of the four. |

Exit code: 1 when a gate failed or, outside `--rehearsal`, was skipped (a gate that did not look has
not passed); 2 on a usage error. `--rehearsal` reaches only `splits_v11` (an unsplit-only failure)
and `decon` (missing manifests): they become `skipped`, never `passed`, and `release publish` still
refuses a build with a skipped gate.

The publish verdict also needs the v1.0 gates (`licence`, `dedup`, `splits`, `load`, `card`,
`croissant`): run `marinedata release gate BUILD_DIR --all` for them first, then `--v11` (a later run
overwrites only the entries it ran). The command ends with `publish verdict: OK|BLOCKED`, the check
`release publish` makes.

An additive release's build holds only the new configs, so give the `--all` call the same `--v1-dir DIR`
as the `--v11` call: `card` counts the frozen configs under DIR as present (`configs_from_v1_dir`, no
`extra_config_section`), `load` reads them from DIR and `masks` scans DIR's mask shards when the build has
none (`v1_dir_scanned`). Without it the card gate fails once per frozen config and `masks` fails with
`no mask rows found`. The release VM's `bootstrap.sh` does this from `GATE_V11_ARGS`, and passes only
`--v1-dir` to the `--all` call: `--hidden-test`, `--benchmark`, `--privacy-review`, `--nc-id`,
`--taxonomy-version` and `--rehearsal` would send that call to the `--v11` gate set.

G8 scans the `images` config only (every new image lives there once; the v1.0 configs are frozen).
The face detector is the pinned YuNet model, loaded through `verified_model_path` (sha256 checked,
never downloaded; `MARINEDATA_YUNET_MODEL` overrides the path); `--detector MODULE:ATTR` replaces
it. A model that cannot be verified, an unreadable image or a detector error FAILS the gate, and a
detection lists the full `image_sha256`.

A detector scores some non-faces over the threshold (a close-up of bleached coral scored 0.86), so G8
reads a human review list, `registry/privacy/review.yaml` (`--privacy-review FILE` replaces it):
`schema_version: 1` and `entries`, each with `image_sha256` (64 hex, unique), `verdict` (`cleared` or
`withhold`), `source_id`, `detector`, `score`, `reason`, `reviewer` and `reviewed_at` (YYYY-MM-DD).
- `cleared`: a detection on that image is not a failure. It is counted under `cleared_reviewed` and
  listed (sha, reviewer, date) in `cleared_reviewed_entries`. EXIF GPS on the image still fails.
- `withhold`: the build drops the image from every table (`V11_TABLES.json` `withheld_images`, the
  privacy ones also in `withheld_images_privacy`); one still in the build fails as
  `withheld_image_present`. Images of the groups the split map withholds are dropped the same way and
  counted in `withheld_images`; `withheld_groups` and `withheld_by_source` count those groups, and G6
  reads them as dropped rows, not `unsplit_images`.
- A detection no entry covers fails as above. A list that does not validate (unknown verdict, bad or
  repeated sha256, missing field) or an explicit `--privacy-review` file that is missing FAILS the gate.

G9 reads the `labelmap` section of `registry/gates.yaml`: `taxonomy_version` (null = the current
registry taxonomy) and `fidelities` (exact, coarsened, approximate, unmappable).

G9 resolves node ids the way the registry writes them. Taxonomy node keys are namespaced
(`rs-taxa-v1:A137039`, `rs-benthic-v1:CCA`, `rs-fauna-v1:...`) while crosswalk edges and table rows
carry bare ids (`A137039`). A bare id resolves when `<namespace>:<id>` is a node key of the pinned
taxonomy; the namespaces are read off the node keys, never listed in the gate. An already-qualified
id resolves as itself. An id no namespace has fails as `unknown_node`. A bare id that resolves in
more than one namespace does not fail; the gate counts the distinct ones under
`counts.ambiguous_bare_ids`. A `label-map` row is compared with its release edge as written (bare
ids on both sides), and a label table row with a node must name a `(source_id, label_native)` that a
mapped `label-map` row carries (`unresolved_label`); the export takes a source's crosswalk from its
registry loader, else from the table builder that wrote its rows.
