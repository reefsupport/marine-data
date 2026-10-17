# Release runbook: the command list (per flavour)

Group near-duplicates first, then generate ONE global split map from those groups, then build each flavour from it (`open` ->
`reefsupport/marine-data`). Where several flavours are built, an image or
near-dup group must have the same split in every flavour. Dry runs always pass `--local-only` (no
pinned-tree fetch). Nothing here uploads except step 9 with `--go`.

1. `marinedata dedup run --staged LABEL=DIR [...] --out dedup/` (once per release)
   Hashes and groups the staged corpus; writes `dedup/groups.parquet` (`sha256`, `split_group_id`,
   `group_upstream_splits`). Near-duplicates never straddle splits, so this runs BEFORE the split map.
2. `marinedata release split-map --release <id> --split-map SPLIT_MAP.json --near-dup dedup/groups.parquet
   --local-only [--local=ID=PATH ...]` (once per release; refuses to overwrite)
   Enumerates every open and restricted-licence source (never-released sources excluded), merges shared images
   and the dedup near-dup groups into one component, honours an upstream test set, pins the benchmark manifests' eval images to test and their second held-out split to val (`pins` header; a pin wins over a v1.0 assignment: `pins.v1_continuity_overridden` records every moved group, the v1.0 configs stay frozen, and the card states the count as a documented v1.0 -> v1.1 split change), and writes the frozen
   map. It FAILS without a near-dup input; `--no-near-dup` is the explicit opt-out (recorded in the map
   header). Pass the `--local` trees of BOTH flavours. Every source with >= 3 groups gets at least one
   group in each of train/val/test (val is taken from train, never from the upstream test). A source
   with no registry split rule takes its group from the `metadata_norm` chain at release time; a source
   with >= 100 rows in < 10 groups logs a warning and groups per image.
   **v1.1.0 and later: inherit the previous map (D1, D2).** The v1.1.0 map embeds the v1.0 `SPLIT_MAP.json` verbatim
   and allocates only the new groups around it. The VM run (exact flags):

   ```
   marinedata release split-map --release v1.1.0 \
     --lock open.lock.yaml --lock nc.lock.yaml \
     --inherit registry/SPLIT_MAP.json --seed <N> --out SPLIT_MAP.json \
     (--near-dup dedup/groups.parquet | --no-near-dup) \
     [--ood-rules splits/v2.yaml | --no-ood] [--ood-metadata ood-metadata.parquet] \
     [--local ID=PATH ...] [--local-only]
   ```

   - `--lock` (repeatable, one per flavour) restricts the enumeration to exactly the sources the locks name; no
     other source is resolved or fetched. `--inherit` is the v1.0 map. `--seed` and `--out` are as above; `--out`
     refuses an existing file. `--ratios` must equal the prior map's (default `70/15/15`), and the prior map must be
     group-keyed and stratified by source, or the command fails.
   - An image whose digest is in a v1.0 group keeps that group's split, whatever its near-dup group says. A new
     group in the same component (shared digest, dHash within 4 bits, or a `dedup run` group) takes that split.
     If the v1.0 map had split a component apart, the new groups of that component are WITHHELD (listed under
     `withheld` in the map, absent from `assignments`) and the v1.0 groups are untouched.
   - A frozen build (`release build --v11`) treats every group under `withheld` as known: it is never allocated or
     appended, and its images are dropped from the metadata spine and every v1.1 table, so nothing of them reaches
     export, croissant, the configs or the stats. `V11_TABLES.json` records `withheld_images`, `withheld_groups` and
     `withheld_by_source` (`{source: {images, groups}}`); `unsplit_images` stays 0. A group in neither `assignments` nor
     `withheld` still aborts the build with `is frozen: group(s) not in the map`: regenerate the map to include it.
     The card generator does not read `V11_TABLES.json`, so the hand-finished card states the withheld counts.
   - A new image 5 to 8 dHash bits from a v1.0 train image is never-eval (forced into train); one 5 to 8 bits from a v1.0
     val/test image is never-train (forced into that image's split). The band is per image since SPLIT_MAP schema 2: a
     component with hit images on both sides keeps the side with more hit images (a tie keeps train), and every hit image
     on the other side, or hitting both sides, is listed under `band_excluded` (`{sha256: [distance, v1.0 sha256, v1.0
     split]}`). Its group keeps its split and the group's other images ship. `release build` drops the listed images from
     every config (RELEASE.json `prior_band_excluded {count, sha256}`), and so do `hf_export` and the v1.1.0 tables
     (`V11_TABLES.json` `band_excluded_images`, `band_excluded_by_source`). A schema-1 map still loads and has no
     `band_excluded`.
   - `release split-map` also unions the gates' own near-duplicate rule: images within 8 bits of the DCT pHash-64
     (`gates/phash.py`, the dedup and splits gates' radius) form one component under the dHash union's chain guard, so
     `near_dup_across_splits` and `near_dup_leak` cannot fail across splits.
   - OOD: the holdouts in `splits/v2.yaml` (default; `--ood-rules` overrides, `--no-ood` opts out) route matching new
     groups, and their whole component, to the extra split `test_ood`, outside the 70/15/15 quota. Rows that need
     per-image fields (geo realm/province, depth, platform, capture time) read them from `--ood-metadata`, a Parquet
     with `image_sha256` plus any of `meow_realm`, `meow_province`, `depth_m`, `platform`, `capture_datetime`; the
     yaml's per-source rules (e.g. DeepFish) need none. A holdout under the yaml's `min_images` / `min_groups` is recorded
     as not constructible (empty, with a note under `ood.holdouts`); one over a ceiling (`max_holdout_share`,
     `max_total_holdout_share`) fails the command. A v1.0 group, an inherited component or a band-forced image is
     never moved to `test_ood`.
   - The map header records `inherits` (prior release, seed, group count, sha256 of the prior file), `ood`
     (rules sha256 and the per-holdout outcome) and `ood_rules` (group to rule names).

3. `marinedata release build --flavour {open|nc} --release <id> --split-map SPLIT_MAP.json
   --out rel --local-only [--local=ID=PATH ...]` (run for each flavour, same `SPLIT_MAP.json`)
   Writes `rel/releases/<id>/<flavour>/` (`tasks/*.tsv`, `RELEASE.json`). With `--lock FILE` only the
   sources the lock names are fetched, and a locked source that cannot be resolved (not admitted, no pinned
   staged tree and no `--local` tree) fails the build, naming it, before the first fetch. A pinned source the registry leaves out
   (`release_skip_reason`, e.g. `deferred to v1.2: ...`) is never locked, demanded by the lock check or
   fetched. A source that cannot be
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
8. `marinedata release card` / upload stages per `docs/formats.md`; check that every config in the card has
   its `data_files` and that `metadata` is listed.
9. `marinedata release publish --release-id <id> --flavour {open|nc} --dir hf-<flavour>
   --release-dir rel/releases/<id>/<flavour>` is a DRY RUN: it prints the target repo, the configs with
   their split rows, the file count and bytes, the parts (below), the card, the lock sha and the tag
   `v<id>`, and refuses unless `GATES.json` shows every required gate passed and `RELEASE.json` matches
   the flavour, repo and lock. `--v1-dir <v1.0 checkout>` adds the delta against the v1.0 tree (the paths
   this publish adds and replaces) and the plan reports G2 (frozen v1.0) and G3 (leak) as PASS or FAIL.
   `--go` (with `HF_TOKEN` set) uploads the parts and tags the card commit; it refuses an existing tag and
   the frozen `v1.0` without `--allow-frozen`. It never creates a repo or changes visibility, and for `nc`
   it refuses unless the repo is already private. What is uploaded is the list under "What `release
   publish` uploads". The build VM in `deploy/release-vm/README.md` runs steps 3 and 6, the card part of
   8, the gates and the dry run of this step; steps 1-2 are replaced there by the frozen
   `registry/SPLIT_MAP.json`, and 4, 5 and 7 are not VM stages.

## Gates: `--all`, then `--v11`, on the same release dir

After the export run both gate sets against the same build and release dir (see [GATES.md](GATES.md)):

```bash
marinedata release gate "$B" --all --require-all --flavour "$F" --release-dir "$R"   # v1.0 gates
marinedata release gate "$B" --v11 --flavour "$F" --release-dir "$R" --v1-dir <v1.0 release dir>  # G1..G9
```

`--v11` does not chain the v1.0 gates: it merges G1..G9 into `GATES.json` and keeps the entries `--all`
wrote, so `release publish` stays BLOCKED until both have run (`--v11` prints `publish verdict:
OK|BLOCKED`). G6 does not recompute the dHash bands: the v1.0 dedup gate covers cross-split
near-duplicates. The privacy gate (G8) runs real YuNet on the pinned model the image bakes at
`/opt/models/face_detection_yunet_2023mar.onnx`; pass that path as `MARINEDATA_YUNET_MODEL` (the VM
wrapper does). The gates verify it by sha256 and never download it.

## What `release publish` uploads

`--go` sends `UPLOAD_PATTERNS` (`src/marinedata/release_publish.py`) and nothing else; a test keeps this
list equal to the code:

<!-- upload-patterns:start -->
- `data/**/*.parquet`
- `README.md`
- `croissant.json`
- `provenance/*`
- `LICENSE`
- `NOTICE`
- `CITATION.cff`
- `assets/*`
- `docs/**`
<!-- upload-patterns:end -->

`provenance/` holds `RELEASE.json` (its public-safe projection), `GATES.json` and the lock, copied there at
upload time: they are uploaded. `SPLIT_MAP.json`, `DEDUP_GATE.json`, `tasks/`, `PUBLISH_STATE.json` and the
build's other bookkeeping files stay on the VM.

The parts, in this order (`release publish ... --list-parts` prints them for a build): one `data:<config>`
per `data/<config>/`, then `provenance` (croissant.json, `provenance/*`, LICENSE, NOTICE, CITATION.cff,
`assets/*`, `docs/**`: everything the card links to), then `card` (README.md alone, last), then `tag`.
`--part PART` runs one; without it all run in that order. Each upload's commit sha is recorded in
`PUBLISH_STATE.json` next to `RELEASE.json`. `card` refuses until every other part is up, and `--part tag`
creates `v<id>` on the sha of the card commit, never on the head of the branch; it refuses if anything was
uploaded after the card. The VM loop (`marinedata-run` on the VM; a failed part stops it):

```bash
for part in data:<config>... provenance card tag; do   # or $(marinedata release publish ... --list-parts)
  marinedata release publish --release-id v1.1.0 --flavour open --dir /data/export/v1.1.0/open \
    --release-dir /data/releases/v1.1.0/open --lock "/app/$LOCK_FILE" --go --part "$part" || break
done
```

### Rollback

Nothing is force-pushed. To take a bad release off the Hub, delete the tag and restore the v1.0 tree in one
commit (the `v1.0` tag keeps pointing at the untouched v1.0 commit):

```python
from huggingface_hub import HfApi

api = HfApi()  # HF_TOKEN in the environment
api.delete_tag("reefsupport/marine-data", tag="v1.1.0", repo_type="dataset")
api.upload_folder(
    repo_id="reefsupport/marine-data",
    repo_type="dataset",
    folder_path="<checkout of the v1.0 tag>",
    delete_patterns=["*"],
    commit_message="rollback: restore the v1.0 tree",
)
```

## Privacy policy (WP-R9)

After `hf_export` (first pass) run `marinedata privacy-scan` on the shards, then re-run
`python -m marinedata.hf_export ... --privacy <privacy.parquet>` and the metadata refresh with the
same `--privacy`. A face score >= 0.85 excludes the image from every config and lists it in
`RELEASE.json` (`privacy_exclusions`: image id, source, score, detector + version); 0.60-0.85 keeps
it with `privacy_flag` = `possible_face` and `face_score` in `metadata`. The thresholds are the two
constants in `src/marinedata/privacy/policy.py`; the dataset card prints them. Without `--privacy`
nothing is excluded or flagged.

## Dataset card and Croissant (offline, read the build folder only)

After `release build` and the HF export, write the two metadata files next to the shards. Neither
command touches the network or the Hub.

```bash
B=<build-dir>                       # holds data/<config>/*.parquet
R=$B/releases/<release-id>/<open|nc> # RELEASE.json, GATES.json, release.lock.yaml, CHANGELOG.md
uv run --frozen marinedata release card "$B" --release-dir "$R" --out "$B/README.md"
uv run --frozen marinedata release croissant "$B" --release-dir "$R" --out "$B/croissant.json"
```

- `release card` renders `docs/card-template.md.tmpl`: the front-matter (`configs`, `dataset_info`,
  licence, tags, size band) and the config/field/licence tables come from the shards and the registry;
  a column without a one-line description in `card_fields.FIELD_DOCS` prints an en dash and a warning.
  A source in `registry/sources` that lacks a field shows `–` instead of failing.
- `release croissant` writes Croissant 1.0 JSON-LD (sha256 and size of every shard, one recordSet per
  config) and validates it with `mlcroissant` when the extra `croissant` is installed; otherwise it says
  `not-validated` and still writes the file.
- The export writes the shards and the summary only. The VM's card stage then renders the card and writes
  `LICENSE`, `NOTICE` and `CITATION.cff` into the export folder, with `python -m marinedata.hf_card
  --release-dir "$R" --summary <export summary> --out "$B" --flavour <f> --citation /app/CITATION.cff`.
  The input is the repository's `CITATION.cff` inside the image, the same file the croissant stage reads;
  the export's `CITATION.cff` is that file with the release's version and date. The v1.1 card links
  `provenance/GATES.json` and `release.lock.yaml`: the `provenance` part ships both, with LICENSE, NOTICE
  and CITATION.cff, before the card.
- Re-run the P1.6 `card` gate on the folder afterwards; it re-reads the shards and fails if the front-matter
  counts, byte sizes or size band no longer match them.
- Regenerate the test goldens with `UPDATE_GOLDEN=1 uv run --frozen pytest tests/test_release_card.py`.
