# Task-label producers

Four one-module-per-source producers under `src/marinedata/task_layers/sources/`, each
writing `data/_tasklabels/<source_id>/<task>.parquet` with a fixed set of columns
(`sha256`, `source_id`, `label_origin`, plus task-specific payload). This doc records
what each one does and its real 2026-09-25 execution result — including two genuine
blockers, documented rather than worked around.

## coralseg.py — Coralseg (UCSD) restage + semseg

Restages the legacy `benthic_datasets/mask_labels/Coralseg/<split>/{Image,Mask}/<stem>`
prefix into the staged-tree layout, reusing `StagedWriter`/`_Uploader`/`checksums`
rather than a new upload path. The task-label interface authorizes opening Coralseg's images (only) so masks
can be keyed by the sha256 the restage itself computes. Mask class_counts come from the
red-channel histogram (0/1 only), per the registry's already-verified
`coralseg-r-channel` loader note.

**Real result:** blocked before any upload. `restage()`'s `DiskGuard.check()` raised
`DiskFloorError` (free 33.8 GiB < the brief's explicit 34 GiB floor) on the very first
pair. No S3 writes happened; the registry entry is **not** flipped to `staged-tree`.
Resume: re-run `coralseg.py main()` once free disk is safely above the floor — no local
scratch was left behind, since each closed batch is deleted immediately after S3
verification.

## seaview.py — SEAVIEW's excluded pickle: scan, then convert-or-stay-images-only

Implements the path for `labelled_data/labelled_segmentation_data.pickle`
(6,956,023,722 B, excluded from the registry's size count as an unsafe format): a
static `pickletools.genops` opcode scan (`scan_globals`/`check_allowlist`, disassembly
only, never `pickle.load`) against an allowlist of `pandas.*`, `numpy.*`, plain
`builtins`/`__builtin__` containers, `datetime`, `collections.OrderedDict`, and
`_codecs.encode` (CPython's own string-pickling helper, added after real-pandas testing
below). Only if the scan is clean does `convert_in_container()` run a restricted
`pickle.Unpickler` inside `docker run --rm --network none --read-only`, with only the
pickle (read-only) and one output dir mounted.

**Real result:** a full paginated `list_objects_v2` of
`benthic_datasets/point_labels/SEAVIEW/labelled_data/` in both `<open-bucket>` and
`<private-bucket>` found no such object in either bucket — it no longer exists.
SEAVIEW stays images-only because there is nothing left to scan or convert, not because
a scan found a disallowed global. The scan/convert code path itself is real and
unit-tested against both a genuine pandas pickle and a malicious
`os.system`-`__reduce__` pickle, for if a copy resurfaces.

**Allowlist finding:** testing `scan_globals` against a real pandas 3.0.6 DataFrame
pickle surfaced two previously-missing, benign globals — `__builtin__` (the Python-2
pickle-protocol-2 spelling of `builtins`, present in any protocol-2 pickle with e.g. a
`slice`) and `_codecs.encode` (CPython's own bytes/str-literal pickling helper, present
in virtually any protocol>=2 pickle) — both now allowlisted. Separately, pandas 3.x
defaults `future.infer_string=True`, which backs even a plain int64-column DataFrame's
column-label Index with PyArrow strings and pulls in `pyarrow.lib` globals; this is
real but **not** added to the allowlist (outside the literal `pandas.*, numpy.*`
scope) — the unit test pins `future.infer_string=False` to test the allowlist as
specified. If SEAVIEW's pickle ever resurfaces and was written by a PyArrow-string-backed
pandas, the scan will (correctly, as the allowlist is written) reject it; that would need a
charter decision, not a code change here.

## rs_labelled.py — Labelbox ndjson rasteriser

Parses the Labelbox `export-result.ndjson` export and keys each row by
`images_from`'s (`reef-support-benthic-own`, `reef-support-seaview-labels`)
already-staged `metadata.parquet` sha256 — no image bytes of rs_labelled's own are ever
read. `rasterize_record()` fills inline `polygon` objects at native resolution via PIL;
an `ImageSegmentationMask` object (Labelbox-hosted mask, never fetched per the
Do-Not-list) makes the whole record unusable rather than guessed at.

**Real result:** 0 real rows. Every one of 2,578 annotation objects sampled across 246
records in `SEAFLOWER_BOLIVAR/export-result.ndjson` is `annotation_kind:
"ImageSegmentationMask"` with only a `mask.url` — zero inline `polygon` geometry. The
brief's "rasterise the polygons" premise doesn't hold for the live export. The
registry's `rs-labelled-masks.split_group.template` was also fixed from a copy-pasted
`"seaview/{group}"` (the earlier placeholder, wrong source) to `"rs_labelled/{group}"`.
Resume: either a future Labelbox export with polygon (not mask) annotations, or a revision of the task-label rules
to permit fetching Labelbox-hosted masks under a different rule.

## coralvqa.py — CoralVQA jsonl -> vqa.parquet

Fetches `CoralVQA_train.jsonl`/`CoralVQA_test.jsonl` anonymously (no token, no login,
`fetch_with_retries` bounded at 3 attempts / 5 min apart) from HF
`CoralReefData/CoralVQA`, and parses both real schemas: train's `conversations`
human/gpt turn pair (question tagged `[qtype]`, e.g. `[vqa]`), and test's flat
`{question_id, image, text, category}` (no ground-truth answer — `answer` is `None`,
not fabricated). `sha256` is left `null`: CoralVQA's real images live only in a single
26.7GB `CoralVQA_Image.zip` that no registry source stages, so hashing them is out of
this brief's scope.

**Real result:** both files fetched successfully (test succeeded on the first real
attempt this pass). `data/_tasklabels/coralvqa/vqa.parquet` written with 254,867 rows
(226,883 train + 27,984 test).

## Tests

`tests/test_task_layers_sources.py` — synthetic fixtures only, no network/S3/Docker:
opcode allowlist accept (real pandas pickle, pinned off the pyarrow-string backend) and
reject (`os.system` `__reduce__`), datetime/OrderedDict/`_codecs` allowlist, ndjson
rasteriser (mask-url unusable, polygon fill, empty-objects, sha keying, unmatched-sha
drop), CoralVQA dual-schema parsing + parquet schema, Coralseg mask class-counts +
parquet schema.
