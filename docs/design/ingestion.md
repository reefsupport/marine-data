# Generic source ingestion (WP-6)

`marinedata ingest-source <adapter> <spec.yaml>` takes any open dataset to
`s3://rs-storage-open/sources/<id>/<version>/`, verified, with bounded local temp.
A new catalog source is a spec yaml, not code. Recipe: `docs/ingest-howto.md`.

## Protocol

An adapter knows one *access pattern*, never one dataset (`src/marinedata/adapters/`):

| adapter | reaches | version pin |
|---|---|---|
| `hf` | open HF dataset repo (parquet shards or loose files) | `rev-<commit sha12>` |
| `http` / `zenodo` | Zenodo record API, or plain URLs | record `metadata.version` / spec `version` |
| `bucket` | anonymous S3/GCS `list-type=2` listing | `list-<listing digest12>` |
| `github` | release assets or a raw tree | tag / `git-<commit sha12>` |

`SourceAdapter` Protocol: `resolve_version() -> enumerate() -> fetch() -> decode()`.

* `resolve_version` pins upstream **before** listing: HF/GitHub refs become commit
  shas, so a re-run reads the same bytes. It also runs the gate check (below).
* `enumerate` is sorted by key and filtered by `include`/`exclude` globs. Loose files
  matching `label_patterns` are kept too.
* `fetch`: tars and single images stream through a sha256+md5 `HashingReader`, with no
  temp copy. Zip and parquet files need random access, so they are spooled to
  `work/tmp/fetch/` and deleted after decoding. Each upstream-declared digest is
  checked (HF LFS oid, Zenodo md5, GitHub `sha256:` digest, S3 single-part ETag), and
  so is the size.
* `decode` dispatches on format: image, tar/tgz (WebDataset grouping by key: a `.json`
  sidecar supplies schema fields and labels; `.cls`/`.txt` become labels), zip
  (imagefolder: a parent directory that is not a split name is the label), or parquet
  (64-row batches; HF `ClassLabel` names from schema metadata). One image in memory.

## Canonical sample schema

`src/marinedata/sample_schema.py` is the single source. `SampleRow` is a frozen
dataclass; its field order is the column order of `metadata.parquet`, and
`arrow_schema()` carries `marinedata.sample_schema=1`.

* Required fields: identity (`sample_id = <source_id>/<stem>`, `stem`, `image_path`,
  `image_sha256`, `image_bytes`, `image_format`), licence/attribution per row (D-C;
  a missing licence is `NOASSERTION`, never guessed), and `fetch_date`.
* Nullable fields: provenance (`upstream_id/url/digest`, `lineage_root_digest`),
  capture (`capture_datetime` in UTC, `lat`, `lon`, `gps_precision_m`, `depth_m`,
  `depth_source`, `platform`, `camera`), ecology (`meow_realm`, `depth_zone`,
  `habitat`), `split_hint`, and `label_refs`.
* **Null means "not stated upstream"**, never "zero" or "unknown-but-guessed". A naive
  datetime becomes null unless the spec sets `naive_datetime_is_utc`. `depth_zone` is
  derived from `depth_m` only.
* `validate_row`/`validate_table`: types, enums, ranges, unique `sample_id`.

## Staged layout (D-D)

```
sources/<id>/<version>/
  images/<stem>.<ext>                     # <= threshold images (default 200k)
  images/shard-NNNNN.tar + index.parquet  # > threshold: ~1 GB WebDataset shards
  labels/files/...  labels/image_labels.parquet
  metadata.parquet  INGEST.json  LICENSE  CHECKSUMS.sha256   # CHECKSUMS last
```

The layout is chosen **before** writing. It comes from `expected_images`, or from the
image count of a loose-file listing, or from `max_images`. An `objects` run that
crosses the threshold raises `LayoutError`; it never silently produces 200k+ objects.
Shards are deterministic (USTAR, mtime 0, uid/gid 0); `index.parquet` (`shard, member,
stem, sha256, size, offset`) allows range reads. Stems: `[A-Za-z0-9_-]`, ≤ 90 chars.

## State machine (one run)

```
PIN -> LIST -> [per item: FETCH -> DECODE -> WRITE -> (temp >= cap/2 ? FLUSH)]
    -> FINALIZE (index, labels, metadata.parquet, INGEST.json, LICENSE)
    -> FLUSH -> CHECKSUMS.sha256 (LAST) -> VERIFY every key -> registry stub
```

FLUSH uploads each closed file, logs it in `work/uploaded.json`, deletes the local
`images/`/`labels/` copy after a verified upload (D-F).

## Resume semantics

Staging is deterministic, so resuming means **re-running the same command**:
* Files already on S3 with equal size and (ETag or `x-amz-meta-sha256`) are skipped.
* A partial multipart upload continues from `work/checkpoints/`. `list_parts` is the
  authority: a part is reused only if its ETag equals the local part md5, so a changed
  file never splices old parts.
* A prefix without `CHECKSUMS.sha256` is an incomplete version. Its presence is the
  "version complete" marker. `root_digest = sha256(CHECKSUMS.sha256)` goes into
  INGEST.json and the registry stub.
* Resume re-downloads upstream (there is no local cache by design: temp is bounded).

## Uploader (`s3_upload.py`)

* 64 MiB multipart parts, one part in memory. The composite ETag
  (`md5(part md5s)-N`) is precomputed locally, so verification is a HEAD.
* `DiskGuard`: `df` free ≥ 40 GiB before each batch, else `DiskFloorError`. Temp is
  capped at 6 GB (D-L), measured by a real walk at each flush -> `peak_temp_bytes`.
* Client from rclone.conf `[rs-hel1]` via configparser; credentials never logged.

## Access policy (D-E)

The HTTP layer is anonymous-only. It never reads `HF_TOKEN`/`GITHUB_TOKEN` and refuses
`Authorization`/`Cookie` headers. A gated, private, or restricted source, or an HTTP
401/403, raises `AccessRefused`. The CLI then exits with code 3 and prints
`NEEDS-YOHAN<TAB><url><TAB><needs>` for the "needs Yohan" list.

## Failure modes

| failure | behaviour |
|---|---|
| upstream digest/size mismatch | `ValueError`, run stops; nothing half-staged is marked complete |
| 429/5xx | 4 retries with backoff, then `RuntimeError` |
| gated/login/terms | `AccessRefused` -> NEEDS-YOHAN, exit 3 |
| disk below the floor / temp above the cap | `DiskFloorError` / `TempCapError` before writing |
| zip or parquet larger than the temp cap | `TempCapError` (split upstream or raise the cap) |
| `objects` layout crosses the threshold | `LayoutError`; re-run with `layout: shards` |
| killed mid-run | re-run: done files are skipped, multipart resumes |
| verify mismatch | `RuntimeError("verify failed: <rel>")`, stub not written |

Known limits: rows are held in memory until finalize; GCS `list-type=2` is untested
live; `hf_*` release modules untouched (WP-2 wires a staged source into a release).
