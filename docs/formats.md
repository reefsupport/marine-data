# Format hardening: MANIFEST, verify, streaming, WebDataset

Three checks and one extra export shape sit on top of the Hub Parquet build
(`marinedata.hf_export`, `hf_parquet`, `hf_card`) once it is frozen and durably staged on
S3 (S59). None of this changes the Hub layout itself.

---

## 1. `marinedata manifest <dir>`

Writes `<dir>/MANIFEST.tsv`: `path`, `size`, `sha256`, `s3_etag`, `rows`, `schema_fingerprint`
(the last two blank for a non-Parquet file). One streaming read per file computes the sha256
and the S3 ETag together (single-part md5, or `md5(concat 64 MiB part md5s)-N` for a file
over one part) — the same convention S3 itself uses, so the manifest can be diffed against a
bucket listing with no download. `MANIFEST.tsv` never lists itself.

```
marinedata manifest ~/dev/reefsupport/data/_hf/v1
```

## 2. `marinedata verify-release <dir | s3://bucket/prefix>`

Checks a build against its manifest. Named `verify-release`, not `verify` — the CLI
already has a `verify` command (registry-declared layouts vs. fetched samples); a different
check, so a new name rather than a collision.

- **local**: re-hashes every file, compares size + sha256 + (Parquet) row count. A manifest
  entry missing on disk is a failure; a file on disk not in the manifest is a warning, or a
  failure under `--strict`.
- **s3**: a flat, paginated `list_objects_v2` (no download) compared by size + ETag.
  `--deep N` (or `--deep all`) additionally downloads N randomly chosen objects to
  `--scratch` (default cwd), checks sha256, and deletes each immediately after — never more
  than one object on disk at a time.

Exit 0/1, one-line summary on stdout.

```
marinedata verify-release ~/dev/reefsupport/data/_hf/v1
marinedata verify-release s3://<open-bucket>/releases/marine-data/v1/hf/ \
  --manifest ~/dev/reefsupport/data/_hf/v1/MANIFEST.tsv --deep 3
```

## 3. Streaming load test

`tests/test_hf_streaming.py` runs `datasets.load_dataset(<dir>, name=<config>, streaming=True)`
for every `config_name` in the card's `README.md` YAML front matter, reads the first 3 rows,
and asserts every row has the same column set. A tiny fixture (built in the test itself)
always runs, proving the mechanism regardless of machine; the real-build parametrization is
marked `slow` and skips cleanly when `_hf/v1` (`$MARINEDATA_HF_DIR`, default
`~/dev/reefsupport/data/_hf/v1`) is absent.

## 4. `marinedata export-wds <hf_dir> <out> [--limit-shards N]`

WebDataset tars beside the Parquet layout: one output `.tar` per input `images`-config
Parquet shard (same stem, `.tar` for `.parquet`), members `<image_sha256>.jpg|png` (raw
embedded bytes, extension from the image's own `path` field) plus `<image_sha256>.json`
(the image row's scalar columns plus a `labels` object joining every label-only config —
`benthic-coarse`, `benthic-l2`, `bleaching-condition`, `coral-health-binary`,
`general-pretraining` — on `image_sha256`). `masks` and `coralscop-pseudo-masks` embed
their own image column and are not joined here.

`--limit-shards N` caps how many input shards are processed (train shards first, in file
order, then validation, then test) — **disk rule: on real data, run with `--limit-shards 1`
only.** The unbounded full export happens server-side later, not on this machine.

```
marinedata export-wds ~/dev/reefsupport/data/_hf/v1 <out>/wds --limit-shards 1
```

Verified with `tarfile` (member names, JSON sidecar content) always, and with `webdataset`
(`WebDataset(path, shardshuffle=False)` reads back the same samples) when that package is
installed.
