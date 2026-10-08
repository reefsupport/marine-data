# Architecture

`marine-data` is a Python package (`src/marinedata`) with a declarative registry (`registry/`). The registry says what
each source is and what may be done with it. The package reads the registry to query sources, build datasets, check
licences and produce releases.

```
registry/  ->  gate + query  ->  loaders  ->  harmonise  ->  builder  ->  release pipeline  ->  published datasets
(YAML)         (what may be      (read each    (shared         (splits,     (dedup, split map,
               used, for what)   source)       label schema)   adapters)    Parquet, checksums)
```

## Registry

`registry/` holds one YAML entry per source and the vocabularies that connect them.

| Path | Content |
|---|---|
| `registry/sources/` | Source entries: licence, legal basis, provenance, access method, capabilities, coverage, counts, citation |
| `registry/licences.yaml` | Licence ids with their tier and flags (`no_derivatives`, `share_alike`, `attribution_required`, and others) |
| `registry/profiles.yaml` | Release profiles: which tiers and flags a given purpose accepts |
| `registry/schemas/`, `registry/crosswalks/` | Label schemas and the mappings from a source's native labels to them |
| `registry/tasks/`, `registry/taxonomy/` | Task definitions and the shared taxon vocabulary |
| `registry/geo/`, `registry/splits/`, `registry/ingest-specs/` | Geographic provenance, split rules and per-source ingestion specs |

Entries are validated on load (`marinedata.registry.Registry`): references between sources, schemas and crosswalks must
resolve, and each entry must carry a verification record (date, method and the primary source consulted).

## Gate and query

`marinedata.gate.evaluate` decides whether a source is admitted under a profile and returns a `Decision` with the
reason. `marinedata.find` wraps it with facets (capability, annotation kind, modality, region, provenance) and returns
a `QueryResult`: the matching sources plus every exclusion and its reason. The profile is a required argument, so there
is no default that returns everything.

## Loaders and harmonisation

`marinedata.loaders` reads each source from its upstream layout, selected by the entry's loader specification. The
harmonisation step maps native labels to a target schema through the crosswalks. A source without a crosswalk raises an
error unless `allow_unmapped=True` is passed, so vocabularies are never mixed silently.

## Builder

`marinedata.DatasetBuilder` combines the admitted sources of a profile into a `Dataset`: group-wise splits, the
`ignore_index` convention for unsupervised label axes, class statistics, and adapters for pandas, PyTorch and
TensorFlow. `marinedata.build_lineage` records which sources contributed, which were excluded and which obligations
(attribution, share-alike) carry forward.

## Release pipeline

The release commands (`marinedata release`, `dedup`, `splits`, `verify-release`) produce the published datasets:

1. Near-duplicate images are grouped so that a group never straddles two splits.
2. A split map is generated once per release and then frozen. The tool refuses to overwrite an existing map.
3. The export writes Parquet shards with images, masks, per-row licence and attribution, class maps and splits, plus a
   `CHECKSUMS.sha256` per configuration and Croissant metadata.
4. Verification compares files and row counts with the build before and after upload.

## Ingestion

`marinedata ingest`, `ingest-source` and `ingest-batch` fetch a source from its upstream distribution into storage and
write a checksum marker. Transfers resume after interruption and finished sources are skipped on re-run. See
[`docs/design/ingestion.md`](design/ingestion.md) and [`docs/ingest-howto.md`](ingest-howto.md).

## Quality and evaluation

Optional extras add face and person screening (`privacy-scan`), label-quality checks, evaluation-set decontamination
(`decon`), captions and benchmarks. They are documented under `docs/` and installed only when needed.

## Credentials

Every read of the open bucket (`rs-storage-open`) resolves one identity through
`marinedata.s3_auth`: (1) the environment, `MARINEDATA_S3_ACCESS_KEY` +
`MARINEDATA_S3_SECRET_KEY` (optionally `MARINEDATA_S3_ENDPOINT`, `MARINEDATA_S3_REGION`);
(2) the rclone remote `$MARINEDATA_RCLONE_REMOTE` (default `rs-hel1`) in
`$MARINEDATA_RCLONE_CONF` (default `~/.config/rclone/rclone.conf`), read with `configparser`;
(3) neither, which means anonymous. Code signs when it holds a key and falls back to anonymous,
never the reverse: a signature refused with 401/403 is retried anonymously, and a signature is
only ever sent to the credentials' own endpoint. The fetchers (`fetchers_remote`),
`manifest_identity`, `metadata_norm.stage`, `task_layers.s3_keyed` and the `manifest` adapter all
use this one resolver. Secrets are never printed or logged (`S3Creds.__repr__` redacts them), and
nothing here writes to the bucket.

What D9 changes: 46 non-open trees in `rs-storage-open` are anonymously readable today. D9
(allowlist + private default) makes anonymous GET return 403 for everything not on the public
allowlist, so only code that signs keeps reading the rest. Because every path above already
signs first, D9 can roll out without a code change; public v1.0 consumers are unaffected.
`marinedata auth check` prints `signed as <env|remote> (endpoint ...)` or `anonymous`, then one
`HEAD` of the bucket and one `HEAD` of a single small known key and reports the HTTP status
(200/403/404). It never lists and always exits 0.

Fetch integrity: a remote GET that has a pinned digest (a staged tree's `CHECKSUMS.sha256`
entry, or `access.sha256`) is hashed before the file is used. A mismatch raises `IntegrityError`
naming the key and both digests and is not retried; files are written through `<path>.part` and
renamed only once whole (and, where pinned, verified); transport faults and truncated bodies are
retried a bounded number of times with backoff (3 by default, 8 for the ~10^5-file staged-tree
fetch). A response is capped at `MAX_GET_BYTES` as received, archive members are capped per member,
in total and by compression ratio, a plaintext `http://` source must declare `access.sha256`, and
an S3 listing that declares a DOCTYPE is refused unparsed.

## Design constraints

- **Licence checks come first.** Every entry records the primary source of its licence, and the profile is explicit.
- **Gaps are recorded, not guessed.** Unknown licences use `NO-LICENCE-STATED`, and unknown counts stay unset.
- **Splits are frozen and grouped.** Related frames and duplicates stay on one side of a split.
- **Provenance travels with the data.** Published rows carry their `licence` and `attribution`.
