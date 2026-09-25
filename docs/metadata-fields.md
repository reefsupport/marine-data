# Metadata field reference — v1

Every column in the `metadata` HF config (see `src/marinedata/metadata_release.py`), one row
per `image_sha256`. Fields not listed below are non-null for all v1 rows (see
`docs/metadata-coverage-v1.md` for the measured percentages).

| Field | Type | Why it is null for v1 |
|---|---|---|
| `upstream_url` | string | no per-item URL resolves; the source-level URL is in the registry |
| `lineage_root_digest` | string | null for first-hop ingests; also null for coralscop-masks-rs (multi-parent images_from — ambiguous per-image lineage) |
| `capture_datetime` | string | no source in v1 stages EXIF or a capture-time column |
| `lat` | double | no source in v1 stages GPS (checked: no EXIF, no location column) |
| `lon` | double | no source in v1 stages GPS (checked: no EXIF, no location column) |
| `gps_precision_m` | double | requires lat/lon, which are null for all of v1 |
| `depth_m` | double | no source in v1 records a per-image depth |
| `depth_source` | string | requires depth_m, which is null for all of v1 |
| `platform` | string | not stated upstream for any v1 source and not safely inferable |
| `camera` | string | not stated upstream for any v1 source |
| `meow_realm` | string | requires lat/lon, which are null for all of v1 |
| `meow_province` | string | requires lat/lon, which are null for all of v1 |
| `meow_ecoregion` | string | requires lat/lon, which are null for all of v1 |
| `depth_zone` | string | requires depth_m, which is null for all of v1 |
| `habitat` | string | the registry has no per-source habitat field yet (open item) |
