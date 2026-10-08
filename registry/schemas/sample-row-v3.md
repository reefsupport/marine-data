# SampleRow schema v3 (PA-P2.2 / P2.4)

`src/marinedata/sample_schema.py` is the one canonical per-sample row. Schema v3 appends 15 columns to the
37 of v2; a v1 (34) or v2 (37) `metadata.parquet` still validates and reads back with the new columns null.
Nothing in a staged tree is rewritten: the exporter joins the resolver output by `image_sha256`.

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Capture day. Month precision stores the first of the month. Null = only a year (or nothing) is known. |
| `date_precision` | `day` \| `month` \| `year` | Required when `date` is set; `year` with `date` null for year-only sources (Tetes: `year=2023`). |
| `year` | int16 | Calendar year; equals `date.year` when both are set. |
| `site_id` `quadrat_id` `survey_id` `region` | string | Null = not resolved. |
| `coord_precision_m` | float64 | Radius `lat`/`lon` are good to; needs a position. |
| `geo_source` `date_source` | string | Table or field the value came from. |
| `license_url` `rights_holder` `license_verified_by` `license_verified_on` `provenance_url` | string/date | Default to the registry entry; `per_row_license` overrides per item. |

`lat lon depth_m platform license attribution upstream_id` already exist (v1). `license`/`attribution` are never
null in a staged `metadata.parquet` (`NOASSERTION` when a source states none). A resolver row may carry a null
`license` when a per-row source's item states none: such a row never ships (`release build` drops it and counts
it per source and reason under `licence_rows_dropped` in RELEASE.json (the key is absent when
nothing was dropped, so existing releases keep their bytes); the metadata gate fails it).

## `platform` controlled vocabulary

`diver-photo-quadrat`, `diver-video`, `auv`, `rov`, `drop-camera`, `bruv`, `aerial`, `web-collected`, `aquarium`,
`lab`, `unknown`. v1/v2 values (`diver snorkel towed lander uav vessel`) stay valid. The registry sets it per source
with `default_platform:` plus a one-line `platform_evidence:` naming the card, paper or README sentence. `unknown`
means "checked, no evidence" and is never replaced by a guess; null means "not derived yet".

## Registry fields

- `per_row_license: {license: <staged field>, license_url: ..., attribution: ..., rights_holder: ...}` for sources
  whose licence is per item. An override replaces the default even when the staged field is empty (-> `license=null`).
- `expect_geo`, `expect_date`, `expect_site`, `expect_depth`: `required` (coverage floor 0.99), `expected` (0.90),
  `none` (default, not checked). `geo` = min(`lat`, `lon`), `date` = day-level `date`.

## Commands

```
marinedata metadata resolve <source> --out <dir>          # writes <dir>/<source>/<version>/{samples.parquet,coverage.json}
marinedata release gate --metadata <dir> [--gates GATES.json]   # coverage floors + licence checks, merged into GATES.json["metadata"]
python -m marinedata.hf_export ... --metadata-dir <dir>          # fill-only join by image_sha256 into `metadata` and 15 `points` columns; no flag = `metadata` unchanged, `points` columns null
```
