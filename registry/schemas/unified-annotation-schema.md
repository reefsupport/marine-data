# Unified annotation schema (v1)

Code: `src/marinedata/annotation_schema.py`. Tests: `tests/test_annotation_schema.py`.
Spec: `UNIFIED-SCHEMA.md` section 3 (open-dataset session, 2026-10-05); this is WP-U1.
Not a registry collection: the registry loader reads `*.yaml` only, so this file is documentation.

One parquet table per task, at `data/_annotations/<task>/<source_id>/<source_version>.parquet`
(`annotation_path`). Written zstd with column statistics, rows sorted by `ann_id` ordinal, so the
bytes are deterministic. Schema metadata carries `marinedata.annotation_schema` (`"1"`) and
`marinedata.annotation_table`.

## Keys

Join key `image_sha256` + `source_id` + `source_version` + `ann_id` (`JOIN_KEY`).
`image_sha256` is 64 lowercase hex and joins to `SampleRow`. `ann_id` is `<source_id>:<ordinal>`,
stable across re-runs, unique per (source_id, source_version). `tracks.image_sha256` is the only
nullable key: null when no frame image was staged (`video_id` + `frame_idx` locate it).

## Common columns

Taxon tables (`boxes masks points image_labels tracks identities`) carry all 22, in this order.
The taxon-free tables (`captions vqa depth pairs`) carry the keys and the provenance group only
(11 columns), with no label, node or `match_type` column.

| Group | Columns |
|---|---|
| keys (4) | `image_sha256`, `source_id`, `source_version`, `ann_id` |
| label (10, taxon tables) | `label_native`, `label_native_id`, `label_set`, `taxon_node_id`, `form_node_id`, `condition_node_id`, `taxon_rank`, `worms_aphia_id` (int64), `rs_benthic_code`, `match_type` |
| provenance (7, all tables) | `annotator_type`, `annotator_detail`, `ann_license`, `ann_attribution`, `confidence` (float32), `upstream_split`, `label_status` |
| extras (1, taxon tables) | `attrs` (JSON object) |

Required: the keys (not `tracks.image_sha256`), `label_set`, `match_type`, `annotator_type`,
`label_status`. `label_native` is required except on a semantic mask row and on `identities`.
Everything else is nullable, and a null always means one thing (see the module docstring).

- `match_type`: `exact | broader | narrower | related | unmapped`. Old fidelity names: coarsened =
  broader, approximate = related, unmappable = unmapped. `narrower` is stored for audit and never
  used as a positive. `unmapped` forces every node and derived column null; any other value needs at
  least one of the three node ids (semantic masks excepted: their classes map per pixel value).
  `taxon_rank`, `worms_aphia_id`, `rs_benthic_code` are derived from the taxon node and only set
  next to it.
- `annotator_type`: `expert | human | crowd | model | pseudo`. From `label-origin.yaml`:
  human_expert = expert, human_crowd = crowd, pseudo_model = pseudo, `derived_rule` = `pseudo` with
  `annotator_detail = "rule"` (`annotator_from_origin`). `unknown` has no mapping and raises.
- `ann_license` / `ann_attribution`: the annotation's own terms, which can differ from the image's
  (CoralSCOP pseudo-masks). SPDX-like id or expression. Null means "same as the image".
- `confidence`: [0, 1], else null. Never invented.
- `upstream_split`: `train | val | test`, normalised with `sample_schema.normalise_split`.
- `label_status`: `ok | conflict | ambiguous | flagged_hard` (the `label_status` values).

## Coordinates

Normalised coordinates (`x_min y_min x_max y_max`, `x y`) are float32 in [0, 1]. Validators accept up
to 1e-6 outside; `to_table` clips them onto the range. Pixel columns are int32 and nullable.
A box needs `x_min < x_max` and `y_min < y_max`. Pixel and normalised boxes must agree within one
pixel when both are given; the pixel columns need `img_w` and `img_h`.

## Tables

Extra columns beyond the common ones (`req` = non-null):

| Table | Cols | Extra columns |
|---|---|---|
| `boxes` | 33 | `x_min y_min x_max y_max` float32 req; `x_min_px y_min_px x_max_px y_max_px` int32; `is_crowd` bool req; `img_w img_h` int32 |
| `masks` | 30 | `mask_kind` req (`semantic`/`instance`); `mask_ref`; `rle`; `polygon` (JSON list); `class_map` (JSON, pixel value -> label_native); `ignore_value` int32; `class_counts`, `canonical_class_counts` (JSON, derived) |
| `points` | 26 | `x y` float32 req; `x_px y_px` int32 (both or neither) |
| `image_labels` | 23 | `label_key` req (default `class`; multi-label = several rows) |
| `tracks` | 36 | `video_id` req; `frame_idx` int32 req; `track_id` req; the box columns (xyxy req, `is_crowd` nullable) |
| `captions` | 14 | `text`, `lang` (BCP-47), `caption_type` (`caption`/`summary`/`qa-answer`), all req |
| `vqa` | 15 | `question`, `answer`, `qa_type`, `lang`, all req |
| `depth` | 15 | `depth_ref` req; `units` req (`m`/`disparity_px`/`relative`); `gt_type` req (`sensor`/`stereo`/`sfm`/`synthetic`/`estimated`); `valid_mask_ref` |
| `pairs` | 13 | `pair_role` req (`degraded`/`enhanced`/`stereo_right`/`sonar`); `ref_image_sha256` req |
| `identities` | 35 | `individual_id` req; `individual_scope` req (`dataset`/`site`); the box columns, all nullable (a box is all-or-none) |

Mask rules. Semantic: one row per image-mask pair; `mask_ref` and a `class_map` are required, `rle`
and `polygon` are not allowed. Instance: one row per instance; at least one of `mask_ref`, `rle`,
`polygon`; no `class_map`; `label_native` required.

## API

`TABLES`, `table_spec(t)`, `arrow_schema(t)`, `validate_row(t, row)` (list of violations),
`validate_rows(t, rows)` (raises, also flags duplicate `ann_id`), `to_table`, `write_annotations`,
`read_annotations`, `validate_table`, `annotation_path`, `annotator_from_origin`, `MatchType`,
`AnnotatorType`, `normalise_split`. Rows are plain mappings; a missing key counts as null.

Not here: the crosswalk floor and `Crosswalk.resolve` (WP-U2), writers per source (WP-U3 onwards).
