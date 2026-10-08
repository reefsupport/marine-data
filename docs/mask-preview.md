# Mask preview column (design note for v1.1)

Status: design only, nothing is implemented. v1.0 on the Hub is frozen, so this ships with v1.1 at the earliest.

## Problem

The `mask` column of `coral-masks`, `scene-masks` and `instance-masks` holds label ids (`uint8` class id, `255` =
ignore; `uint16` instance id, `0` = background). The ids are small numbers, so the Hub dataset viewer and any image
viewer render the masks almost black. Visitors read that as "empty or broken". Colouring `mask` itself would break the
training contract (the pixel value is the label), so the colour version has to be a second column.

## Proposal

- New column `mask_preview` (feature `image`, PNG bytes, last column of the three mask configs). `fish-boxes`,
  `points`, `metadata` and `images` have no mask and do not get it.
- Palette mode PNG, long edge at most 512 px, resized with nearest-neighbour so labels never blend.
- Colour: a fixed, seeded 256-entry palette indexed by the pixel id (`255` and `0` background = black on
  instance masks, `255` ignore = black on semantic masks). The same id has the same colour in every row of a config,
  so the legend in the card ("Reading the masks") stays valid.
- Not part of any training contract: documented as display-only in `card_fields.FIELD_DOCS["mask_preview"]`
  (the test that every layout column has a description fails until that entry exists).

## Cost (measured)

Measured on the three committed Labelbox/Seaview fixture masks (`tests/fixtures/labelbox_rgb/*/masks_stitched`,
1031x1031, 2116x1611 and 4104x3105 px): palette PNG, 32 colours, long edge 512 px.

| | per row | per 1k rows |
|---|---|---|
| mean | 2.6 KB | 2.6 MB |
| max | 6.0 KB | 6.0 MB |

Fixture masks are flat colour regions, so plan with ten times the measurement: about 26 MB per 1k rows. For
scale, `coral-masks` v1.0 is 12.6 GB for 5,997 rows (about 2.1 MB per row), so the preview adds 0.1 % measured and
at most about 1.2 % at the planning figure. Re-measure on a real shard when the exporter flag lands.

## Exporter

`hf_export.export(...)` gets `mask_preview: bool = False`, surfaced as `--mask-preview` on the export step. Off keeps
v1.0 shards byte-identical (the e2e golden tree hashes must not move); v1.1 builds switch it on for the three
mask configs. Acceptance test for the implementation: preview width and height at most 512, mode `P`, not all black, and
the `mask` bytes of the row unchanged (sha256 stable).

## Viewer note for the card

Until `mask_preview` exists, the generated card carries the "Reading the masks" section (colour recipe included). From
v1.1 the section opens with:

> The viewer shows `mask_preview`, a colourised copy at most 512 px wide for looking at. `mask` holds the label ids
> and looks almost black in any image viewer; train on `mask`, never on `mask_preview`.
