# P5 baselines — v2 (design §4.4/§4.5)

## What ran for real

`marinedata eval reproduce` was run twice against a real DINOv2-S/14 checkpoint
(`facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
Apache-2.0, downloaded anonymously — no HF token, no login, D-AA), each time with a
freshly emptied feature cache so both runs genuinely re-extract on MPS rather than
replaying a cache hit.

- **Reproduce check (design §5 acceptance criterion):** macro-F1 delta between the two
  runs was **0.0 pt** on both the `test` (n=400) and `val` (n=200) splits — well inside
  the ±0.5 pt tolerance. (§4's "moves metrics by < 0.1 pt" planning note was
  conservative; MPS forward passes were bit-identical here for this backbone/fixture.)
- **Measured MPS throughput (feature extraction):** **211 img/s** for DINOv2-S/14 at
  batch 32, 224×224, decode-bound — see `results/v2/metrics.json`'s `img_per_s`. This
  **replaces** §4's planning number of ~150 img/s for M-series MPS. Probe-fit time is
  sub-second at n=2,000 and is not the bottleneck; it was not separately profiled.
  Projection for a ~1M-image v2 build at this measured rate: ~1.3 h for feature
  extraction alone (vs. the ~2 h planning estimate) on this Mac; the Job's ~8 img/s CPU
  planning number for probe fits/bootstraps was not re-measured (no server Job access
  from this worker).

## What did NOT run, and why

The manager decision named three probe tasks — `bleaching-condition`
(`bleaching-family-v1` lineage, `label_status in {ok}`, D-U2), `benthic-coarse`
(single-label view, D-Z) and a `source-id-domain` leakage sanity check — plus a real
≤6k-image v1 subset from `<open-bucket>`, stratified by source, split with the
split-v2 allocator, ID vs OOD reported separately.

That data does not exist yet on this integration line. D-Z2 fixes the task-label
contract as `data/_tasklabels/<source_id>/<task>.parquet` (sha256, source_id,
label_origin, + task payload), built by WP-8b and consumed by WP-8c's config builders.
A repo-wide search at merge time (`fa46154` + `fed2608`, this branch's bases) found
**no** `_tasklabels` directory anywhere in the ingest/eval integration line; the one
materialized task-label parquet that exists on any worktree on this machine is
`wp8d-labels`'s `data/_tasklabels/coralvqa/vqa.parquet` — a different task (VQA), not
bleaching or benthic-coarse. Deriving those two tasks' labels directly from raw source
annotations (crosswalks, mask-to-class rollups) is WP-8b/8c's declared job, not P5's,
and reimplementing it here would be well outside this package's named files and this
session's budget.

**Consequence:** `results/v2/metrics.json` and `card_table.md` report the reproduce
determinism check only (DINOv2-S/14, `bleaching-condition` id used as a label
purely to exercise the harness on 4 synthetic classes over 2,000 procedurally-generated
images — not real bleaching labels, not real imagery). No ID/OOD split, no
`benthic-coarse` row, no `source-id-domain` row, and no OpenCLIP ViT-B/16 run are in
this session's results: those all require either real per-task labels (blocked, see
above) or more budget than remained after standing up the harness and clearing the
blocker investigation. OpenCLIP support is implemented in
`src/marinedata/eval/baselines/features.py` (pinned revision
`7288da5a0d6f0b51c4a2b27c624837a9236d0112`, MIT) but was never exercised against real
weights in this session.

## Needs-Yohan

- WP-8b/8c task-label parquets for `bleaching-family-v1` and `benthic-coarse` need to
  land on (or before) the next P5-style run before real ID/OOD baseline numbers can be
  produced.
- Once they land, `eval reproduce` takes the same fixture-parquet contract
  (`image_sha256, image_path, label, split, split_group`) — pointing it at a real
  ≤6k-image stratified sample plus the split-v2 allocator's `split`/`split_group`
  columns is then a data-wiring change, not a harness change.
