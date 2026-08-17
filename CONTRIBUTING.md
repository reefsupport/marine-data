# Contributing

The most valuable contribution is **a verified licence**, not a large upload. We do not
host data — see the README on why.

## Adding a source

Add an entry to the appropriate file under `registry/sources/`, then run:

```bash
pytest
marinedata check --profile research
```

### Two rules

**1. `verified_by` must cite a primary source.**

A licence file you opened. A dataset card. Written permission. A terms-of-use page, with
the date you read it.

Not: "the paper says", "it's on HuggingFace so it's open", "everyone uses it".

This rule exists because we got it wrong ourselves. Our internal catalog recorded
MarineInst20M as "Mixed / Open" on the strength of a paper's phrasing. The `LICENSE.txt`
in the repository said CC-BY-NC-SA — over an image pool including Getty and Shutterstock.
Secondary sources are how that happens.

**2. Tier changes require a second reviewer.**

Raising a source's tier makes it usable in more contexts. That should be as deliberate as
changing a licence.

### Record what you don't know

An honest gap is more useful than a confident guess:

- Unsure of the licence → `NO-LICENCE-STATED` and `legal_basis: unknown`.
- Two sources disagree → set `disputed: true` with a `dispute_note`. The gate will block
  it on shipping profiles until resolved, which is the correct outcome.
- Unknown item counts → leave `items` unset rather than estimating.

### Domain shift is part of the entry

If a source is regionally biased, say so. `missing_classes` in particular: a schema that
omits a taxon abundant in some region will fail there in a way accuracy metrics on the
source's own test set will never reveal. This field is the one thing in the registry that
no other catalog carries, and it is the most common cause of deployment failure in marine
CV.

Measured drops (`known_drops`) need a real citation — a paper, or reproducible internal
measurement described specifically enough to check.

## Code

- `ruff check . && ruff format --check .`
- `pytest` — the gate tests are non-negotiable. If you change gate semantics, the matrix
  in `tests/test_gate.py` must change with it, deliberately and in the same commit.
- Models are frozen (`ConfigDict(frozen=True)`). A registry is a set of assertions; code
  that mutates one in place can silently disagree with a lineage report it already emitted.

## Scope

In scope: marine and aquatic datasets of any modality — imagery, video, geometry, audio,
molecular, remote sensing.

Out of scope: hosting data, model weights (a separate registry may come later), and
anything requiring us to redistribute restrictively-licensed material.

## Conduct

Be accurate, be correctable. If you think a tier is wrong, open an issue — being
corrected is the point.
