# Legal notes

**This project publishes metadata, not legal advice.** Tier assignments record our
reading of primary sources, cited per entry. They are a starting point for your own
review. Nothing here creates a solicitor–client relationship, and no contributor
warrants that any assignment is correct.

If a tier is wrong, open an issue. We would rather be corrected than relied upon.

---

## Why tiers alone are insufficient

Two licences can share a tier and differ in what they actually permit.

`CC-BY-NC` and `CC-BY-NC-ND` are both non-commercial. But **ND forbids derivative works
outright.** Generating a segmentation mask, a crop, a resize, or an augmentation is a
derivative act. An ND source is therefore *untrainable*, not merely non-shippable — a
distinction that is routinely missed, including for datasets in wide research use.

The `no_derivatives` flag exists so the gate can see this, and so it cannot be left in
prose where only a careful reader would find it. `tests/test_registry.py` enforces that
any entry mentioning no-derivatives in prose also sets the flag.

## Facts versus expression

Copyright protects expression, not facts or systems. Label schemes, category names,
taxonomic hierarchies and crosswalk mappings are functional and factual.

In practice this means the *structure* of a restrictively-licensed labelling scheme can
usually inform your own ontology even when the *imagery* cannot be used. The fact that
one scheme's attribute corresponds to another's category is information, not authorship.

Two cautions. First, the EU **sui generis database right** protects substantial
extraction from a database independently of copyright — taking a scheme is different from
bulk-extracting contents. Second, this is a general principle, not a ruling on any
specific dataset.

## Text and data mining (EU)

The `T4_TDM_ONLY` tier covers sources with no licence grant that are nonetheless lawfully
accessible. Any use rests on a statutory exception. In the EU that is DSM Directive
Art. 4, implemented in the Netherlands as **Art. 15o Auteurswet**, which permits mining
for any purpose including commercial, provided:

1. the miner has **lawful access**, and
2. the rightsholder has **not expressly reserved** rights by machine-readable means.

Four constraints make this narrower than it first appears.

**It is a defence, not a permission.** You rely on it when challenged. That is a
materially different posture from holding a licence, and it is a poor fit for products
whose value proposition is auditability.

**Contracts can override it.** DSM Art. 7 makes contractual terms unenforceable against
Arts. 3, 5 and 6 — **Art. 4 is absent from that list.** So a form gate or click-through
can validly exclude commercial TDM where a bare licence arguably cannot. This is what the
`contract_gated` flag records, and it is the sharpest practical dividing line between
sources that are reachable and sources that are not.

**Retention is limited.** Art. 4(2) permits retaining copies only as long as necessary
for the mining. Profiles carry a `retention_days` field for this reason.

**It is jurisdictional.** The exception is EU law. The United States has no equivalent
statutory provision; fair use is the analogue and is fact-specific and contested. A model
trained under an EU exception and deployed elsewhere carries residual exposure the
exception does not answer.

Accordingly, `pretrain-eu` is the only profile admitting T4, it requires a
`legal_opinion_ref` supplied at build time, and it is scoped to pretraining rather than
supervised fine-tuning.

## Provenance defects cannot be cured downstream

Some datasets are redistributed under a licence broader than the rights the redistributor
holds — typically aggregations containing stock or platform imagery. Where that occurs,
the grant is void as to those works.

No downstream act repairs this: not a permissive licence from the redistributor, not
written permission from them, not open-sourcing your derivative, not routing the work
through a research entity. The `provenance_defective` flag is an absolute bar that no
profile can override, and the gate checks it before anything else.

## Public visibility is not a licence

Several significant marine datasets are publicly viewable with no stated terms at all.
That is **not** a grant — it is an absence of one, which is legally weaker than a
restrictive licence, because with a restrictive licence you at least know what you have.
Such sources are tiered `T4_TDM_ONLY`, never permissive.

## Partner data

Imagery received from partners without a written grant is in the same position as
unlicensed public data — friendlier, and easier to fix, but not usable until fixed.
Entries in that position carry `legal_basis: unknown`, which the gate treats as an
absolute bar. `tests/test_registry.py` prevents such entries from being tiered as usable.
