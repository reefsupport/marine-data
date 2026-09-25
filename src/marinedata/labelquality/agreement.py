"""Cross-source label agreement on duplicate images (WP-9).

Two populations of "the same picture, labelled twice":

* identical sha256 whose rows come from more than one source (the same file re-exported);
* dHash distance-0 pairs with different sha256 (re-encoded / resized copies) — the S47
  ``near_dup_pairs`` population at Hamming 0.

Every labelled row is a rater unit ``(sha256, source_id, label)``. A duplicate pair of
images yields the cross product of their rows; a pair of units is *cross-source* when the
two ``source_id`` differ. Cohen's κ needs a rater order, so each unit pair is oriented by
``(source_id, sha256)`` — the lexicographically smaller unit is rater A. The symmetric
(Scott/Fleiss two-rater) κ is reported beside it as a check on that arbitrary order.

Confidence intervals are a cluster bootstrap over duplicate clusters (the dHash bucket or
the sha), n = 1000, percentile — the design §4.3 recipe, because unit pairs inside one
cluster are not independent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

BOOTSTRAP_SEED = 20260925
BOOTSTRAP_N = 1000


@dataclass(frozen=True)
class Unit:
    sha256: str
    source_id: str
    label: str


@dataclass(frozen=True)
class UnitPair:
    cluster: str
    a: Unit
    b: Unit

    @property
    def cross_source(self) -> bool:
        return self.a.source_id != self.b.source_id

    @property
    def agree(self) -> bool:
        return self.a.label == self.b.label


def orient(u: Unit, v: Unit) -> tuple[Unit, Unit]:
    """Deterministic rater order: the smaller ``(source_id, sha256, label)`` is rater A."""
    return (
        (u, v) if (u.source_id, u.sha256, u.label) <= (v.source_id, v.sha256, v.label) else (v, u)
    )


def unit_pairs(
    clusters: Mapping[str, Sequence[str]], units_by_sha: Mapping[str, Sequence[Unit]]
) -> list[UnitPair]:
    """All unit pairs inside each cluster of duplicate sha256 (identical or dHash-0).

    Units of the same sha are paired too (the identical-file case), but a unit is never
    paired with itself.
    """
    out: list[UnitPair] = []
    for cid in sorted(clusters):
        units = [u for sha in sorted(set(clusters[cid])) for u in units_by_sha.get(sha, ())]
        for u, v in combinations(units, 2):
            a, b = orient(u, v)
            out.append(UnitPair(cid, a, b))
    return out


def dhash_clusters(dhashes: Mapping[str, str | int]) -> dict[str, list[str]]:
    """Group sha256 by identical dHash value; only buckets with ≥ 2 distinct sha."""
    buckets: dict[str, list[str]] = {}
    for sha, h in dhashes.items():
        buckets.setdefault(str(h), []).append(sha)
    return {f"dh:{h}": sorted(s) for h, s in buckets.items() if len(s) > 1}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ for two raters. ``nan`` when chance agreement is 1 (one class only)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def scott_pi(a: Sequence[str], b: Sequence[str]) -> float:
    """Symmetric two-rater κ (Scott's π / Fleiss with 2 raters): pooled marginals."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pooled = Counter(a) + Counter(b)
    pe = sum((c / (2 * n)) ** 2 for c in pooled.values())
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def cluster_bootstrap(
    pairs: Sequence[UnitPair],
    stat,  # type: ignore[no-untyped-def]
    *,
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile interval of ``stat(pairs)`` resampling whole clusters."""
    by: dict[str, list[UnitPair]] = {}
    for p in pairs:
        by.setdefault(p.cluster, []).append(p)
    keys = sorted(by)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [p for i in pick for p in by[keys[i]]]
        v = stat(sample)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _kappa_of(pairs: Sequence[UnitPair]) -> float:
    return cohen_kappa([p.a.label for p in pairs], [p.b.label for p in pairs])


def _agree_of(pairs: Sequence[UnitPair]) -> float:
    return float(np.mean([p.agree for p in pairs])) if pairs else float("nan")


@dataclass
class AgreementSummary:
    n_pairs: int
    n_clusters: int
    agreement: float
    agreement_ci: tuple[float, float]
    kappa: float
    kappa_ci: tuple[float, float]
    scott_pi: float
    confusion: dict[str, int] = field(default_factory=dict)
    by_source_pair: dict[str, tuple[int, int]] = field(default_factory=dict)


def summarise(pairs: Sequence[UnitPair], *, n_boot: int = BOOTSTRAP_N) -> AgreementSummary:
    """Agreement, κ (with cluster-bootstrap CIs), π, the label-pair confusion, per source pair."""
    confusion = Counter("|".join(sorted((p.a.label, p.b.label))) for p in pairs)
    per: dict[str, list[int]] = {}
    for p in pairs:
        key = f"{p.a.source_id}~{p.b.source_id}"
        tally = per.setdefault(key, [0, 0])
        tally[0] += 1
        tally[1] += int(p.agree)
    return AgreementSummary(
        n_pairs=len(pairs),
        n_clusters=len({p.cluster for p in pairs}),
        agreement=_agree_of(pairs),
        agreement_ci=cluster_bootstrap(pairs, _agree_of, n=n_boot),
        kappa=_kappa_of(pairs),
        kappa_ci=cluster_bootstrap(pairs, _kappa_of, n=n_boot),
        scott_pi=scott_pi([p.a.label for p in pairs], [p.b.label for p in pairs]),
        confusion=dict(sorted(confusion.items())),
        by_source_pair={k: (v[0], v[1]) for k, v in sorted(per.items())},
    )


def conflicts(units: Iterable[Unit]) -> dict[str, list[Unit]]:
    """sha256 → its units, for every sha whose units carry more than one label."""
    by: dict[str, list[Unit]] = {}
    for u in units:
        by.setdefault(u.sha256, []).append(u)
    return {
        s: sorted(us, key=lambda u: u.source_id)
        for s, us in sorted(by.items())
        if len({u.label for u in us}) > 1
    }
