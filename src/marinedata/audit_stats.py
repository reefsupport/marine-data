"""Stats helpers for the D-I manual privacy audit (WP-5d).

Pure-math, no heavy deps (unlike :mod:`marinedata.privacy`, which needs the
``privacy`` extra) so it runs in the standard ``[dev]`` test environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963985  # two-sided 95% normal quantile


@dataclass(frozen=True)
class WilsonInterval:
    """A Wilson score confidence interval for a binomial proportion."""

    point: float
    low: float
    high: float


def wilson_ci(successes: int, n: int, z: float = Z_95) -> WilsonInterval:
    """Wilson score interval for ``successes`` out of ``n`` trials.

    Preferred over the normal (Wald) approximation for small ``n`` or proportions
    near 0/1 — both apply here (e.g. 0/100 negatives, 18/150 face precision).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError("successes must be within [0, n]")
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    low = max(0.0, (center - adj) / denom)
    high = min(1.0, (center + adj) / denom)
    return WilsonInterval(point=phat, low=low, high=high)


def extrapolate_stratified(
    stratum_hits: dict[str, int],
    stratum_sample_sizes: dict[str, int],
    stratum_population_sizes: dict[str, int],
) -> tuple[float, float, float]:
    """Extrapolate a per-stratum sampled count to the full population.

    Returns ``(point, low, high)`` where ``low``/``high`` sum each stratum's Wilson
    bound scaled by its population size — a conservative (not a proper joint) 95% CI,
    since it assumes the strata's sampling errors are all simultaneously worst-case.
    """
    point = low = high = 0.0
    for key, pop in stratum_population_sizes.items():
        hits = stratum_hits.get(key, 0)
        n = stratum_sample_sizes.get(key, 0)
        if n == 0:
            continue
        ci = wilson_ci(hits, n)
        point += ci.point * pop
        low += ci.low * pop
        high += ci.high * pop
    return point, low, high
