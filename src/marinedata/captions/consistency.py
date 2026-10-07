"""Automated hallucination check: does a caption contradict the labels?

Runs on every row that has a `caption_vlm` (and, as a cheap regression guard, on
`caption_template` — which should always score zero flags since it is built from the
same facts it is checked against). Uses the WP-7 crosswalk vocabulary's *labels*, not
its full crosswalk graph: a small closed keyword set per axis (bleaching condition,
coarse benthic class) is enough to catch a caption naming the wrong state, without
needing NLP.

`wilson_ci` is the interval used for the agent audit's per-claim hallucination rate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Bleaching-condition axis vocabulary (native.yaml / reef-support-bleaching-condition
# crosswalk targets). Keys are lowercase surface forms a caption might use; values are
# the canonical `bleaching-condition` task label they imply.
_BLEACHING_TERMS: dict[str, str] = {
    "healthy": "HEALTHY",
    "bleached": "BLEACHED",
    "bleaching": "BLEACHED",
    "pale": "BLEACHED",
    "unhealthy": "UNHEALTHY",
    "diseased": "UNHEALTHY",
}

# Coarse benthic-class axis vocabulary (rs-benthic-v1.yaml target schema, abbreviated
# forms a caption is unlikely to use verbatim, so both the code and a plain-English
# alias are matched).
_BENTHIC_TERMS: dict[str, str] = {
    "hard coral": "HC",
    "soft coral": "SC",
    "coral rubble": "RB",
    "sand": "SD",
    "algae": "ALG",
    "rock": "RCK",
    "sponge": "SP",
}

_WORD_RE = re.compile(r"[a-z]+(?:\s[a-z]+)?")


def _mentions(text: str, vocab: dict[str, str]) -> set[str]:
    lowered = text.lower()
    return {canon for term, canon in vocab.items() if term in lowered}


@dataclass(frozen=True)
class ConsistencyFlag:
    axis: str
    caption_states: tuple[str, ...]
    label_state: str

    def as_str(self) -> str:
        return (
            f"{self.axis}:caption={','.join(sorted(self.caption_states))}!=label={self.label_state}"
        )


def check_caption(
    caption: str,
    bleaching_status: str | None,
    benthic_dominant: str | None,
) -> list[str]:
    """Return `consistency_flags` for one caption against its row's labels.

    A flag is raised only when the caption *names* a state on an axis the row has a
    label for, and that named state disagrees with the label. A caption that is silent
    on an axis, or that agrees, raises nothing.
    """
    flags: list[str] = []
    if not caption:
        return flags

    if bleaching_status:
        mentioned = _mentions(caption, _BLEACHING_TERMS)
        if mentioned and bleaching_status not in mentioned:
            flags.append(
                ConsistencyFlag("bleaching", tuple(sorted(mentioned)), bleaching_status).as_str()
            )

    if benthic_dominant:
        mentioned = _mentions(caption, _BENTHIC_TERMS)
        if mentioned and benthic_dominant not in mentioned:
            flags.append(
                ConsistencyFlag("benthic", tuple(sorted(mentioned)), benthic_dominant).as_str()
            )

    return flags


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%, z=1.96).

    Used for the agent audit's per-claim hallucination rate, where `n` (a manual
    review count) is small enough that a normal approximation is unreliable.
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))
