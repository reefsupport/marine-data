"""Tier A: deterministic `caption_template` built from `CaptionFacts` only.

`build_caption` is a pure function of its `CaptionFacts` argument: the same facts
always produce the same string (tested in `tests/test_captions_template.py`). Clauses
are emitted in a fixed order; a clause whose backing fact is `None`/empty is skipped
entirely rather than filled with a placeholder — "no fact → no clause".
"""

from __future__ import annotations

from collections.abc import Sequence

from .facts import CaptionFacts

# English only for v2 (manager decision). Fixed clause order — do not reorder without
# updating the determinism test's expected strings.
_HABITAT_WORDS = {
    "coral_reef": "coral reef",
    "seagrass": "seagrass bed",
    "mangrove": "mangrove",
    "rocky_reef": "rocky reef",
    "open_water": "open water",
}


def _humanize(token: str) -> str:
    return _HABITAT_WORDS.get(token, token.replace("_", " "))


def _source_clause(facts: CaptionFacts) -> str | None:
    if not facts.source_id:
        return None
    if facts.platform:
        return f"An underwater image from {facts.source_id}, captured via {facts.platform}"
    return f"An underwater image from {facts.source_id}"


def _habitat_clause(facts: CaptionFacts) -> str | None:
    if not facts.habitat:
        return None
    return f"in a {_humanize(facts.habitat)} habitat"


def _depth_clause(facts: CaptionFacts) -> str | None:
    if not facts.depth_band:
        return None
    return f"at {facts.depth_band}"


def _ecoregion_clause(facts: CaptionFacts) -> str | None:
    if not facts.meow_ecoregion:
        return None
    return f"in the {facts.meow_ecoregion} ecoregion"


def _benthic_clause(facts: CaptionFacts) -> str | None:
    if facts.benthic_dominant:
        clause = f"benthic cover dominated by {facts.benthic_dominant.lower()}"
        if facts.benthic_present:
            others = ", ".join(c.lower() for c in facts.benthic_present)
            clause += f" (also present: {others})"
        return clause
    if facts.benthic_present:
        return "benthic cover including " + ", ".join(c.lower() for c in facts.benthic_present)
    return None


def _bleaching_clause(facts: CaptionFacts) -> str | None:
    if not facts.bleaching_status:
        return None
    return f"coral condition recorded as {facts.bleaching_status.lower()}"


def _taxa_clause(facts: CaptionFacts) -> str | None:
    if not facts.top_taxa:
        return None
    return "with " + ", ".join(facts.top_taxa) + " visible"


# Fixed order: source -> habitat -> depth -> ecoregion -> benthic -> bleaching -> taxa.
_CLAUSE_FNS: Sequence[callable] = (
    _source_clause,
    _habitat_clause,
    _depth_clause,
    _ecoregion_clause,
    _benthic_clause,
    _bleaching_clause,
    _taxa_clause,
)


def build_caption(facts: CaptionFacts) -> str:
    """Deterministically render `facts` into one English sentence.

    Returns an empty string if `facts` carries zero clauses (e.g. only a sha256 with
    every other field null) — callers should treat an empty caption as "no template
    coverage" rather than write a placeholder sentence.
    """
    clauses = [c for c in (fn(facts) for fn in _CLAUSE_FNS) if c]
    if not clauses:
        return ""
    body = ", ".join(clauses)
    return body[0].upper() + body[1:] + "."


def coverage_report(facts_list: Sequence[CaptionFacts], min_facts: int = 3) -> dict:
    """Report the % of rows with `>= min_facts` populated facts (brief item 2)."""
    n = len(facts_list)
    if n == 0:
        return {"rows": 0, "min_facts": min_facts, "at_or_above": 0, "pct": 0.0}
    at_or_above = sum(1 for f in facts_list if f.n_facts() >= min_facts)
    return {
        "rows": n,
        "min_facts": min_facts,
        "at_or_above": at_or_above,
        "pct": round(100.0 * at_or_above / n, 2),
    }
