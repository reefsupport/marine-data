"""WP-13 captions layer: Tier-A deterministic template captions and a Tier-B VLM pilot.

Tier A (`caption_template`, `label_origin=derived`) is built only from fields already
on the v1 release (metadata config + per-task labels): every clause traces to exactly
one source column, and a null field is omitted, never guessed (see `facts.py`).

Tier B (`caption_vlm`, `label_origin=model`) is a stratified pilot that gives an
open-weights VLM the Tier-A facts as grounding context and asks it to describe what is
visible without contradicting them (see `vlm.py`). `consistency.py` flags captions that
name a benthic class, taxon or bleaching state contradicting the labels, and provides
the Wilson-CI helper for the agent audit's per-claim hallucination rate.

See `docs/captions.md` for the method, measured counts and the human-verification
protocol this module cannot itself satisfy.
"""

from __future__ import annotations

from .facts import CaptionFacts, build_facts_table, depth_band, rollup_points_to_facts
from .template import build_caption, coverage_report

__all__ = [
    "CaptionFacts",
    "build_caption",
    "build_facts_table",
    "coverage_report",
    "depth_band",
    "rollup_points_to_facts",
]
