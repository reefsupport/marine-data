"""Tier-A caption facts: deterministic extraction from released metadata + task labels.

Every field on :class:`CaptionFacts` traces to exactly one column of the v1 HF release
(the ``metadata`` config, plus whichever per-task label config a source landed under —
``bleaching-condition``, ``benthic-coarse``, ...). A field that is null, empty or
unresolved on the source row is omitted from the facts — never guessed, never
interpolated from a sibling row. This module makes no network or filesystem call of
its own beyond the dataframes it is handed; see ``cli.py`` for wiring it to disk.

Two fields are deliberately always empty on v1, and callers should not be surprised:

- ``benthic_dominant`` / ``benthic_present`` read the ``label`` column of the
  ``benthic-coarse``/``benthic-l2`` task configs. On the v1 release those columns are
  100% null (WP-8's D-Y points/mask rollup is not wired yet — see
  ``docs/task-layers.md``); this module reads whatever is there rather than
  recomputing the rollup itself, so the fact appears automatically once WP-8 lands it.
- ``top_taxa`` has no populated per-image field anywhere in v1 (no taxon-list task
  config exists yet); the parameter is accepted for forward compatibility only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import pandas as pd

# WP-13 Tier-A depth bands. Not a registry-authoritative vocabulary (there is no
# `registry/schemas` entry for depth bands) — a human-readable bucketing local to
# captioning. depth_m is 0% populated in v1, so this
# path is exercised by unit tests, not yet by any real v1 row.
_DEPTH_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 5.0, "very shallow (under 5 m)"),
    (5.0, 10.0, "shallow (5-10 m)"),
    (10.0, 20.0, "reef-flat / fore-reef depth (10-20 m)"),
    (20.0, 40.0, "upper mesophotic (20-40 m)"),
    (40.0, float("inf"), "deep (40 m or more)"),
)


def depth_band(depth_m: float | None) -> str | None:
    """Bucket a depth in metres into a human-readable band, or ``None`` if absent."""
    if depth_m is None or (isinstance(depth_m, float) and pd.isna(depth_m)):
        return None
    for lo, hi, label in _DEPTH_BANDS:
        if lo <= depth_m < hi:
            return label
    return None


def _clean(value: object) -> str | None:
    """Normalise a scalar cell to ``str | None``: NaN/None/empty → ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_tuple(values: Iterable[object] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out = []
    for v in values:
        c = _clean(v)
        if c:
            out.append(c)
    return tuple(out)


@dataclass(frozen=True)
class CaptionFacts:
    """One image's Tier-A facts. ``None``/``()`` means "not in the source data"."""

    image_sha256: str
    source_id: str | None = None
    source_version: str | None = None
    platform: str | None = None
    habitat: str | None = None
    depth_band: str | None = None
    meow_ecoregion: str | None = None
    benthic_dominant: str | None = None
    benthic_present: tuple[str, ...] = field(default_factory=tuple)
    bleaching_status: str | None = None
    top_taxa: tuple[str, ...] = field(default_factory=tuple)

    def n_facts(self) -> int:
        """Count of populated caption-worthy fields (excludes the join key alone)."""
        n = 1 if self.source_id else 0  # source/platform is one clause
        for v in (
            self.habitat,
            self.depth_band,
            self.meow_ecoregion,
            self.benthic_dominant,
            self.bleaching_status,
        ):
            if v:
                n += 1
        if self.benthic_present:
            n += 1
        if self.top_taxa:
            n += 1
        return n

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["benthic_present"] = list(self.benthic_present)
        d["top_taxa"] = list(self.top_taxa)
        return d

    def to_json(self) -> str:
        """Deterministic (sorted-key) JSON for the ``caption_facts`` parquet column."""
        return json.dumps(self.to_dict(), sort_keys=True)


def _facts_from_row(
    row: pd.Series,
    benthic_label: str | None,
    bleaching_label: str | None,
) -> CaptionFacts:
    return CaptionFacts(
        image_sha256=row["image_sha256"],
        source_id=_clean(row.get("source_id")),
        source_version=_clean(row.get("source_version")),
        platform=_clean(row.get("platform")),
        habitat=_clean(row.get("habitat")),
        depth_band=depth_band(row.get("depth_m")),
        meow_ecoregion=_clean(row.get("meow_ecoregion")),
        benthic_dominant=benthic_label,
        bleaching_status=bleaching_label,
    )


def rollup_points_to_facts(
    points: pd.DataFrame,
    image_col: str,
    class_col: str,
    source_id: str,
    unknown_classes: frozenset[str] = frozenset(),
    dominant_threshold: float = 0.5,
    present_threshold: float = 0.1,
    exclude_unknown_majority: float = 0.5,
) -> list[CaptionFacts]:
    """D-Y points-to-image rollup, for a staged source keyed by its own native image
    id rather than a computed `image_sha256` (e.g. mermaid-aws, pre-WP-6-ingest).

    `benthic_dominant` is a class with `>= dominant_threshold` of an image's points,
    else ``"mixed"``; `benthic_present` is the set of classes with
    `>= present_threshold`. Unknown/unlabelled points are excluded from the
    denominator; an image with `> exclude_unknown_majority` unknown points is dropped
    entirely (D-Y). ``image_sha256`` on the returned facts is the native id, prefixed
    so it is never mistaken for a real sha256 — callers must not write these rows into
    a sha256-keyed parquet without re-keying them once the source is ingested.
    """
    facts: list[CaptionFacts] = []
    for image_id, group in points.groupby(image_col):
        known = group[~group[class_col].isin(unknown_classes)]
        total = len(group)
        if total == 0 or len(known) / total < (1 - exclude_unknown_majority):
            continue
        counts = known[class_col].value_counts()
        fractions = counts / len(known)
        dominant = None
        top_class, top_frac = fractions.index[0], fractions.iloc[0]
        if top_frac >= dominant_threshold:
            dominant = top_class
        elif len(fractions) > 1:
            dominant = "mixed"
        present = tuple(sorted(fractions[fractions >= present_threshold].index))
        facts.append(
            CaptionFacts(
                image_sha256=f"native:{source_id}:{image_id}",
                source_id=source_id,
                benthic_dominant=dominant,
                benthic_present=present,
            )
        )
    return facts


def build_facts_table(
    metadata: pd.DataFrame,
    benthic_coarse: pd.DataFrame | None = None,
    bleaching_condition: pd.DataFrame | None = None,
) -> list[CaptionFacts]:
    """Join metadata with the (optional) task-label tables and extract one
    :class:`CaptionFacts` per ``image_sha256`` in ``metadata``.

    ``benthic_coarse``/``bleaching_condition`` are the per-task label configs
    (columns: ``image_sha256``, ``label``, ...). Only the resolved ``label`` column is
    read — an unresolved/ambiguous row (``label`` null) contributes no fact, which is
    the same gate the manager decision calls ``label_status=ok`` for; v1 has no
    standalone ``label_status`` column yet (WP-9's D-U2 CL gate is not wired), so a
    non-null resolved ``label`` is used as its proxy. See ``docs/captions.md``.
    """
    benthic_map: dict[str, str] = {}
    if benthic_coarse is not None and "label" in benthic_coarse.columns:
        resolved = benthic_coarse.dropna(subset=["label"])
        benthic_map = dict(zip(resolved["image_sha256"], resolved["label"], strict=False))

    bleaching_map: dict[str, str] = {}
    if bleaching_condition is not None and "label" in bleaching_condition.columns:
        resolved = bleaching_condition.dropna(subset=["label"])
        bleaching_map = dict(zip(resolved["image_sha256"], resolved["label"], strict=False))

    facts = []
    for _, row in metadata.iterrows():
        sha = row["image_sha256"]
        facts.append(
            _facts_from_row(
                row,
                benthic_label=_clean(benthic_map.get(sha)),
                bleaching_label=_clean(bleaching_map.get(sha)),
            )
        )
    return facts
