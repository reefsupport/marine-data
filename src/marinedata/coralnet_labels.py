"""CoralNet's public label table and the ``coralnet-label-id`` crosswalk resolver (D-S2).

Every CoralNet label has a global integer id and a public page
(``https://coralnet.ucsd.edu/label/<id>/``: name, functional group, default short code,
verified / duplicate status, description). A CoralNet-derived source's own short codes are
per-source and ambiguous (``Gonio`` vs ``Gonia``), but its ``labelset.csv`` maps each code to
that global id. So the family crosswalks once, by id:

* ``registry/taxonomy/coralnet-labels-<date>.parquet`` — the public table, one row per label
  (``scripts/coralnet_labels_fetch.py`` builds it, ≤ 2 req/s, cached, no login).
* ``registry/crosswalks/coralnet-label-id.yaml`` — curated edges keyed by label id: a WoRMS
  node where the label names a taxon, a functional or non-biotic node otherwise.
* :class:`CoralNetLabelIdResolver` — the curated edge when there is one; otherwise the
  label's functional group through ``coralnet-labelset`` (always ``coarsened`` at best), so an
  id nobody has curated yet still lands on a node instead of dropping silently.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Crosswalk, CrosswalkEdge, Fidelity

LIST_URL = "https://coralnet.ucsd.edu/label/list/"
DETAIL_URL = "https://coralnet.ucsd.edu/label/{id}/"
CROSSWALK_ID = "coralnet-label-id"
GROUP_CROSSWALK_ID = "coralnet-labelset"
TABLE_GLOB = "coralnet-labels-*.parquet"

_ROW = re.compile(r'<tr data-label-id="(\d+)">(.*?)</tr>', re.S)
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")
_PCT = re.compile(r'title="(\d+)%"')
_DUP = re.compile(r'title="Duplicate of ([^"]*)"')
_LINE = re.compile(r'<div class="line">\s*([A-Za-z ]+):(.*?)</div>', re.S)
_STATS = re.compile(r"Used in (\d+) sources\s+and for (\d+) annotations")
_DESC = re.compile(r"<dt>Description:</dt>\s*<dd>(.*?)</dd>", re.S)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", fragment)).split())


def parse_list(page: str) -> list[dict]:
    """Rows of ``/label/list/``: every public label with its list-page columns."""
    rows = []
    for label_id, body in _ROW.findall(page):
        cells = _TD.findall(body)
        status = cells[3]
        pct = _PCT.search(cells[2])
        dup = _DUP.search(status)
        rows.append(
            {
                "label_id": int(label_id),
                "name": _text(cells[0]),
                "functional_group": _text(cells[1]),
                "popularity": int(pct.group(1)) if pct else None,
                "verified": "label-icon-verified" in status,
                "duplicate": "label-icon-duplicate" in status,
                "duplicate_of": html.unescape(dup.group(1)) if dup else None,
                "calcification_rates": "label-icon-calcify" in status,
                "short_code": _text(cells[4]),
            }
        )
    return rows


def parse_detail(page: str) -> dict:
    """Fields of one ``/label/<id>/`` page that the list page does not carry."""
    lines = {k.strip(): _text(v) for k, v in _LINE.findall(page)}
    stats = _STATS.search(page)
    desc = _DESC.search(page)
    return {
        "name": lines.get("Name"),
        "functional_group": lines.get("Functional Group"),
        "short_code": lines.get("Default Short Code"),
        "verified": (lines.get("Verified") or "").startswith("Yes"),
        "used_in_sources": int(stats.group(1)) if stats else None,
        "used_in_annotations": int(stats.group(2)) if stats else None,
        "description": _text(desc.group(1)) if desc else None,
    }


def load_table(registry_root: str | Path) -> dict[int, dict]:
    """The newest ``coralnet-labels-*.parquet`` as ``{label_id: row}``; ``{}`` if absent."""
    paths = sorted((Path(registry_root) / "taxonomy").glob(TABLE_GLOB))
    if not paths:
        return {}
    import pyarrow.parquet as pq

    return {int(r["label_id"]): r for r in pq.read_table(paths[-1]).to_pylist()}


@dataclass(frozen=True)
class CoralNetLabelIdResolver:
    """Crosswalk-shaped view (``id``, ``target_schema``, ``edge``) over label ids."""

    curated: Crosswalk
    groups: Crosswalk
    table: dict[int, dict]

    @property
    def id(self) -> str:
        return self.curated.id

    @property
    def target_schema(self) -> str:
        return self.curated.target_schema

    def edge(self, label_id: str) -> CrosswalkEdge | None:
        found = self.curated.edge(str(label_id))
        if found is not None:
            return found
        row = self.table.get(int(label_id)) if str(label_id).isdigit() else None
        if row is None:
            return None
        group = self.groups.edge(row["functional_group"])
        if group is None:
            return None
        note = (
            f"auto: CoralNet {label_id} {row['name']!r} "
            f"-> functional group {row['functional_group']!r}"
        )
        if group.fidelity is Fidelity.UNMAPPABLE:
            return CrosswalkEdge(
                source_label=str(label_id), fidelity=Fidelity.UNMAPPABLE, note=note
            )
        fidelity = (
            Fidelity.APPROXIMATE if group.fidelity is Fidelity.APPROXIMATE else Fidelity.COARSENED
        )
        return CrosswalkEdge(
            source_label=str(label_id), targets=dict(group.targets), fidelity=fidelity, note=note
        )


def resolver(registry, registry_root: str | Path) -> CoralNetLabelIdResolver:
    return CoralNetLabelIdResolver(
        curated=registry.crosswalk(CROSSWALK_ID),
        groups=registry.crosswalk(GROUP_CROSSWALK_ID),
        table=load_table(registry_root),
    )
