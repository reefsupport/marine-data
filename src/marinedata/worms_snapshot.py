"""Build the pinned WoRMS snapshot — codegen, never runtime.

``marinedata taxonomy snapshot`` queries the open WoRMS REST API (no login) for every
AphiaID and scientific name the registry needs, plus each accepted taxon's full
classification lineage, and writes ``registry/taxonomy/worms-<date>.parquet``.

Everything downstream — the schema checks, the crosswalk audit, the release gate and
the tests — reads ONLY that file. The network is touched here and nowhere else; the
test suite blocks sockets to prove it.

Politeness: WoRMS is a small academic service. Requests are spaced ≥ 0.5 s (≤ 2/s),
batched where the API allows (50 ids or names per call), and every response is cached
on disk so a rerun costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

WORMS = "https://www.marinespecies.org/rest"
USER_AGENT = "marinedata/0.1 (+https://github.com/reefsupport/marine-data)"
MIN_INTERVAL_S = 0.5

SNAPSHOT_COLUMNS = (
    "query",
    "aphia_id",
    "scientific_name",
    "rank",
    "status",
    "accepted_aphia_id",
    "accepted_name",
    "lineage",
    "origin",
    "retrieved_at",
)
"""``lineage`` is a JSON list of ``[rank, name, aphia_id]`` from the root down to and
including the accepted taxon. ``origin`` is ``id`` / ``name`` (asked for) or ``lineage``
(an ancestor recorded so parents resolve offline)."""


class WormsClient:
    """Cached, rate-limited WoRMS REST client."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0

    def _get(self, path: str) -> object:
        key = hashlib.sha256(path.encode()).hexdigest()[:24]
        hit = self.cache / f"{key}.json"
        if hit.exists():
            return json.loads(hit.read_text())
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(f"{WORMS}/{path}", headers={"User-Agent": USER_AGENT})
        body: object = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                    body = json.loads(raw) if raw.strip() else None
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (204, 404):
                    body = None
                    break
                if attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2.0 * (attempt + 1))
        self._last = time.monotonic()
        hit.write_text(json.dumps(body))
        return body

    def records_by_ids(self, ids: Iterable[int]) -> dict[int, dict]:
        ids = sorted(set(ids))
        out: dict[int, dict] = {}
        for i in range(0, len(ids), 50):
            q = "&".join(f"aphiaids[]={x}" for x in ids[i : i + 50])
            for rec in self._get(f"AphiaRecordsByAphiaIDs?{q}") or []:
                if rec:
                    out[int(rec["AphiaID"])] = rec
        return out

    def records_by_names(self, names: Iterable[str]) -> dict[str, list[dict]]:
        names = sorted(set(names))
        out: dict[str, list[dict]] = {}
        for i in range(0, len(names), 50):
            chunk = names[i : i + 50]
            q = "&".join(f"scientificnames[]={urllib.parse.quote(n)}" for n in chunk)
            body = self._get(f"AphiaRecordsByNames?{q}&like=false&marine_only=false") or []
            for name, recs in zip(chunk, body, strict=False):
                out[name] = [r for r in (recs or []) if r]
        return out

    def classification(self, aphia_id: int) -> list[tuple[str, str, int]]:
        node = self._get(f"AphiaClassificationByAphiaID/{aphia_id}")
        chain: list[tuple[str, str, int]] = []
        while isinstance(node, dict) and node.get("AphiaID"):
            chain.append(
                (node.get("rank") or "", node.get("scientificname") or "", int(node["AphiaID"]))
            )
            node = node.get("child")
        return chain


@dataclass(frozen=True)
class NameChoice:
    """Why a name resolved to the record it did — recorded, never silent."""

    name: str
    aphia_id: int | None
    reason: str


def choose(name: str, records: list[dict], prefer_kingdom: tuple[str, ...] = ()) -> NameChoice:
    """Pick one record for a name: an exact accepted match wins; homonyms go to the
    preferred kingdom; an unaccepted-only match follows ``valid_AphiaID``."""
    exact = [r for r in records if (r.get("scientificname") or "").lower() == name.lower()]
    pool = exact or records
    if not pool:
        return NameChoice(name, None, "no WoRMS record")
    accepted = [r for r in pool if r.get("status") == "accepted"]
    cands = accepted or pool
    if len(cands) > 1 and prefer_kingdom:
        pref = [r for r in cands if r.get("kingdom") in prefer_kingdom]
        cands = pref or cands
    rec = cands[0]
    why = "accepted exact" if accepted else f"only {rec.get('status')} match; follows valid_AphiaID"
    if len(cands) > 1:
        why += f"; {len(cands)} homonyms, took {rec.get('kingdom')}"
    return NameChoice(name, int(rec["AphiaID"]), why)


def build_rows(
    client: WormsClient,
    *,
    ids: Iterable[int],
    names: Iterable[str],
    retrieved_at: date | None = None,
    prefer: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[dict], list[NameChoice]]:
    """Resolve ids and names, follow synonyms, and record lineages. Returns rows and
    the per-name choices (so ambiguous or missing names are listed, not dropped)."""
    day = (retrieved_at or date.today()).isoformat()
    by_name = client.records_by_names(names)
    default = ("Animalia", "Plantae", "Chromista")
    choices = [choose(n, recs, (prefer or {}).get(n, default)) for n, recs in by_name.items()]
    asked: dict[int, tuple[str, str]] = {int(x): (str(x), "id") for x in ids}
    for c in choices:
        if c.aphia_id is not None:
            asked.setdefault(c.aphia_id, (c.name, "name"))
    records = client.records_by_ids(asked)
    valid = {int(r["valid_AphiaID"]) for r in records.values() if r.get("valid_AphiaID")}
    records.update(client.records_by_ids(valid - set(records)))
    lineages = {
        a: client.classification(int(records[a].get("valid_AphiaID") or a))
        for a in asked
        if a in records
    }
    ancestors = {lid for chain in lineages.values() for _r, _n, lid in chain}
    records.update(client.records_by_ids(ancestors - set(records)))
    rows: dict[int, dict] = {}
    for aphia, (query, origin) in sorted(asked.items()):
        rec = records.get(aphia)
        if rec is None:
            continue
        acc = int(rec.get("valid_AphiaID") or aphia)
        acc_rec = records.get(acc, rec)
        lineage = lineages[aphia]
        rows[aphia] = {
            "query": query,
            "aphia_id": aphia,
            "scientific_name": rec.get("scientificname") or "",
            "rank": rec.get("rank") or "",
            "status": rec.get("status") or "",
            "accepted_aphia_id": acc,
            "accepted_name": acc_rec.get("scientificname") or "",
            "lineage": json.dumps([list(x) for x in lineage]),
            "origin": origin,
            "retrieved_at": day,
        }
        for i, (rank, sname, lid) in enumerate(lineage):
            if lid not in rows and lid not in asked:
                anc = records.get(lid, {})
                rows[lid] = {
                    "query": "",
                    "aphia_id": lid,
                    "scientific_name": sname,
                    "rank": rank,
                    "status": anc.get("status") or "unknown",
                    "accepted_aphia_id": int(anc.get("valid_AphiaID") or lid),
                    "accepted_name": sname,
                    "lineage": json.dumps([list(x) for x in lineage[: i + 1]]),
                    "origin": "lineage",
                    "retrieved_at": day,
                }
    return [rows[k] for k in sorted(rows)], choices


def write_parquet(rows: list[dict], path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({c: [r[c] for r in rows] for c in SNAPSHOT_COLUMNS})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
