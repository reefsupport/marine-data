"""Taxonomic resolution against OBIS / WoRMS.

**This module is codegen, not runtime.** Nothing in the load path imports it. A
maintainer runs ``marinedata taxa`` locally, it rewrites YAML, and the diff is reviewed.
The committed YAML is the pin — see ``tests/test_registry.py::
test_registry_loads_with_the_network_hard_down``, which fails if a network call ever
creeps into loading, harmonisation or gating.

Why OBIS rather than GBIF: OBIS's ``taxonID`` **is** the WoRMS AphiaID, so one integer
joins us to OBIS occurrences, WoRMS classification, and every other WoRMS-backed marine
dataset, at every rank. WoRMS is the authoritative register for marine taxa; GBIF's
backbone is broader but less careful about them.

Why the frozen fields exist: three of the first eight AphiaIDs entered here by hand were
wrong — one pointed at an unaccepted red-alga family, one at a bryozoan. A bare integer
looks identical whether it is right or wrong. Freezing the scientific name, rank and
status alongside makes an error visible in review and detectable by a drift check.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date

OBIS_TAXON = "https://api.obis.org/v3/taxon"
WORMS_BY_ID = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID"
WORMS_BY_NAME = "https://www.marinespecies.org/rest/AphiaRecordsByName"
USER_AGENT = "marinedata/0.1 (+https://github.com/reefsupport/marine-data)"

RANK_ORDER = (
    "Kingdom",
    "Phylum",
    "Subphylum",
    "Class",
    "Subclass",
    "Order",
    "Suborder",
    "Family",
    "Subfamily",
    "Genus",
    "Subgenus",
    "Species",
    "Subspecies",
)
"""Linnaean ranks, coarse to fine. Informal ranks resolve to position -1."""


class TaxonError(Exception):
    """A name or id could not be resolved."""


@dataclass(frozen=True)
class TaxonRecord:
    """One resolved taxon, as both OBIS and WoRMS understand it."""

    aphia_id: int
    scientific_name: str
    rank: str
    status: str
    accepted_aphia_id: int | None = None
    accepted_name: str | None = None
    classification: dict[str, str] = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def rank_position(self) -> int:
        try:
            return RANK_ORDER.index(self.rank)
        except ValueError:
            return -1

    @property
    def obis_url(self) -> str:
        return f"https://obis.org/taxon/{self.aphia_id}"

    def yaml_fields(self, *, checked_on: date | None = None) -> dict[str, object]:
        """The fields a LabelNode should freeze for this taxon."""
        out: dict[str, object] = {
            "worms_aphia_id": self.accepted_aphia_id or self.aphia_id,
            "worms_scientificname": self.accepted_name or self.scientific_name,
            "worms_rank": self.rank,
            "worms_status": "accepted" if self.accepted_aphia_id else self.status,
            "worms_checked_on": (checked_on or date.today()).isoformat(),
        }
        if self.accepted_aphia_id and self.accepted_aphia_id != self.aphia_id:
            # Keep the trail: a source that asserted the old name still resolves.
            out["worms_aphia_id_asserted"] = self.aphia_id
        return out


def _get(url: str, *, timeout: int = 30, retries: int = 2) -> object:
    """GET and parse JSON, with a courteous retry.

    These are small public academic services. Back off rather than hammer them.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = exc
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise TaxonError(f"could not fetch {url}: {last}")


def _from_obis(record: dict) -> TaxonRecord:
    accepted = record.get("acceptedNameUsageID")
    aphia = int(record["taxonID"])
    classification = {
        rank: record[key]
        for rank, key in (
            ("Kingdom", "kingdom"),
            ("Phylum", "phylum"),
            ("Class", "class"),
            ("Order", "order"),
            ("Family", "family"),
            ("Genus", "genus"),
        )
        if record.get(key)
    }
    return TaxonRecord(
        aphia_id=aphia,
        scientific_name=record.get("scientificName", ""),
        rank=record.get("taxonRank", "") or "",
        status=record.get("taxonomicStatus") or ("accepted" if accepted == aphia else ""),
        accepted_aphia_id=int(accepted) if accepted and int(accepted) != aphia else None,
        accepted_name=record.get("acceptedNameUsage"),
        classification=classification,
    )


def _from_worms(record: dict) -> TaxonRecord:
    aphia = int(record["AphiaID"])
    valid = record.get("valid_AphiaID")
    classification = {
        rank: record[key]
        for rank, key in (
            ("Kingdom", "kingdom"),
            ("Phylum", "phylum"),
            ("Class", "class"),
            ("Order", "order"),
            ("Family", "family"),
            ("Genus", "genus"),
        )
        if record.get(key)
    }
    return TaxonRecord(
        aphia_id=aphia,
        scientific_name=record.get("scientificname", ""),
        rank=record.get("rank", "") or "",
        status=record.get("status", "") or "",
        accepted_aphia_id=int(valid) if valid and int(valid) != aphia else None,
        accepted_name=record.get("valid_name"),
        classification=classification,
    )


def resolve_id(aphia_id: int) -> TaxonRecord | None:
    """Look up an AphiaID. Returns ``None`` if no such record exists.

    Uses WoRMS directly: it is the register of record, and reports ``status`` and
    ``valid_AphiaID`` more reliably than the OBIS mirror for unaccepted names.
    """
    record = _get(f"{WORMS_BY_ID}/{aphia_id}")
    return _from_worms(record) if isinstance(record, dict) else None


class AmbiguousName(TaxonError):
    """A name matches more than one accepted taxon and no expectation disambiguated it.

    Homonyms across kingdoms are common and dangerous. *Turbinaria* is a scleractinian
    coral genus (Dendrophylliidae, AphiaID 206641) **and** a brown alga genus
    (Sargassaceae, AphiaID 206630). Silently taking the first match put a brown alga in
    a coral schema on the first run of this tool. Raising is the only safe default.
    """

    def __init__(self, name: str, candidates: list[TaxonRecord]) -> None:
        self.name = name
        self.candidates = candidates
        detail = "; ".join(
            f"{c.aphia_id} {c.scientific_name} ({c.classification.get('Phylum', '?')}"
            f"/{c.classification.get('Class', '?')})"
            for c in candidates
        )
        super().__init__(
            f"'{name}' matches {len(candidates)} accepted taxa: {detail}. "
            f"Disambiguate with expect_phylum= or expect_class=, or pass the AphiaID."
        )


def resolve_name(
    name: str,
    *,
    expect_phylum: str | None = None,
    expect_class: str | None = None,
    marine_only: bool = True,
) -> TaxonRecord | None:
    """Resolve a scientific name to a single taxon, or raise if it is ambiguous.

    Args:
        expect_phylum: required phylum, e.g. ``"Cnidaria"``. Use it whenever the name
            could be a homonym — which for genus names is more often than you would
            like.
        expect_class: required class, e.g. ``"Hexacorallia"``.

    Raises:
        AmbiguousName: more than one accepted match survives the expectations. Better a
            loud failure than a brown alga filed as a coral.

    Note:
        WoRMS is queried rather than OBIS here. OBIS returns only its best match, which
        *hides* ambiguity — exactly the failure this function exists to prevent.
    """
    quoted = urllib.parse.quote(name)
    params = urllib.parse.urlencode({"like": "false", "marine_only": str(marine_only).lower()})
    payload = _get(f"{WORMS_BY_NAME}/{quoted}?{params}")

    if not isinstance(payload, list) or not payload:
        # Fall back to OBIS: it covers a few names WoRMS name-search misses.
        payload = _get(f"{OBIS_TAXON}/{quoted}")
        if isinstance(payload, dict) and payload.get("results"):
            return _from_obis(payload["results"][0])
        return None

    records = [_from_worms(r) for r in payload]
    accepted = [r for r in records if r.is_accepted] or records

    if expect_phylum:
        accepted = [r for r in accepted if r.classification.get("Phylum") == expect_phylum]
    if expect_class:
        accepted = [r for r in accepted if r.classification.get("Class") == expect_class]

    if not accepted:
        return None
    if len(accepted) > 1:
        raise AmbiguousName(name, accepted)
    return accepted[0]


@dataclass(frozen=True)
class Drift:
    """A committed node disagreeing with the authority today."""

    node_id: str
    field: str
    committed: object
    authority: object

    def line(self) -> str:
        return (
            f"  {self.node_id:<24} {self.field:<22} "
            f"committed={self.committed!r}  authority={self.authority!r}"
        )


def check_node(node, record: TaxonRecord | None) -> list[Drift]:
    """Compare a committed node against a freshly resolved record.

    ``record is None`` means the AphiaID does not exist at all — which is a harder
    failure than drift and is reported as such.
    """
    if record is None:
        return [Drift(node.id, "worms_aphia_id", node.worms_aphia_id, "NO SUCH RECORD")]

    out: list[Drift] = []
    expected = {
        "worms_scientificname": record.accepted_name or record.scientific_name,
        "worms_rank": record.rank,
        "worms_status": "accepted" if record.accepted_aphia_id else record.status,
    }
    for field_name, authority_value in expected.items():
        committed = getattr(node, field_name, None)
        if committed is not None and committed != authority_value:
            out.append(Drift(node.id, field_name, committed, authority_value))

    if record.accepted_aphia_id:
        out.append(Drift(node.id, "worms_aphia_id", node.worms_aphia_id, record.accepted_aphia_id))
    return out
