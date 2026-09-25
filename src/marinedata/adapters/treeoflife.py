"""HF WebDataset member filter (``adapter: hf-member-filter``, WP-6e-A; TreeOfLife-10M).

TreeOfLife-10M ships 63 ``dataset/<src>/image_set_NN.tar.gz`` shards (~32 GB each) of
``<treeoflife_id>.jpg`` members; only ~5% are marine and they are scattered over every
shard. The adapter pins the HF revision (and refuses a gated repo, D-E/D-AA, via
:class:`~marinedata.adapters.hf.HFAdapter`), streams ``metadata/catalog.csv`` once to
build the keep-set, then streams each shard and reads ONLY the kept members — the rest
are skipped by the streaming tar reader without being buffered.

Keep rule (D-AB / SPEC-w3): a catalog row whose phylum/class/order/family/genus is one of
the wholly-marine clades, or whose binomial is in the iNat/WoRMS marine species set;
BIOSCAN rows never qualify; at most ``cap_per_species`` rows per (data_source, species),
chosen by the lowest ``sha256(treeoflife_id)``.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from typing import Any

from . import Decoded, Fetched
from ._http import open_url
from .decode import IMAGE_SUFFIXES
from .hf import HFAdapter
from .inat import load_marine_taxa

MARINE_CLADES = (
    "Anthozoa", "Scyphozoa", "Cubozoa", "Staurozoa", "Echinodermata", "Ctenophora",
    "Tunicata", "Cephalopoda", "Polyplacophora", "Scaphopoda", "Elasmobranchii",
    "Holocephali", "Myxini", "Cetacea", "Phocidae", "Otariidae", "Odobenidae", "Sirenia",
    "Cheloniidae", "Dermochelyidae", "Brachiopoda", "Chaetognatha", "Sipuncula",
    "Hemichordata", "Cephalochordata", "Thecostraca", "Pycnogonida", "Xiphosura",
    "Polychaeta", "Phaeophyceae", "Nudibranchia", "Zosteraceae", "Posidoniaceae",
    "Cymodoceaceae",
)  # fmt: skip
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
DEFAULT_TAXA = "registry/ingest-specs/data/inat-marine-taxa-2026-09-25.csv.gz"


def data_source(row: dict[str, Any]) -> str:
    if (row.get("eol_content_id") or "").strip():
        return "eol"
    if (row.get("inat21_filename") or "").strip():
        return "inat21"
    return "bioscan"


def binomial(row: dict[str, Any]) -> str:
    sp = (row.get("species") or "").strip()
    return sp if " " in sp else f"{(row.get('genus') or '').strip()} {sp}".strip()


def select_marine(
    rows: Iterable[dict[str, Any]], clades: set[str], species: set[str], cap: int
) -> dict[str, dict[str, str]]:
    """``treeoflife_id -> labels`` for the kept rows (pure; see the module rule)."""
    by_stratum: dict[tuple[str, str], list[tuple[str, str, dict[str, str]]]] = {}
    for row in rows:
        src = data_source(row)
        if src == "bioscan":
            continue
        bn = binomial(row)
        rule = (
            "clade"
            if any((row.get(r) or "") in clades for r in RANKS[1:6])
            else ("worms-species" if bn in species else None)
        )
        if rule is None:
            continue
        tid = str(row["treeoflife_id"])
        labels = {r: str(row[r]) for r in RANKS if row.get(r)}
        labels.update(data_source=src, binomial=bn, marine_rule=rule)
        by_stratum.setdefault((src, bn), []).append(
            (hashlib.sha256(tid.encode()).hexdigest(), tid, labels)
        )
    keep: dict[str, dict[str, str]] = {}
    for members in by_stratum.values():
        for _, tid, labels in sorted(members)[:cap]:
            keep[tid] = labels
    return keep


def _catalog_rows(stream: io.RawIOBase | Any) -> Iterator[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    cols = ["treeoflife_id", "eol_content_id", "inat21_filename", *RANKS]
    rd = pacsv.open_csv(
        pa.PythonFile(stream, mode="r"),
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(
            include_columns=cols, column_types={c: pa.string() for c in cols}
        ),
        parse_options=pacsv.ParseOptions(invalid_row_handler=lambda r: "skip"),
    )
    for batch in rd:
        yield from batch.to_pylist()


class HFMemberFilterAdapter(HFAdapter):
    name = "hf-member-filter"

    def __init__(self, params: dict) -> None:
        params = {"include": ["dataset/*.tar.gz"], **dict(params)}
        super().__init__(params)
        self._keep: dict[str, dict[str, str]] | None = None

    def keep_set(self) -> dict[str, dict[str, str]]:
        if self._keep is None:
            if not hasattr(self, "sha"):
                self.resolve_version()
            cat = str(self.params.get("catalog", "metadata/catalog.csv"))
            url = f"{self._base}/datasets/{self._repo}/resolve/{self.sha}/{cat}"
            clades = set(self.params.get("marine_clades") or MARINE_CLADES)
            taxa = load_marine_taxa(str(self.params.get("marine_taxa") or DEFAULT_TAXA))
            species = {s for _, s in taxa.values() if s}
            cap = int(self.params.get("cap_per_species") or 500)
            with open_url(url, timeout=300) as resp:
                raw = io.BufferedReader(resp, 8 << 20)  # type: ignore[arg-type]
                self._keep = select_marine(_catalog_rows(raw), clades, species, cap)
        return self._keep

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        keep = self.keep_set()
        assert fetched.stream is not None, "shards are streamed, never spooled"
        with tarfile.open(fileobj=fetched.stream, mode="r|*") as tar:  # type: ignore[call-overload]
            for m in tar:
                name = PurePosixPath(m.name)
                suffix = name.suffix.lower()
                tid = name.name.split(".", 1)[0]
                if not m.isfile() or suffix not in IMAGE_SUFFIXES or tid not in keep:
                    continue  # skipped without reading: r| tar seeks past the payload
                handle = tar.extractfile(m)
                if handle is None:
                    continue
                yield Decoded(
                    upstream_id=f"{fetched.item.key}#{m.name}",
                    data=handle.read(),
                    suffix=suffix,
                    upstream_url=fetched.item.url,
                    labels={"treeoflife_id": tid, **keep[tid]},
                )
