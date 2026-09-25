"""iNaturalist AWS Open Data, marine subset (``adapter: inat-open-data``, WP-6e-A).

The bucket ``s3://inaturalist-open-data`` is addressed by ``photo_id`` from its metadata
CSVs and can't be listed usefully, so this adapter reads a **manifest** (D-AB subset,
built once by ``scripts/inat_manifest.py`` from ``observations.csv.gz`` +
``photos.csv.gz``) and turns each row into a :class:`RemoteItem` for
``photos/<photo_id>/<size>.<ext>`` (anonymous HTTPS). ``size`` defaults to ``large``
(1024 px long edge, D-AE: ~0.3 TB instead of ~1.4 TB at ``original``); every sample keeps
``photo_id`` and the ``original`` URL so a hi-res pass can be added later.

Subset rule (D-AB, SPEC-w3): research-grade observations of the marine taxon set
(34 wholly-marine clades, or a mixed-clade species WoRMS marks ``isMarine``; the set is
``registry/ingest-specs/data/inat-marine-taxa-*.csv.gz``), one photo per observation
(lowest ``position``), and at most ``cap`` observations per ``taxon_id`` chosen by the
lowest ``sha256(observation_uuid)`` — deterministic, order-independent, no RNG.

Per sample: ``lat``/``lon``, ``capture_datetime`` (= ``observed_on``) and ``license`` go
into ``metadata.parquet`` (the D-K schema, ``upstream_url`` = the fetched ``large`` URL);
``taxon_id``, species, class, the observation uuid, ``photo_id`` and ``original_url`` go
into ``labels/image_labels.parquet`` (the schema has no taxon or second-URL field).
"""

from __future__ import annotations

import gzip
import hashlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import BaseAdapter, Decoded, Fetched, RemoteItem

BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com"
DEFAULT_PHOTO_SIZE = "large"  # D-AE: 1024 px long edge; `original` stays reachable via labels
MANIFEST_COLUMNS = (
    "photo_id",
    "extension",
    "license",
    "observation_uuid",
    "observer_id",
    "taxon_id",
    "latitude",
    "longitude",
    "observed_on",
    "width",
    "height",
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def obs_hash(observation_uuid: str) -> str:
    return hashlib.sha256(observation_uuid.encode()).hexdigest()


def load_marine_taxa(path: str | Path) -> dict[int, tuple[str, str]]:
    """``taxon_id -> (class, species)`` from the committed csv.gz (SPEC-w3's WoRMS rule)."""
    import pyarrow.csv as pacsv

    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    t = pacsv.read_csv(io.BytesIO(raw))
    return {
        int(tid): (str(c or ""), str(s or ""))
        for tid, c, s in zip(
            t["taxon_id"].to_pylist(),
            t["class"].to_pylist(),
            t["species"].to_pylist(),
            strict=True,
        )
    }


def select_subset(rows: Any, cap: int = 100) -> Any:
    """D-AB selection on a pandas frame of joined research-grade marine photo rows:
    one photo per observation (lowest position, then photo_id), then per taxon the ``cap``
    observations with the lowest ``sha256(observation_uuid)``. Returns a new frame sorted
    by (taxon_id, hash); the input is not mutated."""
    one = rows.sort_values(["observation_uuid", "position", "photo_id"]).drop_duplicates(
        "observation_uuid", keep="first"
    )
    one = one.assign(obs_sha=one["observation_uuid"].map(obs_hash))
    picked = one.sort_values(["taxon_id", "obs_sha"]).groupby("taxon_id", sort=False).head(cap)
    return picked.reset_index(drop=True)


def fetch_manifest(url: str, sha256: str | None, cache: Path | None = None) -> Path:
    """The durable (D-AE: S3 ``sources/<id>/_manifest/``) manifest, cached locally once.

    A cached copy is reused only if it matches the pinned ``sha256``; a fresh download that
    does not match raises (a runner must never draw from a different manifest)."""
    from ..fetch import cache_root
    from ._http import download

    dest = (cache or cache_root() / "_manifests") / url.rstrip("/").rsplit("/", 1)[-1]
    if dest.exists() and (not sha256 or _sha256_file(dest) == sha256):
        return dest
    got, _, _ = download(url, dest)
    if sha256 and got != sha256:
        dest.unlink()
        raise ValueError(f"manifest {url}: sha256 {got[:12]} != pinned {sha256[:12]}")
    return dest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class INatOpenDataAdapter(BaseAdapter):
    name = "inat-open-data"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._meta: dict[str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] | None = None
        self._digest = ""

    def _manifest_path(self) -> Path:
        raw = self.params.get("manifest")
        if not raw:
            raise ValueError("inat-open-data needs params.manifest (scripts/inat_manifest.py)")
        if str(raw).startswith(("https://", "http://")):
            return fetch_manifest(str(raw), self.params.get("manifest_sha256"))
        p = Path(str(raw)).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            import pyarrow.parquet as pq

            path = self._manifest_path()
            self._digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self._rows = pq.read_table(path).to_pylist()
        return self._rows

    def resolve_version(self) -> str:
        self._load()
        if self.params.get("version"):
            return str(self.params["version"])
        size = str(self.params.get("photo_size", DEFAULT_PHOTO_SIZE))
        tail = "" if size == DEFAULT_PHOTO_SIZE else f"-{size}"  # a hi-res pass is its own version
        return f"inat-{self.params.get('metadata_date', 'meta')}-{self._digest[:12]}{tail}"

    def list_items(self) -> Iterator[RemoteItem]:
        size = str(self.params.get("photo_size", DEFAULT_PHOTO_SIZE))
        base = str(self.params.get("base_url", BASE_URL)).rstrip("/")
        for r in self._load():
            ext = str(r.get("extension") or "jpg").lower().lstrip(".")
            photo_id = int(r["photo_id"])
            key = f"{photo_id}.{ext}"
            self._meta[key] = {
                "fields": {
                    "lat": r.get("latitude"),
                    "lon": r.get("longitude"),
                    "capture_datetime": r.get("observed_on"),
                    "license": r.get("license"),
                    "attribution": f"iNaturalist observer {r['observer_id']}"
                    if r.get("observer_id") is not None
                    else None,
                },
                "labels": {
                    k: str(r[k])
                    for k in ("taxon_id", "species", "class", "observation_uuid", "observed_on")
                    if r.get(k) not in (None, "")
                }
                | {
                    "photo_id": str(photo_id),
                    "original_url": f"{base}/photos/{photo_id}/original.{ext}",
                },
            }
            yield RemoteItem(key=key, url=f"{base}/photos/{photo_id}/{size}.{ext}")

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        meta = self._meta.get(fetched.item.key, {"fields": {}, "labels": {}})
        extra = {k: v for k, v in meta["fields"].items() if v not in (None, "")}
        for d in super().decode(fetched):
            yield Decoded(
                d.upstream_id,
                d.data,
                d.suffix,
                d.upstream_url or fetched.item.url,
                d.split_hint,
                {**d.fields, **extra},
                {**d.labels, **meta["labels"]},
                d.label_files,
            )
