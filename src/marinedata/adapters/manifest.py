"""``manifest`` adapter (WP-6e-B): a prebuilt manifest (parquet or JSON lines) lists
``key``/``url`` (+ optional ``size``/``sha256``) per image and any per-sample columns.

``field_columns`` maps canonical schema fields (lat, lon, depth_m ...) to manifest
columns and ``label_columns`` names manifest columns kept as inline labels; both are
joined onto each decoded sample by item key. Used by imos-auv (dive track CSVs joined to
left-camera stills) and the MarineInst20M fetchable subset."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from . import BaseAdapter, Decoded, Fetched, RemoteItem

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fetch_remote(url: str, sha256: str | None = None) -> Path:
    """WP-6f: manifests > 10 MB live on S3 (D-AC), so ``manifest`` may be an https URL
    (anonymous GET on rs-storage-open). Cached per URL; ``manifest_sha256`` pins the bytes."""
    from ._http import download

    root = Path(os.environ.get("MARINEDATA_MANIFEST_CACHE", "~/.cache/marinedata/manifests"))
    dest = (
        root.expanduser() / hashlib.sha256(url.encode()).hexdigest()[:16] / url.rsplit("/", 1)[-1]
    )
    if dest.exists() and (not sha256 or hashlib.sha256(dest.read_bytes()).hexdigest() == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    got, _md5, _size = download(url, dest)
    if sha256 and got != sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"manifest {url}: sha256 {got} != pinned {sha256}")
    return dest


def _resolve(path: str, sha256: str | None = None) -> Path:
    if path.startswith(("https://", "http://")):
        return _fetch_remote(path, sha256)
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RowJoinMixin:
    """Join per-key manifest rows onto decoded samples (fields + labels)."""

    params: dict[str, Any]
    _rows: dict[str, Mapping[str, Any]]

    def _join(self, item: RemoteItem, decoded: Decoded) -> Decoded:
        row = self._rows.get(item.key)
        if not row:
            return decoded
        fmap: Mapping[str, str] = self.params.get("field_columns") or {}
        fields = {
            **{f: row[c] for f, c in fmap.items() if row.get(c) is not None},
            **decoded.fields,
        }
        labels = {
            **{
                c: str(row[c])
                for c in self.params.get("label_columns") or []
                if row.get(c) is not None
            },
            **decoded.labels,
        }
        return dataclasses.replace(decoded, fields=fields, labels=labels)

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        for decoded in super().decode(fetched):  # type: ignore[misc]
            yield self._join(fetched.item, decoded)


class ManifestAdapter(RowJoinMixin, BaseAdapter):
    name = "manifest"

    def __init__(self, params: Mapping[str, Any]) -> None:
        super().__init__(params)
        self._rows = {}
        self._path = _resolve(str(self.params["manifest"]), self.params.get("manifest_sha256"))

    def resolve_version(self) -> str:
        if self.params.get("version"):
            return str(self.params["version"])
        return "manifest-" + hashlib.sha256(self._path.read_bytes()).hexdigest()[:12]

    def list_items(self) -> Iterator[RemoteItem]:
        kc = self.params.get("key_column", "key")
        uc = self.params.get("url_column", "url")
        for row in read_manifest(self._path):
            key = str(row[kc])
            self._rows[key] = row
            size = row.get(self.params.get("size_column", "size"))
            yield RemoteItem(
                key=key,
                url=str(row[uc]),
                size=int(size) if size is not None else None,
                sha256=row.get(self.params.get("sha256_column", "sha256")) or None,
            )
