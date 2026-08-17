"""Bounded sample fetching.

The purpose is verification, not bulk download. Roughly 100 items per source is enough
to prove that a declared layout matches reality — which is the failure mode synthetic
fixtures cannot catch. A fixture proves ``ImageMaskPairLoader`` works; it says nothing
about whether *this dataset* is actually laid out as ``images/`` + ``masks/``.

That distinction is not academic. Coralscapes was declared here as image/mask
directories; it is in fact HuggingFace parquet with ``image`` and ``label`` columns. The
loader was fine and the declaration was wrong, and only real data reveals which.

Fetchers **normalise** into the local layout the source declares, so the fetcher owns
source-specific unpacking and the loader stays generic.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .enums import AccessMethod
from .models import Source

DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_LIMIT = 100
USER_AGENT = "marinedata/0.1 (+https://github.com/reefsupport/marine-data)"


class FetchError(Exception):
    """A sample could not be retrieved."""


class FetchNotSupported(FetchError):
    """No fetcher exists for this access method.

    Raised rather than silently returning nothing, so "we cannot automate this" is
    distinguishable from "this dataset is empty". Gated and request-only sources land
    here by design — they require a human to accept terms, which is exactly the
    ``contract_gated`` situation the licence model already tracks.
    """


def cache_root() -> Path:
    """Cache location. ``MARINEDATA_CACHE`` overrides."""
    env = os.environ.get("MARINEDATA_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "marinedata"


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    root: Path
    items: int
    method: str
    truncated: bool
    """True when the source has more items than were fetched — always, for a sample."""

    def manifest(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "items": self.items,
            "method": self.method,
            "truncated": self.truncated,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "note": "Bounded verification sample. NOT the full dataset.",
        }


def _get(url: str, *, timeout: int = 60, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"network error for {url}: {exc.reason}") from exc


def _get_json(url: str, *, timeout: int = 60) -> dict:
    try:
        return json.loads(_get(url, timeout=timeout))
    except json.JSONDecodeError as exc:
        raise FetchError(f"non-JSON response from {url}") from exc


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fetch_huggingface(source: Source, root: Path, limit: int) -> FetchResult:
    """Pull rows via the HF datasets-server and materialise them locally.

    Params (on ``access.params``):
        hf_id: ``owner/name``. Falls back to parsing ``access.uri``.
        config, split: default ``"default"`` / ``"train"``
        image_column: column holding the image. Default ``"image"``
        mask_column: optional column holding a segmentation mask.
        label_column: optional column holding a class label (written as directories).
    """
    params = dict(source.access.params)
    hf_id = str(params.get("hf_id") or "")
    if not hf_id and source.access.uri:
        hf_id = urllib.parse.urlparse(source.access.uri).path.removeprefix("/datasets/").strip("/")
    if not hf_id:
        raise FetchError(f"{source.id}: cannot determine the HuggingFace dataset id")

    config = str(params.get("config", "default"))
    split = str(params.get("split", "train"))
    image_col = str(params.get("image_column", "image"))
    mask_col = params.get("mask_column")
    label_col = params.get("label_column")
    audio_col = params.get("audio_column")

    # The rows endpoint caps `length` at 100 — which is the sample size we want anyway.
    query = urllib.parse.urlencode(
        {
            "dataset": hf_id,
            "config": config,
            "split": split,
            "offset": 0,
            "length": min(limit, 100),
        }
    )
    payload = _get_json(f"{DATASETS_SERVER}/rows?{query}")
    if "error" in payload:
        raise FetchError(f"{source.id}: datasets-server error — {payload['error']}")

    rows = payload.get("rows", [])
    if not rows:
        raise FetchError(f"{source.id}: datasets-server returned no rows for {hf_id}")

    count = 0
    for index, entry in enumerate(rows):
        row = entry.get("row", {})

        if audio_col:
            # Audio rows expose a list of encodings; take the first playable one.
            media = row.get(audio_col)
            if isinstance(media, list) and media:
                media = media[0]
            audio_src = media.get("src") if isinstance(media, dict) else None
            if not audio_src:
                continue
            label = str(row.get(label_col, "unlabelled")) if label_col else "unlabelled"
            suffix = Path(urllib.parse.urlparse(audio_src).path).suffix or ".wav"
            _write(root / label / f"{index:05d}{suffix}", _get(audio_src))
            count += 1
            continue

        image = row.get(image_col)
        src = image.get("src") if isinstance(image, dict) else None
        if not src:
            continue

        stem = f"{index:05d}"
        if label_col and row.get(label_col) is not None:
            # Class-per-directory at the root — the `image-folder` layout.
            _write(root / str(row[label_col]) / f"{stem}.png", _get(src))
        else:
            # Parallel images/ + masks/ — the `image-mask-pairs` layout.
            _write(root / "images" / f"{stem}.png", _get(src))

        if mask_col:
            mask = row.get(mask_col)
            mask_src = mask.get("src") if isinstance(mask, dict) else None
            if mask_src:
                _write(root / "masks" / f"{stem}.png", _get(mask_src))
        count += 1

    if count == 0:
        available = sorted(rows[0].get("row", {}))
        wanted_col = str(audio_col) if audio_col else image_col
        present = wanted_col in available
        reason = (
            f"column '{wanted_col}' exists but carries no resolvable media `src` "
            f"(the upstream datasets-server may be failing to post-process this dataset)"
            if present
            else f"no column named '{wanted_col}'"
        )
        raise FetchError(f"{source.id}: {reason} in {hf_id}. Columns: {available}")

    return FetchResult(source.id, root, count, "huggingface", truncated=True)


def _fetch_http(source: Source, root: Path, limit: int) -> FetchResult:
    """Retrieve a single declared sample archive or file.

    Params:
        sample_url: a small, directly-downloadable file. Required — we deliberately do
            not crawl arbitrary pages looking for data.
    """
    sample_url = source.access.params.get("sample_url")
    if not sample_url:
        raise FetchNotSupported(
            f"{source.id}: no `sample_url` declared. Add one pointing at a small "
            f"downloadable sample, or fetch the data manually."
        )
    payload = _get(str(sample_url))
    name = Path(urllib.parse.urlparse(str(sample_url)).path).name or "sample.bin"
    _write(root / name, payload)
    return FetchResult(source.id, root, 1, "http", truncated=True)


def _fetch_hf(source: Source, root: Path, limit: int) -> FetchResult:
    """Dispatch between the parquet dataset API and a plain repo file tree.

    Both live behind ``method: huggingface``; ``fetch_style: files`` selects the tree.
    """
    if str(source.access.params.get("fetch_style", "rows")) == "files":
        from .fetchers_remote import fetch_hf_files

        return fetch_hf_files(source, root, limit)
    return _fetch_huggingface(source, root, limit)


def _fetch_s3(source: Source, root: Path, limit: int) -> FetchResult:
    from .fetchers_remote import fetch_s3

    return fetch_s3(source, root, limit)


_FETCHERS = {
    AccessMethod.HUGGINGFACE: _fetch_hf,
    AccessMethod.HTTP: _fetch_http,
    AccessMethod.ZENODO: _fetch_http,
    AccessMethod.S3: _fetch_s3,
}


def fetch_sample(
    source: Source,
    *,
    limit: int = DEFAULT_LIMIT,
    root: Path | None = None,
    force: bool = False,
) -> FetchResult:
    """Fetch a bounded verification sample.

    Args:
        source: Registry entry to fetch.
        limit: Maximum items. Kept small on purpose — this verifies layout, not scale.
        root: Destination. Defaults to ``<cache>/<source_id>``.
        force: Re-fetch even if a sample is already cached.

    Raises:
        FetchNotSupported: gated, request-only, S3 and scrape sources need a human.
        FetchError: the fetch failed or produced nothing.
    """
    target = root or (cache_root() / source.id)
    marker = target / "_fetch.json"

    if marker.is_file() and not force:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        return FetchResult(
            source.id,
            target,
            int(existing.get("items", 0)),
            str(existing.get("method", "cache")),
            truncated=True,
        )

    fetcher = _FETCHERS.get(source.access.method)
    if fetcher is None:
        raise FetchNotSupported(
            f"{source.id}: access method '{source.access.method.value}' cannot be "
            f"fetched automatically. "
            + (
                "It is gated — a human must accept the terms first."
                if source.access.gated
                else "Retrieve it manually and point the loader at the local path."
            )
        )

    target.mkdir(parents=True, exist_ok=True)
    result = fetcher(source, target, limit)
    marker.write_text(json.dumps(result.manifest(), indent=2), encoding="utf-8")
    return result


def sample_digest(root: Path) -> str:
    """Stable digest of a fetched sample, for reproducibility in test reports."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "_fetch.json"):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return "sha256:" + digest.hexdigest()[:16]
