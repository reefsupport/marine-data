"""HuggingFace fetchers: datasets-server rows, or a plain repo file tree."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from ..models import Source
from .constants import DATASETS_SERVER
from .core import FetchError, FetchResult
from .integrity import _write
from .transport import _get, _get_json


def _class_dir(value: object, names: list[str] | None) -> str:
    """Directory name for a class label, resolving ClassLabel integers to their names.

    A label of ``74`` becomes ``Sparisoma_viride`` when the feature schema supplies the
    names. Without this the on-disk vocabulary is a set of integers that no crosswalk can
    match and no reviewer can read.
    """
    if names is not None and isinstance(value, int) and 0 <= value < len(names):
        raw = str(names[value])
    else:
        raw = str(value)
    # Directory-safe, but preserve the name — it IS the label.
    return raw.replace("/", "_").replace("\\", "_").strip() or "unlabelled"


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

    # ClassLabel columns come back as INTEGERS; the human-readable names live in the
    # feature schema. Materialising the integer as a directory name produces label
    # vocabularies like {"13", "74", "96"} — structurally valid, semantically useless,
    # and impossible to crosswalk. Resolve them here.
    class_names: dict[str, list[str]] = {}
    for feature in payload.get("features", []):
        ftype = feature.get("type", {})
        if ftype.get("_type") == "ClassLabel" and ftype.get("names"):
            class_names[feature["name"]] = list(ftype["names"])

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
            label = (
                _class_dir(row.get(label_col), class_names.get(str(label_col)))
                if label_col and row.get(label_col) is not None
                else "unlabelled"
            )
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
            _write(
                root / _class_dir(row[label_col], class_names.get(str(label_col))) / f"{stem}.png",
                _get(src),
            )
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


def _fetch_hf(source: Source, root: Path, limit: int) -> FetchResult:
    """Dispatch between the parquet dataset API and a plain repo file tree.

    Both live behind ``method: huggingface``; ``fetch_style: files`` selects the tree.
    """
    if str(source.access.params.get("fetch_style", "rows")) == "files":
        from ..fetchers_remote import fetch_hf_files

        return fetch_hf_files(source, root, limit)
    return _fetch_huggingface(source, root, limit)
