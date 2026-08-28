"""Probing an external dataset host to draft a registry entry.

``marinedata add <hf-id>`` is not registration. It probes the HuggingFace Hub API and
the datasets-server (the same endpoints ``fetch.py`` already reads to *fetch* a
dataset) to infer a layout, columns and a licence *signal*, then prints a draft YAML
entry for a human to review, correct and verify. It never writes into ``registry/``
and never asserts a licence as checked: a probed tag is a dataset-card claim at best,
and this project's own history has a licence being misrecorded on exactly that basis
(see CONTRIBUTING.md's first rule). The draft's ``verification.method`` is always
``dataset-card`` with ``verified_by`` pointing back at what was actually read, never
``primary`` — only a human upgrading it after actually opening a licence file earns
that.

Scoped to HuggingFace only, deliberately: it is the one host with a documented,
machine-readable schema (the datasets-server ``/info`` endpoint) that can be probed
without guessing. Nothing here fetches sample data or writes to disk.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field

from .fetch import DATASETS_SERVER, FetchError, _get_json

HUB_API = "https://huggingface.co/api/datasets"


@dataclass(frozen=True)
class ProbeResult:
    """What could be inferred, plus the draft text and open questions for a human."""

    hf_id: str
    modalities: tuple[str, ...]
    layout: str
    loader_params: dict[str, str]
    licence_tag: str | None
    item_count: int | None
    columns: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    yaml_draft: str = ""


def _parse_hf_id(hf_id_or_url: str) -> str:
    """Accept either ``owner/name`` or a full huggingface.co dataset URL."""
    if hf_id_or_url.startswith(("http://", "https://")):
        path = urllib.parse.urlparse(hf_id_or_url).path
        return path.removeprefix("/datasets/").strip("/")
    return hf_id_or_url.strip("/")


def _first_feature(features: dict, type_names: tuple[str, ...]) -> str | None:
    for name, spec in features.items():
        if spec.get("_type") in type_names:
            return name
    return None


def _infer_layout(features: dict) -> tuple[str, dict[str, str], tuple[str, ...]]:
    """Layout, loader params, and warnings — mirrors exactly what
    ``fetch._fetch_huggingface`` does with these same column roles at fetch time, so a
    drafted entry is guaranteed fetchable, not just plausible-looking.
    """
    warnings: list[str] = []
    audio_col = _first_feature(features, ("Audio",))
    if audio_col:
        return "audio-clips", {"audio_column": audio_col}, ()

    image_col = _first_feature(features, ("Image",))
    if image_col is None:
        warnings.append(
            "no Image or Audio column found — this dataset cannot be fetched via the "
            "rows API as declared. Check whether it needs fetch_style: files instead "
            "(see sweet-corals for that pattern), or is metadata-only."
        )
        return "metadata-only", {}, tuple(warnings)

    label_col = _first_feature(
        {k: v for k, v in features.items() if k != image_col}, ("ClassLabel",)
    )
    other_image_cols = [
        name
        for name, spec in features.items()
        if spec.get("_type") == "Image" and name != image_col
    ]

    params: dict[str, str] = {"image_column": image_col}
    if other_image_cols:
        mask_col = other_image_cols[0]
        params["mask_column"] = mask_col
        if label_col:
            warnings.append(
                f"both a second Image column ('{mask_col}') and a ClassLabel column "
                f"('{label_col}') are present — drafted as image-mask-pairs and kept "
                f"the label column out of loader params; decide which one this "
                f"dataset actually is before shipping the entry."
            )
        return "image-mask-pairs", params, tuple(warnings)

    if label_col:
        params["label_column"] = label_col
        return "image-folder", params, tuple(warnings)

    return "flat-images", params, tuple(warnings)


def _build_draft(hf_id: str, hub_json: dict, info_json: dict) -> ProbeResult:
    """Pure function over already-fetched JSON — kept separate from the network calls
    in :func:`probe_huggingface` so the inference logic is unit-testable without it."""
    warnings: list[str] = []

    card = hub_json.get("cardData") or {}
    licence_tag = card.get("license")
    if not licence_tag:
        for tag in hub_json.get("tags", []):
            if isinstance(tag, str) and tag.startswith("license:"):
                licence_tag = tag.removeprefix("license:")
                break
    if not licence_tag:
        warnings.append("no licence tag found on the dataset card — verify manually before adding.")

    dataset_info = info_json.get("dataset_info")
    if isinstance(dataset_info, dict) and "features" not in dataset_info:
        # Multi-config datasets nest one level deeper, keyed by config name.
        dataset_info = dataset_info.get("default") or next(iter(dataset_info.values()), {})
    dataset_info = dataset_info or {}

    features = dataset_info.get("features", {})
    layout, loader_params, layout_warnings = _infer_layout(features)
    warnings.extend(layout_warnings)

    item_count = None
    splits = dataset_info.get("splits", {})
    if splits:
        item_count = sum(s.get("num_examples", 0) for s in splits.values())

    modalities = []
    if any(spec.get("_type") == "Image" for spec in features.values()):
        modalities.append("image")
    if any(spec.get("_type") == "Audio" for spec in features.values()):
        modalities.append("audio")
    if not modalities:
        modalities.append("image")  # best available default; flagged via warnings above

    slug = hf_id.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    access_params = {"hf_id": hf_id, **loader_params}
    loader_params_yaml = (
        ", ".join(f"{k}: {v}" for k, v in loader_params.items()) if loader_params else ""
    )
    access_params_yaml = ", ".join(f"{k}: {v}" for k, v in access_params.items())
    licence_line = "NO-LICENCE-STATED" if not licence_tag else "TODO-map-" + licence_tag
    licence_claim = licence_tag or "none found"
    warning_lines = "\n".join(f"# WARNING: {w}" for w in warnings)
    loader_params_line = f"      params: {{ {loader_params_yaml} }}" if loader_params_yaml else ""
    items_line = item_count if item_count is not None else "TODO"

    draft = f"""\
# DRAFT — probed {hf_id!r} on huggingface.co. Every value below needs a human review
# pass before this is a real registry entry. In particular:
#   - licence: {licence_tag or "UNKNOWN"!r} is a dataset-card claim, not a primary
#     source (CONTRIBUTING.md rule #1). Open the actual licence file/terms page and
#     update `verified_by` before raising this out of NO-LICENCE-STATED / TDM_ONLY.
#   - legal_basis and provenance below are placeholders — set deliberately.
#   - capabilities, coverage, and annotations.supervises must be set by a human; a
#     probe cannot know what the labels mean scientifically.
{warning_lines}
  - id: {slug}
    name: "{hf_id}"
    version: "unversioned"
    description: TODO
    licence: {licence_line}
    legal_basis: unknown
    provenance: public
    verification:
      verified_on: TODO-today's-date
      verified_by: >
        HuggingFace card {hf_id} states license:{licence_claim} —
        SECONDARY, verify against the actual licence
      method: dataset-card
    access:
      method: huggingface
      uri: https://huggingface.co/datasets/{hf_id}
      params: {{ {access_params_yaml} }}
    modalities: [{", ".join(modalities)}]
    capabilities: [TODO]
    items: {items_line}
    annotations:
      - kind: TODO
        supervises: []
    coverage:
      regions: [TODO]
    loader:
      layout: {layout}
{loader_params_line}
    tags: [needs-review]
"""

    return ProbeResult(
        hf_id=hf_id,
        modalities=tuple(modalities),
        layout=layout,
        loader_params=loader_params,
        licence_tag=licence_tag,
        item_count=item_count,
        columns=tuple(features),
        warnings=tuple(warnings),
        yaml_draft=draft,
    )


def probe_huggingface(hf_id_or_url: str) -> ProbeResult:
    """Probe a HuggingFace dataset and draft a registry entry. Fetches nothing but
    metadata — no sample data, no writes to disk, no registry mutation.

    Raises:
        FetchError: the dataset doesn't exist, or neither API responds usefully.
    """
    hf_id = _parse_hf_id(hf_id_or_url)
    if "/" not in hf_id:
        raise FetchError(
            f"'{hf_id_or_url}' does not look like an owner/name HuggingFace dataset id"
        )

    hub_json = _get_json(f"{HUB_API}/{urllib.parse.quote(hf_id, safe='/')}")
    if "error" in hub_json:
        raise FetchError(f"{hf_id}: {hub_json['error']}")

    info_json = _get_json(f"{DATASETS_SERVER}/info?dataset={urllib.parse.quote(hf_id)}")
    if "error" in info_json:
        # The dataset exists but datasets-server can't process it (as with CoralVQA) —
        # still worth drafting an entry, just with an explicit warning instead of a
        # column inference.
        result = _build_draft(hf_id, hub_json, {})
        extra_warning = (
            f"datasets-server could not process this dataset ({info_json['error']}) — "
            f"columns and layout cannot be inferred automatically. Check the repo's "
            f"file list directly (fetch_style: files may be the right access pattern; "
            f"see sweet-corals or coralvqa for that pattern)."
        )
        return ProbeResult(
            hf_id=result.hf_id,
            modalities=result.modalities,
            layout=result.layout,
            loader_params=result.loader_params,
            licence_tag=result.licence_tag,
            item_count=result.item_count,
            columns=result.columns,
            warnings=(extra_warning, *result.warnings),
            yaml_draft=result.yaml_draft,
        )

    return _build_draft(hf_id, hub_json, info_json)
