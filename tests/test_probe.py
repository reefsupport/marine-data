"""`marinedata add` — drafting a registry entry from a probed HuggingFace dataset.

Tests exercise `_build_draft` directly against synthetic Hub/datasets-server JSON, the
same shape `probe_huggingface` fetches over the network — no network here, matching how
`fetch.py`'s own column-inference logic (image_column/mask_column/label_column) is
tested. The inference rules mirror `_fetch_huggingface` exactly on purpose: a drafted
entry must describe a dataset the existing fetcher can actually read, not merely one
that looks plausible.
"""

from __future__ import annotations

from marinedata.probe import ProbeResult, _build_draft, _parse_hf_id


def test_parses_bare_id() -> None:
    assert _parse_hf_id("owner/name") == "owner/name"


def test_parses_full_url() -> None:
    assert _parse_hf_id("https://huggingface.co/datasets/owner/name") == "owner/name"
    assert _parse_hf_id("https://huggingface.co/datasets/owner/name/viewer") == "owner/name/viewer"


def test_image_and_label_column_drafts_image_folder() -> None:
    info = {
        "dataset_info": {
            "features": {
                "image": {"_type": "Image"},
                "label": {"_type": "ClassLabel", "names": ["Hard Coral", "Soft Coral"]},
            },
            "splits": {"train": {"num_examples": 1000}},
        }
    }
    result = _build_draft("owner/classify", {"cardData": {"license": "cc-by-4.0"}}, info)
    assert result.layout == "image-folder"
    assert result.loader_params == {"image_column": "image", "label_column": "label"}
    assert result.item_count == 1000
    assert result.licence_tag == "cc-by-4.0"
    assert "image" in result.modalities


def test_two_image_columns_drafts_image_mask_pairs() -> None:
    info = {
        "dataset_info": {
            "features": {
                "image": {"_type": "Image"},
                "mask": {"_type": "Image"},
            },
        }
    }
    result = _build_draft("owner/segment", {}, info)
    assert result.layout == "image-mask-pairs"
    assert result.loader_params == {"image_column": "image", "mask_column": "mask"}


def test_image_only_drafts_flat_images() -> None:
    info = {"dataset_info": {"features": {"image": {"_type": "Image"}}}}
    result = _build_draft("owner/pretrain", {}, info)
    assert result.layout == "flat-images"
    assert result.loader_params == {"image_column": "image"}


def test_audio_column_drafts_audio_clips() -> None:
    info = {"dataset_info": {"features": {"audio": {"_type": "Audio"}}}}
    result = _build_draft("owner/sounds", {}, info)
    assert result.layout == "audio-clips"
    assert result.loader_params == {"audio_column": "audio"}
    assert result.modalities == ("audio",)


def test_no_image_or_audio_column_is_metadata_only_with_a_warning() -> None:
    info = {"dataset_info": {"features": {"text": {"_type": "Value"}}}}
    result = _build_draft("owner/text-only", {}, info)
    assert result.layout == "metadata-only"
    assert any("no Image or Audio column" in w for w in result.warnings)


def test_missing_licence_tag_is_flagged_not_guessed() -> None:
    info = {"dataset_info": {"features": {"image": {"_type": "Image"}}}}
    result = _build_draft("owner/no-licence", {}, info)
    assert result.licence_tag is None
    assert any("no licence tag" in w for w in result.warnings)
    assert "NO-LICENCE-STATED" in result.yaml_draft


def test_licence_tag_from_deprecated_tags_list_fallback() -> None:
    """Some cards carry the licence only as a `license:xxx` tag, not cardData.license."""
    info = {"dataset_info": {"features": {"image": {"_type": "Image"}}}}
    result = _build_draft("owner/tagged", {"tags": ["license:mit", "region:us"]}, info)
    assert result.licence_tag == "mit"


def test_draft_never_claims_a_verified_licence() -> None:
    """⭐ The whole point of `marinedata add`: it must never look like a checked claim."""
    info = {"dataset_info": {"features": {"image": {"_type": "Image"}}}}
    result = _build_draft("owner/anything", {"cardData": {"license": "cc-by-4.0"}}, info)
    assert "method: dataset-card" in result.yaml_draft
    assert "legal_basis: unknown" in result.yaml_draft
    assert "SECONDARY" in result.yaml_draft
    assert "method: primary" not in result.yaml_draft


def test_draft_is_a_probe_result() -> None:
    info = {"dataset_info": {"features": {"image": {"_type": "Image"}}}}
    result = _build_draft("owner/x", {}, info)
    assert isinstance(result, ProbeResult)
    assert result.yaml_draft.strip().startswith("# DRAFT")


def test_multi_config_dataset_info_uses_default_or_first_config() -> None:
    """Some datasets nest dataset_info one level deeper, keyed by config name."""
    info = {
        "dataset_info": {
            "some-config": {
                "features": {"image": {"_type": "Image"}},
                "splits": {"train": {"num_examples": 42}},
            }
        }
    }
    result = _build_draft("owner/multi-config", {}, info)
    assert result.layout == "flat-images"
    assert result.item_count == 42
