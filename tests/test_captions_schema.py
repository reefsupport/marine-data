"""WP-13: `registry/captions/schema.yaml` round-trips through `validate_frame`."""

import pandas as pd
import pytest

from marinedata.captions.schema import load_schema, required_columns, validate_frame


@pytest.fixture
def schema():
    return load_schema()


def _valid_frame(schema):
    return pd.DataFrame(
        {
            "image_sha256": ["a", "b"],
            "caption_template": ["A reef photo.", ""],
            "caption_facts": ["{}", "{}"],
            "caption_vlm": [None, "A photo of a coral."],
            "vlm_model": [None, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"],
            "vlm_revision": [None, "main"],
            "prompt_sha256": [None, "deadbeef"],
            "seed": pd.array([None, 0], dtype="Int64"),
            "label_origin": ["derived", "model"],
            "consistency_flags": ["[]", "[]"],
            "audit_verdict": [None, "consistent"],
        }
    )


def test_schema_has_every_brief_required_column(schema):
    cols = set(required_columns(schema))
    for required in (
        "caption_template",
        "caption_facts",
        "caption_vlm",
        "consistency_flags",
        "audit_verdict",
    ):
        assert required in cols


def test_valid_frame_has_no_problems(schema):
    assert validate_frame(_valid_frame(schema), schema) == []


def test_missing_column_is_reported(schema):
    df = _valid_frame(schema).drop(columns=["audit_verdict"])
    problems = validate_frame(df, schema)
    assert any("missing columns" in p and "audit_verdict" in p for p in problems)


def test_extra_column_is_reported(schema):
    df = _valid_frame(schema)
    df["surprise"] = 1
    problems = validate_frame(df, schema)
    assert any("unexpected columns" in p and "surprise" in p for p in problems)


def test_null_in_non_nullable_column_is_reported(schema):
    df = _valid_frame(schema)
    df.loc[0, "label_origin"] = None
    problems = validate_frame(df, schema)
    assert any("label_origin" in p and "nullable" in p for p in problems)


# --- WP-U10: the WP-13 frame feeds the unified ``captions`` table ---------------------------------


def test_wp13_frame_converts_to_valid_unified_captions(schema):
    from marinedata.annotation_schema import validate_rows
    from marinedata.task_layers.captions_table import TEMPLATE_DETAIL, captions_from_frame

    df = _valid_frame(schema)
    df["image_sha256"] = ["ab" * 32, "cd" * 32]  # the unified table wants real 64-hex digests
    rows = captions_from_frame(df, source_id="wp13", source_version="v1")
    validate_rows("captions", rows)
    # the empty template of row two is dropped; the VLM caption of row two is kept
    assert [r["text"] for r in rows] == ["A reef photo.", "A photo of a coral."]
    assert [r["annotator_type"] for r in rows] == ["pseudo", "model"]
    assert rows[0]["annotator_detail"] == TEMPLATE_DETAIL
    assert rows[1]["annotator_detail"].startswith("vlm:mlx-community/Qwen2.5-VL-3B-Instruct-4bit@")
    assert {r["annotator_type"] for r in rows}.isdisjoint({"human", "expert", "crowd"})


def test_unified_captions_licence_class_defaults_to_unknown(schema):
    import json

    from marinedata.task_layers.captions_table import captions_from_frame

    df = _valid_frame(schema)
    df["image_sha256"] = ["ab" * 32, "cd" * 32]
    rows = captions_from_frame(df, source_id="wp13", source_version="v1")
    assert {json.loads(r["attrs"])["licence_class"] for r in rows} == {"unknown"}
    assert all(
        r["caption_type"] == "caption" and r["lang"] == "en" and r["confidence"] is None
        for r in rows
    )


def test_unpinned_vlm_is_named_unspecified_not_nan(schema):
    from marinedata.task_layers.captions_table import captions_from_frame

    df = _valid_frame(schema)
    df["image_sha256"] = ["ab" * 32, "cd" * 32]
    df.loc[1, ["vlm_model", "vlm_revision"]] = [None, None]
    rows = captions_from_frame(df, source_id="wp13", source_version="v1")
    assert rows[1]["annotator_detail"] == "vlm:unspecified@unpinned"
