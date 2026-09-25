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
