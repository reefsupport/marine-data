"""The v1.0 configs only grow (P1.13, P1.14 F1b).

``tests/golden/v1.0-configs-schema.json`` holds the columns and dtypes of the 4 configs published in
v1.0. Today's export layout must contain every one of them, with the same dtype: additive only.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from marinedata.sample_schema import SampleRow
from marinedata.task_layers.mask_configs import INSTANCE_COLUMNS, SEMANTIC_COLUMNS

GOLDEN = json.loads((Path(__file__).parent / "golden" / "v1.0-configs-schema.json").read_text())[
    "configs"
]
MASKS = {
    "coral-masks": SEMANTIC_COLUMNS,
    "scene-masks": SEMANTIC_COLUMNS,
    "instance-masks": INSTANCE_COLUMNS,
}
CODE_DTYPE = {"float64": "double"}
# image bytes, the task tables and the split / licence-class derivations are not SampleRow fields
BUILT_ELSEWHERE = {"image", "points", "boxes", "split", "licence_class"}


def test_the_golden_names_the_four_published_configs():
    assert sorted(GOLDEN) == ["coral-masks", "fish-boxes", "instance-masks", "scene-masks"]


@pytest.mark.parametrize("config", sorted(MASKS))
def test_mask_config_layout_is_a_superset_of_v1_0(config):
    today = dict(MASKS[config])
    for column, dtype in GOLDEN[config].items():
        assert column in today, f"{config}: column {column!r} was removed"
        if isinstance(dtype, str):
            assert today[column] == CODE_DTYPE.get(dtype, dtype), f"{config}.{column} was retyped"


def test_fish_boxes_columns_are_still_produced():
    fields = {f.name for f in dataclasses.fields(SampleRow)}
    missing = [c for c in GOLDEN["fish-boxes"] if c not in BUILT_ELSEWHERE and c not in fields]
    assert missing == [], f"fish-boxes columns no longer in SampleRow: {missing}"
