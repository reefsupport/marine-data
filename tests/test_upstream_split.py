"""WP-R2b step 2 (normaliser half): upstream split spellings and the whole-group test holdout."""

import pytest

from marinedata.upstream_split import normalise_upstream_split, upstream_test_groups


@pytest.mark.parametrize(
    ("raw", "want"),
    [("TEST", "test"), ("Test", "test"), ("testing", "test"), ("val", "val"), ("valid", "val"),
     ("Validation", "val"), ("train", "train"), ("TRAINING", "train"), (" Train ", "train"),
     ("dev", None), ("trainval", None), ("", None), (None, None)],
)  # fmt: skip
def test_normalise_upstream_split(raw, want) -> None:
    assert normalise_upstream_split(raw) == want


def test_a_group_with_any_upstream_test_row_is_held_out_whole() -> None:
    seen = {"g1": {"train", "TEST"}, "g2": {"train"}, "g3": {"Testing"}, "g4": set(), "g5": {"dev"}}
    assert upstream_test_groups(seen) == frozenset({"g1", "g3"})
