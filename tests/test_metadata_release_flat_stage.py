"""A flat staged tree (``image_path``, no ``partition`` column; the HK fish-box sources) keys its
staged metadata by partition ``""`` instead of raising ``KeyError: 'partition'``."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata import metadata_release as mr
from marinedata.registry import Registry


def test_flat_staged_metadata_is_keyed_by_the_empty_partition(tmp_path):
    root = tmp_path / "uiis"
    root.mkdir()
    table = pa.table(
        {"stem": ["train_L_1"], "image_path": ["images/train_L_1.jpg"], "license": ["Apache-2.0"]}
    )
    pq.write_table(table, root / "metadata.parquet")
    source = Registry.load().source("uiis")
    lookup = mr._staged_lookup(tmp_path, source, root)
    assert set(lookup) == {("", "train_L_1")}
    assert lookup[("", "train_L_1")]["license"] == "Apache-2.0"
