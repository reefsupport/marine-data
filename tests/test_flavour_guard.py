"""Release exclusion + no-flavour guard (WP-U14b section 0)."""

from __future__ import annotations

import pytest

from marinedata import licence_class as lc
from marinedata.hf_export import build_layout
from marinedata.task_layers import configs

EXCLUDED = ["floating-marine-debris", "noaa-dsc-csv", "large-scale-fish"]


@pytest.fixture
def guard_on(monkeypatch):
    """Switch the conftest test escape off, as production runs."""
    monkeypatch.setattr(lc, "_UNFLAVOURED_FOR_TESTS", False)


def test_require_flavour_raises_without_flavour(guard_on):
    with pytest.raises(ValueError, match="flavour is required"):
        lc.require_flavour(None, "x")
    assert lc.require_flavour("open", "x") == "open"


def test_escape_is_scoped(guard_on):
    with lc.unflavoured_for_tests():
        assert lc.require_flavour(None, "x") is None
    with pytest.raises(ValueError):
        lc.require_flavour(None, "x")


def test_build_layout_and_all_configs_refuse_none(guard_on, tmp_path):
    with pytest.raises(ValueError, match="build_layout"):
        build_layout({})
    with pytest.raises(ValueError, match="build_all_configs"):
        configs.build_all_configs(object(), tmp_path)


@pytest.mark.parametrize("source", EXCLUDED)
def test_apply_flavour_drops_excluded_without_flavour(source):
    ok = {"source_id": "mermaid-aws", "licence_class": lc.NC, "sha256": "a" * 64}
    bad = {"source_id": source, "licence_class": lc.OPEN, "sha256": "b" * 64}
    result = configs.ConfigResult("points", (ok, bad), {})
    assert configs.apply_flavour(result, None).rows == (ok,)
    assert lc.drop_release_excluded([ok, bad]) == [ok]


@pytest.mark.parametrize("source", EXCLUDED)
def test_build_layout_drops_excluded_without_flavour(source, tmp_path):
    rows = {"t": [_row(tmp_path, 1, source), _row(tmp_path, 2, "zz-keep")]}
    layout = build_layout(rows)  # conftest escape: no flavour
    shas = {
        r.values.get("image_sha256")
        for _spec, splits in layout.values()
        for rs in splits.values()
        for r in rs
    }
    assert f"{1:064x}" not in shas
    assert f"{2:064x}" in shas


@pytest.mark.parametrize("source", EXCLUDED)
def test_ships_in_refuses_excluded_source(source):
    from marinedata.flavours import ships_in
    from marinedata.registry import Registry

    registry = Registry.load()
    for flavour in ("open", "nc"):
        assert ships_in(registry, source, flavour) is False


def _row(tmp_path, i: int, source: str):
    from PIL import Image

    from marinedata.hf_export import SampleRow

    image = tmp_path / f"img{i}.png"
    Image.new("RGB", (8, 8), (i * 20,) * 3).save(image, format="PNG")
    return SampleRow(
        task_id="t", raw_split="train", image_sha256=f"{i:064x}", source_id=source,
        sample_key=f"images/default/img{i}.png", image=image,
    )  # fmt: skip
