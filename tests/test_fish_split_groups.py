"""HK-6: the four staged fish-box sources declare a registry ``split_group`` pattern, so the
staged-tree loader sets a group and the release build no longer refuses them."""

from __future__ import annotations

import pytest

from marinedata.registry import Registry

ROBOFLOW_STEM = (
    "aquarium_pretrain_train_images_IMG_8496_MOV-4_jpg_rf_0f25b61a8d7ab12cbf5ae131582007d5"
)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


@pytest.mark.parametrize(
    ("sid", "stem", "expected"),
    [
        ("uiis", "train_L_1", "uiis/train_L_1"),
        ("uiis10k", "UIIS10K_img_train_7943", "uiis10k/UIIS10K_img_train_7943"),
        ("usis10k", "USIS10K_test_test_00476", "usis10k/USIS10K_test_test_00476"),
        ("roboflow-aquarium", ROBOFLOW_STEM, "roboflow-aquarium/IMG_8496"),
    ],
)  # fmt: skip
def test_fish_sources_resolve_a_group(registry, sid, stem, expected) -> None:
    source = registry.source(sid)
    assert source.split_group.pattern is not None
    assert source.split_group_for(stem=stem, upstream_path="", partition="train") == expected


def test_roboflow_video_frames_and_augmentations_share_a_group(registry) -> None:
    source = registry.source("roboflow-aquarium")
    frame = lambda n, h: f"aquarium_pretrain_train_images_IMG_8497_MOV-{n}_jpg_rf_{h * 32}"  # noqa: E731
    groups = {
        source.split_group_for(stem=frame(n, h), upstream_path="", partition="train")
        for n, h in ((0, "a"), (3, "b"), (5, "c"))
    }
    assert groups == {"roboflow-aquarium/IMG_8497"}
