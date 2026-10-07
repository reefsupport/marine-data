"""Config parsing (design §4.4 baselines config)."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "baselines" / "v2.yaml"


def test_config_loads_and_has_two_permissive_backbones():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    ids = {b["id"] for b in config["backbones"]}
    assert ids == {"dinov2-small", "openclip-vitb16"}
    for b in config["backbones"]:
        assert b["licence"] in {"Apache-2.0", "MIT"}
        assert len(b["revision"]) == 40  # a real HF commit sha, not a floating ref


def test_config_pins_reproduce_tolerance_to_design_sec5():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["reproduce"]["tolerance_pt"] == 0.5


def test_config_probe_tasks_match_manager_decision():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    task_ids = {t["id"] for t in config["tasks"]}
    assert task_ids == {"bleaching-condition", "benthic-coarse", "source-id-domain"}
