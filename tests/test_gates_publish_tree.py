"""G10 publish_tree: the tree the Hub holds after a publish is the one the card declares."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.gates.publish_tree import run_publish_tree_gate
from marinedata.gates.v11_runner import RUNNERS, V11_ALL, V11Options, run_v11_gate

A, B, C, D, H = (f"{i:064x}" for i in (1, 2, 3, 4, 8))  # synthetic ids, never real ones
PRIOR = {
    "images": {"train-src-00000-of-00001.parquet": [A, B]},
    "metadata": {"train-00000-of-00001.parquet": [A, B]},
    "condition-masks": {"train-00000-of-00001.parquet": [H]},
}
EXPORT = {
    "images": {"train-src-v1.2.0-00000-of-00001.parquet": [C, D]},
    "metadata": {"train-00000-of-00001.parquet": [A, B, C, D]},
}
DECLARED = ("images", "metadata", "condition-masks")


def tree(root: Path, spec: dict) -> Path:
    for config, shards in spec.items():
        for name, shas in shards.items():
            path = root / "data" / config / name
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"image_sha256": pa.array(shas, pa.string())}), path)
    return root


def card(root: Path, declared) -> None:
    entries = "".join(f"- config_name: {c}\n" for c in declared)
    (root / "README.md").write_text(f"---\nconfigs:\n{entries}---\n# card\n")


def gate(tmp_path: Path, *, prior=PRIOR, export=EXPORT, declared=DECLARED):
    prior_dir, build = tree(tmp_path / "prior", prior), tree(tmp_path / "build", export)
    if declared is not None:
        card(build, declared)
    return run_publish_tree_gate(build, v1_dir=prior_dir)


def lines(result) -> str:
    return "\n".join(result.failures)


def test_an_additive_export_passes_and_counts_what_stays(tmp_path):
    result = gate(tmp_path)
    assert result.passed, result.failures
    counts = result.counts
    assert (counts["prior_configs"], counts["export_configs"], counts["configs_in_both"]) == (
        3,
        2,
        2,
    )
    assert (counts["collisions"], counts["leftovers"]) == (1, 1)  # metadata is replaced in place
    assert (counts["images_collisions"], counts["images_leftovers"]) == (0, 1)
    assert (counts["lost"], counts["double"], counts["undeclared_prior_configs"]) == (0, 0, 0)
    assert counts["series"] == 4  # images spans two series: the prior one and the release one


def test_an_images_shard_on_a_prior_path_fails(tmp_path):
    export = {**EXPORT, "images": {"train-src-00000-of-00001.parquet": [A, B]}}
    result = gate(tmp_path, export=export)
    assert not result.passed
    assert result.counts["images_collisions"] == 1
    assert "take the path of a prior shard" in lines(result)


def test_a_replaced_shard_that_drops_an_image_loses_it(tmp_path):
    export = {**EXPORT, "metadata": {"train-00000-of-00001.parquet": [A, C, D]}}
    result = gate(tmp_path, export=export)
    assert not result.passed
    assert (result.counts["lost"], result.counts["double"]) == (1, 0)
    assert not any(sha in repr(result) for sha in (A, B, C, D, H))  # no ids in the output


def test_a_leftover_shard_that_repeats_an_export_image_is_double(tmp_path):
    export = {**EXPORT, "metadata": {"train-v1.2.0-00000-of-00001.parquet": [B, C, D]}}
    result = gate(tmp_path, export=export)  # the prior metadata shard stays and holds B too
    assert not result.passed
    assert (result.counts["double"], result.counts["lost"]) == (1, 0)
    assert result.counts["leftovers"] >= 2


def test_a_stale_series_next_to_a_new_one_fails(tmp_path):
    prior = {
        **PRIOR,
        "images": {
            "train-src-00000-of-00002.parquet": [A],
            "train-src-00001-of-00002.parquet": [B],
        },
    }
    export = {**EXPORT, "images": {"train-src-00000-of-00001.parquet": [C, D]}}
    result = gate(tmp_path, prior=prior, export=export)
    assert not result.passed
    assert (result.counts["incomplete_series"], result.counts["failures_all"]) == (1, 1)
    assert "one series of [1, 2] parts" in lines(result)


def test_a_series_with_a_missing_part_fails(tmp_path):
    prior = {
        **PRIOR,
        "images": {
            "train-src-00000-of-00003.parquet": [A],
            "train-src-00002-of-00003.parquet": [B],
        },
    }
    result = gate(tmp_path, prior=prior)
    assert not result.passed and "missing [1]" in lines(result)


def test_a_prior_config_the_card_does_not_declare_fails(tmp_path):
    result = gate(tmp_path, declared=("images", "metadata"))
    assert not result.passed
    assert result.counts["undeclared_prior_configs"] == 1
    assert "condition-masks" in lines(result)


def test_it_fails_without_a_prior_tree_an_export_or_a_card(tmp_path):
    build = tree(tmp_path / "build", EXPORT)
    card(build, DECLARED)
    no_tree = run_publish_tree_gate(build)
    assert not no_tree.passed and "--v1-dir" in lines(no_tree)
    empty = run_publish_tree_gate(build, v1_dir=tmp_path / "empty")
    assert not empty.passed and "nothing to compare" in lines(empty)
    nothing = run_publish_tree_gate(tmp_path / "nothing", v1_dir=tree(tmp_path / "prior", PRIOR))
    assert not nothing.passed and "no data" in lines(nothing)
    no_card = gate(tmp_path / "x", declared=None)
    assert not no_card.passed and "README.md is missing" in lines(no_card)


def test_the_gate_runs_with_the_other_v11_gates(tmp_path):
    assert "publish_tree" in V11_ALL and "publish_tree" in RUNNERS
    prior, build = tree(tmp_path / "prior", PRIOR), tree(tmp_path / "build", EXPORT)
    card(build, DECLARED)
    assert run_v11_gate("publish_tree", build, V11Options(v1_dir=prior)).passed
    assert not run_v11_gate("publish_tree", build, V11Options()).passed
