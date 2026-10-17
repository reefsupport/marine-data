"""G7 ``--hidden-test-published-exception`` (D-HT option B, off by default): a hidden-test image the
frozen v1.0 tree already publishes is exempt, the ones new to the public release keep failing as
``hidden_test_new``, and benchmark overlaps are never touched."""

from __future__ import annotations

import json
from pathlib import Path

from test_decon_gate_policy import REGISTRY, build, manifest, record, sha, write_shard

from marinedata.cli import build_parser, main
from marinedata.cli_release_gate_v11 import wants_v11
from marinedata.gates.decon_gate import run_decon_gate
from marinedata.gates.leak import id_tags

NAMES = {"a1": "train", "b2": "test", "c3": "validation"}
"""The 3-image fixture build: a1 is public in v1.0, b2 is new, c3 is not in the hidden test."""
FLAG = "--hidden-test-published-exception"


def v1_tree(tmp_path: Path, *names: str) -> Path:
    """A fake v1.0 tree: its ``coral-masks`` shard holds the ``image_sha256`` of ``names``."""
    root = tmp_path / "v1"
    write_shard(root, "coral-masks", "train", [{"image_sha256": sha(n)} for n in names])
    return root


def hidden_list(tmp_path: Path, *names: str) -> Path:
    """A hidden-test manifest that lists the release images ``names`` by ``image_sha256``."""
    path = tmp_path / "hidden.json"
    path.write_text(json.dumps({"images": [{"sha256": sha(n)} for n in names]}))
    return path


def clean_benchmark(tmp_path: Path) -> Path:
    return manifest(tmp_path / "coralscapes.parquet", "coralscapes", [sha("nothing")])


def gate(tmp_path: Path, *hidden: str, **kw):
    return run_decon_gate(
        build(tmp_path, NAMES),
        hidden_test=[hidden_list(tmp_path, *hidden)],
        benchmarks=[clean_benchmark(tmp_path)],
        registry=REGISTRY,
        v1_dir=v1_tree(tmp_path, "a1"),
        **kw,
    )


def test_default_off_every_hidden_overlap_fails_even_with_a_v1_tree(tmp_path):
    result = gate(tmp_path, "a1", "b2")
    assert not result.passed
    assert (result.counts["overlaps"], result.counts["failed"]) == (2, 2)
    assert not {"hidden_test_already_published", "hidden_test_new"} & result.counts.keys()
    assert "hidden_test_new" not in " ".join(result.failures)
    assert all("kind" not in f for f in result.counts["failing"])


def test_flag_exempts_the_published_image_and_the_new_one_still_fails(tmp_path):
    result = gate(tmp_path, "a1", "b2", hidden_test_published_exception=True)
    counts = result.counts
    assert not result.passed
    assert (counts["overlaps"], counts["failed"]) == (2, 1)
    assert (counts["hidden_test_already_published"], counts["hidden_test_new"]) == (1, 1)
    sample = counts["hidden_test_already_published_sample"]
    assert [(e["sha256"], e["source"], e["split"]) for e in sample] == [
        (sha("a1"), "src-a", "train")
    ]
    assert [(f["sha256"], f["kind"], f["split"]) for f in counts["failing"]] == [
        (sha("b2"), "hidden_test_new", "test")
    ]
    rec = record(result, "hidden.json")
    assert (rec["overlap"], rec["already_published"], rec["failed"]) == (2, 1, 1)
    assert len(result.failures) == 1 and "hidden_test_new: 1 of 2" in result.failures[0]
    assert "1 already published" in result.failures[0]


def test_flag_passes_when_every_hidden_overlap_is_published_and_only_with_the_flag(tmp_path):
    assert not gate(tmp_path, "a1").passed  # off: the published image fails like any other
    on = gate(tmp_path, "a1", hidden_test_published_exception=True)
    assert on.passed, on.failures
    assert (on.counts["hidden_test_already_published"], on.counts["hidden_test_new"]) == (1, 0)
    assert (on.counts["overlaps"], on.counts["failed"], on.counts["failing"]) == (1, 0, [])
    assert gate(tmp_path, "c3x", hidden_test_published_exception=True).passed  # no overlap at all


def test_benchmark_overlaps_are_never_exempt(tmp_path):
    bench = manifest(tmp_path / "marineeval.parquet", "marineeval", [sha("a1")])  # policy: exclude
    result = run_decon_gate(
        build(tmp_path, NAMES),
        hidden_test=[hidden_list(tmp_path, "a1")],
        benchmarks=[bench],
        registry=REGISTRY,
        v1_dir=v1_tree(tmp_path, "a1"),
        hidden_test_published_exception=True,
    )
    counts = result.counts
    assert not result.passed and counts["failed"] == 1
    assert (counts["hidden_test_already_published"], counts["hidden_test_new"]) == (1, 0)
    assert record(result, "marineeval.parquet")["failed"] == 1
    assert [f["benchmark_id"] for f in counts["failing"]] == ["marineeval"]
    assert "hidden_test_new" not in result.failures[0] and "exclude" in result.failures[0]


def test_flag_without_a_v1_set_fails_closed(tmp_path):
    build_dir, hid, bench = (
        build(tmp_path, NAMES),
        hidden_list(tmp_path, "a1"),
        clean_benchmark(tmp_path),
    )
    for extra in ({}, {"v1_dir": tmp_path / "no-such-tree"}, {"v1_sha256": set()}):
        result = run_decon_gate(
            build_dir,
            hidden_test=[hid],
            benchmarks=[bench],
            registry=REGISTRY,
            hidden_test_published_exception=True,
            **extra,
        )
        assert not result.passed and "hidden_test_new" not in result.counts
        assert len(result.failures) == 1 and "--v1-dir" in result.failures[0]


def test_the_published_sample_is_capped_at_twenty_and_a_given_set_replaces_the_tree(tmp_path):
    names = {f"p{i}": "train" for i in range(25)}
    result = run_decon_gate(
        build(tmp_path, names),
        hidden_test=[hidden_list(tmp_path, *names)],
        benchmarks=[clean_benchmark(tmp_path)],
        registry=REGISTRY,
        v1_sha256={sha(n) for n in names},
        hidden_test_published_exception=True,
    )
    assert result.passed, result.failures
    assert result.counts["hidden_test_already_published"] == 25
    assert len(result.counts["hidden_test_already_published_sample"]) == 20
    assert result.counts["hidden_test_new"] == 0 and result.counts["failing"] == []


def cli(build_dir: Path, hid: Path, bench: Path, v1: Path, *flags: str) -> int:
    gates = ["--only", "decon", "--hidden-test", str(hid), "--benchmark", str(bench)]
    return main(["release", "gate", str(build_dir), *gates, "--v1-dir", str(v1), *flags])


def test_cli_fixture_hidden_test_manifest_and_three_images(tmp_path, capsys):
    """G7 end to end on the fixture: ``release gate --only decon`` reads a hidden-test manifest
    file, the 3-image build and the v1.0 tree, and writes the verdict into GATES.json."""
    build_dir, v1 = build(tmp_path, NAMES), v1_tree(tmp_path, "a1")
    bench = tmp_path / "bench.txt"
    bench.write_text(sha("nothing"))
    hid = hidden_list(tmp_path, "a1", "b2")

    def decon() -> dict:
        return json.loads((build_dir / "GATES.json").read_text())["decon"]

    assert cli(build_dir, hid, bench, v1) == 1  # off: both hidden images fail
    assert decon()["passed"] is False and decon()["counts"]["failed"] == 2
    assert "hidden_test_already_published" not in decon()["counts"]
    capsys.readouterr()
    assert cli(build_dir, hid, bench, v1, FLAG) == 1  # b2 is new to the public release
    counts = decon()["counts"]
    assert (counts["hidden_test_already_published"], counts["hidden_test_new"]) == (1, 1)
    assert "hidden_test_new: 1 of 2" in capsys.readouterr().err
    hidden_list(tmp_path, "a1")  # the hidden test without the new image
    assert cli(build_dir, hid, bench, v1, FLAG) == 0
    assert decon()["passed"] is True and decon()["counts"]["hidden_test_already_published"] == 1
    assert cli(build_dir, hid, bench, v1) == 1  # and off again: it fails


def test_the_flag_alone_selects_the_v11_gates(tmp_path):
    parse = build_parser().parse_args
    assert not wants_v11(parse(["release", "gate", str(tmp_path)]), v10_selected=False)
    args = parse(["release", "gate", str(tmp_path), FLAG])
    assert args.hidden_test_published_exception is True and wants_v11(args, v10_selected=False)


def test_redact_ids_replaces_a_benchmark_id_everywhere_the_gate_publishes_it(tmp_path):
    """Open flavour: GATES.json is uploaded, so a benchmark that stays out of the open release is
    an ``nc_id#`` label in the manifest records, the failing samples and the failure lines (the
    id here is synthetic; the verdict is the same with and without the redaction)."""
    bench = manifest(tmp_path / "ncbench.parquet", "ncbench", [sha("b2")])
    kwargs = {"hidden_test": [hidden_list(tmp_path, "c3x")], "benchmarks": [bench]}
    plain = run_decon_gate(build(tmp_path, NAMES), registry=REGISTRY, **kwargs)
    hidden = run_decon_gate(
        build(tmp_path, NAMES), registry=REGISTRY, redact_ids={"ncbench"}, **kwargs
    )
    tag = id_tags("ncbench", {"ncbench"})[0]
    published = json.dumps(hidden.counts) + " ".join(hidden.failures)
    assert "ncbench" in json.dumps(plain.counts) + " ".join(plain.failures)  # the control
    assert "ncbench" not in published and tag in published
    assert not plain.passed and not hidden.passed
    assert (hidden.counts["overlaps"], hidden.counts["failed"]) == (1, 1)
    assert record(hidden, f"{tag}.parquet")["benchmark_id"] == tag
    assert [f["benchmark_id"] for f in hidden.counts["failing"]] == [tag]


def test_redact_ids_also_covers_a_manifest_that_cannot_be_read(tmp_path):
    result = run_decon_gate(
        build(tmp_path, NAMES),
        hidden_test=[hidden_list(tmp_path, "c3x")],
        benchmarks=[tmp_path / "ncbench.json"],
        registry=REGISTRY,
        redact_ids={"ncbench"},
    )
    tag = id_tags("ncbench", {"ncbench"})[0]
    assert not result.passed and "ncbench" not in " ".join(result.failures)
    assert f"{tag}.json: not found" in " ".join(result.failures)
