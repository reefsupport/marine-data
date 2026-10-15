"""P1.13 delta 1 (D1): the provenance files ``release publish`` ships are public-safe."""

from __future__ import annotations

import json

import yaml

from marinedata.release_provenance import public_release_json, stage_provenance

LOCK = {
    "release_id": "v1.1",
    "created": "2026-10-08T00:00:00+00:00",
    "code_sha": "abc123",
    "sources": [{"id": "open-a", "version": "v1", "root_digest": "d" * 64}],
}
RELEASE = {
    "release": "v1.1",
    "flavour": "open",
    "repo_id": "reefsupport/marine-data",
    "lock_sha256": "e" * 64,
    "configs": {"images": {"splits": {"train": {"rows": 10}}}},
    "tasks": ["pretrain-set"],
    "class_counts": {"registry_sources": {"open": 1, "nc": 3}},
    "excluded_sources": {"nc-b": "licence is cc-by-nc"},
    "skipped_sources": [{"id": "nd-c", "reason": "restricted"}],
    "licence_rows_dropped": {"open-a": {"total": 2}},
    "lock_sources": [{"id": "nc-b"}],
    "skipped_tasks": ["boxes-nd-c"],
    "sources": [{"id": "open-a", "version": "v1"}, {"id": "nc-b", "version": "v2"}],
    "near_dup": {"algorithm": "dhash-64", "cache": "/Users/me/.cache/x", "src": "s3://rs-hel1/b"},
    "out_dir": "/home/ci/out",
}


def test_projection_keeps_counts_names_and_ids_and_drops_the_rest():
    out = public_release_json(RELEASE, LOCK)
    assert (out["release"], out["release_id"], out["code_sha"]) == ("v1.1", "v1.1", "abc123")
    assert out["created"] == LOCK["created"] and out["configs"] == RELEASE["configs"]
    assert out["tasks"] == ["pretrain-set"] and out["class_counts"] == RELEASE["class_counts"]
    assert [s["id"] for s in out["sources"]] == ["open-a"]
    assert out["near_dup"] == {"algorithm": "dhash-64"}
    text = json.dumps(out)
    for private in ("nc-b", "nd-c", "excluded_sources", "skipped_", "licence_rows_dropped"):
        assert private not in text
    for private in ("/Users/", "/home/", "s3://", "rs-hel1", "out_dir"):
        assert private not in text


def test_stage_provenance_writes_the_three_files(tmp_path):
    (tmp_path / "RELEASE.json").write_text(json.dumps(RELEASE))
    (tmp_path / "GATES.json").write_text('{"summary": {}}\n')
    (tmp_path / "release.lock.yaml").write_text(yaml.safe_dump(LOCK))
    target = tmp_path / "provenance"
    stage_provenance(
        target,
        release=tmp_path / "RELEASE.json",
        gates=tmp_path / "GATES.json",
        lock=tmp_path / "release.lock.yaml",
    )
    assert sorted(p.name for p in target.iterdir()) == [
        "GATES.json",
        "RELEASE.json",
        "release.lock.yaml",
    ]
    for name in ("GATES.json", "release.lock.yaml"):
        assert (target / name).read_bytes() == (tmp_path / name).read_bytes()
    assert json.loads((target / "RELEASE.json").read_text()) == public_release_json(RELEASE, LOCK)
