"""Pipeline-level ingest tests (I2, D1 §7): staging integrity and error paths — split
from ``test_ingest.py`` (D2e, 503 ln) by topic. Reproducibility (re-ingest no-op),
checksum-guard failures (changed/added/removed file, ``shasum -c``), and structural
failures (stem collision, timestamp leakage, missing plan, wrong archive member count).
No network: a synthetic zip built in ``tmp_path`` mimics
``SUIM/{train_val,TEST}/{images,masks}``, and ``fetch.get_bytes`` is monkeypatched to
return its bytes instead of hitting the network. Shared builders live in
``tests/_ingest_helpers.py``.

Content and manifest-truth correctness (what's IN the staged tree, not just that it's
self-consistent) lives in ``tests/test_ingest_content.py``.

``tests/test_checksums.py`` already proves the manifest-layer properties this reuses
(no-op rewrite, mutation raises, ``shasum -c`` passes, recorded-not-read — tests 13, 14,
16 in D1 §7 live in ``tests/test_ingest_primitives.py`` instead) — not duplicated here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _ingest_helpers import _build_zip, _profile, _source, _stage, _suim_plan

from marinedata import fetch
from marinedata.checksums import ChecksumError
from marinedata.ingest import IngestError, _stage_with_plan, stage_source


def test_reingesting_identical_bytes_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {}, "b": {}, "c": {}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=3, expected_masks=3)
    source = _source()
    cache_root, out_root, profile = tmp_path / "cache", tmp_path / "out", _profile()

    v1 = _stage_with_plan(source, plan, cache_root=cache_root, out_root=out_root, profile=profile)
    v2 = _stage_with_plan(source, plan, cache_root=cache_root, out_root=out_root, profile=profile)

    assert v1.manifest.root_digest == v2.manifest.root_digest
    assert v1.images == v2.images == 3


def test_staged_root_is_out_sources_id_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, source, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})

    assert result.root == tmp_path / "out" / "sources" / source.id / source.version


def test_a_changed_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    jpg = next((result.root / "images" / "default").glob("*.jpg"))
    data = bytearray(jpg.read_bytes())
    data[0] ^= 0xFF
    jpg.write_bytes(bytes(data))

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


def test_an_added_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    (result.root / "images" / "default" / "stray.jpg").write_bytes(b"not a real image")

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


def test_a_removed_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    mask = next((result.root / "labels" / "masks" / "default").glob("*.png"))
    mask.unlink()

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


def test_shasum_c_passes_on_the_staged_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _source_obj, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "CHECKSUMS.sha256"],
        cwd=result.root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stem_collision_raises_rather_than_suffixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "img#1" and "img_1" both sanitise (non [A-Za-z0-9._-] -> "_") to "img_1".
    specs = {"img#1": {"mask": False}, "img_1": {"mask": False}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=2, expected_masks=0)
    source = _source()
    out_root = tmp_path / "out"

    with pytest.raises(IngestError, match="stem collision"):
        _stage_with_plan(
            source, plan, cache_root=tmp_path / "cache", out_root=out_root, profile=_profile()
        )

    assert not any(out_root.rglob("*_1*"))


def test_no_timestamp_reaches_the_staged_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _source_obj, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    for path in result.root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "fetched_at" not in text
        assert "staged_at" not in text


def test_stage_source_without_a_plan_raises(tmp_path: Path) -> None:
    source = _source(source_id="no-such-plan")
    with pytest.raises(IngestError, match="no ingest plan"):
        stage_source(
            source, cache_root=tmp_path / "cache", out_root=tmp_path / "out", profile=_profile()
        )


def test_wrong_member_count_raises_and_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = _build_zip({"a": {}, "b": {}})
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=5, expected_masks=5)  # zip only has 2 + 2
    source = _source()
    out_root = tmp_path / "out"

    with pytest.raises(IngestError, match="expected 5"):
        _stage_with_plan(
            source, plan, cache_root=tmp_path / "cache", out_root=out_root, profile=_profile()
        )

    assert not (out_root / "sources" / source.id).exists()
