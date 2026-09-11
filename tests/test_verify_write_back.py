"""``marinedata verify`` write-back — `loader.verified_on` must be something code
writes, not a hand-typed assertion nobody checks.

These are text-level tests of `_patch_loader_verification` (no fetch, no registry
load) plus a couple of `write_verification` tests against a bare `Registry(root=...)`
— the write-back only ever needs `registry.root` and a source id, not a validated
`Source`, so building a full valid mini-registry would test more than the function
actually depends on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from marinedata.registry import Registry
from marinedata.verify import VerifyResult, _patch_loader_verification, write_verification

TODAY = date(2026, 9, 11)


def test_inserts_verified_on_and_note_into_a_fresh_block_style_loader() -> None:
    text = (
        "sources:\n"
        "  - id: fixture-one\n"
        "    name: Fixture One\n"
        "    loader:\n"
        "      layout: csv-points\n"
        "      schema_id: worms-genus\n"
        "  - id: fixture-two\n"
        "    name: Fixture Two\n"
    )
    note = "marinedata verify — 12 item(s)"
    patched = _patch_loader_verification(text, "fixture-one", TODAY, note)

    doc = yaml.safe_load(patched)
    loader = doc["sources"][0]["loader"]
    assert loader["verified_on"] == TODAY
    assert loader["verified_note"] == "marinedata verify — 12 item(s)"
    assert loader["layout"] == "csv-points"
    assert loader["schema_id"] == "worms-genus"
    # The untouched sibling entry must survive byte-for-byte.
    assert doc["sources"][1]["name"] == "Fixture Two"


def test_replaces_an_existing_single_line_verified_on_and_folded_verified_note() -> None:
    """The real shape this must handle: a prior ``verify`` run left a single-line
    ``verified_on`` *and* a folded (``>``) multi-line ``verified_note`` — both must be
    fully replaced, with no orphaned continuation lines left behind to corrupt the YAML.
    """
    text = (
        "sources:\n"
        "  - id: reefset-v1\n"
        "    name: ReefSet v1.0\n"
        "    loader:\n"
        "      layout: json-manifest-labels\n"
        "      schema_id: sonotype\n"
        "      params:\n"
        "        manifest: manifest.json\n"
        "      verified_on: 2026-08-30\n"
        "      verified_note: >\n"
        "        The declared audio-clips layout was wrong — clips are NOT pre-sorted\n"
        "        into per-class directories. Confirmed by downloading the full archive.\n"
        "    tags: [reef-audio, permissive]\n"
    )
    patched = _patch_loader_verification(
        text, "reefset-v1", TODAY, "marinedata verify — 57084 item(s), sha256:abc"
    )

    doc = yaml.safe_load(patched)
    entry = doc["sources"][0]
    loader = entry["loader"]
    assert loader["verified_on"] == TODAY
    assert loader["verified_note"] == "marinedata verify — 57084 item(s), sha256:abc"
    assert loader["schema_id"] == "sonotype"
    assert loader["params"]["manifest"] == "manifest.json"
    # A sibling top-level key after the loader block must still parse correctly —
    # proof no continuation line escaped the block it was folded under.
    assert entry["tags"] == ["reef-audio", "permissive"]


def test_flow_style_loader_gains_verified_fields_without_losing_existing_ones() -> None:
    text = (
        "sources:\n"
        "  - id: depth-pairs\n"
        "    name: Depth Pairs\n"
        "    loader: { layout: image-mask-pairs, "
        "params: { images_dir: images, masks_dir: depth } }\n"
    )
    note = "marinedata verify — 5 item(s)"
    patched = _patch_loader_verification(text, "depth-pairs", TODAY, note)

    doc = yaml.safe_load(patched)
    loader = doc["sources"][0]["loader"]
    assert loader["verified_on"] == TODAY
    assert loader["verified_note"] == "marinedata verify — 5 item(s)"
    assert loader["layout"] == "image-mask-pairs"
    assert loader["params"] == {"images_dir": "images", "masks_dir": "depth"}


def test_unknown_source_id_is_a_no_op() -> None:
    text = "sources:\n  - id: only-one\n    loader:\n      layout: csv-points\n"
    assert _patch_loader_verification(text, "nope", TODAY, "note") == text


def test_write_verification_locates_and_patches_the_right_file(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "a.yaml").write_text(
        "sources:\n  - id: alpha\n    loader:\n      layout: csv-points\n", encoding="utf-8"
    )
    (sources_dir / "b.yaml").write_text(
        "sources:\n  - id: beta\n    loader:\n      layout: csv-points\n", encoding="utf-8"
    )
    registry = Registry(sources={}, licences={}, profiles={}, root=tmp_path)
    result = VerifyResult("beta", "verified", items=42, digest="sha256:deadbeef")

    written = write_verification(registry, result)

    assert written == sources_dir / "b.yaml"
    doc = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert doc["sources"][0]["loader"]["verified_on"] == date.today()
    # The other file must be untouched.
    assert "verified_on" not in (sources_dir / "a.yaml").read_text(encoding="utf-8")


def test_write_verification_is_a_no_op_for_a_non_ok_result(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "a.yaml").write_text(
        "sources:\n  - id: alpha\n    loader:\n      layout: csv-points\n", encoding="utf-8"
    )
    registry = Registry(sources={}, licences={}, profiles={}, root=tmp_path)

    assert write_verification(registry, VerifyResult("alpha", "failed")) is None
    assert "verified_on" not in (sources_dir / "a.yaml").read_text(encoding="utf-8")


def test_write_verification_returns_none_without_a_root() -> None:
    registry = Registry(sources={}, licences={}, profiles={})
    result = VerifyResult("alpha", "verified", items=1)
    assert write_verification(registry, result) is None
