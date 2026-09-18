"""D3a2: stage a points source from an anonymous S3 prefix, against a local
ephemeral-port ``http.server`` fixture (never the real bucket) — design §5's 10
named tests plus 4 from the brief.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
from _ingest_s3_helpers import (
    POINTS_PER_STEM,
    PREFIX,
    STEMS,
    build_corpus,
    make_profile,
    make_source,
)

from marinedata import checksums
from marinedata.ingest import IngestError
from marinedata.ingest_s3 import fetch_points, points_for_stems, stage_s3_source
from marinedata.s3_listing import (
    _object_url,
    group_by_stem,
    list_prefix,
    listing_digest,
    select_stems,
)
from marinedata.tables import write_points_table

_TOTAL_POINTS = sum(POINTS_PER_STEM.values())


def _versioned_source(digest: str):
    return make_source(version=f"2026-09-18-{digest[:12]}")


# 1. list_prefix paginates across both pages and yields every listed object.
def test_list_prefix_paginates_across_pages(s3_server):
    corpus = build_corpus(s3_server)
    entries = list(list_prefix(corpus.plan))
    assert sorted(entries) == sorted(corpus.entries)
    assert len(entries) == len(corpus.entries) > 10


# 2. listing_digest is order-independent.
def test_listing_digest_is_order_independent():
    entries = [("a", 1, "e1"), ("b", 2, "e2"), ("c", 3, "e3")]
    assert listing_digest(entries) == listing_digest(list(reversed(entries)))
    assert listing_digest(entries) == listing_digest([entries[1], entries[2], entries[0]])


# 3. select_stems is deterministic and respects a byte cap.
def test_select_stems_deterministic_and_capped(s3_server):
    corpus = build_corpus(s3_server)
    groups = group_by_stem(corpus.entries, corpus.plan)
    image_groups = {stem: fields[".png"] for stem, fields in groups.items()}
    annotated = set(POINTS_PER_STEM)
    cap = corpus.image_sizes[STEMS[0]] + corpus.image_sizes[STEMS[1]]
    first = select_stems(image_groups, annotated, annotated_only=True, cap_bytes=cap)
    second = select_stems(image_groups, annotated, annotated_only=True, cap_bytes=cap)
    assert first == second == [STEMS[0], STEMS[1]]


# 4. select_stems skips stems with no rows in the annotations parquet.
def test_select_stems_skips_unannotated(s3_server):
    corpus = build_corpus(s3_server)
    groups = group_by_stem(corpus.entries, corpus.plan)
    image_groups = {stem: fields[".png"] for stem, fields in groups.items()}
    selected = select_stems(image_groups, set(POINTS_PER_STEM), annotated_only=True, cap_bytes=None)
    assert selected == sorted(POINTS_PER_STEM)
    assert STEMS[3] not in selected
    assert STEMS[4] not in selected


# 5. download_digest's returned sha256 matches an independent hash of the bytes.
def test_download_digest_matches_sha256(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    key = f"{corpus.plan.prefix}{STEMS[0]}.png"
    dest = tmp_path / "img.png"
    digest = checksums.download_digest(_object_url(corpus.plan, key), dest)
    assert digest == hashlib.sha256(dest.read_bytes()).hexdigest()


# 6. points_for_stems carries pixel coords and a nullable growth form.
def test_points_for_stems_pixel_coords_and_nullable_form(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    table = fetch_points(corpus.plan, tmp_path / "cache")
    rows = points_for_stems(table, [STEMS[0]], corpus.plan)
    assert len(rows) == 3
    assert {(r.row, r.col) for r in rows} == {(1, 1), (1, 2), (2, 1)}
    assert any(r.form is None for r in rows)
    assert all(r.schema_id == "mermaid-attributes-test" for r in rows)


# 7. the written points table has no updated_on column and is byte-identical on
# a second write of the same rows.
def test_points_table_no_updated_on_and_byte_identical(s3_server, tmp_path):
    import pyarrow.parquet as pq

    corpus = build_corpus(s3_server)
    table = fetch_points(corpus.plan, tmp_path / "cache")
    rows = points_for_stems(table, [STEMS[0], STEMS[1]], corpus.plan)
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    d1 = write_points_table(p1, rows)
    d2 = write_points_table(p2, rows)
    assert d1 == d2
    assert p1.read_bytes() == p2.read_bytes()
    assert "updated_on" not in pq.read_schema(p1).names


# 8. ANNOTATIONS.json declares exactly one point geometry, rows == N.
def test_annotations_json_declares_one_point_geometry_with_matching_rows(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    digest = listing_digest(list(list_prefix(corpus.plan)))
    source = _versioned_source(digest)
    result = stage_s3_source(
        source, corpus.plan, tmp_path / "cache", tmp_path / "out", make_profile()
    )
    annotations = json.loads((result.root / "ANNOTATIONS.json").read_text())
    geometries = annotations["geometries"]
    assert len(geometries) == 1
    assert geometries[0]["rows"] == _TOTAL_POINTS
    assert geometries[0]["kind"] == "point-label"


# 9. re-ingest is a no-op (no further network needed); a mutated tree raises.
def test_restage_is_noop_then_mutation_raises(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    digest = listing_digest(list(list_prefix(corpus.plan)))
    source = _versioned_source(digest)
    cache_root, out_root = tmp_path / "cache", tmp_path / "out"
    first = stage_s3_source(source, corpus.plan, cache_root, out_root, make_profile())

    s3_server.shutdown()  # proves the no-op path makes zero further requests
    second = stage_s3_source(source, corpus.plan, cache_root, out_root, make_profile())
    assert second.manifest.root_digest == first.manifest.root_digest

    (first.root / "labels" / "points.parquet").write_bytes(b"corrupted")
    with pytest.raises(checksums.ChecksumError):
        stage_s3_source(source, corpus.plan, cache_root, out_root, make_profile())


# 10. `shasum -a 256 -c CHECKSUMS.sha256` passes against the staged tree.
def test_shasum_verifies_checksums_file(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    digest = listing_digest(list(list_prefix(corpus.plan)))
    source = _versioned_source(digest)
    result = stage_s3_source(
        source, corpus.plan, tmp_path / "cache", tmp_path / "out", make_profile()
    )
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", checksums.CHECKSUM_FILE],
        cwd=result.root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# 11. an un-pinned version ("live") raises, naming the 12-hex digest to pin.
def test_version_live_raises_naming_the_digest(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    source = make_source(version="live")
    with pytest.raises(IngestError) as exc:
        stage_s3_source(source, corpus.plan, tmp_path / "cache", tmp_path / "out", make_profile())
    digest = listing_digest(list(list_prefix(corpus.plan)))
    assert digest[:12] in str(exc.value)


# 12. a version pinned to a NOW-stale digest raises, naming the current one.
def test_stale_pinned_version_raises(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    digest = listing_digest(list(list_prefix(corpus.plan)))
    stale = ("0" * 12) if digest[:12] != "0" * 12 else ("1" * 12)
    source = make_source(version=f"2026-09-18-{stale}")
    with pytest.raises(IngestError) as exc:
        stage_s3_source(source, corpus.plan, tmp_path / "cache", tmp_path / "out", make_profile())
    assert digest[:12] in str(exc.value)


# 13. slice_cap_bytes is recorded verbatim in SOURCE.json's _ingest block.
def test_slice_cap_recorded_in_source_json(s3_server, tmp_path):
    corpus = build_corpus(s3_server)
    digest = listing_digest(list(list_prefix(corpus.plan)))
    source = _versioned_source(digest)
    cap = corpus.image_sizes[STEMS[0]]
    result = stage_s3_source(
        source, corpus.plan, tmp_path / "cache", tmp_path / "out", make_profile(), cap
    )
    meta = json.loads((result.root / "SOURCE.json").read_text())["_ingest"]
    assert meta["slice_cap_bytes"] == cap
    assert meta["slice_rule"] == "lexicographic-stem, cumulative image bytes <= cap"
    assert meta["listing_digest"] == digest
    assert meta["annotated_only"] is True
    assert "omitted" in meta


# 14. an unrecognised key under the prefix raises, naming it.
def test_unknown_key_in_prefix_raises(s3_server):
    corpus = build_corpus(s3_server, extra_key=f"{PREFIX}badfile.xyz")
    with pytest.raises(IngestError) as exc:
        group_by_stem(corpus.entries, corpus.plan)
    assert "badfile.xyz" in str(exc.value)
