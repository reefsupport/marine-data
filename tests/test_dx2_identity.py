"""D-X2 (INT-core3): v1 identity at manifest level — task files byte-for-byte, metadata
columns may differ only as enrichment (null -> value), and the ``upstream_id`` duplicate
tie-break reproduces v1's ``hf_export.build_layout`` choice.

The live run against the real v1 release was done once by hand; these tests pin the
mechanism on fixtures, no network, no image bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.checksums import parse_checksums
from marinedata.cli import main
from marinedata.manifest_identity import (
    NAN,
    NULL_SENTINELS,
    ChecksumsDigest,
    ManifestRow,
    classify_v1_column_diffs,
    compare_task_dirs,
    compare_v1_manifest,
    v1_primary_rows,
)
from marinedata.metadata_release import METADATA_COLUMNS
from marinedata.registry import Registry

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_checksums_digest_reads_listing_never_bytes(tmp_path: Path) -> None:
    root = tmp_path / "src-a"
    (root / "images" / "train").mkdir(parents=True)
    image = root / "images" / "train" / "x.jpg"
    image.write_bytes(b"not the bytes the listing hashes")
    digest = ChecksumsDigest({"src-a": root}, {"src-a": {"images/train/x.jpg": SHA_A}})
    assert digest(image) == SHA_A
    with pytest.raises(KeyError, match="not listed"):
        digest(root / "images" / "train" / "y.jpg")
    with pytest.raises(KeyError, match="under no resolved source root"):
        digest(tmp_path / "elsewhere.jpg")


def test_compare_task_dirs_is_byte_level(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "same.tsv").write_bytes(b"image_sha256\tsplit\nx\ttrain\n")
    (b / "same.tsv").write_bytes(b"image_sha256\tsplit\nx\ttrain\n")
    (a / "diff.tsv").write_bytes(b"image_sha256\tsplit\nx\ttrain\n")
    (b / "diff.tsv").write_bytes(b"image_sha256\tsplit\nx\ttrain\r\n")
    (a / "gone.tsv").write_bytes(b"")
    (b / "new.tsv").write_bytes(b"")
    assert compare_task_dirs(a, b) == {
        "diff.tsv": "differs",
        "gone.tsv": "missing",
        "new.tsv": "extra",
        "same.tsv": "identical",
    }


def test_v1_primary_rows_reproduces_build_layout_tie_break() -> None:
    # hf_export.build_layout: primary = min(members, key=(source_id, sample_key)); the
    # staged-tree sample_key is the root-relative image path == ManifestRow.path.
    rows = [
        ManifestRow(SHA_A, "train", "src-b", 10, 10, "images/train/a.jpg"),
        ManifestRow(SHA_A, "train", "src-a", 10, 10, "images/train/z.jpg"),
        ManifestRow(SHA_A, "train", "src-a", 10, 10, "images/train/m.jpg"),
        ManifestRow(SHA_B, "val", "src-a", 10, 10, "images/val/b.jpg"),
    ]
    primary = v1_primary_rows(rows)
    assert [(r.sha256, r.source_id, r.path) for r in primary] == [
        (SHA_A, "src-a", "images/train/m.jpg"),
        (SHA_B, "src-a", "images/val/b.jpg"),
    ]


def test_compare_v1_manifest_maps_val_to_hf_validation() -> None:
    rows = [ManifestRow(SHA_A, "val", "src-a", 10, 20, "images/val/a.jpg")]
    report = compare_v1_manifest(rows, {SHA_A: ("validation", 10)})
    assert report.identical
    assert report.split_mismatches == ()


def test_classify_v1_column_diffs_allows_only_null_to_value() -> None:
    published = {
        SHA_A: {
            "image_sha256": SHA_A,
            "lat": None,
            "geo_precision": "none",
            "habitat": "reef",
            "upstream_id": "x",
        },
        SHA_B: {
            "image_sha256": SHA_B,
            "lat": 1.0,
            "geo_precision": "site",
            "habitat": "reef",
            "upstream_id": "y",
        },
    }
    rebuilt = [
        {
            "image_sha256": SHA_A,
            "lat": 5.0,
            "geo_precision": "site",
            "habitat": "reef",
            "upstream_id": "x",
        },
        {
            "image_sha256": SHA_B,
            "lat": None,
            "geo_precision": "none",
            "habitat": "sand",
            "upstream_id": "y",
        },
    ]
    diffs = classify_v1_column_diffs(rebuilt, published)
    assert diffs["lat"].enriched == (SHA_A,) and diffs["lat"].changed == (SHA_B,)
    # geo_precision's `none` member means "no lat/lon" — the null sentinel.
    assert diffs["geo_precision"].enriched == (SHA_A,)
    assert diffs["geo_precision"].changed == (SHA_B,)
    assert diffs["habitat"].changed == (SHA_B,) and diffs["habitat"].enriched == ()
    assert diffs["upstream_id"].changed == () and diffs["upstream_id"].enriched == ()


def test_absent_sentinels_are_listed_explicitly_per_column() -> None:
    """D-X2a: the absent-sentinel table is explicit per column — pinned here so adding a
    sentinel (or a column) is a reviewed change, never a silent widening."""
    s, f = frozenset({""}), frozenset({NAN})
    assert {
        "geo_precision": frozenset({"none", ""}),
        "geo_source": s,
        "lat": f,
        "lon": f,
        "gps_precision_m": f,
        "capture_datetime": s,
        "depth_m": f,
        "depth_source": s,
        "depth_zone": s,
        "platform": s,
        "camera": s,
        "habitat": s,
        "meow_realm": s,
        "meow_province": s,
        "meow_ecoregion": s,
        "upstream_url": s,
        "lineage_root_digest": s,
    } == NULL_SENTINELS
    kinds = dict(METADATA_COLUMNS)
    assert set(NULL_SENTINELS) <= set(kinds)
    for column, sentinels in NULL_SENTINELS.items():
        # NaN only on float columns, strings only on string columns.
        assert (NAN in sentinels) == (kinds[column] == "double"), column
    # Identity, provenance and byte-derived quality columns have NO sentinel.
    for column in (
        "image_sha256",
        "source_id",
        "source_version",
        "license",
        "attribution",
        "upstream_id",
        "fetch_date",
        "location_generalized",
        "min_side",
        "q_blur",
        "q_clip_lo",
        "q_clip_hi",
        "q_uiqm",
        "q_entropy",
        "q_blank",
        "quality_flags",
    ):
        assert column not in NULL_SENTINELS, column


def test_sentinel_to_value_is_enrichment_only_where_listed() -> None:
    nan = float("nan")
    published = {
        SHA_A: {
            "image_sha256": SHA_A,
            "lat": nan,
            "capture_datetime": "",
            "geo_precision": "",
            "license": "",
            "q_blur": nan,
            "habitat": "none",
        },
    }
    rebuilt = [
        {
            "image_sha256": SHA_A,
            "lat": 1.5,
            "capture_datetime": "2020-01-01",
            "geo_precision": "site",
            "license": "CC-BY-4.0",
            "q_blur": 0.2,
            "habitat": "coral_reef",
        },
    ]
    diffs = classify_v1_column_diffs(rebuilt, published)
    for column in ("lat", "capture_datetime", "geo_precision"):
        assert diffs[column].enriched == (SHA_A,) and diffs[column].changed == (), column
    # Not listed for that column -> a changed value, which fails D-X2.
    for column in ("license", "q_blur", "habitat"):
        assert diffs[column].changed == (SHA_A,) and diffs[column].enriched == (), column
    # A value -> sentinel is a loss, never enrichment.
    back = classify_v1_column_diffs(
        [{"image_sha256": SHA_A, "lat": nan, "geo_precision": "none"}],
        {SHA_A: {"image_sha256": SHA_A, "lat": 2.0, "geo_precision": "site"}},
    )
    assert back["lat"].changed == (SHA_A,) and back["geo_precision"].changed == (SHA_A,)
    # NaN == NaN is identity, not a diff.
    same = classify_v1_column_diffs(
        [{"image_sha256": SHA_A, "q_blur": nan}], {SHA_A: {"image_sha256": SHA_A, "q_blur": nan}}
    )
    assert same["q_blur"] == type(same["q_blur"])((), ())


def test_manifest_only_refuses_generate_and_sources_from_alone(tmp_path: Path) -> None:
    common = ["release", "build", "--release", "v1", "--out", str(tmp_path)]
    split_map = ["--split-map", str(tmp_path / "SPLIT_MAP.json")]
    assert (
        main([*common, *split_map, "--manifest-only", "--generate-split-map", "--no-near-dup"]) == 1
    )
    assert main([*common, *split_map, "--sources-from", str(tmp_path / "R.json")]) == 1


def test_manifest_only_never_fetches_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marinedata import cli_release

    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "empty-cache"))

    def _boom(*_a, **_k):
        raise AssertionError("--manifest-only must never fetch")

    monkeypatch.setattr(cli_release, "fetch_sample", _boom)
    registry = Registry.load()
    with pytest.raises(ValueError, match="never fetches; not staged locally"):
        cli_release._cached_roots(registry, "research", {}, None)
    # Restricted to a source set with nothing admitted outside --local: no error, no fetch.
    assert cli_release._cached_roots(registry, "research", {}, {"no-such-source"}) == {}


def test_manifest_only_task_files_match_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The e2e fixture built twice — full (hashes image bytes) and --manifest-only
    (sha256 from each staged tree's CHECKSUMS.sha256) — yields byte-identical task TSVs."""
    from test_e2e_release import _registry, _stage_all  # sibling module (prepend import mode)

    from marinedata import cli_release

    registry = _registry()
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: registry))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "dhash-cache"))
    roots = _stage_all(tmp_path, monkeypatch)
    local = [f"--local={sid}={root}" for sid, root in roots.items()]

    def _local_checksums(reg, resolved):
        listings = {}
        for source_id, root in resolved.items():
            listings[source_id] = parse_checksums((Path(root) / "CHECKSUMS.sha256").read_text())
        return ChecksumsDigest(resolved, listings)

    monkeypatch.setattr(cli_release, "checksums_digest", _local_checksums)
    full, lite = tmp_path / "full", tmp_path / "lite"
    split_map = full / "SPLIT_MAP.json"
    base = [
        "release",
        "build",
        "--release",
        "r",
        "--flavour",
        "open",
        "--profile",
        "ship-open",
        *local,
    ]
    assert (
        main(
            [
                *base,
                "--out",
                str(full),
                "--split-map",
                str(split_map),
                "--generate-split-map",
                "--no-near-dup",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert main([*base, "--out", str(lite), "--split-map", str(split_map), "--manifest-only"]) == 0
    status = compare_task_dirs(
        full / "releases" / "r" / "open" / "tasks", lite / "releases" / "r" / "open" / "tasks"
    )
    assert status and set(status.values()) == {"identical"}, status
