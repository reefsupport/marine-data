"""``marinedata splitmap generate`` — the release-build SPLIT_MAP.json driver.

Input contract: a TSV (or Parquet) of ``(image_sha256, split_group)``, one row per
admitted image however many sources or tasks it appears in — see
:mod:`marinedata.cli_splitmap` for why this file, not a builder/registry scan, is the
input source today.
"""

from __future__ import annotations

from pathlib import Path

from marinedata.cli import main
from marinedata.splitmap import load_split_map

HEADER = "image_sha256\tsplit_group\n"


def _write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = [HEADER] + [f"{sha}\t{group}\n" for sha, group in rows]
    path.write_text("".join(lines))


def _rows(n_groups: int = 20, per_group: int = 5) -> list[tuple[str, str]]:
    return [
        (f"sha-{g}-{i}", f"group{g}") for g in range(n_groups) for i in range(per_group)
    ]


def test_generate_is_byte_identical_with_pinned_now(tmp_path: Path) -> None:
    tsv = tmp_path / "images.tsv"
    _write_tsv(tsv, _rows())
    out_a = tmp_path / "a" / "SPLIT_MAP.json"
    out_b = tmp_path / "b" / "SPLIT_MAP.json"

    for out in (out_a, out_b):
        code = main(
            [
                "splitmap",
                "generate",
                "--release",
                "r1",
                "--in",
                str(tsv),
                "--now",
                "2026-09-23T00:00:00Z",
                "--seed",
                "0",
                "--out",
                str(out),
            ]
        )
        assert code == 0

    assert out_a.read_text() == out_b.read_text()
    loaded = load_split_map(out_a)
    assert loaded is not None
    assert loaded.by == "group"
    assert loaded.generated_at == "2026-09-23T00:00:00Z"


def test_generate_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    tsv = tmp_path / "images.tsv"
    _write_tsv(tsv, _rows())
    out = tmp_path / "SPLIT_MAP.json"
    out.write_text("{}")

    code = main(
        [
            "splitmap",
            "generate",
            "--release",
            "r1",
            "--in",
            str(tsv),
            "--now",
            "2026-09-23T00:00:00Z",
            "--out",
            str(out),
        ]
    )

    assert code == 1
    assert out.read_text() == "{}", "a refused generate must not touch the existing file"


def test_generate_preseed_pins_matching_groups_before_allocation(tmp_path: Path) -> None:
    tsv = tmp_path / "images.tsv"
    rows = [*_rows(n_groups=10, per_group=100), ("sha-mask-1", "coralscop-masks-rs/p1")]
    _write_tsv(tsv, rows)
    out = tmp_path / "SPLIT_MAP.json"

    code = main(
        [
            "splitmap",
            "generate",
            "--release",
            "r1",
            "--in",
            str(tsv),
            "--now",
            "2026-09-23T00:00:00Z",
            "--preseed",
            "coralscop-masks-rs/*=train",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    loaded = load_split_map(out)
    assert loaded is not None
    assert loaded.assignments["coralscop-masks-rs/p1"] == "train"


def test_generate_dedupes_duplicate_images_across_rows(tmp_path: Path) -> None:
    """The same `image_sha256` appearing twice under the same `split_group` (e.g. seen
    by two sources) must count as one image, not two, in the group's size."""
    tsv = tmp_path / "images.tsv"
    rows = [*_rows(n_groups=5, per_group=20), ("sha-0-0", "group0")]  # duplicate row
    _write_tsv(tsv, rows)
    out = tmp_path / "SPLIT_MAP.json"

    code = main(
        [
            "splitmap",
            "generate",
            "--release",
            "r1",
            "--in",
            str(tsv),
            "--now",
            "2026-09-23T00:00:00Z",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    loaded = load_split_map(out)
    assert loaded is not None
    assert set(loaded.assignments) == {f"group{g}" for g in range(5)}
