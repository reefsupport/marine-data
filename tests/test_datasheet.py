"""WP-5: datasheet completeness, sourced numbers, and Croissant validity.

Every check here is mechanical on purpose. The Limitations table in
``docs/DATASHEET.md`` cites a file (in this repo, never an out-of-repo path — see
``docs/RATING_EVIDENCE_R0.md``'s header) for every row; this test opens that file and
checks the number is actually there, so a future edit cannot silently drift the
datasheet away from its evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASHEET = REPO_ROOT / "docs" / "DATASHEET.md"
CROISSANT_JSON = REPO_ROOT / "docs" / "croissant-v1.json"

GEBRU_SECTIONS = (
    "## 1. Motivation",
    "## 2. Composition",
    "## 3. Collection Process",
    "## 4. Preprocessing / Cleaning / Labeling",
    "## 5. Uses",
    "## 6. Distribution",
    "## 7. Maintenance",
)


@pytest.fixture(scope="module")
def datasheet_text() -> str:
    return DATASHEET.read_text()


def test_all_seven_gebru_sections_present(datasheet_text: str) -> None:
    missing = [heading for heading in GEBRU_SECTIONS if heading not in datasheet_text]
    assert not missing, f"missing Gebru datasheet sections: {missing}"


def test_ethics_subsection_present(datasheet_text: str) -> None:
    assert "### Ethics" in datasheet_text
    for marker in (
        "Diver / face presence",
        "Sensitive-species location policy",
        "Licence per source",
        "Label-quality residuals",
    ):
        assert marker in datasheet_text, f"ethics subsection missing {marker!r}"


def test_no_todo_or_tbd_placeholders(datasheet_text: str) -> None:
    for marker in ("TODO", "TBD", "FACE_N", "FACE_TOTAL", "FACE_PCT"):
        assert marker not in datasheet_text, f"unresolved placeholder {marker!r} in datasheet"


_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")


def _normalize(text: str) -> str:
    return text.replace(",", "").replace("×", "x")


def _significant_numbers(cell: str) -> list[str]:
    """Numeric tokens worth verifying: >=2 significant digits, a percentage or a decimal."""
    tokens = []
    for match in _NUMBER_RE.finditer(cell):
        raw = match.group()
        digits = raw.replace(",", "").replace("%", "").replace(".", "")
        if len(digits) >= 2:
            tokens.append(raw)
    return tokens


def _parse_limitations_table(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Limitation |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 3, f"malformed limitations row: {line!r}"
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def test_limitations_table_is_quantified_and_sourced(datasheet_text: str) -> None:
    rows = _parse_limitations_table(datasheet_text)
    assert len(rows) >= 15, "limitations table looks too short to be the full picture"

    source_text_cache: dict[str, str] = {}
    for label, number_cell, source_cell in rows:
        tokens = _significant_numbers(number_cell)
        assert tokens, f"limitation {label!r} has no quantified number in {number_cell!r}"

        source_name = source_cell.strip("`")
        source_path = REPO_ROOT / source_name
        assert source_path.is_file(), (
            f"{label!r} cites a source file that doesn't exist: {source_name}"
        )

        if source_name not in source_text_cache:
            source_text_cache[source_name] = _normalize(source_path.read_text())
        haystack = source_text_cache[source_name]

        for token in tokens:
            needle = _normalize(token)
            assert needle in haystack, (
                f"{label!r}: number {token!r} not found verbatim in {source_name}"
            )


def test_citation_cff_and_changelog_exist() -> None:
    assert (REPO_ROOT / "CITATION.cff").is_file()
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert "[v1]" in changelog
    for marker in ("TODO", "TBD"):
        assert marker not in changelog


def test_croissant_json_is_valid() -> None:
    mlc = pytest.importorskip("mlcroissant", reason="mlcroissant is a dev/croissant extra")
    assert CROISSANT_JSON.is_file()
    try:
        mlc.Dataset(str(CROISSANT_JSON))
    except mlc.ValidationError as exc:  # pragma: no cover - failure path
        pytest.fail(f"docs/croissant-v1.json failed Croissant validation: {exc}")


def test_croissant_generator_round_trips_a_synthetic_build(tmp_path: Path) -> None:
    """The generator itself (not just the checked-in output) must produce valid Croissant."""
    mlc = pytest.importorskip("mlcroissant", reason="mlcroissant is a dev/croissant extra")
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow is a dev/croissant extra")
    import pyarrow as pa

    from marinedata import croissant

    build_dir = tmp_path / "build"
    config_dir = build_dir / "data" / "widgets"
    config_dir.mkdir(parents=True)
    table = pa.table({"id": ["a", "b"], "label": ["x", "y"]})
    pq.write_table(table, config_dir / "train-00000-of-00001.parquet")

    metadata = croissant.build_metadata(
        build_dir, "reefsupport/synthetic-test", name="synthetic", description="a fixture"
    )
    out = tmp_path / "croissant.json"
    out.write_text(__import__("json").dumps(metadata.to_json()))

    try:
        mlc.Dataset(str(out))
    except mlc.ValidationError as exc:  # pragma: no cover - failure path
        pytest.fail(f"generated Croissant metadata is invalid: {exc}")
