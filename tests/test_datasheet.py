"""WP-5: datasheet completeness, sourced numbers, and Croissant validity.

Every check here is mechanical on purpose. The datasheet must keep its seven sections
and the Ethics subsection, carry no placeholders, list its limitations, and agree with
``README.md`` on the published row counts, so a future edit cannot drift one document
away from the other.
"""

from __future__ import annotations

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
    for marker in ("TODO", "TBD"):
        assert marker not in datasheet_text, f"unresolved placeholder {marker!r} in datasheet"


CONFIG_ROWS = {
    "coral-masks": "5,997",
    "scene-masks": "1,598",
    "instance-masks": "25,296",
    "fish-boxes": "25,912",
}


def _limitations_rows(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Limitation |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 2, f"malformed limitations row: {line!r}"
        rows.append((cells[0], cells[1]))
    return rows


def test_limitations_table_is_populated(datasheet_text: str) -> None:
    rows = _limitations_rows(datasheet_text)
    assert len(rows) >= 6, "limitations table looks too short to be the full picture"
    assert all(label and detail for label, detail in rows)


def test_config_row_counts_match_readme(datasheet_text: str) -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    for config, rows in CONFIG_ROWS.items():
        assert f"`{config}`" in datasheet_text, f"datasheet is missing config {config}"
        assert rows in datasheet_text, f"datasheet is missing the row count of {config}"
        assert rows in readme, f"README and datasheet disagree on the row count of {config}"


def test_citation_cff_and_changelog_exist() -> None:
    assert (REPO_ROOT / "CITATION.cff").is_file()
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert "[1.0.0]" in changelog
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
