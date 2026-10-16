"""PR-M: the v1.1.0 ``metadata`` spine covers both staged ``metadata.parquet`` schemas.

The bucket holds two. The narrow image-labels index (``stem, partition, upstream_path,
upstream_split, width, height`` [+ ``split_group``]) has no ``image_sha256``; the wide
``sample_schema`` table carries ``image_sha256`` and the provenance columns. Before PR-M the spine
read only rows carrying ``image_sha256``, so a narrow tree's images silently left ``metadata``,
``images`` and ``sources`` while their task rows stayed (VM 900 rehearsal, 2026-10-16).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.checksums import file_digest
from marinedata.metadata_release import attribution_for
from marinedata.registry import Registry
from marinedata.release.entries import source_release_entries
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task_layers.configs.common import ConfigResult
from marinedata.task_layers.v11_tables import V11Inputs, build_v11_tables, metadata_columns

NOAA, CORALSCAPES, OWN = "noaa-pifsc-bleaching", "coralscapes", "reef-support-benthic-own"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _image(root: Path, partition: str, name: str, payload: bytes) -> Path:
    path = root / "images" / partition / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _narrow(tmp_path: Path, source_id: str, stems: list[str], *, group: bool = True):
    """A narrow tree written by the image-labels ingest's own writer; ``group=False`` drops the
    ``split_group`` column (coralscapes, suim)."""
    root, rows, shas = tmp_path / source_id, [], {}
    for stem in stems:
        shas[stem] = file_digest(
            _image(root, "default", f"{stem}.PNG", f"{source_id}/{stem}".encode())
        )
        rows.append(
            StagedImage(
                stem=stem, partition="default", upstream_path=f"val/CORAL_BL/{stem}.PNG",
                upstream_split=None, width=224, height=224,
                split_group=f"{source_id}/{stem[:5]}" if group else None,
            )
        )  # fmt: skip
    write_metadata_table(root / "metadata.parquet", rows)
    if not group:
        table = pq.read_table(root / "metadata.parquet")
        pq.write_table(table.drop(["split_group"]), root / "metadata.parquet")
    assert "image_sha256" not in pq.read_schema(root / "metadata.parquet").names
    return root, shas


def _wide(tmp_path: Path):
    """A wide ``sample_schema`` tree (trailing ``partition``); w2's stored sha is wrong."""
    root, rows, shas = tmp_path / OWN, [], {}
    for stem in ("w1", "w2"):
        shas[stem] = file_digest(_image(root, "SEAFLOWER_BOLIVAR", f"{stem}.JPG", stem.encode()))
        rows.append(
            {
                "sample_id": f"{OWN}/{stem}", "source_id": OWN, "source_version": "2026-10-06",
                "stem": stem, "image_path": f"images/SEAFLOWER_BOLIVAR/{stem}.JPG",
                "image_sha256": shas[stem] if stem == "w1" else "0" * 64, "license": "CC-BY-4.0",
                "upstream_url": f"https://example.org/{stem}", "camera": "GoPro HERO9",
                "split_group": f"{OWN}/SEAFLOWER_BOLIVAR", "partition": "SEAFLOWER_BOLIVAR",
            }
        )  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
    return root, shas


def test_narrow_trees_get_one_metadata_row_per_staged_image(registry, tmp_path) -> None:
    noaa, noaa_sha = _narrow(tmp_path, NOAA, ["FFS-B009_2019_06_51", "FFS-B010_2019_06_52"])
    cs, cs_sha = _narrow(tmp_path, CORALSCAPES, ["site1_cs_a", "site2_cs_b"], group=False)
    walk = {e.sha256: e.group for e in source_release_entries(registry.source(CORALSCAPES), cs)}
    inputs = V11Inputs(
        taxonomy_version="3.0.0", roots={NOAA: noaa, CORALSCAPES: cs},
        split_of={s: "train" for s in noaa_sha.values()},
        assignments={g: "test" for g in walk.values()},
    )  # fmt: skip
    out = build_v11_tables(registry, {}, inputs)
    rows = {r["image_sha256"]: r for r in out["metadata"].rows}
    assert set(rows) == {*noaa_sha.values(), *cs_sha.values()}  # main: empty (silent)
    names = {n for n, _ in metadata_columns()}
    for r in rows.values():
        source = registry.source(r["source_id"])
        assert set(r) == names  # the Hub `metadata` schema is unchanged
        assert (r["source_version"], r["license"]) == (source.version, source.licence.id)
        assert r["licence_class"] and r["attribution"] == attribution_for(source)
        assert (r["taxonomy_version"], r["bytes_sha256"]) == ("3.0.0", None)
    row = rows[noaa_sha["FFS-B009_2019_06_51"]]
    assert (row["split_group"], row["split"]) == (f"{NOAA}/FFS-B", "train")  # tree column
    assert row["upstream_id"] == "val/CORAL_BL/FFS-B009_2019_06_51.PNG"
    for sha in cs_sha.values():  # no column: the walk group the split map was built from
        assert (rows[sha]["split_group"], rows[sha]["split"]) == (walk[sha], "test")
    by_source = out["metadata"].notes["report"]["metadata_by_source"]
    assert (by_source[NOAA]["rows"], by_source[CORALSCAPES]["rows"]) == (2, 2)


def test_wide_tree_keys_on_the_file_digest_and_counts_a_stored_sha_mismatch(
    registry, tmp_path
) -> None:
    own, shas = _wide(tmp_path)
    inputs = V11Inputs(
        taxonomy_version="3.0.0", roots={OWN: own}, split_of={s: "val" for s in shas.values()}
    )
    out = build_v11_tables(registry, {}, inputs)
    rows = {r["image_sha256"]: r for r in out["metadata"].rows}
    assert set(rows) == set(shas.values())  # the stored 000... never becomes a key
    row = rows[shas["w1"]]
    assert (row["upstream_url"], row["camera"]) == ("https://example.org/w1", "GoPro HERO9")
    assert (row["split_group"], row["split"]) == (f"{OWN}/SEAFLOWER_BOLIVAR", "val")
    counts = out["metadata"].notes["report"]["metadata_by_source"][OWN]
    assert (counts["rows"], counts["images"], counts["sha256_mismatches"]) == (2, 2, 1)


def test_an_admitted_source_with_task_rows_and_no_metadata_fails_closed(registry, tmp_path):
    from marinedata.task_layers.v11_tables import MetadataSpineError

    root = tmp_path / NOAA
    root.mkdir()  # admitted, but nothing in its tree describes an image
    sha = "a" * 64
    bleaching = ConfigResult("bleaching", ({"image_sha256": sha, "source_id": NOAA},), {})
    inputs = V11Inputs(taxonomy_version="3.0.0", roots={NOAA: root}, split_of={sha: "train"})
    with pytest.raises(MetadataSpineError, match=NOAA):
        build_v11_tables(registry, {"bleaching": bleaching}, inputs)


def test_splits_gate_fails_a_source_the_build_left_without_metadata(tmp_path) -> None:
    import json

    from marinedata.gates.splits_v11 import run_splits_v11_gate

    build = tmp_path / "build"
    shard = build / "data" / "metadata" / "train-00000-of-00001.parquet"
    shard.parent.mkdir(parents=True)
    row = {"image_sha256": "b" * 64, "source_id": "srcA", "split": "train", "split_group": "g"}
    pq.write_table(pa.Table.from_pylist([row]), shard)
    (build / "task_layers").mkdir()
    report = {
        "unsplit_images": 0, "allow_unsplit": False,
        "metadata_by_source": {
            "srcA": {"rows": 1, "shared": 0, "images": 1, "task_rows": 1},
            "srcB": {"rows": 0, "shared": 0, "images": 3, "task_rows": 3},
        },
    }  # fmt: skip
    (build / "task_layers" / "V11_TABLES.json").write_text(json.dumps(report))
    result = run_splits_v11_gate(build)
    assert result.counts.get("failed_metadata_missing") == 1


def test_export_finds_the_image_of_a_fetched_tree_without_checksums(registry, tmp_path) -> None:
    from marinedata.hf_export_v11 import image_rows, staged_index

    noaa, shas = _narrow(tmp_path, NOAA, ["FFS-B011_2019_06_53"])
    assert not (noaa / "CHECKSUMS.sha256").exists()
    sha = shas["FFS-B011_2019_06_53"]
    meta = [{"image_sha256": sha, "source_id": NOAA, "bytes_config": "images", "split": "train"}]
    rows, bytes_sha = image_rows(
        meta, {NOAA: noaa}, locate=lambda sid: staged_index(registry, sid, noaa)
    )
    assert bytes_sha == {sha: sha}
    assert rows[0][1].file == noaa / "images" / "default" / "FFS-B011_2019_06_53.PNG"
