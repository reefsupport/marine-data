"""``scripts/registry_sync_bucket.py``: argument parsing and a dry run on a tmp registry (F8)."""

import importlib
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TREES_CSV = (
    "bucket,source_id,version,bytes,n_objects,has_checksums,has_ingest_json,"
    "has_metadata_parquet,images_in_listing\n"
    "rs-storage-private,demo-src,v1,1000,10,false,false,false,\n"
)
ENTRY = """sources:
- id: demo-src
  version: v1
  access: {method: s3, params: {bucket: rs-storage-private}}
"""


@pytest.fixture
def sync(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module("registry_sync_bucket")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "demo.yaml").write_text(ENTRY)
    (tmp_path / "trees.csv").write_text(TREES_CSV)
    monkeypatch.setattr(module, "REG", tmp_path / "sources")
    monkeypatch.setattr(module, "SPECS", tmp_path / "specs")
    monkeypatch.setattr(module, "TREES", tmp_path / "trees.csv")
    monkeypatch.setattr(module, "EVIDENCE", tmp_path / "evidence.json")
    return module


def test_unknown_flags_are_rejected(sync, monkeypatch):
    monkeypatch.setattr("sys.argv", ["registry_sync_bucket.py", "--nope"])
    with pytest.raises(SystemExit) as exc:
        sync.main()
    assert exc.value.code == 2


def test_probe_dry_run_reads_nothing_private_and_edits_nothing(sync, monkeypatch, tmp_path, capsys):
    before = (tmp_path / "sources" / "demo.yaml").read_text()
    monkeypatch.setattr("sys.argv", ["registry_sync_bucket.py", "--probe"])
    assert sync.main() == 0
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["gets"] == 0 and evidence["trees"][0]["skipped"].startswith("private bucket")
    assert (tmp_path / "sources" / "demo.yaml").read_text() == before
    assert capsys.readouterr().out.strip()
