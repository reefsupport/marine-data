"""WP-6e-B: subset filter, video frame sampler, manifest + GBIF adapters, MarineInst20M
overlay, IMOS track manifest. Fixtures only, no network."""

from __future__ import annotations

import importlib.util
import io
import json
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from marinedata.adapters import Decoded, Fetched, RemoteItem, make_adapter
from marinedata.adapters._http import HashingReader
from marinedata.adapters.gbif import INAT_DATASET, media_ext, occurrence_row
from marinedata.adapters.video_frames import effective_step, frame_indices, sample_video
from marinedata.ingest_source import IngestSpec
from marinedata.overlay_marineinst import match_rates, overlay_rows
from marinedata.subset_filter import SubsetFilter, expected_total, hash_unit, select_bottom_k

ROOT = Path(__file__).resolve().parents[1]
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _dec(uid: str, **labels: str) -> Decoded:
    return Decoded(upstream_id=uid, data=JPEG, suffix=".jpg", labels=labels)


# --- subset filter -------------------------------------------------------------------
def test_subset_block_is_documentation_unless_enforced():
    assert SubsetFilter.from_spec({"target_items": 5, "stratify": ["a"], "rule": "x"}) is None
    assert SubsetFilter.from_spec(None) is None
    flt = SubsetFilter.from_spec({"enforce": True, "stratify": ["a"], "cap_per_stratum": 2})
    assert flt is not None and flt.cap == 2


def test_predicates_and_cap():
    flt = SubsetFilter.from_spec(
        {
            "enforce": True,
            "stratify": ["dataset", "label"],
            "cap_per_stratum": 2,
            "filters": [
                {"field": "root", "op": "eq", "value": "living"},
                {"field": "label", "op": "not_in", "value": ["other", "unknown"]},
            ],
        }
    )
    got = [
        flt.admit(_dec(f"f#{i}", dataset="d", root=r, label=lab))
        for i, (r, lab) in enumerate(
            [
                ("living", "a"),
                ("detritus", "a"),
                ("living", "other"),
                ("living", "a"),
                ("living", "a"),
                ("living", "b"),
            ]
        )
    ]
    assert got == [True, False, False, True, False, True]
    assert flt.stats() == {
        "seen": 6,
        "kept": 3,
        "groups": 2,
        "rejected_filter": 2,
        "rejected_hash": 0,
        "rejected_cap": 1,
    }


def test_hash_choice_is_deterministic_and_order_independent():
    keys = [f"p.parquet#{i}" for i in range(5000)]
    counts = {"d|a": 5000}

    def run(order):
        flt = SubsetFilter(stratify=("dataset", "label"), cap=100000, group_counts=counts)
        flt.cap, flt.oversample = 500, 1.0
        return {k for k in order if flt.admit(_dec(k, dataset="d", label="a"))}

    shuffled = keys[:]
    random.Random(7).shuffle(shuffled)
    a, b = run(keys), run(shuffled)
    threshold = {k for k in keys if hash_unit("marinedata-subset-v1", k) < 500 / 5000}
    assert a == b == threshold  # under the cap: pure hash draw, fetch order irrelevant
    assert 400 < len(a) <= 500


def test_select_bottom_k_exact_and_order_independent():
    rows = [(f"k{i}", {"g": str(i % 3)}) for i in range(300)]
    flt = SubsetFilter(stratify=("g",), cap=10)
    one = select_bottom_k(rows, flt)
    two = select_bottom_k(list(reversed(rows)), flt)
    assert one == two and len(one) == 30
    assert expected_total({"a": 5, "b": 50}, 10) == 15


def test_planktonzilla_spec_enforces_the_d_ab_subset():
    spec = IngestSpec.load(ROOT / "registry/ingest-specs/planktonzilla.yaml")
    flt = SubsetFilter.from_spec(spec.subset, ROOT)
    assert flt is not None and flt.cap == 2000 and flt.stratify == ("dataset", "proposed_label")
    assert expected_total(flt.group_counts, flt.cap) == spec.subset["target_items"]
    assert not flt.admit(
        _dec("x#1", dataset="global_uvp5", root_class="detritus", proposed_label="detritus")
    )


# --- video ---------------------------------------------------------------------------
def test_effective_step_caps_at_one_fps():
    assert effective_step(30.0) == 30
    assert effective_step(5.0) == 10
    assert effective_step(None) == 10
    assert frame_indices(95, 5.0) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_sample_video_labels_and_split_group(tmp_path):
    video = tmp_path / "seq_0007.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=25:duration=3",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    out = list(sample_video(video, "train/seq_0007.mp4", {"frame_step": 10, "max_fps": 1.0}))
    assert [i for i, _, _ in out] == [0, 25, 50]
    assert all(lab["split_group_id"] == lab["video_id"] == "seq_0007" for _, _, lab in out)
    assert out[1][2]["frame_idx"] == "25" and out[0][1][:3] == b"\xff\xd8\xff"
    adapter = make_adapter("hf", {"repo": "x/y", "frame_step": 10})
    item = RemoteItem("train/seq_0007.mp4", "https://x/seq_0007.mp4")
    decoded = list(adapter.decode(Fetched(item, path=video)))
    assert [d.upstream_id for d in decoded][-1] == "train/seq_0007.mp4#frame_000050"
    assert decoded[0].split_hint == "train"


# --- manifest + GBIF adapters --------------------------------------------------------
def test_manifest_adapter_joins_fields_and_labels(tmp_path):
    man = tmp_path / "m.jsonl"
    man.write_text(
        json.dumps(
            {
                "key": "c/d/PR_1_LC16.tif",
                "url": "https://h/PR_1_LC16.tif",
                "lat": -39.1,
                "lon": 146.3,
                "depth_m": 23.3,
                "dive": "d1",
            }
        )
        + "\n"
    )
    ad = make_adapter(
        "manifest",
        {
            "manifest": str(man),
            "label_columns": ["dive"],
            "field_columns": {"lat": "lat", "lon": "lon", "depth_m": "depth_m"},
        },
    )
    assert ad.resolve_version().startswith("manifest-")
    (item,) = list(ad.enumerate())
    fetched = Fetched(item, stream=HashingReader(io.BytesIO(JPEG)))
    (d,) = list(ad.decode(fetched))
    assert d.fields == {"lat": -39.1, "lon": 146.3, "depth_m": 23.3} and d.labels["dive"] == "d1"


def _occ(gid, ds, sp, fmt="image/jpeg"):
    return {
        "gbifID": gid,
        "datasetKey": ds,
        "speciesKey": sp,
        "species": f"S{sp}",
        "decimalLatitude": 1.5,
        "decimalLongitude": 2.5,
        "eventDate": "2020-01-02",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "license": "http://creativecommons.org/licenses/by/4.0/",
        "media": [
            {"type": "Sound", "identifier": "https://x/a.mp3"},
            {"type": "StillImage", "identifier": f"https://img/{gid}", "format": fmt},
        ],
    }


def test_gbif_adapter_excludes_inat_and_caps_per_species(monkeypatch):
    ad = make_adapter(
        "gbif-occurrence-media",
        {"taxon_keys": [206], "per_species_cap": 2, "snapshot": "api-20260925"},
    )

    def fake_get(query):
        q = dict(query)
        if q.get("facet") == "datasetKey":
            return {
                "facets": [
                    {"counts": [{"name": "ds-a", "count": 3}, {"name": INAT_DATASET, "count": 9}]}
                ]
            }
        assert q["datasetKey"] == "ds-a"
        return {
            "endOfRecords": True,
            "results": [_occ(1, "ds-a", 7), _occ(2, "ds-a", 7), _occ(3, "ds-a", 7)],
        }

    monkeypatch.setattr(ad, "_get", fake_get)
    items = list(ad.enumerate())
    assert [i.key for i in items] == [
        "ds-a/1.jpg",
        "ds-a/2.jpg",
    ] and ad.resolve_version() == "api-20260925"
    (d,) = list(ad.decode(Fetched(items[0], stream=HashingReader(io.BytesIO(JPEG)))))
    assert (
        d.fields["lat"] == 1.5
        and d.labels["license"].endswith("by/4.0/")
        and d.labels["species_key"] == "7"
    )


def test_gbif_row_helpers():
    assert occurrence_row({"media": [{"type": "Sound", "identifier": "u"}]}) is None
    assert media_ext({"identifier": "https://x/y.JPEG"}) == ".jpg"
    assert media_ext({"format": "image/png", "identifier": "https://x/y"}) == ".png"


# --- MarineInst20M overlay -----------------------------------------------------------
def test_overlay_rows_and_match_rates(tmp_path):
    zp = tmp_path / "UIIS.zip"
    doc = {
        "images": [{"id": 0, "file_name": "images/0029.jpg", "width": 4, "height": 3}],
        "annotations": [
            {
                "image_id": 0,
                "segmentation": {"size": [3, 4], "counts": "Q1"},
                "bbox": [0, 0, 1, 1],
                "category_name": "fish",
                "caption": "a small fish",
            }
        ],
    }
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("jsons/0029.json", json.dumps(doc))
        zf.writestr(
            "jsons/0030.json",
            json.dumps({**doc, "images": [{**doc["images"][0], "file_name": "images/0030.jpg"}]}),
        )
    rows = list(overlay_rows(zp, "UIIS", "Public_Datasets"))
    assert [r["basename"] for r in rows] == ["0029", "0030"]
    assert json.loads(rows[0]["captions"]) == ["a small fish"] and rows[0]["n_instances"] == 1
    rates = match_rates(rows, {"uiis": ["train/images/0029.JPG"], "uiis10k": []})
    assert rates["uiis"]["matched_basename"] == 1 and rates["uiis"]["match_pct"] == 50.0
    assert rates["uiis10k"]["status"] == "not staged"


# --- IMOS track manifest ---------------------------------------------------------------
def _imos():
    spec = importlib.util.spec_from_file_location("bim", ROOT / "scripts/build_imos_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_imos_track_rows_take_left_camera_with_position():
    mod = _imos()
    text = (
        "NOTE preamble\nDive origin\n\n"
        "year,month,day,hour,minute,second,local_north,local_east,depth,"
        "latitude,longitude,roll,pitch,heading,altitude,leftimage,rightimage,label\n"
        "2016,3,17,1,37,53.92,47.9,-25.7,23.29,-39.09,146.32,0,0,2.5,1.97,"
        "PR_20160317_013753_929_LC16.png,PR_20160317_013753_929_RM16.png,0\n"
    )
    (row,) = mod.track_rows(
        "Wilsonsprom201603SS",
        "IMOS/AUV/W/r2016_01/",
        "IMOS/AUV/W/r2016_01/i2016_gtif/",
        mod.parse_track(text),
    )
    assert row["key"].endswith("i2016_gtif/PR_20160317_013753_929_LC16.tif")
    assert row["pair_right"].endswith("RM16.png") and row["depth_band"] == "20-40m"
    assert row["capture_datetime"] == "2016-03-17T01:37:53.920Z" and row["lat"] == -39.09
