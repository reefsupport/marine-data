"""WP-6e-A deep-source adapters: network-free fixture tests (iNat, ToL, Commons, FathomNet)."""

from __future__ import annotations

import io
import tarfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.adapters import Fetched, RemoteItem, make_adapter
from marinedata.adapters._http import HashingReader
from marinedata.adapters.fathomnet import FathomNetAdapter
from marinedata.adapters.inat import INatOpenDataAdapter, load_marine_taxa, obs_hash, select_subset
from marinedata.adapters.treeoflife import HFMemberFilterAdapter, select_marine
from marinedata.adapters.wikimedia import USER_AGENT, CommonsAdapter


def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 90, 160)).save(buf, "JPEG")
    return buf.getvalue()


def _fetched(key: str, data: bytes) -> Fetched:
    return Fetched(
        RemoteItem(key=key, url=f"https://x/{key}"), stream=HashingReader(io.BytesIO(data))
    )


# ---- iNat -------------------------------------------------------------------------------


def _inat_rows() -> pd.DataFrame:
    rows = []
    for i in range(5):  # taxon 1: five observations, two photos each
        for pos in (1, 0):
            rows.append(
                {
                    "observation_uuid": f"o{i}",
                    "taxon_id": 1,
                    "photo_id": 10 * i + pos,
                    "position": pos,
                }
            )
    rows.append({"observation_uuid": "p0", "taxon_id": 2, "photo_id": 99, "position": 0})
    return pd.DataFrame(rows)


def test_inat_select_subset_is_deterministic_one_photo_per_obs_and_capped() -> None:
    rows = _inat_rows()
    a = select_subset(rows, cap=3)
    b = select_subset(rows.sample(frac=1, random_state=7), cap=3)
    assert a[["observation_uuid", "photo_id"]].equals(b[["observation_uuid", "photo_id"]])
    assert a["observation_uuid"].is_unique and (a["position"] == 0).all()
    t1 = a[a["taxon_id"] == 1]
    assert len(t1) == 3 and len(a[a["taxon_id"] == 2]) == 1
    want = sorted((f"o{i}" for i in range(5)), key=obs_hash)[:3]
    assert list(t1["observation_uuid"]) == want
    assert len(rows) == 11  # input not mutated


def test_inat_marine_taxa_file_matches_spec_count() -> None:
    taxa = load_marine_taxa("registry/ingest-specs/data/inat-marine-taxa-2026-09-25.csv.gz")
    assert len(taxa) == 34872  # SPEC-w3 measured.n_taxa


def test_inat_adapter_urls_and_per_sample_metadata(tmp_path) -> None:
    man = tmp_path / "m.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "photo_id": 123,
                    "extension": "jpeg",
                    "license": "CC-BY-NC",
                    "observation_uuid": "u1",
                    "observer_id": 7,
                    "taxon_id": 47533,
                    "latitude": -16.5,
                    "longitude": 145.9,
                    "observed_on": "2021-05-04",
                    "width": 4,
                    "height": 4,
                    "species": "Acropora millepora",
                },
            ]
        ),
        man,
    )
    ad = make_adapter("inat-open-data", {"manifest": str(man), "metadata_date": "2026-09"})
    assert isinstance(ad, INatOpenDataAdapter)
    assert ad.resolve_version().startswith("inat-2026-09-")
    (item,) = list(ad.enumerate())
    # D-AE: fetch `large` (1024 px); keep photo_id + the `original` URL for a later hi-res pass.
    base = "https://inaturalist-open-data.s3.amazonaws.com/photos/123"
    assert item.url == f"{base}/large.jpeg"
    (d,) = list(ad.decode(_fetched(item.key, _jpeg())))
    assert d.fields["lat"] == -16.5 and d.fields["lon"] == 145.9
    assert d.fields["license"] == "CC-BY-NC" and d.fields["capture_datetime"] == "2021-05-04"
    assert d.labels["taxon_id"] == "47533" and d.labels["observation_uuid"] == "u1"
    assert d.labels["photo_id"] == "123"
    assert d.labels["original_url"] == f"{base}/original.jpeg"
    fetched = Fetched(item, stream=HashingReader(io.BytesIO(_jpeg())))
    (d_real,) = list(ad.decode(fetched))
    assert d_real.upstream_url == f"{base}/large.jpeg"
    # An explicit photo_size still wins (a later hi-res pass).
    hi = make_adapter("inat-open-data", {"manifest": str(man), "photo_size": "original"})
    (hi_item,) = list(hi.enumerate())
    assert hi_item.url == f"{base}/original.jpeg"
    assert hi.resolve_version().endswith("-original") and not ad.resolve_version().endswith(
        "-large"
    )


# ---- TreeOfLife-10M member filter -------------------------------------------------------


def test_tol_select_marine_rules_and_cap() -> None:
    base = {"eol_content_id": "1", "inat21_filename": "", "kingdom": "Animalia"}
    rows = [
        {
            **base,
            "treeoflife_id": f"c{i}",
            "class": "Anthozoa",
            "genus": "Acropora",
            "species": "millepora",
        }
        for i in range(4)
    ] + [
        {
            **base,
            "treeoflife_id": "w1",
            "class": "Actinopterygii",
            "genus": "Chromis",
            "species": "viridis",
        },
        {
            **base,
            "treeoflife_id": "n1",
            "class": "Actinopterygii",
            "genus": "Esox",
            "species": "lucius",
        },
        {
            "treeoflife_id": "b1",
            "eol_content_id": "",
            "inat21_filename": "",
            "class": "Anthozoa",
            "genus": "X",
            "species": "y",
        },
    ]
    keep = select_marine(rows, {"Anthozoa"}, {"Chromis viridis"}, cap=2)
    assert set(keep) >= {"w1"} and "n1" not in keep and "b1" not in keep
    assert sum(1 for k in keep if k.startswith("c")) == 2
    assert keep["w1"]["marine_rule"] == "worms-species" and keep["w1"]["data_source"] == "eol"


def test_tol_decode_reads_only_kept_members() -> None:
    buf = io.BytesIO()
    img = _jpeg()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("keep-1.jpg", "drop-1.jpg", "keep-2.jpg"):
            info = tarfile.TarInfo(name)
            info.size = len(img)
            tar.addfile(info, io.BytesIO(img))
    ad = HFMemberFilterAdapter({"repo": "imageomics/TreeOfLife-10M"})
    ad._keep = {"keep-1": {"class": "Anthozoa"}, "keep-2": {"class": "Asteroidea"}}
    out = list(ad.decode(_fetched("dataset/EOL/image_set_01.tar.gz", buf.getvalue())))
    assert [d.labels["treeoflife_id"] for d in out] == ["keep-1", "keep-2"]
    assert out[1].labels["class"] == "Asteroidea" and out[0].data == img
    assert ad.params["include"] == ["dataset/*.tar.gz"]


# ---- Wikimedia Commons ------------------------------------------------------------------


def _commons_fake(calls: list[dict]):
    def page(title: str, url: str, lic: str) -> dict:
        em = {
            "LicenseShortName": {"value": lic},
            "Artist": {"value": '<a href="x">Ann &amp; Bo</a>'},
            "GPSLatitude": {"value": "12.5"},
            "GPSLongitude": {"value": "-70.1"},
        }
        return {
            "title": title,
            "imageinfo": [{"url": url, "size": 10, "sha1": "ab", "extmetadata": em}],
        }

    def fake(query: dict) -> dict:
        calls.append(query)
        if query.get("list") == "categorymembers":
            if query["cmtitle"] == "Category:Underwater photographs":
                subs = ["Category:Underwater photographs of fish", "Category:Fish of Aruba"]
                return {"query": {"categorymembers": [{"title": t} for t in subs]}}
            return {"query": {"categorymembers": []}}
        if query["gcmtitle"] == "Category:Underwater photographs" and "gcmcontinue" not in query:
            return {
                "query": {
                    "pages": {"1": page("File:Reef B.jpg", "https://u/b.jpg", "CC BY-SA 4.0")}
                },
                "continue": {"gcmcontinue": "n2", "continue": "gcmcontinue||"},
            }
        if query["gcmtitle"] == "Category:Underwater photographs":
            return {"query": {"pages": {"2": page("File:Reef A.jpg", "https://u/a.jpg", "CC0")}}}
        return {"query": {"pages": {"3": page("File:Reef A.jpg", "https://u/a.jpg", "CC0")}}}

    return fake


def test_commons_walk_filters_subcats_follows_continue_and_carries_licence() -> None:
    calls: list[dict] = []
    ad = make_adapter("commons-api", {"min_interval_s": 0})
    assert isinstance(ad, CommonsAdapter)
    ad._get = _commons_fake(calls)  # type: ignore[method-assign]
    v1 = ad.resolve_version()
    items = list(ad.enumerate())
    assert ad.categories == [
        "Category:Underwater photographs",
        "Category:Underwater photographs of fish",
    ]
    assert [i.key for i in items] == ["Reef_A.jpg", "Reef_B.jpg"]  # deduped, sorted
    assert any("gcmcontinue" in c for c in calls)
    (d,) = list(ad.decode(_fetched("Reef_B.jpg", _jpeg())))
    assert d.fields["license"] == "CC BY-SA 4.0" and d.fields["attribution"] == "Ann & Bo"
    assert d.fields["lat"] == 12.5 and d.labels["commons_category"] == "Underwater photographs"
    ad2 = make_adapter("commons-api", {"min_interval_s": 0})
    ad2._get = _commons_fake([])  # type: ignore[method-assign]
    assert ad2.resolve_version() == v1 and "reef.support" in USER_AGENT


# ---- FathomNet dedupe + per-sample licence ----------------------------------------------


def test_fathomnet_dedupes_sha256_and_excludes_gfisher(tmp_path) -> None:
    excl = tmp_path / "gfisher.txt"
    excl.write_text("CC" * 32 + "\n")

    def entry(uuid: str, sha: str, lic: str) -> dict:
        return {
            "uuid": uuid,
            "url": f"https://f/{uuid}.jpg",
            "sha256": sha,
            "latitude": 36.7,
            "longitude": -122.0,
            "depthMeters": 812.0,
            "boundingBoxes": [{"concept": "Sebastes", "annotationLicense": lic}],
        }

    body = {
        "content": [
            entry("u1", "aa" * 32, "CC0-1.0"),
            entry("u2", "aa" * 32, "CC0-1.0"),
            entry("u3", "cc" * 32, "CC-BY-4.0"),
            entry("u4", "dd" * 32, "CC-BY-NC-4.0"),
        ],
        "totalItems": 4,
    }
    ad = FathomNetAdapter({"version": "v", "full": True, "exclude_sha256": [str(excl)]})
    ad._get = lambda path: body if "page=0&" in path else {"content": []}  # type: ignore[method-assign]
    keys = [i.key for i in ad.enumerate()]
    assert keys == ["u1.jpg", "u4.jpg"] and ad.duplicates == 1 and ad.excluded == 1
    (d,) = list(ad.decode(_fetched("u4.jpg", _jpeg())))
    assert d.fields["license"] == "CC-BY-NC-4.0" and d.fields["depth_m"] == 812.0


@pytest.mark.parametrize("name", ["inat-open-data", "hf-member-filter", "commons-api"])
def test_new_adapters_registered_everywhere(name: str) -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("sd", Path("scripts/spec_dryrun.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert name in mod.SUPPORTED_ADAPTERS
    src = Path("src/marinedata/cli_ingest_source.py").read_text()
    assert f'"{name}"' in src


def test_inat_manifest_from_url_is_cached_and_sha_pinned(tmp_path, monkeypatch) -> None:
    import hashlib

    from marinedata.adapters import _http
    from marinedata.adapters.inat import fetch_manifest

    blob = b"PAR1-manifest-bytes"
    calls: list[str] = []

    def fake_download(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest(), "", len(blob)

    monkeypatch.setattr(_http, "download", fake_download)
    url = "https://b.example/sources/inat-marine/_manifest/m.parquet"
    sha = hashlib.sha256(blob).hexdigest()
    p = fetch_manifest(url, sha, cache=tmp_path)
    assert p == tmp_path / "m.parquet" and p.read_bytes() == blob
    assert fetch_manifest(url, sha, cache=tmp_path) == p and len(calls) == 1  # cache hit
    with pytest.raises(ValueError, match="pinned"):
        fetch_manifest(url, "0" * 64, cache=tmp_path)  # stale cache refetched, then refused
    assert not p.exists()
