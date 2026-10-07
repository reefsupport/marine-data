"""D-AF runner tests: a dead item lands in ``MISSING.tsv`` on S3 and the source still
completes; past a threshold the source aborts before the ``CHECKSUMS`` marker is pushed.
Local HTTP server + moto, like ``test_ingest_source``."""

from __future__ import annotations

import datetime as dt
import json
import urllib.error

import pytest

pytest.importorskip("pyarrow")
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from _wp6_fixtures import LocalServer, png  # noqa: E402

from marinedata import ingest_missing as im  # noqa: E402
from marinedata.ingest_missing import TooManyMissing  # noqa: E402
from marinedata.ingest_source import IngestSpec, run_ingest  # noqa: E402
from marinedata.s3_upload import DiskGuard  # noqa: E402

GOOD = 12


def _env(tmp_path, dead: dict[str, int], garbage_tar: bool = False, good: int = GOOD):
    """``good`` loose PNGs + ``dead[name] = status`` URLs (+ an undecodable .tar)."""
    srv = LocalServer()
    names = []
    for i in range(good):
        srv.add(f"/img{i:03d}.png", png(i, 16))
        names.append(f"img{i:03d}.png")
    for name, status in dead.items():
        srv.add(f"/{name}", b"gone", status=status, ctype="text/plain")
        names.append(name)
    if garbage_tar:
        srv.add("/broken.tar", b"this is not a tar archive" * 40)
        names.append("broken.tar")
    urls = "".join(f"    - url: {srv.base}/{n}\n" for n in sorted(names))
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-dead\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Reef Lab\n"
        f"params:\n  version: '1'\n  urls:\n{urls}\nbucket: bkt\n"
    )
    return srv, IngestSpec.load(spec_path, "http")


def _run(spec, tmp_path, s3, jobs=1):
    guard = DiskGuard(tmp_path / "w", temp_cap_bytes=10**7, floor_bytes=0)
    return run_ingest(
        spec, tmp_path / "w", client=s3, guard=guard, jobs=jobs, fetch_date=dt.date(2026, 9, 25)
    )


def _keys(s3):
    return {o["Key"] for o in s3.list_objects_v2(Bucket="bkt").get("Contents", [])}


@pytest.mark.parametrize("jobs", [1, 4])
def test_runner_skips_404_410_and_undecodable_into_missing_tsv(tmp_path, jobs):
    srv, spec = _env(tmp_path, {"gone404.png": 404, "gone410.png": 410}, garbage_tar=True)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        rep = _run(spec, tmp_path, s3, jobs=jobs)
        pre = f"sources/syn-dead/{rep.version}"
        assert (rep.images, rep.missing) == (GOOD, 3)
        assert f"{pre}/MISSING.tsv" in _keys(s3) and f"{pre}/CHECKSUMS.sha256" in _keys(s3)
        body = s3.get_object(Bucket="bkt", Key=f"{pre}/MISSING.tsv")["Body"].read().decode()
        ingest = json.loads(s3.get_object(Bucket="bkt", Key=f"{pre}/INGEST.json")["Body"].read())
        sums = s3.get_object(Bucket="bkt", Key=f"{pre}/CHECKSUMS.sha256")["Body"].read()
    srv.close()
    status = {line.split("\t")[0]: line.split("\t")[2] for line in body.splitlines()[1:]}
    assert status == {
        "broken.tar": "decode-error:ReadError",
        "gone404.png": "http-404",
        "gone410.png": "http-410",
    }
    assert ingest["missing_items"] == 3 and len(ingest["upstream"]) == GOOD
    assert b"MISSING.tsv" in sums  # part of the staged tree
    stub = (tmp_path / "w" / "registry-stub-syn-dead.yaml").read_text()
    assert "missing_items: 3" in stub


def test_clean_source_has_no_missing_tsv(tmp_path):
    srv, spec = _env(tmp_path, {})
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        rep = _run(spec, tmp_path, s3)
        ingest = json.loads(
            s3.get_object(Bucket="bkt", Key=f"sources/syn-dead/{rep.version}/INGEST.json")[
                "Body"
            ].read()
        )
        assert not any(k.endswith("MISSING.tsv") for k in _keys(s3))
    srv.close()
    assert rep.missing == 0 and "missing_items" not in ingest


@pytest.mark.parametrize(
    ("patch", "dead", "match"),
    [
        ({"MAX_CONSECUTIVE": 3}, {f"z{i}.png": 404 for i in range(3)}, "3 consecutive"),
        ({"MIN_ATTEMPTS": 10}, {"a0.png": 404}, "1/10"),  # 10% after the 10th attempt
    ],
)
def test_runner_aborts_past_a_threshold_and_uploads_no_marker(
    tmp_path, monkeypatch, patch, dead, match
):
    for name, value in patch.items():
        monkeypatch.setattr(im, name, value)
    srv, spec = _env(tmp_path, dead)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        with pytest.raises(TooManyMissing, match=match):
            _run(spec, tmp_path, s3)
        assert not any(k.endswith("CHECKSUMS.sha256") for k in _keys(s3))
    srv.close()


def test_runner_aborts_when_every_item_is_dead(tmp_path):
    srv, spec = _env(tmp_path, {"a.png": 404, "b.png": 410}, good=0)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        with pytest.raises(TooManyMissing, match="all 2"):
            _run(spec, tmp_path, s3)
        assert _keys(s3) == set()
    srv.close()


def test_runner_still_aborts_on_a_non_dead_failure(tmp_path):
    srv, spec = _env(tmp_path, {"bad.png": 400})
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        with pytest.raises(urllib.error.HTTPError):
            _run(spec, tmp_path, s3)
    srv.close()
