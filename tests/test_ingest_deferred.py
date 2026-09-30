"""D-AJ: a tape-recall 503 defers an item instead of failing it (see ``ingest_deferred``
and ``adapters._http.TapeRecall``). Local HTTP server + moto, like ``test_ingest_source``."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("pyarrow")
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from _wp6_fixtures import LocalServer, png  # noqa: E402

from marinedata.ingest_source import IngestSpec, run_ingest  # noqa: E402
from marinedata.s3_upload import DiskGuard  # noqa: E402

GOOD = 3
TAPE_BODY = b"This file is loading from tape. Please retry shortly."


def _env(tmp_path, tape: dict[str, int]):
    """``GOOD`` loose PNGs + ``tape[name] = 503`` urls answering a tape-recall body."""
    srv = LocalServer()
    names = []
    for i in range(GOOD):
        srv.add(f"/img{i:03d}.png", png(i, 16))
        names.append(f"img{i:03d}.png")
    for name in tape:
        srv.add(f"/{name}", TAPE_BODY, status=503, ctype="text/plain")
        names.append(name)
    urls = "".join(f"    - url: {srv.base}/{n}\n" for n in sorted(names))
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: syn-tape\nadapter: http\nlicense: CC-BY-4.0\nattribution: Synthetic Reef Lab\n"
        f"params:\n  version: '1'\n  urls:\n{urls}\nbucket: bkt\n"
    )
    return srv, IngestSpec.load(spec_path, "http")


def _run(spec, tmp_path, s3):
    guard = DiskGuard(tmp_path / "w", temp_cap_bytes=10**7, floor_bytes=0)
    return run_ingest(
        spec, tmp_path / "w", client=s3, guard=guard, jobs=1, fetch_date=dt.date(2026, 9, 25)
    )


def test_tape_503_is_deferred_not_missing_then_fetched_next_pass(tmp_path):
    tape_names = {"tape1.png": 503, "tape2.png": 503}
    srv, spec = _env(tmp_path, tape_names)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")

        rep = _run(spec, tmp_path, s3)
        assert (rep.images, rep.missing) == (GOOD, 0)  # deferred never counts as missing

        ledger = tmp_path / "w" / "deferred" / f"syn-tape-{rep.version}.tsv"
        pending = {ln.split("\t")[0] for ln in ledger.read_text().splitlines()[1:]}
        assert pending == set(tape_names)

        # the recall completes: same URLs now answer 200
        for name in tape_names:
            srv.add(f"/{name}", png(99, 16))

        rep2 = _run(spec, tmp_path, s3)
        assert (rep2.images, rep2.missing) == (GOOD + len(tape_names), 0)
        assert rep2.plan["first_keys"][0] in tape_names  # retried first, per D-AJ
        assert ledger.read_text().strip() == "key\turl"  # nothing pending once resolved
    srv.close()
