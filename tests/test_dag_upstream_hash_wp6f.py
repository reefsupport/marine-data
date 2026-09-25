"""D-AG (WP-6f-b): a provider's declared hash that disagrees with the served bytes is recorded,
the bytes are kept; only a byte-count mismatch (short transfer) still raises."""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from marinedata.adapters import Fetched, RemoteItem, _check_declared
from marinedata.adapters._http import HashingReader
from marinedata.adapters.fathomnet import FathomNetAdapter


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_check_declared_records_not_raises():
    data = _png()
    sha, md5 = hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()
    ok = RemoteItem("a.png", "u", sha256=sha.upper(), md5=md5)
    bad = RemoteItem("a.png", "u", sha256="0" * 64)
    assert _check_declared(ok, sha, md5, len(data)) is True
    assert _check_declared(bad, sha, md5, len(data)) is False
    assert _check_declared(RemoteItem("a.png", "u"), sha, md5, len(data)) is None
    with pytest.raises(ValueError, match="bytes != declared"):
        _check_declared(RemoteItem("a.png", "u", size=1), sha, md5, len(data))


@pytest.mark.parametrize("declared_ok", [True, False])
def test_fathomnet_keeps_mismatched_bytes(declared_ok):
    data = _png()
    ours = hashlib.sha256(data).hexdigest()
    declared = ours if declared_ok else "f" * 64
    ad = FathomNetAdapter({"version": "v"})
    ad._meta["u1"] = {"fields": {}, "labels": {"concepts": "x"}, "raw": {"sha256": declared}}
    item = RemoteItem("u1.png", "https://x/u1.png", sha256=declared)
    fetched = Fetched(item, stream=HashingReader(io.BytesIO(data)))
    out = list(ad.decode(fetched))
    fetched.close()  # must not raise on a declared-hash mismatch (D-AG)
    assert len(out) == 1 and out[0].data == data
    assert out[0].fields["upstream_sha256"] == declared
    assert out[0].fields["upstream_sha256_match"] is declared_ok
    assert out[0].labels["upstream_sha256_match"] == str(declared_ok).lower()
    assert fetched.sha256 == ours and fetched.upstream_match is declared_ok
    assert ad.upstream_mismatch == (0 if declared_ok else 1)


def test_fathomnet_undecodable_still_fails():
    ad = FathomNetAdapter({"version": "v"})
    ad._meta["u2"] = {"fields": {}, "labels": {}, "raw": {"sha256": "f" * 64}}
    fetched = Fetched(
        RemoteItem("u2.zzz", "https://x/u2.zzz"), stream=HashingReader(io.BytesIO(b"x"))
    )
    with pytest.raises(ValueError):
        list(ad.decode(fetched))
