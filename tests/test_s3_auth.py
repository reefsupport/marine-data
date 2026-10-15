"""Credential resolver, signed-read helpers, fetch integrity. Offline: no network, no real keys."""

from __future__ import annotations

import hashlib
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from marinedata import cli_auth, fetch
from marinedata.fetch import FetchError, IntegrityError, get_to_file
from marinedata.s3_auth import (
    S3Creds,
    auth_headers_for,
    make_s3_client,
    resolve_s3_credentials,
)
from marinedata.task_layers import s3_keyed

OPEN_HOST = "rs-storage-open.hel1.your-objectstorage.com"
SECRET = "s3cr3t/with%percent"
ENV = {"MARINEDATA_S3_ACCESS_KEY": "AKENV0001", "MARINEDATA_S3_SECRET_KEY": "envsecret"}


def _conf(path: Path, remote: str = "rs-hel1", secret: str = SECRET) -> Path:
    path.write_text(
        f"[{remote}]\ntype = s3\naccess_key_id = AKRCLONE99\nsecret_access_key = {secret}\n"
        "endpoint = hel1.your-objectstorage.com\nregion = hel1\n"
    )
    return path


def _creds() -> S3Creds:
    return S3Creds("AKTEST1234", SECRET, "hel1.your-objectstorage.com", "hel1", "rs-hel1")


# ------------------------------------------------------------------ resolver


def test_env_wins_over_rclone(tmp_path: Path) -> None:
    conf = _conf(tmp_path / "rclone.conf")
    env = {
        **ENV,
        "MARINEDATA_S3_ENDPOINT": "https://s3.example.test/",
        "MARINEDATA_S3_REGION": "eu-x",
    }
    creds = resolve_s3_credentials({**env, "MARINEDATA_RCLONE_CONF": str(conf)})
    assert creds is not None
    assert (creds.origin, creds.access_key, creds.secret_key) == ("env", "AKENV0001", "envsecret")
    assert (creds.endpoint, creds.endpoint_host, creds.region) == (
        "https://s3.example.test",
        "s3.example.test",
        "eu-x",
    )


def test_env_defaults_to_the_open_bucket_endpoint() -> None:
    creds = resolve_s3_credentials(ENV)
    assert creds is not None
    assert (creds.endpoint, creds.region) == ("hel1.your-objectstorage.com", "hel1")


def test_rclone_fallback_reads_default_remote_and_keeps_percent_secrets(tmp_path: Path) -> None:
    conf = _conf(tmp_path / "rclone.conf")
    creds = resolve_s3_credentials({"MARINEDATA_RCLONE_CONF": str(conf)})
    assert creds is not None
    assert (creds.origin, creds.access_key, creds.secret_key) == ("rs-hel1", "AKRCLONE99", SECRET)
    assert creds.endpoint_url == "https://hel1.your-objectstorage.com"


def test_rclone_remote_is_selectable(tmp_path: Path) -> None:
    conf = _conf(tmp_path / "rclone.conf", remote="other")
    env = {"MARINEDATA_RCLONE_CONF": str(conf)}
    assert resolve_s3_credentials(env) is None  # default remote rs-hel1 is absent
    picked = resolve_s3_credentials({**env, "MARINEDATA_RCLONE_REMOTE": "other"})
    assert picked is not None and picked.origin == "other"


def test_anonymous_fallback(tmp_path: Path) -> None:
    assert (
        resolve_s3_credentials({"MARINEDATA_RCLONE_CONF": str(tmp_path / "missing.conf")}) is None
    )
    bad = tmp_path / "bad.conf"
    bad.write_text("not an ini file\n[unterminated\n")
    assert resolve_s3_credentials({"MARINEDATA_RCLONE_CONF": str(bad)}) is None
    half = {"MARINEDATA_S3_ACCESS_KEY": "AKONLY", "MARINEDATA_RCLONE_CONF": str(tmp_path / "x")}
    assert resolve_s3_credentials(half) is None  # half a pair is never used


def test_repr_and_str_redact_the_secret() -> None:
    creds = _creds()
    for text in (repr(creds), str(creds), f"{creds}", f"{[creds]}"):
        assert SECRET not in text and "AKTEST1234" not in text
        assert "secret_key='***'" in text or "***" in text


def test_make_s3_client_unsigned_without_creds_and_signed_with() -> None:
    pytest.importorskip("boto3")
    from botocore import UNSIGNED

    anon = make_s3_client(None)
    assert anon.meta.config.signature_version is UNSIGNED
    assert anon.meta.endpoint_url == "https://hel1.your-objectstorage.com"
    signed = make_s3_client(_creds(), concurrent=True)
    assert signed.meta.config.signature_version is not UNSIGNED
    assert signed.meta.endpoint_url == "https://hel1.your-objectstorage.com"


def test_auth_headers_sign_only_the_credentials_own_host() -> None:
    url = f"https://{OPEN_HOST}/sources/suim/2020/CHECKSUMS.sha256"
    own = auth_headers_for(url, _creds())
    assert own["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKTEST1234/")
    assert SECRET not in " ".join(own.values())
    assert auth_headers_for("https://evil.example.com/x", _creds()) == {}
    assert auth_headers_for(url, None) == {}


# ------------------------------------------------------------------ _get: signed, then anonymous


class _Resp:
    """Minimal urlopen response: ``read(amt)`` chunks, optional declared length."""

    def __init__(self, body: bytes, *, declared: int | None = None) -> None:
        self._body, self._pos = body, 0
        self.headers = {"Content-Length": str(len(body) if declared is None else declared)}

    def read(self, amt: int = -1) -> bytes:
        end = len(self._body) if amt is None or amt < 0 else self._pos + amt
        out, self._pos = self._body[self._pos : end], min(end, len(self._body))
        return out

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "x", {}, None)  # type: ignore[arg-type]


def test_get_signs_then_falls_back_to_anonymous_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    url = f"https://{OPEN_HOST}/k"

    def urlopen(request: urllib.request.Request, timeout: int) -> _Resp:
        signed = request.get_header("Authorization") is not None
        seen.append(signed)
        if signed:
            raise _http_error(url, 403)
        return _Resp(b"public")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch._get(url, retries=1, creds=_creds()) == b"public"
    assert seen == [True, False]  # signed first, anonymous second, never the reverse


def test_get_without_creds_never_signs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def urlopen(request: urllib.request.Request, timeout: int) -> _Resp:
        seen.append(request.get_header("Authorization"))
        return _Resp(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert fetch._get(f"https://{OPEN_HOST}/k") == b"ok"
    assert seen == [None]


def test_get_retries_a_truncated_body(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = iter([_Resp(b"abcd", declared=10), _Resp(b"0123456789")])
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: next(bodies))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch._get("https://example.test/k") == b"0123456789"


# ------------------------------------------------------------------ get_to_file: integrity + .part


def test_get_to_file_verified_download_renames_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"x" * 3000
    dest = tmp_path / "a" / "b.bin"
    trace: list[tuple[bool, bool]] = []

    class Tracing(_Resp):
        def read(self, amt: int = -1) -> bytes:
            trace.append((dest.exists(), dest.with_name("b.bin.part").exists()))
            return super().read(amt)

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: Tracing(body))
    digest = hashlib.sha256(body).hexdigest()
    assert get_to_file("https://example.test/b", dest, expected_sha256=f"sha256:{digest}") == digest
    assert dest.read_bytes() == body and not dest.with_name("b.bin.part").exists()
    assert all(not final for final, _ in trace), "dest must not exist while the body streams"
    assert any(part for _, part in trace[1:]), "bytes are staged in <dest>.part first"


def test_get_to_file_mismatch_raises_integrity_error_and_keeps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Resp(b"TAMPERED"))
    dest = tmp_path / "img.jpg"
    expected = hashlib.sha256(b"original").hexdigest()
    actual = hashlib.sha256(b"TAMPERED").hexdigest()
    with pytest.raises(IntegrityError) as err:
        get_to_file(
            "https://example.test/img", dest, expected_sha256=expected, label="src:images/img.jpg"
        )
    assert isinstance(err.value, FetchError)
    message = str(err.value)
    assert "src:images/img.jpg" in message and expected in message and actual in message
    assert not dest.exists() and not dest.with_name("img.jpg.part").exists()


def test_get_to_file_partial_download_never_reaches_dest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"y" * 2000
    dest = tmp_path / "f.bin"
    state = {"calls": 0}

    class Dropping(_Resp):
        def read(self, amt: int = -1) -> bytes:
            if state["calls"] == 1 and self._pos:
                raise http.client.IncompleteRead(b"y")  # connection dropped mid-body
            return super().read(amt)

    def urlopen(request: urllib.request.Request, timeout: int) -> _Resp:
        state["calls"] += 1
        assert not dest.exists()
        return Dropping(body[:1000] if state["calls"] == 1 else body, declared=len(body))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    get_to_file("https://example.test/f", dest, expected_sha256=hashlib.sha256(body).hexdigest())
    assert state["calls"] == 2 and dest.read_bytes() == body


def test_get_to_file_retries_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def urlopen(request: urllib.request.Request, timeout: int) -> _Resp:
        calls.append(1)
        raise _http_error("https://example.test/f", 503)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    with pytest.raises(FetchError, match="failed after 3 attempt"):
        get_to_file("https://example.test/f", tmp_path / "f.bin")
    assert len(calls) == 3 and sleeps == [1.5, 3.0]
    assert not (tmp_path / "f.bin").exists() and not (tmp_path / "f.bin.part").exists()


def test_write_goes_through_part_and_leaves_no_part(tmp_path: Path) -> None:
    fetch._write(tmp_path / "d" / "x.bin", b"data")
    assert (tmp_path / "d" / "x.bin").read_bytes() == b"data"
    assert not (tmp_path / "d" / "x.bin.part").exists()


# ------------------------------------------------------------------ fetch_small + CLI


def test_fetch_small_signs_when_creds_resolve_then_drops_to_anonymous(tmp_path: Path) -> None:
    targets: list[object] = []

    def opener(target: object, timeout: float) -> _Resp:
        targets.append(target)
        if isinstance(target, urllib.request.Request):
            raise _http_error(target.full_url, 403)
        return _Resp(b"label")

    data = s3_keyed.fetch_small(
        "sources/t/v/labels/a.png", opener=opener, sleep=lambda s: None, creds=_creds()
    )
    assert data == b"label"
    assert isinstance(targets[0], urllib.request.Request) and isinstance(targets[1], str)
    anon = s3_keyed.fetch_small(
        "sources/t/v/labels/a.png", opener=opener, sleep=lambda s: None, creds=None
    )
    assert anon == b"label" and isinstance(targets[-1], str)


def test_auth_check_reports_identity_without_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    heads: list[str] = []
    monkeypatch.setattr(cli_auth, "_head", lambda url, creds: heads.append(url) or "200")
    monkeypatch.setenv("MARINEDATA_S3_ACCESS_KEY", "AKCLI00001")
    monkeypatch.setenv("MARINEDATA_S3_SECRET_KEY", "clisecret")
    assert cli_auth._cmd_check(None) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "signed as env (endpoint hel1.your-objectstorage.com)"
    assert "clisecret" not in out and "AKCLI00001" not in out
    assert len(heads) == 2 and heads[0].endswith("/") and heads[1].endswith("CHECKSUMS.sha256")
    monkeypatch.delenv("MARINEDATA_S3_ACCESS_KEY")
    assert cli_auth._cmd_check(None) == 0  # type: ignore[arg-type]
    assert capsys.readouterr().out.splitlines()[0] == "anonymous"
