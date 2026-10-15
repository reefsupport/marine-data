"""``s3_client.get_client``: the one S3 client factory (P1.4b). Readers fall back to anonymous,
writers refuse it, and no module builds its own boto3 client or reads an rclone file."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("boto3")
from botocore import UNSIGNED

from marinedata import s3_upload
from marinedata.s3_auth import (
    ENV_ACCESS_KEY,
    ENV_RCLONE_CONF,
    ENV_SECRET_KEY,
    creds_from_job_env,
    creds_from_rclone,
)
from marinedata.s3_client import (
    S3CredentialsError,
    client_from_env,
    client_from_rclone,
    get_client,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "marinedata"
KEY, SECRET = "AKIATESTKEY0001", "s3cr3t/with%percent"
WRITERS = (
    "ingest_source.py",
    "ingest_stream.py",
    "cli_ingest_batch.py",
    "metadata_release.py",
    "task_layers/sources/coralseg.py",
    "task_layers/sources/coralseg_stream.py",
)


def _conf(path: Path, remote: str = "rs-hel1") -> Path:
    path.write_text(
        f"[{remote}]\ntype = s3\nendpoint = https://hel1.your-objectstorage.com\nregion = hel1\n"
        f"access_key_id = {KEY}\nsecret_access_key = {SECRET}\n"
    )
    return path


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ENV_ACCESS_KEY, ENV_SECRET_KEY, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_reader_falls_back_to_anonymous() -> None:
    assert get_client().meta.config.signature_version is UNSIGNED


def test_writer_refuses_anonymous_with_a_clear_message() -> None:
    with pytest.raises(S3CredentialsError) as exc:
        get_client(signed_required=True)
    assert isinstance(exc.value, KeyError)  # the old constructors raised KeyError
    assert "MARINEDATA_S3_ACCESS_KEY" in str(exc.value)
    assert SECRET not in str(exc.value)


def test_env_key_signs_and_satisfies_a_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ACCESS_KEY, KEY)
    monkeypatch.setenv(ENV_SECRET_KEY, SECRET)
    client = get_client(signed_required=True)
    assert client.meta.config.signature_version is not UNSIGNED
    assert client.meta.endpoint_url == "https://hel1.your-objectstorage.com"


def test_remote_picks_the_rclone_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_RCLONE_CONF, str(_conf(tmp_path / "rclone.conf", remote="other")))
    assert get_client(signed_required=True, remote="other").meta.config.signature_version is not (
        UNSIGNED
    )
    with pytest.raises(S3CredentialsError):  # the default remote is not in that file
        get_client(signed_required=True)


def test_aws_pair_is_read_only_for_the_cluster_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("S3_ENDPOINT", "s3.example.test")
    monkeypatch.setenv("S3_REGION", "eu-test-1")
    with pytest.raises(S3CredentialsError):  # a developer's real AWS key is never picked up
        get_client(signed_required=True)
    client = get_client(signed_required=True, job_env=True)
    assert client.meta.endpoint_url == "https://s3.example.test"


def test_job_creds_without_endpoint_mean_the_sdk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    creds = creds_from_job_env()
    assert creds is not None and creds.endpoint_url == ""
    assert SECRET not in repr(creds)
    assert "hel1" not in client_from_env().meta.endpoint_url


def test_compat_constructors_never_return_anonymous(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        client_from_rclone("rs-hel1", tmp_path / "absent.conf")
    with pytest.raises(KeyError):
        client_from_env()
    conf = _conf(tmp_path / "rclone.conf")
    assert client_from_rclone("rs-hel1", conf).meta.config.signature_version is not UNSIGNED
    assert s3_upload.client_from_rclone is client_from_rclone  # public names kept
    assert s3_upload.client_from_env is client_from_env


def test_creds_from_rclone_reads_one_remote_without_interpolation(tmp_path: Path) -> None:
    conf = _conf(tmp_path / "rclone.conf")
    creds = creds_from_rclone("rs-hel1", conf)
    assert creds is not None and creds.secret_key == SECRET and creds.origin == "rs-hel1"
    assert SECRET not in repr(creds)
    assert creds_from_rclone("nope", conf) is None
    assert creds_from_rclone("rs-hel1", tmp_path / "absent.conf") is None


def test_error_message_is_not_repr_quoted() -> None:
    assert str(S3CredentialsError("plain message")) == "plain message"


@pytest.mark.parametrize("rel", WRITERS)
def test_writers_ask_for_a_signed_client(rel: str) -> None:
    tree = ast.parse((SRC / rel).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "get_client"
    ]
    assert calls, f"{rel} no longer obtains its client through get_client"
    for call in calls:
        flag = {kw.arg: kw.value for kw in call.keywords}.get("signed_required")
        assert isinstance(flag, ast.Constant) and flag.value is True, rel
