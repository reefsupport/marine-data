"""One credential resolver for every read of the marine-data bucket (D9 groundwork).

Order: environment (``MARINEDATA_S3_ACCESS_KEY`` + ``MARINEDATA_S3_SECRET_KEY``, optional
``MARINEDATA_S3_ENDPOINT`` / ``MARINEDATA_S3_REGION``), else the rclone remote
(``$MARINEDATA_RCLONE_REMOTE``, default ``rs-hel1``, in ``$MARINEDATA_RCLONE_CONF`` or
``~/.config/rclone/rclone.conf``, read with ``configparser``), else ``None`` = anonymous.
Callers sign when they hold a key and fall back to anonymous, never the reverse.
The cluster Job's legacy ``AWS_*`` pair is a separate opt-in (:func:`creds_from_job_env`).

Secrets are never printed, logged or put in an exception: :class:`S3Creds` redacts its own
``repr``. The module is import-light (stdlib only); boto3 is imported inside
:func:`make_s3_client`. SigV4 for plain ``urllib`` GET/HEAD lives here too
(:func:`auth_headers_for`) so the dependency-free fetchers can sign without boto3.
"""

from __future__ import annotations

import configparser
import hashlib
import hmac
import os
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OPEN_BUCKET = "rs-storage-open"
OPEN_ENDPOINT = "hel1.your-objectstorage.com"
OPEN_REGION = "hel1"
DEFAULT_REMOTE = "rs-hel1"
DEFAULT_RCLONE_CONF = Path("~/.config/rclone/rclone.conf")

ENV_ACCESS_KEY = "MARINEDATA_S3_ACCESS_KEY"
ENV_SECRET_KEY = "MARINEDATA_S3_SECRET_KEY"
ENV_ENDPOINT = "MARINEDATA_S3_ENDPOINT"
ENV_REGION = "MARINEDATA_S3_REGION"
ENV_RCLONE_CONF = "MARINEDATA_RCLONE_CONF"
ENV_RCLONE_REMOTE = "MARINEDATA_RCLONE_REMOTE"


@dataclass(frozen=True, repr=False)
class S3Creds:
    """A resolved S3 identity. ``origin`` is ``"env"`` or the rclone remote's name."""

    access_key: str
    secret_key: str
    endpoint: str
    region: str
    origin: str

    def __repr__(self) -> str:  # the secret key (and most of the key id) never leaves the object
        return (
            f"S3Creds(origin={self.origin!r}, endpoint={self.endpoint!r}, "
            f"region={self.region!r}, access_key='...{self.access_key[-4:]}', secret_key='***')"
        )

    @property
    def endpoint_url(self) -> str:
        """The endpoint as a URL; ``""`` (cluster Job identity only) means the SDK default."""
        if not self.endpoint or self.endpoint.startswith(("http://", "https://")):
            return self.endpoint
        return f"https://{self.endpoint}"

    @property
    def endpoint_host(self) -> str:
        return urllib.parse.urlsplit(self.endpoint_url).hostname or ""


def _build(
    access: str, secret: str, endpoint: str | None, region: str | None, origin: str
) -> S3Creds:
    ep = (endpoint or "").strip().rstrip("/") or OPEN_ENDPOINT
    host = urllib.parse.urlsplit(ep if "//" in ep else f"https://{ep}").hostname
    default_region = OPEN_REGION if host == OPEN_ENDPOINT else "us-east-1"
    return S3Creds(access, secret, ep, (region or "").strip() or default_region, origin)


def _from_env(environ: Mapping[str, str]) -> S3Creds | None:
    access = (environ.get(ENV_ACCESS_KEY) or "").strip()
    secret = (environ.get(ENV_SECRET_KEY) or "").strip()
    if not (access and secret):  # half a pair is ignored, never mixed with another source
        return None
    return _build(access, secret, environ.get(ENV_ENDPOINT), environ.get(ENV_REGION), "env")


def creds_from_rclone(
    remote: str = DEFAULT_REMOTE, conf: Path | str | None = None
) -> S3Creds | None:
    """The key of one named rclone remote, or ``None`` when the file, the section or the key is
    absent. The only place in the package that parses an rclone config."""
    path = Path(conf or DEFAULT_RCLONE_CONF).expanduser()
    parser = configparser.ConfigParser(interpolation=None)  # secrets may contain '%'
    try:
        if not parser.read(path) or not parser.has_section(remote):
            return None
    except (configparser.Error, OSError, UnicodeDecodeError):
        return None
    sec = parser[remote]
    access = (sec.get("access_key_id") or "").strip()
    secret = (sec.get("secret_access_key") or "").strip()
    if not (access and secret):
        return None
    return _build(access, secret, sec.get("endpoint"), sec.get("region"), remote)


def _from_rclone(environ: Mapping[str, str], remote: str | None = None) -> S3Creds | None:
    return creds_from_rclone(
        remote or environ.get(ENV_RCLONE_REMOTE) or DEFAULT_REMOTE, environ.get(ENV_RCLONE_CONF)
    )


def resolve_s3_credentials(
    environ: Mapping[str, str] | None = None, *, remote: str | None = None
) -> S3Creds | None:
    """The S3 identity this process should read with, or ``None`` for anonymous. ``remote`` picks
    the rclone remote to try (default ``$MARINEDATA_RCLONE_REMOTE`` or ``rs-hel1``); a key in the
    environment still wins."""
    env = os.environ if environ is None else environ
    return _from_env(env) or _from_rclone(env, remote)


def creds_from_job_env(
    environ: Mapping[str, str] | None = None,
    *,
    endpoint_url: str | None = None,
    region: str | None = None,
) -> S3Creds | None:
    """The cluster Job's identity: ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY`` with optional
    ``S3_ENDPOINT`` / ``S3_REGION`` (``deploy/ingest-job``), or ``None`` when either key is
    missing; an empty endpoint means the SDK default. Deliberately not part of
    :func:`resolve_s3_credentials`: a developer's real AWS key must never be sent to the Hetzner
    endpoint, so only callers that ask for the Job path read these variables."""
    env = os.environ if environ is None else environ
    access = (env.get("AWS_ACCESS_KEY_ID") or "").strip()
    secret = (env.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if not (access and secret):
        return None
    endpoint = (endpoint_url or env.get("S3_ENDPOINT") or "").strip().rstrip("/")
    return S3Creds(
        access, secret, endpoint, (region or env.get("S3_REGION") or "").strip(), "job-env"
    )


def make_s3_client(creds: S3Creds | None, *, concurrent: bool = False) -> Any:
    """boto3 S3 client: signed with ``creds``, or an ``UNSIGNED`` anonymous client on the
    open bucket's endpoint when ``creds`` is ``None``. ``concurrent=True`` sizes the pool for
    ~24 parallel GETs and bounds a hung socket (connect 15 s, read 30 s)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    pool = (
        {"max_pool_connections": 48, "connect_timeout": 15, "read_timeout": 30}
        if concurrent
        else {}
    )
    retries = {"max_attempts": 8, "mode": "standard"}
    if creds is None:
        return boto3.client(
            "s3",
            endpoint_url=f"https://{OPEN_ENDPOINT}",
            region_name=OPEN_REGION,
            config=Config(signature_version=UNSIGNED, retries=retries, **pool),
        )
    return boto3.client(
        "s3",
        endpoint_url=creds.endpoint_url or None,
        region_name=creds.region or None,
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        config=Config(
            retries=retries,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            **pool,
        ),
    )


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sigv4_headers(
    method: str, host: str, path: str, query: str, region: str, access: str, secret: str
) -> dict[str, str]:
    """Minimal AWS SigV4 for GET requests with an empty payload."""
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical = f"{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    scope = f"{date_stamp}/{region}/s3/aws4_request"
    to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    )

    signing_key = _sign(
        _sign(_sign(_sign(f"AWS4{secret}".encode(), date_stamp), region), "s3"), "aws4_request"
    )
    signature = hmac.new(signing_key, to_sign.encode(), hashlib.sha256).hexdigest()

    return {
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }


def auth_headers_for(url: str, creds: S3Creds | None, method: str = "GET") -> dict[str, str]:
    """SigV4 headers for an empty-payload ``method`` request to ``url``, or ``{}`` (anonymous)
    when there is no key or the host is not the credentials' own endpoint — a signature is
    never sent to a third-party host."""
    if creds is None:
        return {}
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    own = creds.endpoint_host
    if not own or not (host == own or host.endswith(f".{own}")):
        return {}
    return _sigv4_headers(
        method, parts.netloc, parts.path or "/", parts.query, creds.region,
        creds.access_key, creds.secret_key,
    )  # fmt: skip
