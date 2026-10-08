"""One factory for every boto3 S3 client in this package (D9, WP-3).

Every module that talks to the marine-data buckets calls :func:`get_client`; none builds its own
boto3 client or reads an rclone file. The identity comes from the single resolver in
:mod:`marinedata.s3_auth` (environment, then the ``rs-hel1`` rclone remote, else anonymous) and
the client from :func:`marinedata.s3_auth.make_s3_client`. Credentials are never printed, logged
or put in an exception.

* Readers call ``get_client()``: no key resolves -> an anonymous client on the open bucket.
* Writers call ``get_client(signed_required=True)``: no key -> :class:`S3CredentialsError`, never
  an anonymous client that would only fail at the first PUT.
* The cluster Job (``deploy/ingest-job``) injects ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
  / ``S3_ENDPOINT``. Only ``job_env=True`` reads them, so a developer's real AWS key can never be
  sent to the Hetzner endpoint by accident.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .s3_auth import (
    creds_from_job_env,
    creds_from_rclone,
    make_s3_client,
    resolve_s3_credentials,
)

__all__ = ["S3CredentialsError", "client_from_env", "client_from_rclone", "get_client"]


class S3CredentialsError(KeyError):
    """A write path found no signing key. A ``KeyError`` because the constructors this replaces
    raised ``KeyError`` for a missing key, so existing handlers keep working."""

    def __str__(self) -> str:  # KeyError would repr()-quote the message
        return str(self.args[0]) if self.args else ""


def get_client(
    *,
    signed_required: bool = False,
    remote: str | None = None,
    concurrent: bool = False,
    job_env: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """boto3 S3 client from the one credential resolver.

    ``signed_required=True`` is for writers (and readers of a private bucket): with no key it
    raises :class:`S3CredentialsError` instead of returning an anonymous client. ``remote`` picks
    the rclone remote to try (default ``$MARINEDATA_RCLONE_REMOTE`` or ``rs-hel1``); a key in the
    environment still wins. ``concurrent=True`` sizes the pool for ~24 parallel GETs and bounds a
    hung socket. ``job_env=True`` puts the cluster Job's ``AWS_*`` pair first."""
    creds = (creds_from_job_env(environ) if job_env else None) or resolve_s3_credentials(
        environ, remote=remote
    )
    if creds is None and signed_required:
        raise S3CredentialsError(
            "no S3 credentials: set MARINEDATA_S3_ACCESS_KEY and MARINEDATA_S3_SECRET_KEY, or "
            "configure the rs-hel1 rclone remote; a write path never falls back to anonymous"
        )
    return make_s3_client(creds, concurrent=concurrent)


def client_from_rclone(
    remote: str = "rs-hel1", conf: Path | None = None, *, concurrent: bool = False
) -> Any:
    """Signed client for one named rclone remote (compatibility wrapper; prefer
    :func:`get_client`). Raises :class:`S3CredentialsError` when the remote or its key is
    missing, never returns an anonymous client."""
    creds = creds_from_rclone(remote, conf)
    if creds is None:
        raise S3CredentialsError(f"rclone remote {remote!r} not found")
    return make_s3_client(creds, concurrent=concurrent)


def client_from_env(endpoint_url: str | None = None, region: str | None = None) -> Any:
    """Signed client from the cluster Job's ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
    (``S3_ENDPOINT`` / ``S3_REGION`` optional); compatibility wrapper, prefer
    ``get_client(job_env=True)``. Raises :class:`S3CredentialsError` when either key is missing."""
    creds = creds_from_job_env(endpoint_url=endpoint_url, region=region)
    if creds is None:
        raise S3CredentialsError("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not set")
    return make_s3_client(creds)
