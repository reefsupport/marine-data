"""``marinedata auth check`` — who does this process read the open bucket as? (D9 groundwork)

Prints ``signed as <env|remote> (endpoint <host>)`` or ``anonymous`` (the identity
:func:`marinedata.s3_auth.resolve_s3_credentials` picks), then makes exactly two ``HEAD``
requests: the open bucket (``head_bucket``) and one small known key (``head_object``: the
``CHECKSUMS.sha256`` of the registry's ``suim`` source). No listing, no walks, no writes,
no secret in the output. Exit 0 whatever the answer: it is a diagnostic.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.parse
import urllib.request

from .checksums import CHECKSUM_FILE
from .fetch import USER_AGENT
from .s3_auth import OPEN_BUCKET, OPEN_ENDPOINT, auth_headers_for, resolve_s3_credentials

_PROBE_SOURCE = "suim"
_FALLBACK_KEY = f"sources/suim/2020/{CHECKSUM_FILE}"


def _probe_target() -> tuple[str, str, str]:
    """``(bucket, endpoint, key)`` of the probe, read from the registry's ``suim`` entry."""
    try:
        from .registry import Registry, _default_root

        params = Registry.load(_default_root()).source(_PROBE_SOURCE).access.params
        return (
            str(params.get("bucket") or OPEN_BUCKET),
            str(params.get("endpoint") or OPEN_ENDPOINT),
            f"{params.get('prefix', '')}{CHECKSUM_FILE}",
        )
    except Exception:  # registry unavailable: the documented constants still probe the bucket
        return OPEN_BUCKET, OPEN_ENDPOINT, _FALLBACK_KEY


def _head(url: str, creds) -> str:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": USER_AGENT, **auth_headers_for(url, creds, "HEAD")},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return str(response.status)
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except (urllib.error.URLError, OSError) as exc:
        return f"error ({type(exc).__name__})"


def _cmd_check(args: argparse.Namespace) -> int:
    creds = resolve_s3_credentials()
    if creds is None:
        print("anonymous")
    else:
        print(f"signed as {creds.origin} (endpoint {creds.endpoint})")
    bucket, endpoint, key = _probe_target()
    base = f"https://{bucket}.{endpoint}"
    print(f"head_bucket {bucket}: {_head(f'{base}/', creds)}")
    print(f"head_object {key}: {_head(f'{base}/{urllib.parse.quote(key)}', creds)}")
    return 0


def add_auth_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("auth", help="S3 credential diagnostics (read-only)")
    sub = p.add_subparsers(dest="auth_command", required=True)
    check = sub.add_parser(
        "check",
        help="Show the identity bucket reads use; one head_bucket + one head_object, no listing",
    )
    check.set_defaults(func=_cmd_check)
