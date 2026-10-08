"""SigV4 signer + S3-listing XML-safety tests. No network: the signer is pure
string/HMAC math, and the XML tests feed literal bytes straight to `_parse_s3_xml`.

SigV4 vectors, honestly: this worker had no network access to fetch AWS's own
published SigV4 test-suite fixtures (and embedding a live fetch into a test would
itself violate "no network calls in the test suite"), and reciting a remembered hex
signature from training data risked planting a wrong "expected" constant that nothing
here could independently check — exactly the fabrication risk this programme exists to
avoid. So instead: a clean-room reference implementation, typed directly from AWS's
publicly documented Signature Version 4 algorithm (independent of `_sigv4_headers`'s
code, not merely re-imported), is diffed against the real function's output across
several varied inputs, plus sensitivity checks that each signed input actually changes
the signature. This does not prove byte-for-byte agreement with AWS's own fixture
file, but it does prove internal correctness against the documented algorithm from a
second, independently-written implementation — see `## Open` in the MD-4 report.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from marinedata.fetch import FetchError
from marinedata.fetchers_remote import _parse_s3_xml, _sigv4_headers

EMPTY_PAYLOAD_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
"""SHA-256 of b"" — a fixed mathematical constant, verified live in this sandbox via
`hashlib.sha256(b"").hexdigest()` before being hardcoded here, not recalled from
memory: `_sigv4_headers` always signs an empty body (GET-only), so this is what
`x-amz-content-sha256` must equal on every call regardless of any other input."""


def _reference_sigv4(
    method: str,
    host: str,
    path: str,
    query: str,
    region: str,
    access: str,
    secret: str,
    amz_date: str,
) -> tuple[str, str]:
    """Independent re-derivation of AWS's documented SigV4 algorithm (canonical
    request -> string to sign -> 4-round HMAC key derivation -> signature), typed
    fresh here rather than reusing any part of `fetchers_remote._sigv4_headers`.
    Returns (signature_hex, credential_scope).
    """
    payload_hash = hashlib.sha256(b"").hexdigest()
    date_stamp = amz_date[:8]

    canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = (
        f"{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )

    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_date = _hmac(f"AWS4{secret}".encode(), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, "s3")
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return signature, scope


_CASES = [
    (
        "GET",
        "examplebucket.s3.amazonaws.com",
        "/",
        "list-type=2",
        "us-east-1",
        "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    (
        "GET",
        "my-mermaid-mirror.s3.eu-west-2.amazonaws.com",
        "/",
        "list-type=2&prefix=images%2F",
        "eu-west-2",
        "AKIDEXAMPLE",
        "anotherSecretKeyValueEXAMPLEEXAMPLE12345",
    ),
    (
        "GET",
        "reef-support-data.s3.us-west-1.amazonaws.com",
        "/",
        "",
        "us-west-1",
        "AKIATESTACCESSKEY0001",
        "SecretKeyForTestingOnlyNotARealCredential",
    ),
]


@pytest.mark.parametrize("method,host,path,query,region,access,secret", _CASES)
def test_sigv4_signature_matches_independent_reference(
    method: str, host: str, path: str, query: str, region: str, access: str, secret: str
) -> None:
    """⭐ Acceptance: SigV4 signer tested against published-algorithm vectors — see the
    module docstring for why these are a clean-room re-derivation rather than literal
    AWS fixture-file hex. `_sigv4_headers` stamps its own `now()`, so the real call's
    `x-amz-date` is captured and fed into the reference rather than assumed."""
    headers = _sigv4_headers(method, host, path, query, region, access, secret)
    amz_date = headers["x-amz-date"]
    expected_signature, expected_scope = _reference_sigv4(
        method, host, path, query, region, access, secret, amz_date
    )
    assert headers["x-amz-content-sha256"] == EMPTY_PAYLOAD_SHA256
    expected_auth = (
        f"AWS4-HMAC-SHA256 Credential={access}/{expected_scope}, "
        f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={expected_signature}"
    )
    assert headers["Authorization"] == expected_auth


def test_sigv4_authorization_header_shape() -> None:
    headers = _sigv4_headers(
        "GET", "bucket.s3.amazonaws.com", "/", "list-type=2", "us-east-1", "AKID", "secret"
    )
    auth = headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKID/")
    assert "SignedHeaders=host;x-amz-content-sha256;x-amz-date" in auth
    signature = auth.rsplit("Signature=", 1)[1]
    assert len(signature) == 64
    int(signature, 16)  # must be valid hex


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "HEAD"),
        ("host", "other-bucket.s3.amazonaws.com"),
        ("path", "/other"),
        ("query", "list-type=2&prefix=x"),
        ("region", "eu-west-2"),
        ("access", "AKIDIFFERENT"),
        ("secret", "a-totally-different-secret"),
    ],
)
def test_sigv4_signature_is_sensitive_to_every_signed_input(field: str, value: str) -> None:
    """Changing any single signed input must change the signature — catches the class
    of bug where a field is accepted but silently dropped from the canonical request
    (e.g. the path or query never actually reaching the signature)."""
    base = dict(
        method="GET",
        host="bucket.s3.amazonaws.com",
        path="/",
        query="list-type=2",
        region="us-east-1",
        access="AKID",
        secret="secret",
    )
    baseline = _sigv4_headers(**base)["Authorization"]  # type: ignore[arg-type]
    varied = dict(base, **{field: value})
    changed = _sigv4_headers(**varied)["Authorization"]  # type: ignore[arg-type]
    assert changed != baseline


# ── S3 listing XML safety (`_parse_s3_xml`) ───────────────────────────────────────


def test_parse_s3_xml_returns_element_for_a_normal_listing() -> None:
    payload = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<Name>bucket</Name></ListBucketResult>"
    )
    el = _parse_s3_xml("fixture", "bucket.s3.amazonaws.com", payload)
    assert el.tag.endswith("ListBucketResult")


def test_parse_s3_xml_rejects_a_doctype() -> None:
    """⭐ The billion-laughs guard: a DOCTYPE must be refused *before* `ET.fromstring`
    ever sees it, closing the entity-expansion vector verified live against this
    interpreter (see `_parse_s3_xml`'s docstring) — not merely assumed safe."""
    bomb = b'<?xml version="1.0"?><!DOCTYPE lolz ['
    bomb += b'<!ENTITY a "1234567890">'
    bomb += b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
    bomb += b"]><lolz>&b;</lolz>"
    with pytest.raises(FetchError, match="DOCTYPE"):
        _parse_s3_xml("fixture", "bucket.s3.amazonaws.com", bomb)


def test_parse_s3_xml_raises_fetcherror_on_malformed_xml() -> None:
    with pytest.raises(FetchError, match="unparseable"):
        _parse_s3_xml("fixture", "bucket.s3.amazonaws.com", b"not xml at all <<<")
