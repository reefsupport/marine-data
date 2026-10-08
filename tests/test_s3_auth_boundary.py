"""The S3 credential boundary: one module resolves credentials, one family builds clients.

`s3_auth.py` is the only file under ``src/`` that names the rclone config file. `s3_auth.py` and
`s3_client.py` are the only files that build a boto3 S3 client or session. Anything else that
needs a bucket goes through ``s3_client.get_client``, so a new constructor elsewhere fails here
instead of in review.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDENTIAL_FILES = {"src/marinedata/s3_auth.py"}
CLIENT_FILES = {"src/marinedata/s3_auth.py", "src/marinedata/s3_client.py"}
# scripts/build_imos_manifest.py reads the public AWS IMOS bucket anonymously (another endpoint,
# not our storage credentials), so it builds its own unsigned client and is exempt from the
# client rule.
CLIENT_EXEMPT = {"scripts/build_imos_manifest.py"}

CREDENTIAL_RE = re.compile(r"rclone\.conf|RCLONE_CONFIG")
CLIENT_RE = re.compile(r"""boto3\.client\(\s*["']s3["']|boto3\.Session\b""")


def _files_matching(pattern: re.Pattern[str], *roots: str) -> set[str]:
    found = set()
    for root in roots:
        for path in (ROOT / root).rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                found.add(path.relative_to(ROOT).as_posix())
    return found


def test_rclone_config_is_named_only_in_s3_auth():
    hits = _files_matching(CREDENTIAL_RE, "src")
    assert hits == CREDENTIAL_FILES, (
        f"rclone.conf / RCLONE_CONFIG outside the resolver: {sorted(hits - CREDENTIAL_FILES)}"
    )


def test_boto3_s3_clients_are_built_only_in_the_client_modules():
    hits = _files_matching(CLIENT_RE, "src", "scripts") - CLIENT_EXEMPT
    assert hits <= CLIENT_FILES, (
        f"boto3 S3 client built outside s3_auth/s3_client: {sorted(hits - CLIENT_FILES)}"
    )
    assert "src/marinedata/s3_auth.py" in hits, (
        "the scan no longer finds the real constructor: fix CLIENT_RE"
    )


def test_the_exempt_script_says_why():
    text = (ROOT / "scripts/build_imos_manifest.py").read_text(encoding="utf-8")
    assert CLIENT_RE.search(text) and "Exempt from tests/test_s3_auth_boundary.py" in text
