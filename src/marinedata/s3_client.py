"""A boto3 S3 client built from an rclone remote (WP-3).

Same convention used elsewhere in this package for the authenticated bucket
(``rs-storage-open`` on the ``rs-hel1`` Hetzner endpoint): ``configparser`` reads
``~/.config/rclone/rclone.conf`` directly, credentials are never printed, logged, or passed
through a shell tool. This module is new (not yet on this branch's copy of the sibling
``s3_upload.py`` work) so WP-3's S3 verify path does not depend on unmerged work.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any


def client_from_rclone(remote: str = "rs-hel1", conf: Path | None = None) -> Any:
    """boto3 S3 client for ``remote``'s section of an rclone config file."""
    import boto3
    from botocore.config import Config

    parser = configparser.ConfigParser()
    parser.read(conf or Path.home() / ".config/rclone/rclone.conf")
    if remote not in parser:
        raise KeyError(f"rclone remote {remote!r} not found")
    sec = parser[remote]
    return boto3.client(
        "s3",
        endpoint_url=sec.get("endpoint"),
        region_name=sec.get("region") or None,
        aws_access_key_id=sec.get("access_key_id"),
        aws_secret_access_key=sec.get("secret_access_key"),
        config=Config(retries={"max_attempts": 8, "mode": "standard"}),
    )
