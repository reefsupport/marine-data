"""Shared fixtures for ``tests/test_ingest_s3.py`` (D3a2): a local, ephemeral-port
``http.server`` that answers path-style S3 REST requests — anonymous ``ListObjectsV2``
XML (2 pages) plus the object bytes themselves (tiny PNGs, a few sidecars, one small
annotations parquet) — split out purely so the test file itself stays readable and
under its own 400-line norm.
"""

from __future__ import annotations

import http.server
import io
import re
import threading
import urllib.parse
import xml.sax.saxutils as saxutils
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import pytest

from marinedata.enums import (
    AccessMethod,
    AnnotationKind,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import (
    Access,
    Annotation,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    Verification,
)
from marinedata.s3_listing import S3Plan

BUCKET = "test-bucket"
PREFIX = "mermaid/"
STEM_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
STEMS = [f"00000000-0000-4000-8000-00000000000{i}" for i in range(5)]  # s0..s4, sorted
ANNOTATIONS_KEY = f"{PREFIX}mermaid_confirmed_annotations.parquet"


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buf, format="PNG")
    return buf.getvalue()


def _points_table_bytes() -> bytes:
    """s0: 3 rows (one with ``growth_form_name=None``), s1: 2 rows, s2: 5 rows.
    s3/s4 have none — the "unannotated, skipped" stems. Extra upstream-only columns
    (``id``, ``point_id``, ``growth_form_id``, ``updated_on``) prove the staged
    schema drops them rather than inheriting whatever the source happens to carry.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        (STEMS[0], 1, 1, "A1", "Acropora", None, None, "Pacific"),
        (STEMS[0], 1, 2, "A1", "Acropora", "g1", "branching", "Pacific"),
        (STEMS[0], 2, 1, "A2", "Porites", "g2", "massive", "Pacific"),
        (STEMS[1], 1, 1, "A1", "Acropora", "g1", "branching", "Pacific"),
        (STEMS[1], 2, 2, "A3", "Sand", None, None, "Pacific"),
        (STEMS[2], 1, 1, "A1", "Acropora", "g1", "branching", "Pacific"),
        (STEMS[2], 1, 2, "A1", "Acropora", "g1", "branching", "Pacific"),
        (STEMS[2], 1, 3, "A2", "Porites", "g2", "massive", "Pacific"),
        (STEMS[2], 2, 1, "A3", "Sand", None, None, "Pacific"),
        (STEMS[2], 2, 2, "A3", "Sand", None, None, "Pacific"),
    ]
    table = pa.table(
        {
            "id": [f"row-{i}" for i in range(len(rows))],
            "point_id": [f"pt-{i}" for i in range(len(rows))],
            "image_id": [r[0] for r in rows],
            "row": [r[1] for r in rows],
            "col": [r[2] for r in rows],
            "benthic_attribute_id": [r[3] for r in rows],
            "benthic_attribute_name": [r[4] for r in rows],
            "growth_form_id": [r[5] for r in rows],
            "growth_form_name": [r[6] for r in rows],
            "region_name": [r[7] for r in rows],
            "updated_on": ["2026-01-01T00:00:00Z"] * len(rows),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


POINTS_PER_STEM = {STEMS[0]: 3, STEMS[1]: 2, STEMS[2]: 5}


class _S3Handler(http.server.BaseHTTPRequestHandler):
    pages: ClassVar[dict[str | None, bytes]] = {}
    objects: ClassVar[dict[str, bytes]] = {}

    def log_message(self, *_args: object) -> None:  # quiet the test output
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("list-type") == ["2"]:
            token = query.get("continuation-token", [None])[0]
            body = type(self).pages.get(token)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self._send(body, "application/xml")
            return
        body = type(self).objects.get(parsed.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self._send(body, "application/octet-stream")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def s3_server():
    _S3Handler.pages = {}
    _S3Handler.objects = {}
    server = http.server.HTTPServer(("127.0.0.1", 0), _S3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


@dataclass(frozen=True)
class Corpus:
    plan: S3Plan
    entries: list[tuple[str, int, str]]
    image_sizes: dict[str, int]


def _listing_xml(
    entries: list[tuple[str, int, str]], *, truncated: bool, next_token: str | None
) -> bytes:
    parts = ["<?xml version='1.0' encoding='UTF-8'?><ListBucketResult>"]
    for key, size, etag in entries:
        parts.append(
            f"<Contents><Key>{saxutils.escape(key)}</Key><Size>{size}</Size>"
            f"<ETag>&quot;{etag}&quot;</ETag></Contents>"
        )
    parts.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    if next_token:
        parts.append(f"<NextContinuationToken>{next_token}</NextContinuationToken>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode("utf-8")


def build_corpus(s3_server: http.server.HTTPServer, *, extra_key: str | None = None) -> Corpus:
    """Populate ``s3_server`` with a 5-stem corpus (s0..s2 annotated, s3/s4 not),
    4 objects each (image + 3 sidecars) plus one annotations parquet, paginated
    into exactly 2 ``ListObjectsV2`` pages. ``extra_key`` optionally injects one
    more listed key with an unrecognised suffix (the "unknown key" test)."""
    port = s3_server.server_address[1]
    endpoint = f"http://127.0.0.1:{port}"
    entries: list[tuple[str, int, str]] = []
    image_sizes: dict[str, int] = {}

    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120), (130, 140, 150)]
    for stem, color in zip(STEMS, colors, strict=True):
        payload = _png_bytes(color)
        image_sizes[stem] = len(payload)
        _S3Handler.objects[f"/{BUCKET}/{PREFIX}{stem}.png"] = payload
        entries.append((f"{PREFIX}{stem}.png", len(payload), f"etag-{stem}-img"))
        for suffix, body in (
            ("_thumbnail.png", b"thumb"),
            ("_featurevector", b"feat"),
            ("_annotations.csv", b"csv"),
        ):
            key = f"{PREFIX}{stem}{suffix}"
            _S3Handler.objects[f"/{BUCKET}/{key}"] = body
            entries.append((key, len(body), f"etag-{stem}-{suffix}"))

    points_bytes = _points_table_bytes()
    _S3Handler.objects[f"/{BUCKET}/{ANNOTATIONS_KEY}"] = points_bytes
    entries.append((ANNOTATIONS_KEY, len(points_bytes), "etag-annotations"))

    if extra_key is not None:
        _S3Handler.objects[f"/{BUCKET}/{extra_key}"] = b"???"
        entries.append((extra_key, 3, "etag-extra"))

    entries.sort(key=lambda e: e[0])
    split = len(entries) // 2
    _S3Handler.pages[None] = _listing_xml(entries[:split], truncated=True, next_token="page2")
    _S3Handler.pages["page2"] = _listing_xml(entries[split:], truncated=False, next_token=None)

    plan = S3Plan(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint=endpoint,
        image_suffix=".png",
        sidecar_suffixes=("_thumbnail.png", "_featurevector", "_annotations.csv"),
        annotations_key=ANNOTATIONS_KEY,
        stem_pattern=STEM_PATTERN,
        partition="default",
        annotated_only=True,
        license_text="CC BY-NC-SA 4.0 (fixture)\n",
        points_schema_id="mermaid-attributes-test",
    )
    return Corpus(plan=plan, entries=entries, image_sizes=image_sizes)


def make_source(*, version: str, source_id: str = "mermaid-fixture") -> Source:
    return Source(
        id=source_id,
        name="MERMAID fixture",
        description="Synthetic S3 ingest fixture source.",
        version=version,
        licence=Licence(id="CC-BY-NC-SA-4.0", name="CC BY-NC-SA 4.0", tier=Tier.NONCOMMERCIAL),
        verification=Verification(
            verified_on=date(2026, 8, 17), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.S3,
            uri=f"s3://{BUCKET}",
            params={"bucket": BUCKET, "endpoint": "s3.amazonaws.com", "prefix": PREFIX},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_CLASSIFICATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout="flat-images", schema_id="mermaid-attributes", crosswalk_id=None),
        annotations=(Annotation(kind=AnnotationKind.POINT_LABEL, supervises=("taxon", "form")),),
    )


def make_profile() -> Profile:
    return Profile(
        id="research",
        description="fixture profile",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )
