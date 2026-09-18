"""Stage one source's declared version into ``sources/<id>/<version>/`` (D1, §5 step 7).

Orchestration only — pixel and table bytes come from :mod:`marinedata.normalise` and
:mod:`marinedata.tables`; the three metadata files come from :mod:`marinedata.manifests`.
This module owns ordering, the archive-facts-per-source lookup, and the checksum flow.

**Re-staging an already-staged tree never re-fetches (D1 §3).** If
``<out_root>/<source_id>/<version>/CHECKSUMS.sha256`` already exists,
:func:`_stage_with_plan` skips straight to :func:`marinedata.checksums.write_checksums`
over the existing tree — which *is* ``assert_unchanged`` semantics: identical bytes are
a no-op, any drift (a changed, added or removed file) raises ``ChecksumError``. A
pipeline that instead re-copied every file from the archive on every re-stage would
silently repair external tampering before the checksum scan ever saw it, which is
exactly the failure the design's no-op/raise test pair exists to catch.
"""

from __future__ import annotations

import re
import shutil
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import checksums, fetch, gate
from .ingest_dispatch import stage_source  # noqa: F401 — re-exported, dispatch moved out (D3f)
from .manifests import AnnotationCounts, write_annotations_json, write_license, write_source_json
from .models import Profile, Source
from .normalise import bmp_mask_to_indexed_png, copy_image
from .tables import (
    BoxRow,
    PointRow,
    StagedImage,
    write_boxes_table,
    write_metadata_table,
    write_points_table,
)


class IngestError(Exception):
    """A source cannot be staged as requested — no plan, a short archive, a stem
    collision, or an annotation row that does not resolve to a staged image."""


@dataclass(frozen=True)
class StagedVersion:
    """What one :func:`stage_source` call produced."""

    source_id: str
    version: str
    root: Path
    images: int
    manifest: checksums.ChecksumManifest


@dataclass(frozen=True)
class ArchivePlan:
    """Per-source archive facts (I2 override, brief 2026-09-18-I2).

    D1 §7c hard-codes 1598/1598 for ``suim`` directly in ``stage_source``'s body; this
    dataclass moves those facts (and the member-matching patterns needed to find them)
    into one module-level, per-source lookup instead, so ``_stage_with_plan`` — the
    actual pipeline — never depends on the real 173 MB SUIM zip, and tests can drive it
    with a small fixture plan built in ``tmp_path``.
    """

    image_pattern: re.Pattern[str]
    """Matches an archive member's POSIX path, relative to the extracted archive root.
    Must carry named groups ``stem`` (upstream basename minus extension, before
    sanitisation) and ``split`` (the upstream split directory — documentation only,
    D1 §2)."""

    mask_pattern: re.Pattern[str]
    """Same shape as ``image_pattern``, for the paired mask member."""

    expected_images: int
    expected_masks: int
    partition: str
    version: str
    classes: int

    license_text: str
    """D1 §2's ``LICENSE`` file is the licence text verbatim, but nothing else in this
    package carries it: ``models.Licence`` has only ``id``/``name``/``url``/``notes``,
    never full text, and the archive itself ships none. This field is an addition to
    D1 §5's ArchivePlan field list (which names only the member patterns, expected
    counts, partition, version and class count) — noted in the I2 report."""

    ignore_index: int | None = None
    """The upstream raster's unlabelled/ignore class index, written verbatim into
    ``ANNOTATIONS.json``'s ``raster_ignore_value`` (D2c). ``None`` when the source has
    no reserved ignore index and every observed value is a real class — writing a
    hardcoded ``255`` regardless of whether it ever occurs is exactly the defect this
    field exists to close."""


_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")


def _plan_stems(members: Sequence[str]) -> dict[str, str]:
    """Map each upstream archive member path to its sanitised stem (D1 §2's Image id).

    ``stem`` is the upstream basename minus extension, sanitised via
    ``re.sub(r"[^A-Za-z0-9._-]", "_", ...)``. Injective-or-raise: two members that
    sanitise to the same stem never fall back to a ``_1`` suffix, which would depend on
    iteration order and so be unstable across re-ingest.
    """
    result: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for member in members:
        raw = Path(member).stem
        stem = _SANITIZE.sub("_", raw)
        collision = by_stem.get(stem)
        if collision is not None and collision != member:
            raise IngestError(
                f"stem collision: {member!r} and {collision!r} both sanitise to {stem!r}"
            )
        by_stem[stem] = member
        result[member] = stem
    return result


def fetch_archive(source: Source, *, cache_root: Path) -> tuple[Path, str, int]:
    """Download the whole archive named by ``access.params['sample_url']`` and extract
    every member into scratch work space under ``cache_root`` (D1 §5, step 7b).

    Returns ``(work_dir, archive_sha256, archive_size_bytes)``. Never calls
    ``fetch_sample``: that discards the archive, hard-codes ``truncated=True``, and
    writes a ``_fetch.json`` carrying ``fetched_at`` — none of which belongs in a
    staged tree. The cache root is never the staged root (``out_root``); ``work_dir``
    is discarded once its images/masks have been copied into the version tree.

    Calls ``fetch.get_bytes``/``fetch.extract_archive`` through the module (not as
    bound names) so a caller's ``monkeypatch.setattr(fetch, "get_bytes", ...)`` — the
    no-network path every test in this package uses — actually takes effect.
    """
    sample_url = str(source.access.params.get("sample_url") or "")
    if not sample_url:
        raise IngestError(f"{source.id}: no sample_url declared in access.params")
    name = Path(urllib.parse.urlparse(sample_url).path).name or "archive.zip"

    payload = fetch.get_bytes(sample_url)
    archive_path = cache_root / source.id / "_archive" / name
    archive_sha256 = checksums.write_digest(archive_path, payload)

    work = cache_root / source.id / "_work"
    if work.exists():
        shutil.rmtree(work)
    fetch.extract_archive(payload, work, name, 0)
    return work, archive_sha256, len(payload)


def _mit_suim_text() -> str:
    """Verbatim upstream text fetched 2026-09-18 from
    raw.githubusercontent.com/IRVLab/SUIM/master/LICENSE (byte-for-byte, copyright
    line included). Moved out of a module constant (D2c) to keep this module at or
    under the 400-line cap — ``src/marinedata/licenses/mit-suim.txt`` carries the
    text, loaded the same way :func:`marinedata.ingest_parquet._apache_2_0_text`
    loads ``apache-2.0.txt``."""
    from importlib.resources import files

    return files("marinedata").joinpath("licenses", "mit-suim.txt").read_text(encoding="utf-8")


_PLANS: Mapping[str, ArchivePlan] = MappingProxyType(
    {
        "suim": ArchivePlan(
            image_pattern=re.compile(
                r"^SUIM/(?P<split>train_val|TEST)/images/(?P<stem>[^/]+)\.jpg$"
            ),
            mask_pattern=re.compile(r"^SUIM/(?P<split>train_val|TEST)/masks/(?P<stem>[^/]+)\.bmp$"),
            expected_images=1598,
            expected_masks=1598,
            partition="default",
            version="2020",
            classes=8,
            license_text=_mit_suim_text(),
            ignore_index=None,  # D2c: BW 0-7 are all real classes; 255 never occurs.
        ),
    }
)


def _stage_with_plan(
    source: Source,
    plan: ArchivePlan,
    *,
    cache_root: Path,
    out_root: Path,
    profile: Profile,
    points: Sequence[PointRow] = (),
    boxes: Sequence[BoxRow] = (),
) -> StagedVersion:
    """The actual pipeline (D1 §7 a-g), driven by ``plan`` rather than hard-coded facts.

    ``points``/``boxes`` are an I2 addition: D1 §5 gives ``_stage_with_plan`` no path
    to receive point- or box-geometry rows, and test 5 (an image with a mask AND
    points appears once) needs one — see the I2 report.
    """
    # a. the existing gate, before any byte is written. No second gate.
    decision = gate.evaluate(source, profile)
    decision.raise_if_denied()

    version_root = out_root / "sources" / source.id / source.version
    manifest_path = version_root / checksums.CHECKSUM_FILE

    if manifest_path.is_file():
        # Re-stage over an already-staged tree (D1 §3): verify, never re-fetch.
        manifest = checksums.write_checksums(version_root)
        images = len(list((version_root / "images" / plan.partition).glob("*.jpg")))
        return StagedVersion(source.id, source.version, version_root, images, manifest)

    # b. fetch the whole archive into scratch work space.
    work, archive_sha256, archive_bytes = fetch_archive(source, cache_root=cache_root)
    sample_url = str(source.access.params.get("sample_url") or "")

    # c. assert the declared member counts; never stage a short tree.
    members = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file())
    image_members = [m for m in members if plan.image_pattern.match(m)]
    mask_members = [m for m in members if plan.mask_pattern.match(m)]
    if len(image_members) != plan.expected_images or len(mask_members) != plan.expected_masks:
        raise IngestError(
            f"{source.id}: expected {plan.expected_images} image(s) + "
            f"{plan.expected_masks} mask(s), found {len(image_members)} + "
            f"{len(mask_members)} — refusing to stage a short tree"
        )

    # d. stems, injective-or-raise.
    image_stems = _plan_stems(image_members)
    mask_stems = _plan_stems(mask_members)
    masks_by_stem = {stem: member for member, stem in mask_stems.items()}
    staged_stems = set(image_stems.values())

    # e's own guard, run before any byte is written: every annotation row (mask or
    # sparse geometry) must resolve to a staged image (LABEL-ORG §7.4, test 17).
    orphan_masks = sorted(set(masks_by_stem) - staged_stems)
    if orphan_masks:
        raise IngestError(f"{source.id}: mask(s) with no matching image: {orphan_masks[:5]}")
    for row in (*points, *boxes):
        if row.stem not in staged_stems:
            raise IngestError(
                f"{source.id}: annotation row stem {row.stem!r} does not resolve to "
                f"any staged image"
            )

    # e. per image: copy byte-for-byte; if a mask exists, re-encode it. An image with
    # no mask does not raise — the image still appears, with no mask file.
    recorded: dict[str, str] = {}
    staged_rows: list[StagedImage] = []
    images_without_annotation = 0
    observed_indices: set[int] = set()

    for member in sorted(image_members):
        stem = image_stems[member]
        match = plan.image_pattern.match(member)
        split = match.group("split") if match else None

        dest = version_root / "images" / plan.partition / f"{stem}.jpg"
        digest, width, height = copy_image(work / member, dest)
        recorded[dest.relative_to(version_root).as_posix()] = digest

        mask_member = masks_by_stem.get(stem)
        if mask_member is not None:
            mask_dest = version_root / "labels" / "masks" / plan.partition / f"{stem}.png"
            mask_digest, observed = bmp_mask_to_indexed_png(
                work / mask_member, mask_dest, classes=plan.classes
            )
            recorded[mask_dest.relative_to(version_root).as_posix()] = mask_digest
            observed_indices |= observed
        else:
            images_without_annotation += 1

        staged_rows.append(
            StagedImage(
                stem=stem,
                partition=plan.partition,
                upstream_path=member,
                upstream_split=split,
                width=width,
                height=height,
            )
        )

    # f. tables, then the three metadata files.
    metadata_path = version_root / "metadata.parquet"
    recorded[metadata_path.relative_to(version_root).as_posix()] = write_metadata_table(
        metadata_path, staged_rows
    )

    if points:
        points_path = version_root / "labels" / "points.parquet"
        recorded[points_path.relative_to(version_root).as_posix()] = write_points_table(
            points_path, list(points)
        )
    if boxes:
        boxes_path = version_root / "labels" / "boxes.parquet"
        recorded[boxes_path.relative_to(version_root).as_posix()] = write_boxes_table(
            boxes_path, list(boxes)
        )

    annotation = source.annotations[0] if source.annotations else None
    ingest_meta = {
        "ingest_version": 1,
        "stem_rule": "basename-no-extension",
        "partition_rule": f"literal:{plan.partition}",
        "mask_encoder": "pillow/11.0.0",
        "upstream": [{"url": sample_url, "sha256": archive_sha256, "bytes": archive_bytes}],
        "fetched_uri": sample_url,
        "gate": {"profile": profile.id, "allowed": decision.allowed, "reason": decision.reason},
    }
    source_json_path = version_root / "SOURCE.json"
    recorded[source_json_path.relative_to(version_root).as_posix()] = write_source_json(
        source_json_path, source, ingest_meta
    )

    geometries: list[Mapping[str, object]] = []
    if mask_members:
        geometries.append(
            {
                "kind": annotation.kind.value if annotation else "dense-mask",
                "path": "labels/masks/",
                "format": "png-indexed",
                "schema_id": source.loader.schema_id if source.loader else "dataset-native",
                "crosswalk_id": source.loader.crosswalk_id if source.loader else None,
                "classes": plan.classes,
                "raster_ignore_value": plan.ignore_index,
                "images_covered": len(mask_members),
                "rows": None,
                "supervises": list(annotation.supervises) if annotation else [],
                "observed_indices": sorted(observed_indices),
            }
        )
    if points:
        geometries.append(
            {
                "kind": "point-label",
                "path": "labels/points.parquet",
                "format": "parquet",
                "schema_id": "dataset-native",
                "crosswalk_id": None,
                "classes": None,
                "raster_ignore_value": None,
                "images_covered": len({row.stem for row in points}),
                "rows": len(points),
                "supervises": [],
                "observed_indices": [],
            }
        )

    counts = AnnotationCounts(
        images=len(staged_rows),
        images_without_annotation=images_without_annotation,
        geometries=tuple(geometries),
    )
    annotations_json_path = version_root / "ANNOTATIONS.json"
    recorded[annotations_json_path.relative_to(version_root).as_posix()] = write_annotations_json(
        annotations_json_path, source, counts
    )

    license_path = version_root / "LICENSE"
    recorded[license_path.relative_to(version_root).as_posix()] = write_license(
        license_path, source, plan.license_text
    )

    # g. last — pins the whole version.
    manifest = checksums.write_checksums(version_root, recorded=recorded)
    return StagedVersion(source.id, source.version, version_root, len(staged_rows), manifest)
