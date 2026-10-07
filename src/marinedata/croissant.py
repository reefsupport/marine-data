"""Croissant 1.0 (+ RAI) metadata for the Hugging Face build (WP-5, S58).

Reads the *built* export directory directly (``data/<config>/*.parquet``) and emits a
single ``croissant.json`` describing every config as one ``RecordSet``, backed by a
``FileSet`` of individually sha256-hashed ``FileObject`` shards. This module never
writes into the HF build directory itself and never touches ``hf_card``/``hf_export``/
``hf_parquet`` (D-M — those are S59's); it only reads the build they produce.

Usage::

    python -m marinedata.croissant --build-dir <_hf/v1> --repo-id <id> --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlcroissant as mlc
import pyarrow.parquet as pq

REPO_URL_TEMPLATE = "https://huggingface.co/datasets/{repo_id}"

# The only non-text column in this export is the embedded-image column (see
# hf_export.py's IMAGE_SPEC); everything else is a plain string column.
IMAGE_COLUMNS = frozenset({"image"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConfigBuild:
    """One HF config folder: its shard files, columns and total row count."""

    name: str
    files: tuple[Path, ...]
    columns: tuple[str, ...]
    num_rows: int


def scan_config(config_dir: Path) -> ConfigBuild:
    files = tuple(sorted(config_dir.glob("*.parquet")))
    if not files:
        raise ValueError(f"no parquet shards under {config_dir}")
    schema = pq.ParquetFile(files[0]).schema_arrow
    columns = tuple(field.name for field in schema)
    num_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    return ConfigBuild(config_dir.name, files, columns, num_rows)


def scan_build_dir(build_dir: Path) -> list[ConfigBuild]:
    data_dir = build_dir / "data"
    return [scan_config(p) for p in sorted(data_dir.iterdir()) if p.is_dir()]


def _file_objects(ctx: mlc.Context, cfg: ConfigBuild, repo_id: str) -> list[mlc.FileObject]:
    base_url = REPO_URL_TEMPLATE.format(repo_id=repo_id)
    objects = []
    for fp in cfg.files:
        rel = f"data/{cfg.name}/{fp.name}"
        objects.append(
            mlc.FileObject(
                ctx=ctx,
                id=f"{cfg.name}/{fp.name}",
                name=f"{cfg.name}-{fp.name}",
                description=f"Parquet shard of the `{cfg.name}` config, hashed at build time.",
                content_url=f"{base_url}/resolve/main/{rel}",
                encoding_formats=["application/x-parquet"],
                sha256=_sha256_file(fp),
            )
        )
    return objects


def _record_set(ctx: mlc.Context, cfg: ConfigBuild) -> mlc.RecordSet:
    file_set_id = f"{cfg.name}-files"
    fields = [
        mlc.Field(
            ctx=ctx,
            id=f"{cfg.name}/{col}",
            name=col,
            description=f"`{col}` column of the `{cfg.name}` config ({cfg.num_rows} rows).",
            data_types=[mlc.DataType.IMAGE_OBJECT if col in IMAGE_COLUMNS else mlc.DataType.TEXT],
            source=mlc.Source(file_set=file_set_id, extract=mlc.Extract(column=col)),
        )
        for col in cfg.columns
    ]
    return mlc.RecordSet(
        ctx=ctx,
        id=cfg.name,
        name=cfg.name,
        description=(
            f"{cfg.num_rows} rows across {len(cfg.files)} Parquet shard(s) in the "
            f"`{cfg.name}` config."
        ),
        fields=fields,
    )


# Responsible-AI fields (D11 level 5 / RAI extension). These describe the *release*
# process and known issues; every factual number below is sourced in docs/DATASHEET.md,
# which cites the primary evidence file for each one.
RAI_FIELDS: dict[str, object] = {
    "data_collection": (
        "Nine imagery sources (NOAA PIFSC, five Roboflow Universe community exports, "
        "Reef Support's own benthic and bleaching surveys, and CoralSCOP model output) "
        "were staged, sha256-verified and deduplicated by dHash-64 (Hamming <=4 union, "
        "<=8 never-eval exclusion). See docs/DATASHEET.md #Collection Process."
    ),
    "data_annotation_protocol": [
        "Labels are taken as-shipped from each upstream source (native class sets), "
        "crosswalked onto Reef Support's condition/taxon ontology where a crosswalk "
        "exists (28% of image sources; see docs/DATASHEET.md #Preprocessing/Labeling). "
        "CoralSCOP masks are machine-generated pseudo-labels, never ground truth."
    ],
    "machine_annotation_tools": ["CoralSCOP (instance segmentation, weak supervision only)"],
    "data_biases": [
        "CoralSCOP alone is 54% of images (single-source concentration).",
        "100% shallow coral reef habitat; no depth, region or platform recorded per sample (D8).",
        "NOAA PIFSC images are 224x224 px crops; ~23k Roboflow images are resized to "
        "640x640 px — neither carries a resolution flag (D9).",
    ],
    "data_limitations": [
        "7,085 images are byte-distinct duplicates at Hamming distance 0 (no confirm "
        "hash yet run); 14,265 near-dup pairs exist under the release's Hamming<=4 "
        "union rule.",
        "Only 2 of 8 possible ground-truth task types are covered (classification, "
        "segmentation); no detection, points, captions, VQA, tracking or enhancement.",
        "`coral-genus-caribbean` carries 0 image labels and is dropped from this "
        "release's HF configs (D-A).",
    ],
    "personal_sensitive_information": [
        "A subset of in-water photographs may show a diver's face or body; see "
        "docs/DATASHEET.md #Ethics and docs/ETHICS_FACE_AUDIT.md for the measured count "
        "and method. No sample in v1 carries per-sample GPS coordinates (D8); the "
        "sensitive-species location-rounding policy in the datasheet applies to future "
        "releases once location fields ship."
    ],
    "data_release_maintenance_plan": (
        "See CHANGELOG.md for the per-release changelog and docs/DATASHEET.md "
        "#Maintenance for the erratum process."
    ),
}


def build_metadata(build_dir: Path, repo_id: str, *, name: str, description: str) -> mlc.Metadata:
    ctx = mlc.Context()
    configs = scan_build_dir(build_dir)

    distribution: list[mlc.FileObject | mlc.FileSet] = []
    record_sets: list[mlc.RecordSet] = []
    for cfg in configs:
        distribution.extend(_file_objects(ctx, cfg, repo_id))
        distribution.append(
            mlc.FileSet(
                ctx=ctx,
                id=f"{cfg.name}-files",
                name=f"{cfg.name}-files",
                description=f"All Parquet shards of the `{cfg.name}` config.",
                includes=[f"data/{cfg.name}/*.parquet"],
                encoding_formats=["application/x-parquet"],
            )
        )
        record_sets.append(_record_set(ctx, cfg))

    return mlc.Metadata(
        ctx=ctx,
        name=name,
        description=description,
        url=REPO_URL_TEMPLATE.format(repo_id=repo_id),
        license=["other"],
        cite_as="See CITATION.cff at the repository root.",
        version="1.0.0",
        date_published=datetime(2026, 9, 24, tzinfo=timezone.utc),
        distribution=distribution,
        record_sets=record_sets,
        **RAI_FIELDS,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", type=str, default="ReefSupport Marine Data v1")
    parser.add_argument(
        "--description",
        type=str,
        default=(
            "Licence-aware, multi-source coral-reef imagery corpus with condition, "
            "bleaching and segmentation labels. See docs/DATASHEET.md for the full "
            "datasheet."
        ),
    )
    args = parser.parse_args(argv)

    metadata = build_metadata(
        args.build_dir, args.repo_id, name=args.name, description=args.description
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metadata.to_json(), indent=2, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
