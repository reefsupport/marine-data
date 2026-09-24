"""Tests for the ``staged-tree`` layout (WS-D 7i).

Fixtures are built with the repo's own writers (:mod:`marinedata.tables`), not
hand-rolled parquet — the point is to prove the loader reads exactly what
``write_metadata_table``/``write_points_table`` actually produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import _touch_image, make_source

from marinedata.loaders import LoaderError, build_loader
from marinedata.models import SplitGroupRule
from marinedata.scan import group_key
from marinedata.schema import Axis
from marinedata.tables import (
    ImageLabelRow,
    PointRow,
    StagedImage,
    write_image_labels_table,
    write_metadata_table,
    write_points_table,
)


def _stage(
    root: Path,
    rows: list[StagedImage],
    *,
    points: list[PointRow] | None = None,
    image_labels: list[ImageLabelRow] | None = None,
    write_images: bool = True,
) -> Path:
    if write_images:
        for row in rows:
            _touch_image(root / "images" / row.partition / f"{row.stem}.jpg")
    write_metadata_table(root / "metadata.parquet", rows)
    if points:
        write_points_table(root / "labels" / "points.parquet", points)
    if image_labels:
        write_image_labels_table(root / "labels" / "image_labels.parquet", image_labels)
    return root


def test_sample_count(tmp_path: Path) -> None:
    root = _stage(
        tmp_path,
        [
            StagedImage(stem=f"img{i}", partition="default", upstream_path=f"orig/{i}.jpg",
                        upstream_split=None, width=10, height=10)
            for i in range(3)
        ],
    )
    samples = list(build_loader(make_source("staged-tree"), root))
    assert len(samples) == 3
    assert {s.key for s in samples} == {
        "images/default/img0.jpg",
        "images/default/img1.jpg",
        "images/default/img2.jpg",
    }


def test_split_group_pass_through(tmp_path: Path) -> None:
    """The parquet's ``split_group`` must be what ``group_key(by='group')`` sees —
    not whatever ``source.split_group_for`` would derive fresh (the default rule's
    fallback template is ``<source_id>/<partition>``, which this deliberately does
    not match)."""
    root = _stage(
        tmp_path,
        [
            StagedImage(
                stem="img0",
                partition="default",
                upstream_path="orig/0.jpg",
                upstream_split=None,
                width=10,
                height=10,
                split_group="pinned-at-staging/0",
            )
        ],
    )
    source = make_source("staged-tree")
    sample = next(iter(build_loader(source, root)))

    assert sample.meta["split_group"] == "pinned-at-staging/0"
    assert group_key(sample, "group") == "pinned-at-staging/0"
    # The registry's own default rule would compute something else — proving the
    # loader did not silently re-derive it.
    assert source.split_group_for(stem="img0", upstream_path="orig/0.jpg", partition="default") != (
        "pinned-at-staging/0"
    )


def test_points_join(tmp_path: Path) -> None:
    rows = [
        StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                    upstream_split=None, width=10, height=10),
        StagedImage(stem="img1", partition="default", upstream_path="orig/1.jpg",
                    upstream_split=None, width=10, height=10),
    ]
    points = [
        PointRow(stem="img0", partition="default", row=1, col=2, label="Hard Coral",
                  schema_id="fixture-schema"),
        PointRow(stem="img0", partition="default", row=3, col=4, label="Soft Coral",
                  schema_id="fixture-schema"),
    ]
    root = _stage(tmp_path, rows, points=points)
    samples = {s.key: s for s in build_loader(make_source("staged-tree"), root)}

    with_points = samples["images/default/img0.jpg"]
    assert with_points.points == ((1, 2), (3, 4))
    assert with_points.meta["native_labels"] == ["Hard Coral", "Soft Coral"]
    assert with_points.meta["n_points"] == 2

    without_points = samples["images/default/img1.jpg"]
    assert without_points.points == ()
    assert "native_labels" not in without_points.meta


def test_image_labels_join(tmp_path: Path) -> None:
    """``labels/image_labels.parquet`` resolves through ``_resolve`` the same way
    points do — one row per positive class, ``meta["native_image_labels"]`` recorded,
    absent from an image with no rows (WS-D S23)."""
    rows = [
        StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                    upstream_split=None, width=10, height=10),
        StagedImage(stem="img1", partition="default", upstream_path="orig/1.jpg",
                    upstream_split=None, width=10, height=10),
    ]
    image_labels = [
        ImageLabelRow(
            stem="img0", partition="default", label="Unhealthy", schema_id="fixture-schema"
        ),
    ]
    root = _stage(tmp_path, rows, image_labels=image_labels)
    samples = {s.key: s for s in build_loader(make_source("staged-tree"), root)}

    labelled = samples["images/default/img0.jpg"]
    assert labelled.meta["native_image_labels"] == ["Unhealthy"]

    unlabelled = samples["images/default/img1.jpg"]
    assert "native_image_labels" not in unlabelled.meta


def test_image_labels_conflict_on_same_axis_is_not_supervised(tmp_path: Path) -> None:
    """Two image-level labels that resolve to DIFFERENT targets on the same axis must
    leave that axis unsupervised (counted + reported in ``meta``), never
    last-write-wins (WS-D S24 D2)."""
    rows = [
        StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                    upstream_split=None, width=10, height=10),
    ]
    image_labels = [
        ImageLabelRow(
            stem="img0", partition="default", label="Healthy", schema_id="fixture-schema"
        ),
        ImageLabelRow(
            stem="img0", partition="default", label="Unhealthy", schema_id="fixture-schema"
        ),
    ]
    root = _stage(tmp_path, rows, image_labels=image_labels)
    samples = {s.key: s for s in build_loader(make_source("staged-tree"), root)}

    sample = samples["images/default/img0.jpg"]
    assert sample.meta["native_image_labels"] == ["Healthy", "Unhealthy"]
    assert sample.meta["multi_label_conflicts"] == ["taxon"]
    assert sample.labels == {}
    assert sample.supervised == frozenset()


def test_missing_image_raises(tmp_path: Path) -> None:
    root = _stage(
        tmp_path,
        [
            StagedImage(stem="ghost", partition="default", upstream_path="orig/ghost.jpg",
                        upstream_split=None, width=10, height=10)
        ],
        write_images=False,
    )
    loader = build_loader(make_source("staged-tree"), root)
    with pytest.raises(LoaderError, match="ghost"):
        list(loader)


def test_missing_image_skipped_when_partial(tmp_path: Path) -> None:
    root = _stage(
        tmp_path,
        [
            StagedImage(stem="ghost", partition="default", upstream_path="orig/ghost.jpg",
                        upstream_split=None, width=10, height=10),
            StagedImage(stem="real", partition="default", upstream_path="orig/real.jpg",
                        upstream_split=None, width=10, height=10),
        ],
        write_images=False,
    )
    _touch_image(root / "images" / "default" / "real.jpg")
    loader = build_loader(make_source("staged-tree"), root, partial=True)
    samples = list(loader)
    assert [s.key for s in samples] == ["images/default/real.jpg"]


def test_missing_metadata_parquet_raises(tmp_path: Path) -> None:
    (tmp_path / "images" / "default").mkdir(parents=True)
    _touch_image(tmp_path / "images" / "default" / "img0.jpg")
    loader = build_loader(make_source("staged-tree"), tmp_path)
    with pytest.raises(LoaderError, match=r"metadata\.parquet"):
        list(loader)


def test_null_split_group_raises_when_source_declares_a_rule(tmp_path: Path) -> None:
    """S28: v6i/v1's registry entries declare a source-specific split_group pattern,
    but the converter that staged them hardcoded ``split_group=None`` — the loader
    must refuse that tree rather than silently letting shared-stem leakage checks pass
    on an unproven grouping."""
    root = _stage(
        tmp_path,
        [
            StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                        upstream_split=None, width=10, height=10, split_group=None)
        ],
    )
    rule = SplitGroupRule(pattern=r"^(.+?)_jpg\.rf\.", match_field="stem", template="rf/{group}")
    source = make_source("staged-tree").model_copy(update={"split_group": rule})
    with pytest.raises(LoaderError, match="null/empty split_group"):
        list(build_loader(source, root))


def test_null_split_group_ok_when_source_has_no_explicit_rule(tmp_path: Path) -> None:
    """A source on the bare default rule (no per-entry ``split_group:`` override) is
    not held to this guard — many pre-D1 staged trees legitimately have a null
    split_group column (StagedImage.split_group's docstring)."""
    root = _stage(
        tmp_path,
        [
            StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                        upstream_split=None, width=10, height=10, split_group=None)
        ],
    )
    samples = list(build_loader(make_source("staged-tree"), root))
    assert len(samples) == 1


def test_mask_values_param_emitted_for_dense_mask(tmp_path: Path) -> None:
    """WS-D S31: no staged-tree source had a way to declare what a dense mask's own
    pixel values mean — this is that path, mirroring the shape
    ``DualConditionMaskLoader`` builds from ``positive_label``/``negative_label``."""
    rows = [
        StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                    upstream_split=None, width=10, height=10),
    ]
    root = _stage(tmp_path, rows)
    _touch_image(root / "labels" / "masks" / "default" / "img0.png")

    source = make_source(
        "staged-tree", {"mask_values": "0=unlabelled,1=bleached,2=non_bleached"}
    )
    sample = next(iter(build_loader(source, root)))

    assert sample.meta["mask_is_dense"] is True
    assert sample.meta["mask_values"] == {
        "0": "unlabelled",
        "1": "bleached",
        "2": "non_bleached",
    }


def test_mask_values_absent_when_not_declared(tmp_path: Path) -> None:
    """A staged-tree source with a mask but no declared ``mask_values`` param keeps
    behaving exactly as before this change — ``mask_is_dense`` only."""
    rows = [
        StagedImage(stem="img0", partition="default", upstream_path="orig/0.jpg",
                    upstream_split=None, width=10, height=10),
    ]
    root = _stage(tmp_path, rows)
    _touch_image(root / "labels" / "masks" / "default" / "img0.png")

    sample = next(iter(build_loader(make_source("staged-tree"), root)))

    assert sample.meta["mask_is_dense"] is True
    assert "mask_values" not in sample.meta


def test_bleaching_mask_pixel_values_harmonise_through_real_crosswalk(
    tmp_path: Path, registry
) -> None:
    """The registry's own ``reef-support-bleaching`` entry + its
    ``reef-support-bleaching-condition`` crosswalk: pixel 1 ("bleached") harmonises to
    BLEACHED, pixel 2 ("non_bleached") harmonises to the crosswalk's declared target
    for it (HEALTHY — see the crosswalk's coarsened-fidelity note), and the source's
    own ``mask_values`` param is what a caller reads to know which pixel is which."""
    source = registry.source("reef-support-bleaching")
    mask_values = source.loader.params["mask_values"]
    assert mask_values == "0=unlabelled,1=bleached,2=non_bleached"

    rows = [
        StagedImage(stem="img0", partition="UNAL_BLEACHING_TAYRONA",
                    upstream_path="orig/0.jpg", upstream_split=None, width=10, height=10,
                    split_group="rs-colombia/UNAL_BLEACHING_TAYRONA/CB1"),
    ]
    root = _stage(tmp_path, rows)
    _touch_image(root / "labels" / "masks" / "UNAL_BLEACHING_TAYRONA" / "img0.png")

    loader = build_loader(source, root)
    harmonizer = registry.harmonizer_for("reef-support-bleaching")
    loader.bind_harmonizer(harmonizer)
    sample = next(iter(loader))

    assert sample.meta["mask_values"] == {
        "0": "unlabelled",
        "1": "bleached",
        "2": "non_bleached",
    }
    bleached = harmonizer.map_label(sample.meta["mask_values"]["1"])
    non_bleached = harmonizer.map_label(sample.meta["mask_values"]["2"])
    assert bleached.labels[Axis.CONDITION].node_id == "BLEACHED"
    assert non_bleached.labels[Axis.CONDITION].node_id == "HEALTHY"
