"""pandas adapter.

A flat table of samples for exploration, stratification checks and joins against
external metadata. Deliberately does *not* decode images — a DataFrame of decoded
pixels is a memory problem, and every question a DataFrame is good for (class balance,
site coverage, licence mix, leakage checks) is answerable from the paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..labelindex import IGNORE_INDEX
from ..schema import Axis

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..builder import Dataset


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is not installed. Install it with: pip install 'marinedata[pandas]'"
        ) from exc
    return pd


def to_dataframe(dataset: Dataset, *, split: str | None = None) -> pd.DataFrame:
    """Flatten a dataset to a DataFrame, one row per sample.

    Columns: identity (``source_id``, ``key``, ``split``, ``partition``), provenance
    (``licence_tier``), paths (``image``, ``mask``, ``audio``), per-axis labels as both
    node id and encoded index, and per-axis supervision booleans.

    The supervision columns are the ones worth looking at first. ``form_supervised``
    being 12% is not a bug — it is what combining ragged sources actually looks like —
    but it explains a weak growth-form head far better than a loss curve does.
    """
    pd = _require_pandas()

    samples = dataset.split_samples(split) if split else dataset.samples
    positions = dataset.splits.get(split, []) if split else range(len(dataset.samples))
    split_of: dict[int, str] = {}
    if not split:
        for name, idxs in dataset.splits.items():
            for i in idxs:
                split_of[i] = name

    rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(samples):
        position = list(positions)[offset] if split else offset
        encoded = dataset.label_index.encode(sample, projector=dataset.projector)
        row: dict[str, Any] = {
            "source_id": sample.source_id,
            "key": sample.key,
            "split": split or split_of.get(position),
            "partition": sample.meta.get("partition"),
            "licence_tier": sample.licence_tier.value if sample.licence_tier else None,
            "image": str(sample.image) if sample.image else None,
            "mask": str(sample.mask) if sample.mask else None,
            "audio": str(sample.audio) if sample.audio else None,
            "n_boxes": len(sample.boxes),
            "n_points": len(sample.points),
        }
        for axis in Axis:
            value = sample.labels.get(axis)
            row[f"{axis.value}"] = value.node_id if value else None
            row[f"{axis.value}_fidelity"] = value.fidelity.value if value else None
            row[f"{axis.value}_index"] = encoded.get(axis, IGNORE_INDEX)
            row[f"{axis.value}_supervised"] = axis in sample.supervised
        rows.append(row)

    frame = pd.DataFrame(rows)
    # Drop axes nothing in this dataset touches, so the table stays readable.
    for axis in Axis:
        if not frame[f"{axis.value}_supervised"].any():
            frame = frame.drop(
                columns=[
                    f"{axis.value}",
                    f"{axis.value}_fidelity",
                    f"{axis.value}_index",
                    f"{axis.value}_supervised",
                ]
            )
    return frame


def leakage_report(dataset: Dataset) -> pd.DataFrame:
    """Groups appearing in more than one split — i.e. leakage.

    An empty frame is the expected result of ``split(by="site")``. A non-empty one after
    ``by="random"`` shows precisely how much the metrics are being flattered.
    """
    pd = _require_pandas()

    seen: dict[str, set[str]] = {}
    for name, positions in dataset.splits.items():
        for position in positions:
            sample = dataset.samples[position]
            group = f"{sample.source_id}/{sample.meta.get('partition', '')}"
            seen.setdefault(group, set()).add(name)

    leaked = [
        {"group": group, "splits": ", ".join(sorted(names)), "n_splits": len(names)}
        for group, names in sorted(seen.items())
        if len(names) > 1
    ]
    return pd.DataFrame(leaked, columns=["group", "splits", "n_splits"])
