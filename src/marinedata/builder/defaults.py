"""Default split ratios, the split-name type, scalar label kinds, the pretrain vocabulary."""

from __future__ import annotations

from pathlib import Path

from ..enums import AnnotationKind
from ..splitmap import load_split_map

SUPERVISED_DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

SELF_SUPERVISED_DEFAULT_RATIOS = {"train": 0.95, "probe": 0.05}
"""A self-supervised corpus has no ground truth to score a "val"/"test" split against —
the 70/15/15 supervised default would carve off 30% of a pretraining corpus for nothing.
The held-out slice here is named ``probe`` rather than ``val``/``test`` on purpose: its
job is monitoring representation quality (e.g. a linear probe or k-NN check), not a task
accuracy metric, and reusing supervised split names would suggest otherwise. Final
evaluation of a self-supervised encoder happens on a separate, labelled downstream task
— not inside the pretraining corpus itself."""

SplitName = str


def _default_ratios(
    ratios: dict[SplitName, float] | None,
    *,
    self_supervised: bool,
    split_map: str | Path | None,
) -> dict[SplitName, float]:
    """Fill in the default split ratios, adopting a persisted map's ratios first.

    A self-supervised task defaults to ``SELF_SUPERVISED_DEFAULT_RATIOS`` (95/5), but a
    shared ``split_map`` may already have been generated at 70/15/15 for the whole
    corpus (see :mod:`marinedata.splitmap`). Without this, pretraining would raise on
    every release build purely because its own default disagrees with the map it was
    told to use. An explicit ``ratios=`` from the caller always wins — this only fills
    in the *default*.
    """
    if ratios is not None:
        return ratios
    if split_map is not None:
        existing = load_split_map(split_map)
        if existing is not None:
            return dict(existing.ratios)
    return SELF_SUPERVISED_DEFAULT_RATIOS if self_supervised else SUPERVISED_DEFAULT_RATIOS


def _to_pretrain_vocabulary(
    assignment: dict[str, SplitName], ratios: dict[SplitName, float]
) -> tuple[dict[str, SplitName], dict[SplitName, float]]:
    """Translate a shared map's train/val/test assignment into the pretrain vocabulary.

    ``train`` stays ``train``. ``val`` demotes to ``probe`` — the self-supervised
    representation-quality check, not a task metric (see
    ``SELF_SUPERVISED_DEFAULT_RATIOS``). ``test`` is dropped from the returned
    ``assignment`` entirely: a labelled task's held-out evaluation images must never
    enter a pretraining corpus, so a group the map assigned ``test`` gets no entry here
    and ``Dataset.split()`` excludes its samples rather than assigning them a split.
    A map already speaking the pretrain vocabulary directly (no ``val``/``test`` key)
    passes through unchanged.
    """
    new_assignment = {
        key: ("probe" if name == "val" else name)
        for key, name in assignment.items()
        if name != "test"
    }
    new_ratios = {
        ("probe" if name == "val" else name): value
        for name, value in ratios.items()
        if name != "test"
    }
    return new_assignment, new_ratios


_SCALAR_LABEL_KINDS = frozenset(
    {
        AnnotationKind.IMAGE_LABEL,
        AnnotationKind.MULTILABEL,
        AnnotationKind.POINT_LABEL,
        AnnotationKind.BBOX,
        AnnotationKind.TRACK,
        AnnotationKind.AUDIO_EVENT,
    }
)
"""Annotation kinds that yield a per-sample scalar label, and can therefore leak an
unmapped vocabulary into the label index. Dense masks and point clouds carry classes in
the raster instead, which the index never touches."""
