"""The opening of the v1.2.0 dataset card: what the dataset is.

``## Dataset summary`` of the v1.2.0 card describes the dataset (two paragraphs and a facts table).
The release history that used to open it is now the first paragraph of ``## Versions and
changelog`` (:func:`~.hf_card_v11.render_v11_card`). Every number comes from the facts of the
build, the same ones that the Supported tasks, Splits and Licensing sections print; the GitHub
``README.md`` opens its "Published datasets" section with the same two paragraphs
(:func:`summary_paragraphs` with ``closing=CLOSING_README``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from . import card_v12_text as v12
from .release_card import display_licence

V2_CLASSES = sum(v12.EDGES.values())
"""Classes of Coralscapes V2: every class has one edge in the `coralscapes-95` crosswalk."""
LICENCE_ORDER = ("Apache-2.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "US-GOV-PD")
"""How the facts table lists the licences of the release: these first, any other one by id."""
LICENCE_NAMES = {"US-GOV-PD": "US Government public domain"}
"""The licence ids that :func:`~.release_card.display_licence` would leave as an id."""
CLOSING_CARD = "; the full history is under [Versions and changelog](#versions-and-changelog)."
CLOSING_README = ". Release history below."
UNMAPPED = "unmapped"
"""The `match_type` of a source label that no taxonomy node reaches (the row is kept)."""


def licence_families(rows: Iterable[Mapping]) -> list[str]:
    """The licences of ``rows`` (the sources of the Licensing matrix), by name, in card order."""
    found = {r["licence"] for r in rows if r.get("licence")}
    ordered = [i for i in LICENCE_ORDER if i in found] + sorted(found - set(LICENCE_ORDER))
    return [LICENCE_NAMES.get(i) or display_licence(i) for i in ordered]


def sources_fact(rows: Sequence[Mapping], n_sources: int) -> str:
    """``16, all permitting commercial use (Apache-2.0, CC BY 4.0, ...)``: the sources of the
    release and, from the licence rows, the licences they carry."""
    permissive = bool(rows) and all(r.get("licence_class") == "open" for r in rows)
    families = licence_families(rows)
    return (
        f"{n_sources}"
        + (", all permitting commercial use" if permissive else "")
        + (f" ({', '.join(families)})" if families else "")
    )


def labels_mapped(rows: Iterable[Mapping]) -> tuple[int, int]:
    """``(mapped, all)`` over the rows of the `label-map` config: one row per source label, the
    ones whose ``match_type`` is ``unmapped`` being the labels that no taxonomy node reaches."""
    rows = list(rows)
    return sum(1 for r in rows if r.get("match_type") != UNMAPPED), len(rows)


def summary_paragraphs(
    *, name: str, images: int, sources: int, nodes: int, configs: int, version: str,
    published: str, closing: str = CLOSING_CARD,
) -> str:  # fmt: skip
    """The two paragraphs that open the v1.2.0 summary; ``closing`` ends the last sentence."""
    what = (
        f"{name} is a collection of {images:,} underwater images from {sources} openly licensed "
        "sources, labelled for coral-reef and marine computer vision: semantic masks, instance "
        "masks, bounding boxes, point annotations and image-level bleaching labels, plus derived "
        f"benthic cover. Labels are mapped to one taxonomy ({nodes:,} nodes), so labels from "
        "different sources can be trained and evaluated together, and every row carries the "
        "`image_sha256` of its pixels and the licence and attribution of its source."
    )
    how = (
        f"The data ships as {configs} streamable Parquet configurations. Image bytes are stored "
        "once per `image_sha256`, in `images` or in the frozen v1.0 config that "
        "`metadata.bytes_config` names; every label config joins to them on that key. Use it for "
        f"benthic and coral segmentation (including the {V2_CLASSES}-class Coralscapes V2), "
        "instance segmentation, object detection, bleaching classification, point-based cover "
        "estimation, retrieval and pretraining. "
        f"The current release is v{version} (published {published}){closing}"
    )
    return f"{what}\n\n{how}"
