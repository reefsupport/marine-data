"""Static v1.2.0 statements: Coralscapes V2 joins ``semseg`` and ``instances``.

One set of sentences, rendered into the open card by :func:`~.hf_card_v11.render_v11_card` when
the export carries ``coralscapes-v2`` and mirrored in CHANGELOG.md and docs/label-matching.md. The
numbers were checked on 2026-10-29: the twin and split counts against the pinned upstream
``instances`` parquet (HF revision 8a4567b2), the
class counts against ``registry/crosswalks/coralscapes-95.yaml`` and the totals against
``results/label-matching-v1.2.0.json``. Nothing here names a repo outside the open release, a
bucket, an image digest or a source that stays out of the open release
(``hf_export_v11.assert_public_safe`` re-checks the rendered card). The V1 statements of
:mod:`~.card_v11_text` are not edited; the one that V2 makes incomplete is replaced by
:func:`limitations`, which refuses to run when its anchor has moved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

SOURCE_ID = "coralscapes-v2"
VERSION = "1.2.0"
GITHUB = "https://github.com/reefsupport/marine-data"
LABEL_MATCHING_URL = f"{GITHUB}/blob/main/docs/label-matching.md"
LABEL_MATCHING_JSON_URL = f"{GITHUB}/blob/main/results/label-matching-v1.2.0.json"
TAXONOMY = "3.1.0"
V2_REPO = "josauder/coralscapesV2"
V2_REVISION = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"
V2_CARD_DATE = "2026-09-15"
V2_PAPER = "https://arxiv.org/abs/2609.12826"
V2_FISH_MASKS = 65_738
"""Fish instance masks as counted by the upstream card; not recounted here."""

SUPERSEDES = (
    "V2 supersedes V1 for Coralscapes tasks; V1 configs are kept frozen for reproducibility."
)

V2_FRAMES = 2_433
V2_SPLITS = {"train": 1_790, "validation": 170, "test": 473}
TEST_PAIR = ("site23_000097_017682", "site23_000152_017682")
BENCHMARK = V2_SPLITS["test"] + V2_SPLITS["validation"]
"""Rows of the ``coralscapes-v2`` benchmark manifest; it has one digest fewer than rows."""
TWINS = 2_074
V1_ONLY = 1
V2_NEW = {"train": 273, "validation": 4, "test": 82}
SITES = {"V1": 35, "V2": 41}
EDGES = {"exact": 60, "coarsened": 25, "approximate": 9, "unmappable": 1}
LEGACY = {"same": 64, "refines": 22, "retarget": 8, "ignore-to-mapped": 1}
NEW_NODES = {"CN_ANTIPATHARIA": 22549, "SC_ALCYONIIDAE": 125269, "SC_NEPHTHEIDAE": 146762}
"""The three ``rs-benthic-v1`` nodes of taxonomy 3.1.0 and their WoRMS AphiaIDs (read live)."""
REGISTRY = {
    "units": 74, "with_crosswalk": 57, "no_crosswalk": 17, "labels": 3_670, "edges": 3_258,
    "crosswalks": 44, "exact": 2_251, "coarsened": 777, "approximate": 71,
    "unmappable": 159, "lossy_without_note": 66,
}  # fmt: skip
"""Registry-wide totals of ``results/label-matching-v1.2.0.json``; a test asserts they match."""

_ANCHOR = "Coralscapes has no soft-coral or octocoral class."

SUMMARY = (
    "v1.2.0 adds Coralscapes V2 (source id `coralscapes-v2`) to `semseg` and `instances` and adds "
    "no config. " + SUPERSEDES
)
VERSION_NOTE = (
    "Additive: Coralscapes V2 added to `semseg` and `instances`; every v1.0, v1.1.0 and v1.1.1 "
    "file, row, id and path is unchanged."
)
LIMITATION = (
    "Coralscapes V1 (`coralscapes`) has no soft-coral or octocoral class. Coralscapes V2 "
    "(`coralscapes-v2`) has soft-coral and black-coral classes, so use V2 for Coralscapes tasks. "
    "`label-map` records the classes each source cannot contain, so a model trained on V1 alone "
    "cannot predict them and cover for them is undercounted. A V1 frame and its V2 twin are the "
    f"same frame: the counts of the two sources are not additive ({TWINS:,} frames are in both)."
)
CHANGELOG = (
    "**Added (additive; no v1.0, v1.1.0 or v1.1.1 file, row, id or path changed):** Coralscapes "
    f"V2 (source id `coralscapes-v2`, {V2_FRAMES:,} frames, 95 classes) in `semseg` and "
    "`instances`. No config is added or renamed.",
    "**Superseded, not removed:** " + SUPERSEDES,
    "**Label space:** taxonomy 3.1.0 adds three `rs-benthic-v1` nodes (`CN_ANTIPATHARIA`, "
    "`SC_ALCYONIIDAE`, `SC_NEPHTHEIDAE`) and the 95 `coralscapes-95` edges; nothing in 3.0.0 "
    "changes. The `coral-binary` view counts black coral (`CN_ANTIPATHARIA`) as coral.",
    f"**Splits:** the {TWINS:,} frames V1 and V2 share are linked by stem, so a twin pair is one "
    "group; the official V2 split is kept (train "
    f"{V2_SPLITS['train']:,} / validation {V2_SPLITS['validation']} / test {V2_SPLITS['test']}) "
    "and no frame moves.",
    "**Changed:** this card (text only).",
)

INTRO = (
    f"**{SUPERSEDES}** [Coralscapes V2](https://huggingface.co/datasets/{V2_REPO}) "
    f"(Apache-2.0, [arXiv:2609.12826]({V2_PAPER})) re-annotates and extends the Coralscapes V1 "
    f"reef scenes: {V2_FRAMES:,} frames of 1024 x 2048 pixels from {SITES['V2']} Red Sea dive "
    "sites, labelled with 95 classes instead of 39, including soft corals and black coral. This "
    f"release pins the upstream revision `{V2_REVISION}` (card dated {V2_CARD_DATE}) and keeps "
    f"the official split: {V2_SPLITS['train']:,} train, {V2_SPLITS['validation']} validation and "
    f"{V2_SPLITS['test']} test frames."
)
CLASSES = (
    "### New classes and taxonomy 3.1.0\n\n"
    'V2 separates classes that V1 folded into "other coral". The `rs-benthic-v1` scheme gains '
    "three nodes (taxonomy 3.1.0 is additive: 3 nodes and 95 edges are added and nothing in "
    "3.0.0 changes): `CN_ANTIPATHARIA` (order Antipatharia, black corals, WoRMS AphiaID "
    f"{NEW_NODES['CN_ANTIPATHARIA']}), `SC_ALCYONIIDAE` (family Alcyoniidae, "
    f"{NEW_NODES['SC_ALCYONIIDAE']}) and `SC_NEPHTHEIDAE` (family Nephtheidae, "
    f"{NEW_NODES['SC_NEPHTHEIDAE']}). Eight V2 classes leave the hard-coral target of the V1 "
    "class they came from: `cirrhipathes` (black coral); `leather coral alive`, "
    "`nephtheidae alive`, `xeniidae alive` and the two `soft coral (malacalcyonacea)` classes "
    "(alive, bleached), which map to soft-coral nodes; and `millepora bleached` and `millepora "
    "dead` (fire coral). In the `coral-binary` view `CN_ANTIPATHARIA` counts as coral: the V1 "
    'class "other coral alive", which `cirrhipathes` came from, is hard coral, and counting it '
    "keeps a V1 frame and its V2 twin consistent."
)
FROZEN = (
    "### What stays frozen\n\n"
    "The four v1.0 configs (`coral-masks`, `scene-masks`, `instance-masks`, `fish-boxes`) and "
    "every v1.1.0 and v1.1.1 config keep their files, rows, ids and paths; v1.2.0 only adds "
    "files. Coralscapes V1 (`coralscapes`) stays where it is, with its V1 splits, so earlier tags "
    "reproduce exactly. Pin a tag with `revision` to read an earlier release. Downstream model "
    "pins are unchanged."
)
# Open flavour only: the v1.2.0 section is rendered for the open card alone. The numbers are those
# of the frozen release map; tests/test_card_v12_text.py pins every one of them to
# releases/marine-data/v1.2.0/SPLIT_MAP.json (the moved groups also to v1.1.0's). Id-free: no
# digest, group id or frame name.
SPLIT_MAP = (
    "### Split map of v1.2.0\n\n"
    "**Inherited.** The split map inherits the v1.1.0 map (`inherits` in `SPLIT_MAP.json`): its "
    "35,825 groups carry over, 4,382 of them kept verbatim, and the splits that v1.1.1 published "
    "per image are the prior for the rest. Re-resolving them changed the split of 9 inherited "
    "groups and of no other: 8 are near-duplicate groups from the Roboflow bleaching exports "
    "that no published v1.1.x table references, and 1 is a UIIS10K image that v1.1.1 published "
    "in two splits (train in `fish-boxes` and `instance-masks`, validation in `metadata`); "
    "v1.2.0 puts it in train in every table. No benchmark pin overrides an inherited split "
    "(`pins.v1_continuity_overridden`: 0 images, 0 groups).\n\n"
    "**Coralscapes V2.** 2,432 of the 2,433 V2 frames are assigned: 1,790 to train, 170 to "
    "validation and 472 to test (the official V2 split has 473 test frames; the one missing is "
    "withheld, below). The V2 benchmark pins hold 472 test and 170 validation images, counted by "
    "digest, so the byte-identical test pair counts once; the V1 pins are unchanged at 391 test "
    "and 166 validation images. Stem links join the V1 and V2 twins: 2,074 twins, 359 V2-only "
    "frames, 1 V1-only frame, 0 ambiguous stems and 0 conflicts (a conflict is a twin whose "
    "upstream V2 split differs from the split v1.0 published).\n\n"
    "**Withheld and left out.** One group is withheld: a V2 benchmark frame inside the "
    "near-duplicate band of train images of a prior release. The band says never-eval and the "
    "pin says eval, so the frame ships in no table of any config. The split map also leaves out "
    "18 images that sit in the prior-release near-duplicate band (dHash distance 8 or less) of "
    "prior-release train and evaluation images, on the side their near-duplicate group did not "
    "keep, and 34 images that each collide on the DCT pHash-64 with an image of another split "
    "(one image of the pair goes, never an image that an earlier release published); these 52 "
    "images ship in no split of any config.\n\n"
    "**Near-duplicate unions; no dedup run.** A dHash candidate pair (distance 4 or less) is "
    "unioned only when its pHash-256 confirms it (at most 32 of 256 bits apart, both images "
    "carrying at least 4 bits of grey entropy); a pair that fails the confirm links nothing, and "
    "a new image paired with a prior-release image then joins the prior-release band "
    "(`near_dup.union_confirm`). No near-duplicate groups file was supplied for v1.2.0 "
    "(`near_dup.dedup_groups` reads `disabled by --no-near-dup`): the v1.1.0 groups file cannot "
    "be regenerated from the repository, so v1.2.0 ships without a dedup run rather than with "
    "one that cannot be reproduced. The dHash and pHash unions above still ran inside the "
    "split-map inheritance. The build-time split-leak gate is off unless a groups file is "
    "supplied."
)
_BIB_AUTHORS = (
    "Sauder, Jonathan", "Ruckli, Thomas", r"Strodomskyt{\.e}, Gabriel{\.e}",
    "Abdallah, Ibrahim Souleiman", "Abdi, Rahma Hassan", "Awaleh, Djama Goumaneh",
    "Farah, Mohamed Houssein", "Nour, Moustapha", "Saad, Osama Sharhubil",
    "Altaib, Mustafa Mohammed Khalafallah", "Kteifan, Maysoon", "Alsoqi, Farah", "Zgool, Eyad",
    "Al-Omari, Jafar", "Gebreluel, Temesgen Gebremeskel", "Abdulkerim, Zekaria Zekeria",
    "Ghirmay, Meron", "Beraki, Teklehaimanot", "Tuia, Devis", "Banc-Prandi, Guilhem",
)  # fmt: skip
BIBTEX_V2 = (
    "@inproceedings{sauder2026coralscapesv2,\n"
    "  title={CoralscapesV2: Panoptic and Fine-Grained Visual Scene Understanding in Coral "
    "Reefs},\n"
    "  author={" + " and ".join(_BIB_AUTHORS) + "},\n"
    "  booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshop on "
    "Marine Vision},\n"
    "  year={2026}\n"
    "}"
)
BIBTEX_V1 = (
    "@misc{sauder2025coralscapesdatasetsemanticscene,\n"
    "  title={The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs},\n"
    "  author={Jonathan Sauder and Viktor Domazetoski and Guilhem Banc-Prandi and Gabriela Perna "
    "and Anders Meibom and Devis Tuia},\n"
    "  year={2025},\n"
    "  eprint={2503.20000},\n"
    "  archivePrefix={arXiv},\n"
    "  primaryClass={cs.CV},\n"
    "  url={https://arxiv.org/abs/2503.20000}\n"
    "}"
)
CITATION = (
    "### Citing Coralscapes V2\n\n"
    f"If you use the `coralscapes-v2` rows, cite the V2 paper ([arXiv:2609.12826]({V2_PAPER})) as "
    "its dataset card asks; V2 extends the original Coralscapes dataset, which the card also "
    "asks you to cite. Both entries are copied from the upstream card.\n\n"
    f"```bibtex\n{BIBTEX_V2}\n\n{BIBTEX_V1}\n```"
)


# History rows of the published v1.1.0 / v1.1.1 cards (the v1.2.0 card is a superset of them).
V110_DATE = "2026-10-17"  # build date printed in the published v1.1.0 card
V110_NOTE = "Additive: 11 new configurations."
V111_DATE = "2026-10-24"
V111_NOTE = (
    "Additive: one new configuration, `condition-masks` (634 Tayrona bleaching rasters, `train`). "
    "Every v1.1.0 file is unchanged."
)
V111_CHANGELOG = (
    "Additive: one new configuration, `condition-masks` (634 Tayrona bleaching rasters, `train`). "
    "Every v1.1.0 file is unchanged.",
)
RELEASE_MONTH = "2026-10"
"""The date cell of the v1.2.0 row when the renderer got no build date."""


def version_rows(v1_date: str, v1_note: str, built: str | None) -> list[tuple[str, str, str]]:
    """The four rows of the v1.2.0 card's version table: v1.0.0, v1.1.0, v1.1.1 and v1.2.0."""
    return [
        ("v1.0.0", v1_date, v1_note),
        ("v1.1.0", V110_DATE, V110_NOTE),
        ("v1.1.1", V111_DATE, V111_NOTE),
        (f"v{VERSION}", built or RELEASE_MONTH, VERSION_NOTE),
    ]


# Named gate slots of the v1.2.0 "Build and gates" text. The build host fills them from
# ``provenance/GATES.json`` with :func:`fill_gates` (release runbook, step 7); the card stage does
# it when it gets ``--gates-json`` (:func:`fill_from_gates`).
GATE_SLOTS = ("V120_GATES_RUN", "V120_DEDUP_LINKED", "V120_SPLIT_TWIN_GROUPS")
GATES_PARAGRAPH = (
    "**v1.2.0 gates.** [[V120_GATES_RUN]] gates ran on the build host. The dedup gate reports "
    "[[V120_DEDUP_LINKED]] V1/V2 twin pairs as one link group (`LINKED`), not as a cross-split "
    "leak, and the splits gate finds [[V120_SPLIT_TWIN_GROUPS]] twin groups on one side of the "
    "split. The numbers are in `provenance/GATES.json`."
)


def applies(sources: Iterable[Mapping]) -> bool:
    """True when the export carries ``coralscapes-v2`` rows: only then the card is a v1.2.0 card."""
    return any(s.get("source_id") == SOURCE_ID for s in sources)


def images(sources: Iterable[Mapping]) -> int | None:
    """``n_images`` of the ``coralscapes-v2`` row of the ``sources`` config (the build fills it)."""
    for s in sources:
        if s.get("source_id") == SOURCE_ID:
            return int(s["n_images"]) if s.get("n_images") else None
    return None


def limitations(base: tuple[str, ...]) -> tuple[str, ...]:
    """``base`` with its Coralscapes bullet replaced; raises when the bullet is not first."""
    if not base or not base[0].startswith(_ANCHOR):
        raise ValueError(f"v1.2.0 card: limitation anchor {_ANCHOR!r} is not the first bullet")
    return (LIMITATION, *base[1:])


def _changed(sources: Iterable[Mapping]) -> str:
    n = images(sources)
    count = f" ({n:,} images in this release)" if n else ""
    return (
        "### What changed in `semseg` and `instances`\n\n"
        f"- Both configs gain rows under the new source id `coralscapes-v2`{count}; the `sources` "
        "config and the Splits table carry the counts per source, config and split. No config is "
        "added or renamed, and no existing file, row, id or path changes.\n"
        "- `semseg` holds the V2 semantic masks as `uint8` PNGs with the upstream class ids 1 to "
        '95. Pixel `0` is the upstream "unlabelled" value and is not counted as a class; '
        "`label-map` records the ignore semantics per source.\n"
        "- `instances` holds the V2 instance masks as `uint16` PNGs, unchanged from upstream: a "
        "pixel below 2000 is the class id of a stuff class, a pixel of 2000 or more is "
        "`class_id * 2000 + instance_index`, and fish is class 25. The upstream card reports "
        f"{V2_FISH_MASKS:,} fish instance masks."
    )


def _matching() -> str:
    e, g = EDGES, REGISTRY
    return (
        "### Label matching\n\n"
        "Every V2 class resolves into `rs-benthic-v1` through the crosswalk `coralscapes-95`: "
        f"{sum(e.values())} edges, {e['exact']} exact, {e['coarsened']} coarsened and "
        f"{e['approximate']} approximate. One, `hard substrate`, is unmappable and its pixels "
        "train as ignore (`255`). No label is unresolved, ambiguous or missing. The edge note of "
        "each V2 class names the V1 class it came from, and the check pairs all 95 with the V1 "
        f"edge of that name: {LEGACY['same']} keep the V1 target, {LEGACY['refines']} refine it "
        "(for example a coral genus under a V1 morphology class), "
        f"{LEGACY['retarget']} move off hard coral (above) and 1 (`human-made structure`) is "
        'mapped where V1\'s "unknown hard substrate" is not. The same verification runs over '
        f"every source of the registry that has a crosswalk: {g['labels']:,} labels and "
        f"{g['edges']:,} edges in {g['crosswalks']} crosswalks ({g['exact']:,} exact, "
        f"{g['coarsened']} coarsened, {g['approximate']} approximate, {g['unmappable']} "
        "unmappable, each unmappable edge with a note), with no unresolved, ambiguous, missing or "
        f"duplicate label. Method, per-source tables and the unmappable labels: "
        f"[docs/label-matching.md]({LABEL_MATCHING_URL})."
    )


def _splits() -> str:
    n = V2_NEW
    return (
        "### Splits: V1 and V2 twins\n\n"
        f"{TWINS:,} V2 frames are V1 frames: a V1 file name is the V2 stem "
        "(`site<N>_<frame>_<timestamp>`) plus `_leftImg8bit.png`. V2 re-encoded the images and "
        "re-annotated the masks, so the bytes and the pixels of a twin differ from V1 and the "
        "sha256 keys that carry the v1.0 splits forward cannot link them. The split map therefore "
        "links twins by stem and treats each pair as one group. The dedup and splits gates count "
        "a pair in one split as one link group and fail a pair that sits in two splits. For all "
        f"{TWINS:,} twins the release split of the V1 frame equals the official V2 split, so no "
        f"frame moves. V2 adds {sum(n.values())} frames ({n['train']} train, {n['validation']} "
        f"validation, {n['test']} test) that keep their official V2 split; the V2 test and "
        "validation frames are pinned as benchmark splits "
        # 472: tests/test_card_v12_text.py pins it to releases/marine-data/v1.2.0/SPLIT_MAP.json
        # (`pins.benchmarks`, coralscapes-v2 test count).
        f"(472 of the {V2_SPLITS['test']} test frames and all {V2_SPLITS['validation']} "
        "validation frames; the remaining test frame is withheld, see the split map below). "
        f"{V1_ONLY} V1 frame, a V1 test frame, has no V2 twin and exists only in "
        f"the frozen V1 configs. V2 covers {SITES['V2']} dive sites (V1: {SITES['V1']}; the six "
        "new ones are site36 to site41). The 55 V2-only frames of site30 are V2 train frames while "
        "the frames of site30 shared with V1 are validation frames; the official split is kept "
        "and the gates re-check that boundary at build time. V1 and V2 counts are not additive: "
        f"summing them counts the {TWINS:,} shared frames twice."
    )


def _test_pair() -> str:
    a, b = TEST_PAIR
    return (
        "### A byte-identical test pair\n\n"
        f"Two V2 frames, `{a}` and `{b}`, have identical image bytes and different masks, and "
        "both are in the upstream `test` split. That is how the upstream data are published, and "
        "the release keeps them as published: neither frame is dropped, merged or moved. The "
        f"benchmark manifest of the V2 test and validation frames therefore has {BENCHMARK} rows "
        f"and {BENCHMARK - 1} unique image digests."
    )


def section(sources: Iterable[Mapping]) -> str:
    """The ``## Coralscapes V2`` card section; the V2 image count is read from ``sources``."""
    parts = (
        "## Coralscapes V2 (new in v1.2.0)",
        INTRO,
        _changed(sources),
        CLASSES,
        _matching(),
        _splits(),
        SPLIT_MAP,
        _test_pair(),
        FROZEN,
        CITATION,
    )
    return "\n\n".join(parts)


def labels_section() -> str:
    """The ``## Using labels across sources`` card section; its figures are :data:`REGISTRY`."""
    g = REGISTRY
    first, second, third = NEW_NODES
    nodes = f"`{first}`, `{second}` and `{third}`"
    return (
        "## Using labels across sources\n\n"
        "Every source names its classes in its own way: a pixel value, a label from an "
        "annotation platform, a species concept, a survey code. To use several sources "
        "together, express every label in the ids of one reference vocabulary, the "
        "`rs-benthic-v1` scheme of taxonomy release "
        f"{TAXONOMY} (additive: it adds the three nodes {nodes}, and nothing in 3.0.0 changes). "
        "Each source that has labels to map has one crosswalk: one edge per native label, naming "
        "the node it resolves to and how closely. An edge is `exact` (both mean the same thing), "
        "`coarsened` (the node is a strict ancestor: detail is lost, nothing is invented), "
        "`approximate` (best effort: the concepts overlap imperfectly) or `unmappable` (no node "
        "exists).\n\n"
        "### Verified for every source\n\n"
        "A check on the registry alone, which reads no image, resolves every native label of every "
        f"source into `rs-benthic-v1`. It covers {g['units']} units (sources, and crosswalks that "
        f"no source claims): {g['with_crosswalk']} have a crosswalk and pass, "
        f"{g['no_crosswalk']} are label-less or have no crosswalk. The {g['crosswalks']} "
        f"crosswalks hold {g['labels']:,} labels and {g['edges']:,} edges: {g['exact']:,} exact, "
        f"{g['coarsened']} coarsened, {g['approximate']} approximate and {g['unmappable']} "
        "unmappable, and every unmappable edge carries a note that says why. No label is "
        "unresolved, ambiguous, missing or duplicated. Method, per-source tables and the "
        f"unmappable labels: [docs/label-matching.md]({LABEL_MATCHING_URL}). The result as data: "
        f"[results/label-matching-v1.2.0.json]({LABEL_MATCHING_JSON_URL}).\n\n"
        "### How the training views apply it\n\n"
        f"The `marinedata` package ([code]({GITHUB})) applies the crosswalks: "
        "`labels.lut(source_id, scheme)` and `labels.remap_mask(mask, source_id, scheme)` turn the "
        "native values of any source into the ids of one training view. `coral-binary` is coral "
        "against everything else (`CN_ANTIPATHARIA` counts as coral), `benthic-coarse` has six "
        "classes (hard coral, fire coral, soft coral, algae, abiotic, other fauna) and `scene` "
        "has the eight SUIM classes. `255` is the ignore value in all of them: it carries no "
        "gradient and is not counted in any metric, and it is what an unmappable label and every "
        "pixel value that a source does not declare become, so a source cannot add a class by "
        "accident. A head finer than a view reads the `rs-benthic-v1` node of each edge instead, "
        "and supervises a `coarsened` label at its node and the ancestors of the node, never at "
        "one finer class below it.\n\n"
        "### What an edge cannot promise\n\n"
        "A crosswalk makes labels comparable, not identical. A `coarsened` edge loses detail and "
        "an `approximate` edge is the closest node, not the same thing: read the edge note before "
        "you rely on a fine class. An unmappable label supervises nothing on the axis it cannot "
        f"fill. {g['lossy_without_note']} lossy edges of older crosswalks have no note yet; the "
        "check pins that number so that it can only fall. The two coarse views can also disagree "
        "on a label: the V1 label `other coral alive` and its V2 twin `cirrhipathes` (black "
        "coral) are both coral in `coral-binary`, but only the V1 label is hard coral in "
        "`benthic-coarse`, where the V2 label is ignored."
    )


def mask_note() -> str:
    """The v1.1.1 condition-masks note: the v1.2.0 card keeps it (the card is a superset)."""
    from .condition_masks_docs import MASK_NOTE

    return MASK_NOTE


def fill_gates(readme: str, numbers: Mapping[str, object]) -> str:
    """Release runbook step 7: fill the named gate slots of a rendered v1.2.0 card.

    Raises ``ValueError`` on a missing or unknown slot name or a card without the slot, so a
    published card never carries a ``[[SLOT]]`` marker."""
    missing = [s for s in GATE_SLOTS if s not in numbers]
    unknown = sorted(set(numbers) - set(GATE_SLOTS))
    if missing or unknown:
        raise ValueError(f"gate slots: missing {missing}, unknown {unknown}")
    out = readme
    for slot in GATE_SLOTS:
        token = f"[[{slot}]]"
        if token not in out:
            raise ValueError(f"the card has no {token} slot")
        out = out.replace(token, str(numbers[slot]))
    return out


_SLOT = re.compile(r"\[\[(V120_[A-Z0-9_]+)\]\]")


def pending_slots(readme: str) -> tuple[str, ...]:
    """The ``[[V120_...]]`` gate markers still in ``readme``, each once, in card order."""
    return tuple(dict.fromkeys(_SLOT.findall(readme)))


def gate_numbers(gates: Mapping[str, object]) -> dict[str, int]:
    """The slot numbers read from a ``GATES.json`` dict (shape: :mod:`marinedata.gates`).

    ``V120_GATES_RUN`` counts the entries that did not skip, ``V120_DEDUP_LINKED`` is
    ``dedup.counts.near_pairs_linked_twins`` and ``V120_SPLIT_TWIN_GROUPS`` is
    ``splits.counts.stem_link_twins``. Raises ``ValueError`` instead of printing a number the
    gates did not measure: a ``dedup`` or ``splits`` gate that did not pass, a missing counter, or
    a twin pair the splits gate found in two splits."""
    from .gates import SUMMARY_KEY, gate_status
    from .gates.dedup import LINKED

    status = {k: gate_status(v) for k, v in gates.items() if k != SUMMARY_KEY}

    def counter(gate: str, key: str) -> int:
        if status.get(gate) != "passed":
            raise ValueError(
                f"GATES.json: the {gate} gate is {status.get(gate, 'missing')}, not passed; "
                "run the gates before the card is filled"
            )
        entry = gates[gate]
        counts = entry.get("counts") if isinstance(entry, dict) else None
        value = counts.get(key) if isinstance(counts, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"GATES.json: {gate}.counts.{key} is missing or not a count")
        return value

    apart = counter("splits", "stem_link_pairs_apart")
    if apart:
        raise ValueError(f"GATES.json: the splits gate found {apart} twin pair(s) in two splits")
    return {
        "V120_GATES_RUN": sum(s != "skipped" for s in status.values()),
        "V120_DEDUP_LINKED": counter("dedup", LINKED),
        "V120_SPLIT_TWIN_GROUPS": counter("splits", "stem_link_twins"),
    }


def fill_from_gates(readme: str, gates: Mapping[str, object]) -> str:
    """:func:`fill_gates` with the numbers :func:`gate_numbers` reads from a ``GATES.json`` dict."""
    return fill_gates(readme, gate_numbers(gates))
