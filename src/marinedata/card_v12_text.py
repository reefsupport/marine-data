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

from collections.abc import Iterable, Mapping

SOURCE_ID = "coralscapes-v2"
VERSION = "1.2.0"
GITHUB = "https://github.com/reefsupport/marine-data"
LABEL_MATCHING_URL = f"{GITHUB}/blob/main/docs/label-matching.md"
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
TWINS = 2_074
V1_ONLY = 1
V2_NEW = {"train": 273, "validation": 4, "test": 82}
SITES = {"V1": 35, "V2": 41}
EDGES = {"exact": 60, "coarsened": 25, "approximate": 9, "unmappable": 1}
LEGACY = {"same": 64, "refines": 22, "retarget": 8, "ignore-to-mapped": 1}
NEW_NODES = {"CN_ANTIPATHARIA": 22549, "SC_ALCYONIIDAE": 125269, "SC_NEPHTHEIDAE": 146762}
"""The three ``rs-benthic-v1`` nodes of taxonomy 3.1.0 and their WoRMS AphiaIDs (read live)."""
REGISTRY = {
    "labels": 3_670, "edges": 3_258, "crosswalks": 44, "exact": 2_251, "coarsened": 777,
    "approximate": 71, "unmappable": 159,
}  # fmt: skip
"""Registry-wide totals of ``results/label-matching-v1.2.0.json``."""

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
V110_DATE = "2026-10-17"
V110_NOTE = "Additive: 11 new configurations."
V111_DATE = "2026-10-24"
V111_NOTE = (
    "Additive: one new configuration, `condition-masks` (635 Tayrona bleaching rasters, `train`). "
    "Every v1.1.0 file is unchanged."
)
V111_CHANGELOG = (
    "Additive: one new configuration, `condition-masks` (635 Tayrona bleaching rasters, `train`). "
    "Every v1.1.0 file is unchanged.",
)

# Named gate slots of the v1.2.0 "Build and gates" text. The build host fills them from
# ``provenance/GATES.json`` with :func:`fill_gates` (release runbook, step 7).
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
        f"validation, {n['test']} test) that keep their official V2 split; the V2 test "
        f"({V2_SPLITS['test']}) and validation ({V2_SPLITS['validation']}) frames are pinned as "
        f"benchmark splits. {V1_ONLY} V1 frame, a V1 test frame, has no V2 twin and exists only in "
        f"the frozen V1 configs. V2 covers {SITES['V2']} dive sites (V1: {SITES['V1']}; the six "
        "new ones are site36 to site41). The 55 V2-only frames of site30 are V2 train frames while "
        "the frames of site30 shared with V1 are validation frames; the official split is kept "
        "and the gates re-check that boundary at build time. V1 and V2 counts are not additive: "
        f"summing them counts the {TWINS:,} shared frames twice."
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
        FROZEN,
        CITATION,
    )
    return "\n\n".join(parts)


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
