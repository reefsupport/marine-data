"""The frozen base of a patch release (``release gate --patch-of v1.1.0``).

A patch (v1.1.1 ``condition-masks``) adds configs to a published release and leaves every file of it
alone. The 20 GB prior tree is never downloaded: the gates read its file list with LFS sha256s
(anonymous ``list_repo_tree(expand=True)``) and its card (one small file) from the Hub. Everything
here is anonymous (``token=False``): a stored Hugging Face login is never read.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .shards import CardError
from .v11_common import card_configs, front_matter

DEFAULT_REPO = "reefsupport/marine-data"


@dataclass(frozen=True)
class PatchBase:
    """The prior release as the patch gates see it: ``tree`` is ``{path: lfs sha256 | None}`` and
    ``card`` the path of its ``README.md``. ``release`` is the tag (``v1.1.0``)."""

    release: str
    tree: Mapping[str, str | None]
    card: Path
    repo: str = DEFAULT_REPO
    card_entries: Mapping[str, dict] = field(default_factory=dict)
    dataset_info: Mapping[str, dict] = field(default_factory=dict)
    size_categories: tuple[str, ...] = ()

    @property
    def tree_configs(self) -> frozenset[str]:
        """Config folders of the prior tree (``data/<config>/...``)."""
        return frozenset(
            p.split("/")[1] for p in self.tree if p.startswith("data/") and p.count("/") > 1
        )

    @property
    def configs(self) -> frozenset[str]:
        """Every config the prior release lists: card entries plus tree folders."""
        return frozenset(self.card_entries) | self.tree_configs

    def resolves(self, pattern: str) -> bool:
        """True when a card ``data_files`` glob matches a file of the prior tree."""
        return any(fnmatch.fnmatchcase(path, pattern) for path in self.tree)


def build_base(release: str, tree: Mapping[str, str | None], card: Path, repo: str = DEFAULT_REPO):
    """A :class:`PatchBase` with the card's config entries, ``dataset_info`` and size bands read."""
    card = Path(card)
    if not card.is_file():
        raise CardError(f"prior card {card} not found")
    front = front_matter(card)
    info = front.get("dataset_info") or []
    info = [info] if isinstance(info, dict) else info
    bands = front.get("size_categories") or []
    return PatchBase(
        release=release,
        tree=dict(tree),
        card=card,
        repo=repo,
        card_entries=card_configs(card),
        dataset_info={str(e.get("config_name", "default")): e for e in info if isinstance(e, dict)},
        size_categories=tuple([bands] if isinstance(bands, str) else map(str, bands)),
    )


def load_patch_base(
    release: str,
    *,
    repo: str = DEFAULT_REPO,
    prior_dir: Path | None = None,
    tree_json: Path | None = None,
    cache_dir: Path | None = None,
) -> PatchBase:
    """The prior release from the Hub (anonymous), or offline: ``prior_dir`` holds its
    ``README.md`` and ``tree_json`` its ``{path: sha256}`` listing."""
    from .condition_masks import hub_tree

    tree = json.loads(Path(tree_json).read_text()) if tree_json else hub_tree(repo, release)
    if prior_dir is not None:
        card = Path(prior_dir) / "README.md"
    else:
        from huggingface_hub import hf_hub_download

        card = Path(
            hf_hub_download(
                repo,
                "README.md",
                repo_type="dataset",
                revision=release,
                token=False,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
    return build_base(release, tree, card, repo)
