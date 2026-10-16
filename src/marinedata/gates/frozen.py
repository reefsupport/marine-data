"""G2 frozen v1.0: the v1.1.0 build never rewrites the v1.0 tree.

Every build file whose path exists in the v1.0 tree (``--v1-dir``) must be byte-identical, except
``README.md`` (the card is rewritten, text only). No new file may land inside a v1.0 config
folder (``data/<v1 config>/``: that would add rows to a frozen config). The v1.0 ``configs:``
entries of the card front matter must be unchanged; new configs only add entries. A gate with no
v1.0 tree to compare against fails: it would have checked nothing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import GateResult, finish
from .v11_common import canonical, card_configs, file_sha256, tree_files

NAME = "frozen"
CARD = "README.md"


def run_frozen_gate(build_dir: Path, *, v1_dir: Path | None) -> GateResult:
    build = Path(build_dir)
    if v1_dir is None or not Path(v1_dir).is_dir():
        return finish(NAME, {}, ["no v1.0 tree: pass --v1-dir (a checkout of the v1.0 repo)"])
    v1 = Path(v1_dir)
    data = v1 / "data"
    v1_configs = {p.name for p in data.iterdir() if p.is_dir()} if data.is_dir() else set()
    counts: Counter = Counter()
    counts["v1_files"] = sum(1 for p in tree_files(v1) if p.relative_to(v1).as_posix() != CARD)
    failures: list[str] = []
    for path in tree_files(build):
        rel = path.relative_to(build).as_posix()
        if rel == CARD:
            continue
        counts["build_files"] += 1
        ref = v1 / rel
        if ref.is_file():
            counts["overlap"] += 1
            if ref.stat().st_size != path.stat().st_size or file_sha256(ref) != file_sha256(path):
                counts["changed"] += 1
                failures.append(f"{rel} changed_v1_file: differs from the v1.0 file (frozen)")
        else:
            parts = rel.split("/")
            if len(parts) > 2 and parts[0] == "data" and parts[1] in v1_configs:
                counts["extended_v1_config"] += 1
                failures.append(f"{rel} extends_v1_config: new file in frozen config {parts[1]}")
    old_card, new_card = v1 / CARD, build / CARD
    if not old_card.is_file() or not new_card.is_file():
        failures.append(f"{CARD} missing in the v1.0 tree or the build: cannot compare configs")
    else:
        old, new = card_configs(old_card), card_configs(new_card)
        counts["v1_configs"] = len(old)
        if not old:
            failures.append(f"{CARD} configs: the v1.0 card lists no configs")
        for name, entry in old.items():
            if name not in new:
                failures.append(f"{CARD} configs: v1.0 config {name} removed")
            elif canonical(new[name]) != canonical(entry):
                failures.append(f"{CARD} configs: v1.0 config {name} changed")
    return finish(NAME, counts, failures)
