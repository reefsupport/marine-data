"""v1.1.1 card, Croissant, CITATION and lock: v1.1.0's documents with ``condition-masks`` added.

Every edit is anchored on text that must occur exactly where expected; a missing anchor raises
(:class:`~.hf_export_v11.V11ExportError`) instead of writing a card that silently lacks the change.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from .hf_export_v11 import V11ExportError, assert_public_safe

CONFIG = "condition-masks"
VERSION = "1.1.1"
PRIOR = "1.1.0"
_DTYPE = {"string": "string", "image": "image", "int64": "int64", "double": "double"}
MASK_NOTE = (
    "**Condition masks (v1.1.1).** `condition-masks` holds the Tayrona bleaching rasters of "
    "`reef-support-bleaching` as palette PNGs, `train` only, one row per image (hidden-test "
    "images withheld). The image bytes are not repeated: `metadata.bytes_config` names the "
    "config that holds them. Pixel `255` is ignore (unlabelled in the source raster, value 0), "
    "`1` is `bleached` (condition `BLEACHED`) and `2` is `non_bleached` (condition `HEALTHY`: "
    "the source only recorded bleached or not). `class_map` keeps the source pixel values; "
    "`attrs.class_resolution` carries the condition node. The v1.1.0 `label-map` has no rows for "
    "this source; this note is the definition."
)


def _once(text: str, anchor: str, new: str, *, before: bool = False, nth: int = 1) -> str:
    """``text`` with ``new`` placed before/after the ``nth`` occurrence of ``anchor``."""
    pos, count = -1, 0
    while count < nth:
        pos = text.find(anchor, pos + 1)
        if pos < 0:
            raise V11ExportError(f"card anchor not found (#{nth}): {anchor[:60]!r}")
        count += 1
    cut = pos if before else pos + len(anchor)
    return text[:cut] + new + text[cut:]


def _features(columns) -> str:
    return "".join(f"  - name: {n}\n    dtype: {_DTYPE[k]}\n" for n, k in columns)


def patch_card(readme: str, columns, *, rows: int, date: str) -> str:
    """The v1.1.0 card with the ``condition-masks`` config, version line, note and changelog."""
    if f"config_name: {CONFIG}" in readme:
        raise V11ExportError("card already lists condition-masks")
    entry = (
        f"- config_name: {CONFIG}\n  data_files:\n  - split: train\n"
        f"    path: data/{CONFIG}/train-*.parquet\n"
    )
    info = f"- config_name: {CONFIG}\n  features:\n{_features(columns)}"
    out = _once(readme, "- config_name: instances\n", entry, before=True, nth=1)
    out = _once(out, "- config_name: instances\n", info, before=True, nth=2)
    out = out.replace(
        "| Version | v1.1.0, built", f"| Version | v{VERSION} (v{PRIOR} plus `{CONFIG}`), built", 1
    )
    if f"v{VERSION}" not in out:
        raise V11ExportError("card anchor not found: the Version row")
    out = re.sub(r"\| Configurations \| 11: ", "| Configurations | 12: ", out, count=1)
    out = re.sub(
        r"(?m)^(\| Configurations \|.*`semseg`), ",
        lambda m: f"{m.group(1)}, `{CONFIG}`, ",
        out,
        count=1,
    )
    if f"`semseg`, `{CONFIG}`" not in out:
        raise V11ExportError("card anchor not found: the Configurations row")
    names = ", ".join(f"`{n}`" for n, _ in columns)
    bullet = f"- `{CONFIG}` ({len(columns)} columns): {names}.\n"
    out = re.sub(
        r"(?m)^(- `semseg` \(\d+ columns\):.*\n)", lambda m: m.group(1) + bullet, out, count=1
    )
    if bullet not in out:
        raise V11ExportError("card anchor not found: the semseg structure bullet")
    out = _once(out, "## Label space", MASK_NOTE + "\n\n", before=True)
    row = (
        f"| v{VERSION} | {date} | Additive: one new configuration, `{CONFIG}` ({rows:,} Tayrona "
        "bleaching rasters, `train`). Every v1.1.0 file is unchanged. |\n"
    )
    out = re.sub(r"(?m)^(\| v1\.1\.0 \|.*\n)", lambda m: m.group(1) + row, out, count=1)
    if row not in out:
        raise V11ExportError("card anchor not found: the v1.1.0 changelog row")
    return out.replace(f"version = {{{PRIOR}}}", f"version = {{{VERSION}}}")


def patch_croissant(doc: dict, columns, files: list[dict]) -> dict:
    """A new Croissant dict: the ``semseg`` file set / record set cloned for ``condition-masks``."""
    out, keep = copy.deepcopy(doc), {n for n, _ in columns}

    def rename(obj):
        return json.loads(json.dumps(obj).replace("semseg", CONFIG))

    sets = {r["@id"]: r for r in out["recordSet"]}
    if "semseg" not in sets or "semseg_splits" not in sets:
        raise V11ExportError("croissant: no semseg record set to clone")
    record = rename(sets["semseg"])
    record["field"] = [f for f in record["field"] if f["name"].split("/", 1)[1] in keep]
    record["description"] = f"{CONFIG}: {sum(f['rows'] for f in files):,} rows from 1 source(s)."
    splits = rename(sets["semseg_splits"])
    splits["data"] = [{f"{CONFIG}_splits/split_name": "train"}]
    base = next(d for d in out["distribution"] if d.get("name", "").startswith("semseg/"))
    fileset = rename(next(d for d in out["distribution"] if d.get("@id") == "semseg-parquet"))
    root = base["contentUrl"].rsplit("/data/semseg/", 1)[0]
    objs = [
        {**rename(base), "@id": f"{CONFIG}/{f['name']}", "name": f"{CONFIG}/{f['name']}",
         "contentUrl": f"{root}/data/{CONFIG}/{f['name']}",
         "contentSize": f"{f['bytes']} B", "sha256": f["sha256"]}
        for f in files
    ]  # fmt: skip
    out["distribution"] = [*out["distribution"], *objs, fileset]
    out["recordSet"] = [*out["recordSet"], record, splits]
    out["version"] = VERSION
    return out


def patch_lock(lock: dict, *, created: str, code_sha: str | None = None) -> dict:
    """The v1.1.0 lock with ``release_id`` v1.1.1 and the new config (a new dict)."""
    configs = [*lock.get("configs", [])]
    if CONFIG not in configs:
        configs.append(CONFIG)
    out = {**lock, "release_id": f"v{VERSION}", "created": created, "configs": configs}
    return {**out, "code_sha": code_sha} if code_sha else out


def patch_citation(text: str, date: str) -> str:
    out = re.sub(r"(?m)^version: .*$", f'version: "{VERSION}"', text, count=1)
    return re.sub(r"(?m)^date-released: .*$", f'date-released: "{date}"', out, count=1)


def _read(path, what: str) -> str:
    if not path or not Path(path).is_file():
        raise V11ExportError(f"--patch needs {what}")
    return Path(path).read_text()


def write_docs(args, summary: dict, report: dict) -> list[str]:
    """Write ``README.md``, ``croissant.json``, ``CITATION.cff`` and the provenance lock
    into ``args.out`` from the v1.1.0 ones (``--prior-card/-croissant/-lock/-citation``)."""
    import yaml

    from .condition_masks import COLUMNS

    out = Path(args.out)
    now = dt.datetime.now(dt.UTC)
    date = args.release_date or now.date().isoformat()
    files = []
    for entry in summary["configs"][CONFIG]["written"]:
        data = (out / entry["name"]).read_bytes()
        files.append({
            "name": Path(entry["name"]).name, "bytes": entry["bytes"], "rows": entry["rows"],
            "sha256": hashlib.sha256(data).hexdigest(),
        })  # fmt: skip
    card = patch_card(
        _read(args.prior_card, "--prior-card"), COLUMNS, rows=report["rows"], date=date
    )
    assert_public_safe(card, "open", where="README.md")
    doc = patch_croissant(
        json.loads(_read(args.prior_croissant, "--prior-croissant")), COLUMNS, files
    )
    lock = patch_lock(
        yaml.safe_load(_read(args.prior_lock, "--prior-lock")),
        created=now.isoformat(timespec="seconds"), code_sha=getattr(args, "code_sha", None),
    )  # fmt: skip
    cff = patch_citation(_read(args.prior_citation, "--prior-citation"), date)
    written = {
        "README.md": card, "croissant.json": json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        "CITATION.cff": cff, "provenance/release.lock.yaml": yaml.safe_dump(lock, sort_keys=False),
    }  # fmt: skip
    for rel, text in written.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    return sorted(written)
