"""Open (non-gated, public) Hugging Face dataset repos over plain HTTP — no login (D-E).

The revision is pinned to a commit sha before anything is listed, so a re-run reads the
exact same files. ``gated``/``private``/``disabled`` repos raise :class:`AccessRefused`
with the URL for the "needs Yohan" list; this adapter never sends a token.

Params: ``repo`` (required), ``revision`` (default ``main``), ``endpoint``
(default ``https://huggingface.co``; tests point it at a local server), ``version``.
Without ``include``: the repo's parquet files if it has any, else its image files.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator

from . import AccessRefused, BaseAdapter, RemoteItem, suffix_of
from ._http import get_json, get_json_pages


class HFAdapter(BaseAdapter):
    name = "hf"

    @property
    def _base(self) -> str:
        return str(self.params.get("endpoint", "https://huggingface.co")).rstrip("/")

    @property
    def _repo(self) -> str:
        repo = str(self.params.get("repo") or "")
        if repo.count("/") != 1:
            raise ValueError("hf adapter: params.repo must be '<owner>/<name>'")
        return repo

    def resolve_version(self) -> str:
        rev = urllib.parse.quote(str(self.params.get("revision", "main")), safe="")
        info = get_json(f"{self._base}/api/datasets/{self._repo}/revision/{rev}")
        assert isinstance(info, dict)
        page = f"{self._base}/datasets/{self._repo}"
        if info.get("gated"):
            raise AccessRefused(page, f"gated ({info['gated']}): accept the terms on the page")
        if info.get("private") or info.get("disabled"):
            raise AccessRefused(page, "private or disabled repo")
        self.sha = str(info["sha"])
        return str(self.params.get("version") or f"rev-{self.sha[:12]}")

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "sha"):
            self.resolve_version()
        url = f"{self._base}/api/datasets/{self._repo}/tree/{self.sha}?recursive=true"
        files = [
            e
            for page in get_json_pages(url)
            for e in page  # type: ignore[union-attr]
            if isinstance(e, dict) and e.get("type") == "file"
        ]
        if not self.params.get("include") and any(
            suffix_of(e["path"]) == ".parquet" for e in files
        ):
            files = [e for e in files if suffix_of(e["path"]) == ".parquet"]
        for e in files:
            lfs = e.get("lfs") or {}
            yield RemoteItem(
                key=e["path"],
                url=f"{self._base}/datasets/{self._repo}/resolve/{self.sha}/"
                f"{urllib.parse.quote(e['path'])}",
                size=int(e.get("size") or 0) or None,
                sha256=lfs.get("oid"),
            )
