"""GitHub release assets or raw repo files, anonymously (``adapter: github``).

Release mode: ``params.release`` (a tag, or ``latest``) -> the release's assets; the
version is the tag and each asset's ``digest: sha256:…`` is verified when GitHub gives
one. Raw mode: ``params.ref`` (branch/tag/sha) is pinned to a commit sha first, then
either ``params.raw`` (explicit paths) or the recursive tree is listed. Unauthenticated
API calls are limited to 60/hour — fine for listing, never used for bytes.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterator

from . import BaseAdapter, RemoteItem
from ._http import get_json

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class GitHubAdapter(BaseAdapter):
    name = "github"

    @property
    def _api(self) -> str:
        return str(self.params.get("api", "https://api.github.com")).rstrip("/")

    @property
    def _repo(self) -> str:
        return str(self.params["repo"])

    def resolve_version(self) -> str:
        if self.params.get("release"):
            tag = str(self.params["release"])
            path = (
                "releases/latest"
                if tag == "latest"
                else f"releases/tags/{urllib.parse.quote(tag, safe='')}"
            )
            rel = get_json(f"{self._api}/repos/{self._repo}/{path}")
            assert isinstance(rel, dict)
            self._assets = rel.get("assets") or []
            return str(self.params.get("version") or rel["tag_name"])
        ref = str(self.params.get("ref") or "")
        if not ref:
            raise ValueError("github adapter: set params.release or params.ref")
        if not _SHA40.match(ref):
            commit = get_json(f"{self._api}/repos/{self._repo}/commits/{ref}")
            assert isinstance(commit, dict)
            ref = str(commit["sha"])
        self._sha = ref
        return str(self.params.get("version") or f"git-{ref[:12]}")

    def list_items(self) -> Iterator[RemoteItem]:
        if not hasattr(self, "_assets") and not hasattr(self, "_sha"):
            self.resolve_version()
        if hasattr(self, "_assets"):
            for a in self._assets:
                digest = str(a.get("digest") or "")
                yield RemoteItem(
                    key=str(a["name"]),
                    url=str(a["browser_download_url"]),
                    size=a.get("size"),
                    sha256=digest[7:] if digest.startswith("sha256:") else None,
                )
            return
        raw_base = str(self.params.get("raw_base", "https://raw.githubusercontent.com"))
        paths = self.params.get("raw")
        if paths:
            entries = [{"path": p, "size": None} for p in paths]
        else:
            tree = get_json(f"{self._api}/repos/{self._repo}/git/trees/{self._sha}?recursive=1")
            assert isinstance(tree, dict)
            entries = [t for t in tree.get("tree") or [] if t.get("type") == "blob"]
        for t in entries:
            yield RemoteItem(
                key=str(t["path"]),
                size=t.get("size"),
                url=f"{raw_base.rstrip('/')}/{self._repo}/{self._sha}/"
                f"{urllib.parse.quote(str(t['path']))}",
            )
