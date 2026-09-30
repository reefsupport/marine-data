"""Open (non-gated, public) Hugging Face dataset repos over plain HTTP — no login (D-E).

The revision is pinned to a commit sha before anything is listed, so a re-run reads the
exact same files. ``gated``/``private``/``disabled`` repos raise :class:`AccessRefused`
with the URL for the "needs Yohan" list; this adapter never sends a token.

Params: ``repo`` (required), ``revision`` (default ``main``), ``endpoint``
(default ``https://huggingface.co``; tests point it at a local server), ``version``.
Without ``include``: the repo's parquet files if it has any, else its image files.

WP-6i: streamed files go through :class:`~._range.ResumableStream` (HF's CDN drops a
connection idle ~30 s; the stream resumes with a Range request instead of ending short).
``remote_zip: true`` expands every ``.zip`` into one item per member, read over HTTP
Range from the remote central directory — the archive is never spooled. With
``zip_manifest`` (globs over the listed files, e.g. ``['*.jsonl']``) only the members whose
basename a manifest record names (field ``zip_manifest_field``, default ``image``) are kept.
"""

from __future__ import annotations

import fnmatch
import io
import json
import logging
import threading
import urllib.parse
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from . import SPOOLED, AccessRefused, BaseAdapter, Fetched, RemoteItem, suffix_of
from ._http import HashingReader, get_json, get_json_pages
from ._range import RemoteFile, ResumableStream, open_remote_zip, read_member

log = logging.getLogger(__name__)


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

    def _item(self, e: dict) -> RemoteItem:
        lfs = e.get("lfs") or {}
        return RemoteItem(
            key=e["path"],
            url=f"{self._base}/datasets/{self._repo}/resolve/{self.sha}/"
            f"{urllib.parse.quote(e['path'])}",
            size=int(e.get("size") or 0) or None,
            sha256=lfs.get("oid"),
        )

    def list_items(self) -> Iterator[RemoteItem]:
        # WP-6m: stream the tree page-by-page instead of buffering the whole listing —
        # a 60-100k-file repo paginates the HF tree API dozens of times, and a bounded
        # ``--fetch-only --limit N`` probe must not pay for every page just to answer
        # "give me N". The "prefer parquet if the repo has any" rule (D-E) can only be
        # decided from a full scan in general; we approximate it from the first page
        # only (HF pages are large, and parquet-backed repos put their parquet files at
        # the top of a recursive tree listing in practice) so the common case is exact
        # and the pathological case (parquet appearing only deep in a huge tree) degrades
        # to "include everything", never to a full-tree scan before the first item.
        if not hasattr(self, "sha"):
            self.resolve_version()
        url = f"{self._base}/api/datasets/{self._repo}/tree/{self.sha}?recursive=true"
        pages = get_json_pages(url)
        first_page = next(pages, [])
        first_files = [
            e for e in first_page if isinstance(e, dict) and e.get("type") == "file"
        ]
        prefer_parquet = not self.params.get("include") and any(
            suffix_of(e["path"]) == ".parquet" for e in first_files
        )

        def _files() -> Iterator[dict]:
            yield from first_files
            for page in pages:
                yield from (
                    e for e in page if isinstance(e, dict) and e.get("type") == "file"  # type: ignore[union-attr]
                )

        for e in _files():
            if prefer_parquet and suffix_of(e["path"]) != ".parquet":
                continue
            yield self._item(e)

    # -- WP-6i: resumable streams + remote zip members ----------------------------------
    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        member = getattr(self, "_zip_members", {}).get(item.key)
        if member is not None:
            zf, rf, info = member
            with self._zip_lock:  # one cache window per archive: serialise member reads
                data = read_member(zf, rf, info)
            return Fetched(item, stream=HashingReader(io.BytesIO(data)))
        if item.parts or (suffix_of(item.key) in SPOOLED and not self.is_label(item.key)):
            return super().fetch(item, tmp_dir)
        return Fetched(item, stream=HashingReader(ResumableStream(item.url)))  # type: ignore[arg-type]

    def enumerate(self, *, limit: int | None = None) -> Iterator[RemoteItem]:
        if not self.params.get("remote_zip"):
            yield from super().enumerate(limit=limit)
            return
        # remote_zip needs the manifest names up front to pick zip members, so it
        # always needs the full listing regardless of ``limit``.
        items = list(super().enumerate())
        wanted = self._manifest_names(items)
        for item in items:
            if suffix_of(item.key) == ".zip" and not self.is_label(item.key):
                yield from self._expand_zip(item, wanted)
            else:
                yield item

    def _manifest_names(self, items: list[RemoteItem]) -> set[str] | None:
        globs = list(self.params.get("zip_manifest") or [])
        if not globs:
            return None
        field = str(self.params.get("zip_manifest_field") or "image")
        names: set[str] = set()
        for item in items:
            if not any(fnmatch.fnmatch(item.key, g) for g in globs):
                continue
            stream = ResumableStream(item.url)
            try:
                for line in io.TextIOWrapper(io.BufferedReader(stream), "utf-8"):  # type: ignore[arg-type]
                    rec = json.loads(line) if line.strip() else None
                    ref = rec.get(field) if isinstance(rec, dict) else None
                    if isinstance(ref, str) and ref:
                        names.add(PurePosixPath(ref).name)
            finally:
                stream.close()
        return names

    def _expand_zip(self, item: RemoteItem, wanted: set[str] | None) -> Iterator[RemoteItem]:
        if item.size is None:
            raise ValueError(f"{item.key}: remote_zip needs the declared size (HF tree)")
        try:
            zf, rf = open_remote_zip(item.url, item.size)
        except zipfile.BadZipFile:
            # WP-6n (marineevt): a nested remote_zip tree can ship one malformed/
            # non-standard member (e.g. a genuinely corrupt upstream archive) among many
            # otherwise-good zips -- the declared size and tree metadata both check out
            # (ruled out: a directory, a stale/pointer size), the bytes at this url just
            # never yield a central directory. One bad archive must not crash enumerate()
            # for the whole nested tree: stage it as a normal (unexpanded) item instead,
            # so ingest_missing's existing per-item skip (D-AF, "decode-error:BadZipFile")
            # drops just this one item when it is actually fetched/decoded.
            log.warning(
                "%s: not a valid remote zip (no central directory found), staged whole",
                item.key,
            )
            yield item
            return
        if not hasattr(self, "_zip_members"):
            self._zip_members: dict[str, tuple[zipfile.ZipFile, RemoteFile, zipfile.ZipInfo]] = {}
            self._zip_lock = threading.Lock()
        infos = [
            i
            for i in zf.infolist()
            if not i.is_dir()
            and "__MACOSX" not in i.filename
            and (wanted is None or PurePosixPath(i.filename).name in wanted)
        ]
        found = {PurePosixPath(i.filename).name for i in infos}
        log.info(
            "%s: remote zip %d entries, %d kept%s (%d range requests, %d B read)",
            item.key, len(zf.infolist()), len(infos),
            "" if wanted is None else f", {len(wanted - found)} manifest names not in the zip",
            rf.requests, rf.fetched,
        )  # fmt: skip
        for info in sorted(infos, key=lambda i: i.filename):
            key = f"{item.key}#{info.filename}"
            self._zip_members[key] = (zf, rf, info)
            yield RemoteItem(key=key, url=item.url, size=info.file_size)
