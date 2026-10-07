"""Local filesystem tree, already on disk (``adapter: local_dir``).

For datasets Yohan downloaded by hand (Kaggle/Drive zips with no anonymous API) and
unzipped under a local directory: list every real file under ``params.path`` in
sorted order and hand each one to the *existing* fetch/decode pipeline via a
``file://`` URL — ``open_url``/``download`` already speak ``file://`` through
``urllib.request`` with no adapter-specific code, so images decode/verify exactly
like any other source, and ``include``/``exclude``/``label_patterns`` (inherited
from :class:`BaseAdapter`) decide which non-image files (CSV/metadata/video) are
carried as plain label files instead of being dropped or decoded.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from . import BaseAdapter, RemoteItem


class LocalDirAdapter(BaseAdapter):
    name = "local_dir"

    @property
    def _root(self) -> Path:
        return Path(str(self.params["path"])).expanduser().resolve()

    def resolve_version(self) -> str:
        return str(self.params.get("version") or f"local-dir-{self._root.name}")

    def list_items(self) -> Iterator[RemoteItem]:
        root = self._root
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            # Never follow symlinked directories (os.walk already won't descend into
            # them with followlinks=False; dropping them from dirnames here also keeps
            # them out of any future walk and documents the intent).
            dirnames[:] = sorted(d for d in dirnames if not (base / d).is_symlink())
            for name in sorted(filenames):
                path = base / name
                if path.is_symlink():
                    continue  # never follow symlinks (files either)
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue  # never yield a path outside `path`
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                key = path.relative_to(root).as_posix()
                yield RemoteItem(key=key, url=path.as_uri(), size=size)
