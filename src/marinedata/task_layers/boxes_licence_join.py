"""Per-image licence join for the FathomNet competition sets (WP-U6b).

``fathomnet-fgvc23`` / ``fathomnet-fgvc25`` ship FathomNet images; their LICENSE (CC-BY-4.0) covers
the *annotations* only. The image's own licence is the one on the matching FathomNet image record
(``imageLicense`` in the ``fathomnet`` source's per-image JSON), joined on the FathomNet image
uuid, which names the FGVC image (``<uuid>.png``, ``images_zip_content_images_<uuid>``).

Conservative by construction: an FGVC image with no staged FathomNet record, or whose record carries
no recognised licence, is not looked up further; the caller gives its rows the strictest FathomNet
class (``restricted-nd``), never ``open`` by default.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
MATCHED, UNMATCHED = "matched", "unmatched"


def image_uuid(*names: str | None) -> str | None:
    """The first FathomNet image uuid found in a stem, file name or url (lower-case)."""
    for name in names:
        if name and (m := _UUID.search(str(name))):
            return m.group(0).lower()
    return None


@dataclass(frozen=True)
class JoinedLicence:
    status: str  # MATCHED | UNMATCHED
    licence: str | None  # the FathomNet ``imageLicense`` text of a matched image


class FathomnetLicenceJoin:
    """``uuid -> JoinedLicence`` against the staged ``fathomnet`` records.

    ``staged`` is the set of FathomNet uuids that have a staged label record (loaded once, on first
    use); ``fetch_record(uuid)`` returns that record's JSON bytes (``OSError`` when unreadable).
    Lookups are cached; a uuid outside ``staged`` costs no request.
    """

    def __init__(
        self,
        staged: Callable[[], Collection[str]],
        fetch_record: Callable[[str], bytes],
        workers: int = 16,
    ) -> None:
        self._staged_loader, self._fetch, self._workers = staged, fetch_record, workers
        self._staged: frozenset[str] | None = None
        self._cache: dict[str, JoinedLicence] = {}

    def _read(self, uuid: str) -> JoinedLicence:
        try:
            doc = json.loads(self._fetch(uuid))
        except (OSError, ValueError):
            return JoinedLicence(UNMATCHED, None)
        return JoinedLicence(MATCHED, doc.get("imageLicense") if isinstance(doc, dict) else None)

    def lookup(self, uuids: Iterable[str | None]) -> dict[str | None, JoinedLicence]:
        """One :class:`JoinedLicence` per distinct input (``None`` and unknown uuids: unmatched)."""
        uuids = list(uuids)
        if self._staged is None:
            self._staged = frozenset(u.lower() for u in self._staged_loader())
        wanted = {u for u in uuids if u}
        todo = sorted(u for u in wanted - self._cache.keys() if u in self._staged)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            fetched = dict(zip(todo, pool.map(self._read, todo), strict=True))
        self._cache = {**self._cache, **fetched}
        none = JoinedLicence(UNMATCHED, None)
        return {u: self._cache.get(u, none) if u else none for u in set(uuids)}


def match_rate(joined: Iterable[JoinedLicence]) -> dict[str, int]:
    return dict(Counter(j.status for j in joined))
