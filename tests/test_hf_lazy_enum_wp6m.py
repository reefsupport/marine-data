"""WP-6m: the HF adapter must not touch a full 60-100k file tree to answer a small
``--limit`` — ``ingest-source hf koi-rgb-sonar.yaml --fetch-only --limit 2`` and
``sonarsweep.yaml`` hung >120s enumerating their (99,986 / 63,982-item) repo trees
because ``list_items()`` built one big list over every paginated page before
``BaseAdapter.enumerate()`` sorted it and *then* the caller sliced to ``limit``.
"""

from __future__ import annotations

from collections.abc import Iterator

import marinedata.adapters.hf as hf_mod
from marinedata.adapters.hf import HFAdapter

PAGE_SIZE = 1000
N_PAGES = 100  # 100,000 entries total — mirrors the real hung repos' scale


def _fake_tree_pages(touched: list[int]) -> Iterator[list[dict]]:
    """A fake paginated HF tree: 100 pages of 1,000 file entries each. ``touched``
    counts every entry actually constructed/iterated, standing in for a page fetch
    the real adapter would otherwise make an HTTP round trip for."""
    for p in range(N_PAGES):
        page = []
        for i in range(PAGE_SIZE):
            touched[0] += 1
            page.append({"type": "file", "path": f"dir{p}/f{i}.jpg", "size": 10})
        yield page


def _adapter(monkeypatch, touched: list[int]) -> HFAdapter:
    monkeypatch.setattr(hf_mod, "get_json_pages", lambda url: _fake_tree_pages(touched))
    adapter = HFAdapter({"repo": "owner/name"})
    adapter.sha = "deadbeef"  # skip resolve_version()'s HTTP call entirely
    return adapter


def test_enumerate_with_limit_touches_a_bounded_number_of_entries(monkeypatch) -> None:
    touched = [0]
    adapter = _adapter(monkeypatch, touched)

    items = list(adapter.enumerate(limit=2))

    assert len(items) == 2
    # bounded by (at most) one tree page, nowhere near the 100,000-entry tree
    assert touched[0] <= PAGE_SIZE
    assert touched[0] < N_PAGES * PAGE_SIZE


def test_list_items_is_itself_lazy(monkeypatch) -> None:
    touched = [0]
    adapter = _adapter(monkeypatch, touched)

    it = adapter.list_items()
    next(it)
    next(it)

    assert touched[0] <= PAGE_SIZE


def test_enumerate_without_limit_is_unchanged_full_sorted_listing(monkeypatch) -> None:
    """No behaviour change for real (unbounded) ingestion: still sorted, still full."""
    touched = [0]
    adapter = _adapter(monkeypatch, touched)

    items = list(adapter.enumerate())

    assert len(items) == N_PAGES * PAGE_SIZE
    assert [i.key for i in items] == sorted(i.key for i in items)
