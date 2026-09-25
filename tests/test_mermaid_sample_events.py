"""WP-2d: the MERMAID `summarysampleevents` pager, against a fixture page — no network.

The endpoint has no image id (see docs/geo-provenance.md), so this doesn't feed
mermaid-aws's own geo backfill yet; it's tested on its own as a ready join table.
"""

from __future__ import annotations

from marinedata import mermaid_sample_events as mse

PAGE_1 = {
    "count": 3,
    "next": "https://api.datamermaid.org/v1/summarysampleevents/?limit=2&page=2",
    "previous": None,
    "results": [
        {
            "sample_event_id": "07b7b990-d707-4910-b3bc-23228da1e65b",
            "latitude": -17.97855,
            "longitude": 179.2251,
            "site_id": "9b2238bd-9ff2-4a04-be1d-44892006097c",
            "site_name": "AA",
            "project_id": "9bf0538e-99c7-405b-a54b-a1568c8a757e",
        },
        {
            # a second sample event at the SAME site — site_table keeps the first
            "sample_event_id": "aaaaaaaa-0000-0000-0000-000000000000",
            "latitude": -17.9,
            "longitude": 179.2,
            "site_id": "9b2238bd-9ff2-4a04-be1d-44892006097c",
            "site_name": "AA",
            "project_id": "9bf0538e-99c7-405b-a54b-a1568c8a757e",
        },
    ],
}
PAGE_2 = {
    "count": 3,
    "next": None,
    "previous": PAGE_1["next"],
    "results": [
        {
            # no position published for this sample event -> dropped, never a fake 0,0
            "sample_event_id": "bbbbbbbb-0000-0000-0000-000000000000",
            "latitude": None,
            "longitude": None,
            "site_id": "cccccccc-0000-0000-0000-000000000000",
            "site_name": "no-position",
            "project_id": "9bf0538e-99c7-405b-a54b-a1568c8a757e",
        }
    ],
}
FIXTURE_PAGES = {mse.API_BASE: PAGE_1, PAGE_1["next"]: PAGE_2}


def test_parse_page_skips_rows_with_no_position():
    assert len(mse.parse_page(PAGE_1)) == 2
    assert mse.parse_page(PAGE_2) == []


def test_page_through_follows_next_and_stops(tmp_path):
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        return FIXTURE_PAGES[url]

    slept: list[float] = []
    records = mse.page_through(
        fetch,
        cache_dir=tmp_path,
        rate_limit_hz=2.0,
        sleep=slept.append,
    )

    assert calls == [mse.API_BASE, PAGE_1["next"]]
    assert len(records) == 2  # PAGE_2's positionless row is dropped
    assert slept == [0.5]  # rate-limited once, between the two page fetches, never on the first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["page-00000.json", "page-00001.json"]


def test_page_through_respects_max_pages(tmp_path):
    def fetch(url: str):
        return FIXTURE_PAGES[url]

    records = mse.page_through(fetch, cache_dir=tmp_path, max_pages=1, sleep=lambda _: None)
    assert len(records) == 2
    assert next(iter(tmp_path.iterdir())).name == "page-00000.json"


def test_site_table_dedups_by_site_first_wins():
    records = mse.parse_page(PAGE_1)
    table = mse.site_table(records)
    assert table == {"9b2238bd-9ff2-4a04-be1d-44892006097c": {"lat": -17.97855, "lon": 179.2251}}
