#!/usr/bin/env python3
"""Wakes a PANGAEA series' tape-archived files by touching each with a 1-byte GET (WP-6o-b).

PANGAEA's ``hs.pangaea.de``/``download.pangaea.de`` fronts store older campaigns on tape;
a cold file answers 503 until something requests it, after which it stages onto disk (the
delay this trigger exists for — see ``ingest_deferred.py`` for the ingest-time side, which
*defers* those 503s instead of failing the run). This script lists a series' files the same
way ``adapter: pangaea``/``pangaea-series`` does (no new listing code — it loads the spec
and calls the real adapter) and issues a ``Range: bytes=0-0`` GET per file, never a full
download, to trigger the tape recall without moving bytes.

Progress is a flat tsv (``url\\tstatus\\tts``) appended to as it goes, so a restart skips
any url whose *last* recorded status is already 200 or 206. Non-ready urls are retried in
a loop, sleeping ``--rate``-limited between requests and 300s between loops, until every
url is ready or ``--max-hours`` has elapsed.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from marinedata.adapters import make_adapter  # noqa: E402
from marinedata.ingest_source import IngestSpec  # noqa: E402

USER_AGENT = "marinedata-recall/1.0"
SLEEP_BETWEEN_LOOPS_S = 300.0
READY_STATUSES = ("200", "206")
# WP-6o-b: this trigger is a throwaway operational tool (one series, one session), so its
# tsv/log default lives in the session scratchpad, not the repo — override with --out-dir
# for any other run.
DEFAULT_OUT_DIR = Path(
    "/private/tmp/claude-501/-Users-yohanrunhaar-dev-reefsupport/"
    "0ca12ad3-aada-4ede-ab99-14fec1fa7cc2/scratchpad/pangaea-recall"
)


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class UrllibSession:
    """Minimal ``requests.Session``-shaped ``.get`` backed by urllib (no new dependency)."""

    def get(self, url: str, headers: dict[str, str], timeout: float) -> _Response:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _Response(resp.status)
        except urllib.error.HTTPError as exc:
            return _Response(exc.code)
        except urllib.error.URLError:
            return _Response(0)


def series_urls(series_id: str, specs_dir: Path | None = None) -> list[str]:
    """The series' file URLs, listed exactly the way the pangaea adapter lists them."""
    specs_dir = specs_dir or ROOT / "registry" / "ingest-specs"
    spec = IngestSpec.load(specs_dir / f"{series_id}.yaml")
    adapter = make_adapter(spec.adapter, spec.params)
    return [item.url for item in adapter.list_items()]


def _ready_urls(tsv_path: Path) -> set[str]:
    """Urls whose *last* recorded status in the tsv is 200/206 (restart skip, D-AJ-style)."""
    if not tsv_path.exists():
        return set()
    last: dict[str, str] = {}
    for line in tsv_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        url, status, _ts = parts
        last[url] = status
    return {url for url, status in last.items() if status in READY_STATUSES}


def _append_row(tsv_path: Path, url: str, status: str, ts: float) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("a") as f:
        f.write(f"{url}\t{status}\t{ts:.0f}\n")


def recall(
    urls: list[str],
    tsv_path: Path,
    *,
    max_hours: float,
    rate: float,
    session: UrllibSession | None = None,
    clock=time.monotonic,
    wall_clock=time.time,
    sleep=time.sleep,
) -> int:
    """Loop over the non-ready urls until all are ready or ``max_hours`` elapses.

    Returns the number of loops run. Rate-limited to ``rate`` requests/s within a loop;
    sleeps :data:`SLEEP_BETWEEN_LOOPS_S` between loops when urls remain.
    """
    session = session or UrllibSession()
    interval = 1.0 / rate if rate > 0 else 0.0
    deadline = clock() + max_hours * 3600.0
    pending = [u for u in urls if u not in _ready_urls(tsv_path)]
    loops = 0
    while pending and clock() < deadline:
        loops += 1
        still_pending: list[str] = []
        for i, url in enumerate(pending):
            if i > 0 and interval:
                sleep(interval)
            resp = session.get(
                url,
                headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT},
                timeout=30,
            )
            status = str(resp.status_code)
            _append_row(tsv_path, url, status, wall_clock())
            if status not in READY_STATUSES:
                still_pending.append(url)
        pending = still_pending
        if pending and clock() < deadline:
            sleep(SLEEP_BETWEEN_LOOPS_S)
    return loops


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", required=True, help="ingest-spec id, e.g. pangaea-ps96-ofos-weddell"
    )
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--rate", type=float, default=1.0, help="max requests/s")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    urls = series_urls(args.series)
    tsv_path = args.out_dir / f"{args.series}.tsv"
    print(f"{args.series}: {len(urls)} urls, tsv={tsv_path}", flush=True)
    loops = recall(urls, tsv_path, max_hours=args.max_hours, rate=args.rate)
    ready = len(_ready_urls(tsv_path))
    print(f"{args.series}: {loops} loop(s), {ready}/{len(urls)} ready", flush=True)
    return 0 if ready == len(urls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
