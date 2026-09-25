"""Fetch CoralNet's public label table (D-S2): ≤ 2 req/s, cached, resumable, no login.

    python scripts/coralnet_labels_fetch.py fetch <cache> \
        [--priority IDS.txt] [--all] [--import-html DIR]
    python scripts/coralnet_labels_fetch.py build <cache> <out.parquet>

``fetch`` stores ``/label/list/`` once (``<cache>/label_list.html``: id, name, group, short
code, verified / duplicate / calcification flags for every public label), then walks
``/label/<id>/`` for the description and usage stats. Only the parsed fields are kept, one
JSON line per id in ``<cache>/details.jsonl`` (the pages themselves are ~14 KB each; the
disk is tight). A re-run skips ids already in the file. ``--priority`` ids go first;
``--all`` then walks every listed id. ``--import-html`` parses pages another run already
cached as ``<id>.html`` (the WP-7c Reefolution cache) instead of fetching them again.

``build`` joins the list rows to the details into the registry parquet.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marinedata.coralnet_labels import DETAIL_URL, LIST_URL, parse_detail, parse_list

MIN_INTERVAL = 0.5  # seconds between request starts: at most 2 requests per second
USER_AGENT = "marinedata-taxonomy/1.3 (+https://reef.support; public label pages only)"
_last = [0.0]


def _get(url: str) -> tuple[int, str]:
    wait = _last[0] + MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return exc.code, ""
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
        _last[0] = time.monotonic()
    return 0, ""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(cache: Path, priority: Path | None, walk_all: bool, import_html: Path | None) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    listing = cache / "label_list.html"
    if not listing.exists():
        status, body = _get(LIST_URL)
        if status != 200:
            sys.exit(f"label list: HTTP {status}")
        listing.write_text(body)
    ids = [r["label_id"] for r in parse_list(listing.read_text())]
    out = cache / "details.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["label_id"] for line in out.read_text().splitlines() if line}
    order = [int(x) for x in priority.read_text().split()] if priority else []
    if walk_all:
        order += ids
    with out.open("a") as fh:
        if import_html:
            for page in sorted(import_html.glob("*.html")):
                lid = int(page.stem)
                if lid not in done:
                    row = {"label_id": lid, "http": 200, "retrieved_at": "2026-09-25 (WP-7c cache)"}
                    fh.write(json.dumps(row | parse_detail(page.read_text())) + "\n")
                    done.add(lid)
        n = 0
        for lid in dict.fromkeys(order):
            if lid in done:
                continue
            status, body = _get(DETAIL_URL.format(id=lid))
            row = {"label_id": lid, "http": status, "retrieved_at": _now()}
            if status == 200:
                row |= parse_detail(body)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.add(lid)
            n += 1
            if n % 200 == 0:
                print(f"{_now()} fetched {n}, total {len(done)}/{len(ids)}", flush=True)
    print(f"{_now()} done: {len(done)}/{len(ids)} detail rows", flush=True)


def build(cache: Path, dest: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = parse_list((cache / "label_list.html").read_text())
    details = {}
    for line in (cache / "details.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("http") == 200:
            details[d["label_id"]] = d
    for r in rows:
        d = details.get(r["label_id"], {})
        r["description"] = d.get("description")
        r["used_in_sources"] = d.get("used_in_sources")
        r["used_in_annotations"] = d.get("used_in_annotations")
        r["detail_fetched"] = bool(d)
        r["retrieved_at"] = d.get("retrieved_at")
    table = pa.Table.from_pylist(rows)
    meta = {
        "source": LIST_URL + " + " + DETAIL_URL,
        "rows": str(len(rows)),
        "details": str(sum(r["detail_fetched"] for r in rows)),
    }
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in meta.items()})
    pq.write_table(table, dest, compression="zstd")
    print(f"{dest}: {len(rows)} labels, {meta['details']} with description/stats")


if __name__ == "__main__":
    cmd, cache = sys.argv[1], Path(sys.argv[2])
    if cmd == "build":
        build(cache, Path(sys.argv[3]))
    else:
        args = sys.argv[3:]

        def opt(flag: str) -> Path | None:
            return Path(args[args.index(flag) + 1]) if flag in args else None

        fetch(cache, opt("--priority"), "--all" in args, opt("--import-html"))
