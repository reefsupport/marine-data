"""Fetchers for S3, HuggingFace file trees and GitHub.

Deliberately dependency-free: S3 listing uses the REST XML API and SigV4 is signed by
hand rather than pulling in boto3. A verification tool that needs a 50 MB dependency to
check whether a directory exists is a tool people will skip.

What each unlocks:

- **S3 (anonymous)** — public buckets such as the MERMAID open-data mirror.
- **S3 (credentialed)** — our own Hetzner buckets, so our own layouts get verified
  rather than assumed. Those are the layouts that matter most and the ones nobody else
  can check for us.
- **HuggingFace files** — repos published as a file tree rather than a parquet dataset.
- **GitHub** — repos shipping small sample data in-tree.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .checksums import CHECKSUM_FILE, file_digest, parse_checksums
from .fetch import FetchError, FetchNotSupported, FetchResult, _get, _write
from .models import Source

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
FETCHABLE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".csv", ".json", ".ndjson", ".wav", ".txt"}
)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sigv4_headers(
    method: str, host: str, path: str, query: str, region: str, access: str, secret: str
) -> dict[str, str]:
    """Minimal AWS SigV4 for GET requests with an empty payload."""
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical = f"{method}\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    scope = f"{date_stamp}/{region}/s3/aws4_request"
    to_sign = (
        f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    )

    signing_key = _sign(
        _sign(_sign(_sign(f"AWS4{secret}".encode(), date_stamp), region), "s3"), "aws4_request"
    )
    signature = hmac.new(signing_key, to_sign.encode(), hashlib.sha256).hexdigest()

    return {
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }


def _stratify(keys: list[tuple[str, int]], prefix: str, limit: int) -> list[tuple[str, int]]:
    """Sample evenly across the first path segment below the prefix.

    Taking the first N keys alphabetically silently breaks paired-directory datasets:
    ``images/`` sorts before ``masks/``, so a naive sample contains only images and the
    loader then reports a missing masks directory that in fact exists. Stratifying by
    sub-directory keeps the sample representative of the actual structure.

    Filenames are also aligned across groups where possible, so image/mask pairs survive
    the sample rather than being two disjoint halves.
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for key, size in keys:
        relative = key[len(prefix) :].lstrip("/")
        head = relative.split("/", 1)[0] if "/" in relative else ""
        groups.setdefault(head, []).append((key, size))

    if len(groups) <= 1:
        return keys[:limit]

    # Prefer stems present in every group — that is what makes a pair a pair.
    stems = [
        {Path(k).stem.removesuffix("_mask") for k, _ in members} for members in groups.values()
    ]
    shared = sorted(set.intersection(*stems)) if stems else []

    per_group = max(1, limit // len(groups))
    out: list[tuple[str, int]] = []
    for members in groups.values():
        if shared:
            picked = [kv for kv in members if Path(kv[0]).stem.removesuffix("_mask") in shared]
        else:
            picked = members
        out.extend(picked[:per_group])
    return out[:limit]


def _s3_endpoint(source: Source) -> tuple[str, str, str]:
    """Return (bucket, host, region) for a source's S3 access."""
    params = source.access.params
    uri = source.access.uri or ""
    bucket = str(params.get("bucket") or urllib.parse.urlparse(uri).netloc or "")
    if not bucket:
        raise FetchError(f"{source.id}: cannot determine the S3 bucket from '{uri}'")
    endpoint = str(params.get("endpoint", "s3.amazonaws.com"))
    region = str(params.get("region", "us-east-1"))
    return bucket, f"{bucket}.{endpoint}", region


MANIFEST_FETCH_WORKERS = 16
"""Concurrent GETs for a pinned staged tree. One-at-a-time was latency-bound at ~110
files/min against ``rs-storage-open`` — ~17 h for the v1 release's ~115k files.
``MARINEDATA_FETCH_WORKERS`` overrides."""
MANIFEST_FETCH_RETRIES = 8
"""Per-file attempts (backoff 1.5, 3, 6, 12, 24, 30, 30 s) — at ~10^5 GETs a 3-attempt
budget turns an ordinary burst of 5xx/429/timeouts into a failed release."""
MANIFEST_FETCH_TIMEOUT = 30
"""Per-attempt socket timeout. ``rs-storage-open`` normally answers in ~0.2 s but
intermittently stalls one key for tens of seconds (observed 17-21 s TTFB on IPv4 and
IPv6 alike, and five straight 60 s read timeouts on one v1 image during S44): a stalled
GET is abandoned and retried sooner, and the retries span ~6 min rather than ~5."""
MANIFEST_FETCH_MAX_FAILURES = 50
"""Files that may fail outright before the rest of the fetch is cancelled."""


def _manifest_workers() -> int:
    raw = os.environ.get("MARINEDATA_FETCH_WORKERS", "").strip()
    try:
        return max(1, int(raw)) if raw else MANIFEST_FETCH_WORKERS
    except ValueError:
        return MANIFEST_FETCH_WORKERS


def _is_pinned_staged_tree(source: Source) -> bool:
    """Whether ``source`` can be fetched by digest-pinned manifest instead of a listing.

    Anonymous ``GetObject`` is allowed on ``rs-storage-open`` but anonymous
    ``ListBucket`` is not (7i) — every fetch that starts by listing the prefix 403s
    there. A pinned staged tree does not need to list: ``checksums.root_digest`` names
    the exact ``CHECKSUMS.sha256`` key to fetch, and that manifest names every other
    file by key too, so the whole fetch is GETs by known key. Sources without a pin, or
    without the ``staged-tree`` layout the manifest format is written for, keep the
    listing path unchanged.
    """
    checksums = source.checksums
    loader = source.loader
    return (
        checksums is not None
        and bool(checksums.root_digest)
        and loader is not None
        and loader.layout == "staged-tree"
    )


def staged_root_params(source: Source) -> dict[str, str] | None:
    """The ``access.params.staged_*`` keys (bucket, endpoint, region, prefix) of a source whose
    release root is a staged tree that is NOT its fetch method, else ``None``.

    ``atlantis-synthetic-depth`` must stay ``method: api`` / ``client: atlantis`` (its upstream
    is a Kaggle download, and ``tests/test_fetch_api_dispatch.py`` guards that it is never
    routed to another client), yet a release build roots it from the staged ``atlantis`` tree.
    Only a pinned staged tree (``checksums.root_digest`` + ``layout: staged-tree``) with a
    bucket and prefix qualifies.
    """
    params = source.access.params
    staged = {
        key: str(params[f"staged_{key}"])
        for key in ("bucket", "endpoint", "region", "prefix")
        if params.get(f"staged_{key}")
    }
    if "bucket" not in staged or "prefix" not in staged:
        return None
    return staged if _is_pinned_staged_tree(source) else None


def fetch_staged_root(source: Source, root: Path, limit: int) -> FetchResult:
    """Fetch the staged tree named by ``access.params.staged_*`` by its pinned manifest."""
    staged = staged_root_params(source)
    if staged is None:
        raise FetchNotSupported(
            f"{source.id}: no pinned staged tree declared in access.params.staged_*"
        )
    host = f"{staged['bucket']}.{staged.get('endpoint', 's3.amazonaws.com')}"
    return _fetch_s3_manifest(source, root, limit, host, staged["prefix"])


def _fetch_s3_manifest(
    source: Source, root: Path, limit: int, host: str, prefix: str
) -> FetchResult:
    """Fetch a pinned staged tree by its ``CHECKSUMS.sha256`` manifest, never listing.

    1. GET ``<prefix>CHECKSUMS.sha256`` by its known key and check its own sha256
       against ``checksums.root_digest`` — a stale or tampered manifest is caught
       before a single file it names is trusted.
    2. Parse it with the repo's own :func:`marinedata.checksums.parse_checksums` (the
       one place this ``<sha256>  <relpath>`` format is read) and fetch
       ``metadata.parquet`` plus the first ``limit`` remaining files (the images) in
       manifest order — sorted-path order, the same every run, so a ``--limit`` sample
       is reproducible rather than whatever a listing's pagination happened to hand
       back. A file under ``labels/`` is fetched only if its filename stem matches one
       of the sampled images (``images/default/<stem>.jpg`` <-> ``labels/masks/<stem>.png``);
       a label file whose stem matches no image at all — a shared manifest like
       reefolution's small ``labels/points.parquet`` — is treated like ``metadata.parquet``
       and always fetched in full. Per-image mask sources like coralscop-masks-rs have
       ~39k label files for a handful of sampled images, and fetching every one in full
       made ``verify --limit N`` take an hour instead of seconds.
    3. Verify each downloaded file's sha256 against the manifest before writing it, so
       a partial or corrupted transfer raises naming the exact file rather than
       silently staging bad bytes. Files are fetched :data:`MANIFEST_FETCH_WORKERS` at a
       time; a file already on disk with the manifest's sha256 is not fetched again, so
       an interrupted release fetch resumes instead of restarting.
    4. Check the result on disk: a full fetch (``limit`` covering every image) must
       leave every manifest file present, a sample every file it chose — otherwise
       raise with the missing count. A partial tree is never returned as if complete
       (the staged-tree loader walks ``images/`` on disk, so missing files would silently
       shrink the release rather than fail it).
    """
    manifest_key = f"{prefix}{CHECKSUM_FILE}"
    manifest_bytes = _get(f"https://{host}/{urllib.parse.quote(manifest_key)}")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    expected_digest = source.checksums.root_digest  # type: ignore[union-attr]
    if manifest_digest != expected_digest:
        raise FetchError(
            f"{source.id}: {CHECKSUM_FILE} digest {manifest_digest} does not match "
            f"checksums.root_digest {expected_digest} — the pinned manifest is stale "
            f"or has been tampered with"
        )

    digests = parse_checksums(manifest_bytes.decode("utf-8"))
    metadata = [rel for rel in digests if rel == "metadata.parquet"]
    rest = [rel for rel in digests if rel != "metadata.parquet" and not rel.startswith("labels/")]
    image_stems = {Path(rel).stem for rel in rest}

    labels_by_stem: dict[str, list[str]] = {}
    shared_labels: list[str] = []
    for rel in digests:
        if not rel.startswith("labels/"):
            continue
        stem = Path(rel).stem
        if stem in image_stems:
            labels_by_stem.setdefault(stem, []).append(rel)
        else:
            shared_labels.append(rel)

    sampled = rest[:limit]
    sampled_labels = [rel for path in sampled for rel in labels_by_stem.get(Path(path).stem, [])]
    # De-duplicated (two partitions can share a stem) so no two workers write one file.
    wanted = list(dict.fromkeys(metadata + shared_labels + sampled_labels + sampled))
    truncated = len(rest) > limit

    def fetch_one(relative: str) -> None:
        dest = root / relative
        if dest.is_file() and file_digest(dest) == digests[relative]:
            return  # already staged with the right bytes (an earlier, interrupted run)
        key = f"{prefix}{relative}"
        payload = _get(
            f"https://{host}/{urllib.parse.quote(key)}",
            timeout=MANIFEST_FETCH_TIMEOUT,
            retries=MANIFEST_FETCH_RETRIES,
        )
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != digests[relative]:
            raise FetchError(
                f"{source.id}: {relative} downloaded with sha256 {actual_digest}, the "
                f"manifest expects {digests[relative]} — the transfer is corrupt or "
                f"the tree changed under a pinned version"
            )
        _write(dest, payload)

    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=_manifest_workers()) as pool:
        futures = {pool.submit(fetch_one, relative): relative for relative in wanted}
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                failures[futures[future]] = exc
                if len(failures) >= MANIFEST_FETCH_MAX_FAILURES:
                    for pending in futures:
                        pending.cancel()  # the endpoint is down, not flaky — stop early
                    break

    # Completeness is checked on disk, not inferred from "no exception": a full fetch
    # must leave every file the manifest names in place, a sample every file it chose.
    required = list(digests) if not truncated else wanted
    missing = sorted(rel for rel in required if not (root / rel).is_file())
    if failures or missing:
        first = sorted(failures)[0] if failures else missing[0]
        detail = f"{first}: {failures[first]}" if failures else first
        raise FetchError(
            f"{source.id}: {'full' if not truncated else 'sample'} fetch incomplete — "
            f"{len(missing)} of {len(required)} file(s) missing under {root} "
            f"({len(failures)} failed to download); first: {detail}"
        )

    return FetchResult(source.id, root, len(wanted), "s3-manifest", truncated=truncated)


def fetch_s3(source: Source, root: Path, limit: int) -> FetchResult:
    """List a bucket prefix and download up to ``limit`` objects.

    Params:
        bucket, endpoint, region, prefix
        credentials_env: prefix for env vars, e.g. ``S3`` → ``S3_ACCESS_KEY_ID``.
            Omit for anonymous buckets.

    A source with ``checksums.root_digest`` and ``loader.layout: staged-tree`` — a
    pinned tree this repo's own writer produced — is instead fetched by
    :func:`_fetch_s3_manifest`, which never lists (see :func:`_is_pinned_staged_tree`).
    Credentialed sources (``credentials_env`` set) always keep this listing path: they
    can list, so there is no need to trade the loop-detects-drift benefit of a listing
    for a manifest fetch.
    """
    bucket, host, region = _s3_endpoint(source)
    prefix = str(source.access.params.get("prefix", ""))
    env_prefix = source.access.params.get("credentials_env")

    if env_prefix is None and _is_pinned_staged_tree(source):
        return _fetch_s3_manifest(source, root, limit, host, prefix)

    access_key = secret_key = None
    if env_prefix:
        access_key = os.environ.get(f"{env_prefix}_ACCESS_KEY_ID")
        secret_key = os.environ.get(f"{env_prefix}_SECRET_ACCESS_KEY")
        if not (access_key and secret_key):
            raise FetchNotSupported(
                f"{source.id}: needs credentials in {env_prefix}_ACCESS_KEY_ID / "
                f"{env_prefix}_SECRET_ACCESS_KEY. This is a private source — set them "
                f"or fetch manually."
            )

    # SigV4 requires the canonical query string sorted by parameter name and
    # RFC3986-encoded. urlencode's default quote_plus would also break the signature.
    def _list(list_prefix: str, max_keys: int, delimiter: str = "") -> ET.Element:
        params = {"list-type": "2", "prefix": list_prefix, "max-keys": str(max_keys)}
        if delimiter:
            params["delimiter"] = delimiter
        query = urllib.parse.urlencode(
            sorted(params.items()), quote_via=urllib.parse.quote, safe=""
        )
        headers = (
            _sigv4_headers("GET", host, "/", query, region, access_key, secret_key)
            if access_key
            else {}
        )
        try:
            return ET.fromstring(_get(f"https://{host}/?{query}", headers=headers))
        except ET.ParseError as exc:
            raise FetchError(f"{source.id}: unparseable S3 listing from {host}") from exc

    def _contents(tree: ET.Element) -> list[tuple[str, int]]:
        return [
            (el.findtext("s3:Key", "", S3_NS), int(el.findtext("s3:Size", "0", S3_NS)))
            for el in tree.findall("s3:Contents", S3_NS)
        ]

    def _usable(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return [
            (key, size)
            for key, size in items
            if Path(key).suffix.lower() in FETCHABLE_SUFFIXES and 0 < size < 25_000_000
        ]

    # Discover immediate sub-directories first. Listing flat and taking the first N keys
    # silently breaks paired layouts: `images/` sorts before `masks/`, and with hundreds
    # of images the mask directory never appears in the first page at all — so the
    # loader reports a missing directory that actually exists.
    top = _list(prefix, 1000, delimiter="/")
    subdirs = [
        el.findtext("s3:Prefix", "", S3_NS) for el in top.findall("s3:CommonPrefixes", S3_NS)
    ]

    if subdirs:
        per_dir = max(1, limit // len(subdirs))
        # Top-level files sit alongside the sub-directories and are often the manifest
        # the loader needs (an export ndjson, a labels CSV). Dropping them because they
        # are not in a sub-directory breaks the layout for no good reason.
        gathered = list(_usable(_contents(top)))
        for sub in subdirs:
            gathered.extend(_usable(_contents(_list(sub, per_dir * 3))))
        wanted = _stratify(gathered, prefix, limit)
    else:
        keys = _usable(_contents(top))
        wanted = keys[:limit]

    if not wanted:
        raise FetchError(
            f"{source.id}: no fetchable objects under s3://{bucket}/{prefix} "
            f"({len(subdirs)} sub-directories listed)."
        )

    for key, _ in wanted:
        relative = key[len(prefix) :].lstrip("/") or Path(key).name
        obj_path = "/" + urllib.parse.quote(key)
        obj_headers = (
            _sigv4_headers("GET", host, obj_path, "", region, access_key, secret_key)
            if access_key
            else {}
        )
        _write(root / relative, _get(f"https://{host}{obj_path}", headers=obj_headers))

    return FetchResult(source.id, root, len(wanted), "s3", truncated=True)


def fetch_hf_files(source: Source, root: Path, limit: int) -> FetchResult:
    """Download files from a HuggingFace repo published as a file tree.

    Listed recursively and written preserving each file's path relative to
    ``subpath`` — not flattened into one ``images/`` directory — so a repo using
    directory structure to carry the label (an ``imagefolder``-built dataset whose
    class subdirectories, for whatever reason, never made it into the datasets-server
    schema as a queryable column) still produces a real ``image-folder`` layout on
    disk. NOAA's PIFSC bleaching dataset is exactly this: its rows API exposes only an
    ``image`` column — no label at all, checked 2026-08-28 — while the repo's own file
    tree has the class right there as ``train/CORAL/`` vs ``train/CORAL_BL/``.

    Params:
        hf_id, repo_type (``datasets`` default), subpath, revision (``main`` default)
    """
    params = source.access.params
    hf_id = str(params.get("hf_id") or "")
    if not hf_id:
        raise FetchError(f"{source.id}: `hf_id` is required for the hf-files fetcher")

    repo_type = str(params.get("repo_type", "datasets"))
    revision = str(params.get("revision", "main"))
    subpath = str(params.get("subpath", "")).strip("/")

    api = f"https://huggingface.co/api/{repo_type}/{hf_id}/tree/{revision}"
    if subpath:
        api = f"{api}/{urllib.parse.quote(subpath)}"
    api += "?recursive=true"

    import json

    entries = json.loads(_get(api))
    files = [
        e
        for e in entries
        if e.get("type") == "file"
        and Path(e.get("path", "")).suffix.lower() in FETCHABLE_SUFFIXES
        and 0 < int(e.get("size", 0)) < 25_000_000
    ][:limit]

    if not files:
        kinds = {e.get("type") for e in entries}
        raise FetchError(
            f"{source.id}: no small fetchable files under {hf_id}/{subpath or '.'} "
            f"(entry types: {sorted(kinds)}). Point `subpath` at a directory of images."
        )

    for entry in files:
        path = entry["path"]
        url = (
            f"https://huggingface.co/{repo_type}/{hf_id}/resolve/{revision}/"
            f"{urllib.parse.quote(path)}"
        )
        relative = Path(path).relative_to(subpath) if subpath else Path(path)
        _write(root / relative, _get(url))

    return FetchResult(source.id, root, len(files), "hf-files", truncated=True)


FATHOMNET_API = "https://database.fathomnet.org/api"
"""fathomnet.org itself is now a Wix marketing site (301s away from any API path) —
the real host, confirmed live and matching the official fathomnet-py client
(github.com/fathomnet/fathomnet-py, src/fathomnet/api/images.py), is database.fathomnet.org."""


def fetch_fathomnet(source: Source, root: Path, limit: int) -> FetchResult:
    """Sample images with bounding-box annotations from the FathomNet REST API.

    No auth needed for reads: ``GET /api/images/list/all`` is the same paginated bulk
    endpoint the official client calls. Writes a COCO-style ``images/`` +
    ``annotations.json`` — CocoJsonLoader reads it directly, so this fetcher exists to
    normalise the source into that convention, same as every other fetcher here.

    Only images that carry at least one bounding box are kept — FathomNet hosts plenty
    of unannotated imagery too, which is out of scope for a source declared as
    detection data.

    Params: ``page`` (default 0), useful for sampling past the same images twice.
    """
    import json as _json

    page = int(source.access.params.get("page", 0))
    query = urllib.parse.urlencode({"page": page, "size": limit})
    payload = _json.loads(_get(f"{FATHOMNET_API}/images/list/all?{query}"))
    entries = payload.get("content", [])
    if not entries:
        raise FetchError(f"{source.id}: FathomNet returned no images for page={page}")

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    category_ids: dict[str, int] = {}
    written = 0

    for index, entry in enumerate(entries):
        url = entry.get("url")
        boxes = entry.get("boundingBoxes") or []
        if not url or not boxes:
            continue

        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
        filename = f"{index:05d}{suffix}"
        _write(root / "images" / filename, _get(url))
        coco_images.append(
            {
                "id": index,
                "file_name": filename,
                "width": entry.get("width"),
                "height": entry.get("height"),
            }
        )
        for box in boxes:
            concept = box.get("concept")
            if not concept:
                continue
            category_id = category_ids.setdefault(concept, len(category_ids) + 1)
            coco_annotations.append(
                {
                    "id": len(coco_annotations),
                    "image_id": index,
                    "category_id": category_id,
                    "bbox": [
                        box.get("x", 0),
                        box.get("y", 0),
                        box.get("width", 0),
                        box.get("height", 0),
                    ],
                }
            )
        written += 1

    if written == 0:
        raise FetchError(
            f"{source.id}: none of the {len(entries)} images on page={page} carried a "
            f"bounding box — try a different page"
        )

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": cid, "name": name} for name, cid in category_ids.items()],
    }
    _write(root / "annotations.json", _json.dumps(coco).encode())
    return FetchResult(source.id, root, written, "fathomnet-api", truncated=True)


def _parse_pangaea_tab(text: str) -> list[dict[str, str]]:
    """PANGAEA ``.tab`` exports carry a metadata preamble (citation, parameters,
    licence) ending in a bare ``*/`` line, then a normal tab-delimited table. Nothing
    in the standard library reads this directly, and it is common enough across
    PANGAEA-hosted marine monitoring archives to be worth a small, reusable parser
    rather than one-off string slicing in the fetcher itself.
    """
    import csv

    lines = text.splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.strip() == "*/") + 1
    except StopIteration:
        raise FetchError(
            "not a recognised PANGAEA .tab file (no '*/' preamble terminator)"
        ) from None
    return list(csv.DictReader(lines[header_index:], delimiter="\t"))


def fetch_pangaea_manifest(source: Source, root: Path, limit: int) -> FetchResult:
    """Fetch a PANGAEA dataset-collection manifest and download the images its
    ``*_links-to-photos.tab`` tables individually address.

    PANGAEA (and older marine-monitoring archives generally) often publish field-photo
    metadata as tab-delimited tables carrying a URL per image rather than embedding the
    bytes — the ``?format=zip`` export bundling every child dataset in a collection is
    itself tiny (a few MB) even when it describes tens of thousands of photos. Built
    for Heron Reef's 17-year GBR time series; written generically enough (configurable
    column names) to serve another PANGAEA photo-link manifest without new code.

    Params:
        sample_url: the manifest zip (a PANGAEA collection's ``?format=zip`` export).
        url_column: default ``"URL thumb"`` — PANGAEA publishes a full-resolution
            ``"URL image"`` and a small ``"URL thumb"`` per photo; thumbnails keep a
            verification sample a verification sample.
        filename_column: default ``"File name"``.
    """
    import io
    import zipfile

    params = source.access.params
    manifest_url = str(params.get("sample_url") or "")
    if not manifest_url:
        raise FetchError(f"{source.id}: no sample_url declared for the manifest")
    url_column = str(params.get("url_column", "URL thumb"))
    filename_column = str(params.get("filename_column", "File name"))

    payload = _get(manifest_url)
    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        tab_files = sorted(n for n in zf.namelist() if n.endswith("links-to-photos.tab"))
        if not tab_files:
            raise FetchError(
                f"{source.id}: manifest at {manifest_url} has no *links-to-photos.tab files"
            )

        for tab_name in tab_files:
            if written >= limit:
                break
            rows = _parse_pangaea_tab(zf.read(tab_name).decode("utf-8", errors="replace"))
            for row in rows:
                if written >= limit:
                    break
                image_url = row.get(url_column)
                filename = row.get(filename_column)
                if not image_url or not filename:
                    continue
                _write(root / "images" / filename, _get(image_url))
                written += 1

    if written == 0:
        raise FetchError(f"{source.id}: no images referenced by the manifest at {manifest_url}")
    return FetchResult(source.id, root, written, "pangaea-manifest", truncated=True)
