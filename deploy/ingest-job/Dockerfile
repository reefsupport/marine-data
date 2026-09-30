# syntax=docker/dockerfile:1
#
# Server-side ingest Job image (WP-6c): runs `marinedata ingest-batch` next to the bucket
# (Hetzner hel1) instead of over the Mac's ~10 MB/s uplink. See ../../docs/ingest-howto.md
# for what a run actually does; ../README.md for build/push/apply.
#
# Base image is pinned by digest (multi-arch manifest list), resolved 2026-09-25 from the
# public Docker Hub API with an anonymous pull token — no login required:
#   curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull"
#   curl -s -H "Authorization: Bearer <token>" \
#     -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
#     "https://registry-1.docker.io/v2/library/python/manifests/3.12-slim" -D - -o /dev/null \
#     | grep -i docker-content-digest
# Re-run that (or `docker pull python:3.12-slim` + `docker inspect --format '{{index .RepoDigests 0}}'`)
# to re-pin when the base image needs a security update.
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

WORKDIR /app

# This worktree (base feat/wp6-ingest @ 8150c8f, checked 2026-09-25) has no uv.lock, so we
# install from pyproject.toml with pip rather than `uv sync --frozen`. If a later merge adds
# uv.lock, switch this to `uv sync --frozen --extra ingest` (installing uv first).
#
# boto3 is NOT yet a declared pyproject dependency — s3_upload.py imports it lazily so the
# rest of the CLI works without it — so it is pinned explicitly here for the ingest-batch
# entrypoint, which needs it unconditionally.
COPY pyproject.toml README.md ./
COPY src ./src
COPY registry ./registry
RUN pip install --no-cache-dir --root-user-action=ignore '.[ingest]' 'boto3>=1.34,<2'

RUN groupadd --system --gid 10001 marinedata \
    && useradd --system --uid 10001 --gid marinedata --no-create-home \
       --shell /usr/sbin/nologin marinedata \
    && mkdir -p /work && chown marinedata:marinedata /work

USER marinedata
WORKDIR /work

ENTRYPOINT ["marinedata", "ingest-batch"]
