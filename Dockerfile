# syntax=docker/dockerfile:1
#
# Release-build image (Phase A, P1.7): runs `marinedata release build|gate|publish` on a
# temporary VM next to the bucket (see deploy/release-vm/README.md). It is not the cluster
# ingest Job image (deploy/ingest-job/Dockerfile).
#
#   make image          build marinedata:$(git describe)
#   make image-check    --version, `check --profile research`, and the IMAGE_MAX_BYTES size cap (2 GB)
#
# `quality` is not optional here: `marinedata.cli` imports cv2 (opencv-headless) at import time.
# `privacy` (torch, torchvision, opencv: the v1.1.0 privacy gate G8) is in. uv.lock resolves torch
# and torchvision from the CPU-only index off macOS (pyproject.toml [tool.uv.sources]), so no CUDA
# wheels (nvidia-*, triton) reach the image; the cap is IMAGE_MAX_BYTES (Makefile). `baselines`
# stays out on purpose. Change the set with `--build-arg EXTRAS=...`. `data/` is not baked in (see
# .dockerignore): the VM mounts the clone's data/ at /app/data read-only, which is where
# `release build --tasks v2` looks.
#
# The base image is pinned by digest (multi-arch manifest list, resolved 2026-09-25, same pin as
# deploy/ingest-job/Dockerfile); re-pin there and here together.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
ARG UV_VERSION=0.11.4

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${PYTHON_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

ARG EXTRAS="ingest hf croissant quality privacy"

# The venv lives outside /app so `COPY . .` can never overwrite it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MARINEDATA_CACHE=/data/cache \
    HF_HOME=/data/hf

WORKDIR /app

# Dependencies first (cached until pyproject.toml or uv.lock change), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)
COPY . .
RUN uv sync --frozen --no-dev $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)

# The pinned YuNet face model (src/marinedata/gates/yunet_model.py): fetched here, at image build
# time, and checked against its sha256 by `ADD --checksum` and again at load; it is never downloaded
# at gate time. --chmod=0444: ADD from a URL is 0600 root, and the container runs as uid 10001.
ADD --checksum=sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 --chmod=0444 \
    https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /opt/models/face_detection_yunet_2023mar.onnx
ENV MARINEDATA_YUNET_MODEL=/opt/models/face_detection_yunet_2023mar.onnx

RUN groupadd --system --gid 10001 marinedata \
    && useradd --system --uid 10001 --gid marinedata --create-home \
       --shell /usr/sbin/nologin marinedata \
    && mkdir -p /data && chown marinedata:marinedata /data

USER marinedata

ENTRYPOINT ["marinedata"]
CMD ["--help"]
