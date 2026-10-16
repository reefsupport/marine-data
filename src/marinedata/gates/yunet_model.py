"""The pinned YuNet face model: OpenCV Zoo URL + sha256, baked into the image, verified at load.

The release Dockerfile fetches ``YUNET_URL`` with ``ADD --checksum=sha256:<YUNET_SHA256>`` at image
build time and sets ``MARINEDATA_YUNET_MODEL`` to the file. Nothing here (or anywhere at gate time)
touches the network: a missing or altered model raises, it is never downloaded."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/"
    f"{YUNET_COMMIT}/models/face_detection_yunet/{YUNET_FILE}"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_BYTES = 232589
ENV_VAR = "MARINEDATA_YUNET_MODEL"


class YuNetModelError(RuntimeError):
    """The YuNet model is missing or is not the pinned file."""


def verified_model_path(path: str | Path | None = None) -> Path:
    """``path`` (default ``$MARINEDATA_YUNET_MODEL``) once its sha256 is :data:`YUNET_SHA256`."""
    raw = path or os.environ.get(ENV_VAR)
    if not raw:
        raise YuNetModelError(
            f"{ENV_VAR} is not set: the release image bakes the pinned model in at build time"
        )
    model = Path(raw)
    if not model.is_file():
        raise YuNetModelError(
            f"YuNet model {model} not found (it is never downloaded at gate time)"
        )
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != YUNET_SHA256:
        raise YuNetModelError(
            f"YuNet model {model.name}: sha256 {digest[:12]} != pinned {YUNET_SHA256[:12]}"
        )
    return model
