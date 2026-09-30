"""Video frame sampler (WP-6e-B, D-AB): 1 frame per ``frame_step`` frames (default 10),
capped at ``max_fps`` frames per second (default 1). ffmpeg when present, else PyAV.

Every frame carries ``video_id`` and ``frame_idx`` (0-based index in the source video)
and ``split_group_id = video_id``, so a video never straddles splits."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


def effective_step(fps: float | None, frame_step: int = 10, max_fps: float = 1.0) -> int:
    """Frames between kept frames: ``frame_step``, raised so kept frames <= ``max_fps``/s."""
    step = max(1, int(frame_step))
    if fps and fps > 0 and max_fps > 0:
        step = max(step, math.ceil(fps / max_fps - 1e-9))
    return step


def frame_indices(
    n_frames: int, fps: float | None, frame_step: int = 10, max_fps: float = 1.0
) -> list[int]:
    return list(range(0, max(0, n_frames), effective_step(fps, frame_step, max_fps)))


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_fps(path: Path) -> float | None:
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    out = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        num, _, den = json.loads(out.stdout)["streams"][0]["avg_frame_rate"].partition("/")
        return float(num) / float(den or 1) if float(den or 1) else None
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return None


def _ffmpeg_frames(path: Path, step: int) -> Iterator[tuple[int, bytes]]:
    with tempfile.TemporaryDirectory(prefix="vidstep-") as td:
        subprocess.run(
            [
                _ffmpeg(),
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select=not(mod(n\\,{step}))",
                "-fps_mode",
                "passthrough",
                "-qscale:v",
                "2",
                str(Path(td) / "f-%08d.jpg"),
            ],
            check=True,
        )
        for i, frame in enumerate(sorted(Path(td).glob("f-*.jpg"))):
            yield i * step, frame.read_bytes()


def _pyav_frames(path: Path, step: int) -> Iterator[tuple[int, bytes]]:
    import io

    import av  # type: ignore[import-not-found]

    with av.open(str(path)) as container:
        for n, frame in enumerate(container.decode(video=0)):
            if n % step == 0:
                buf = io.BytesIO()
                frame.to_image().save(buf, format="JPEG", quality=92)
                yield n, buf.getvalue()


def sample_video(
    path: Path, key: str, params: Mapping[str, Any]
) -> Iterator[tuple[int, bytes, dict[str, str]]]:
    """Yield ``(frame_idx, jpeg_bytes, labels)`` for the kept frames of one video."""
    video_id = str(params.get("video_id_prefix", "")) + PurePosixPath(key).stem
    fps = probe_fps(path) or (float(params["assume_fps"]) if params.get("assume_fps") else None)
    step = effective_step(fps, int(params.get("frame_step", 10)), float(params.get("max_fps", 1.0)))
    frames = _ffmpeg_frames(path, step) if _ffmpeg() else _pyav_frames(path, step)
    for idx, data in frames:
        labels = {"video_id": video_id, "frame_idx": str(idx), "split_group_id": video_id}
        if fps:
            labels["frame_ts"] = f"{idx / fps:.3f}"
        yield idx, data, labels
