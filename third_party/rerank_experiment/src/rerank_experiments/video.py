"""Video probing and frame sampling.

Kept deliberately small and dependency-light (OpenCV + PIL) so it can be swapped
for a hardware-accelerated decoder without touching the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds (0.0 if it cannot be read)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frame_count <= 0:
            return 0.0
        return float(frame_count / fps)
    finally:
        cap.release()


def probe_durations(video_dir: Path, video_names: list[str]) -> dict[str, float]:
    """Probe durations for a set of videos (by name, without extension)."""
    durations: dict[str, float] = {}
    for name in video_names:
        durations[name] = probe_duration(video_dir / f"{name}.mp4")
    return durations


def _uniform_timestamps(start: float, end: float, num_frames: int) -> list[float]:
    """Evenly spaced sample times centred inside ``[start, end)``."""
    if num_frames <= 1:
        return [(start + end) / 2.0]
    # Sample at the centre of ``num_frames`` equal sub-intervals to avoid the
    # exact boundaries (which can land on shot cuts or black frames).
    step = (end - start) / num_frames
    return [start + (i + 0.5) * step for i in range(num_frames)]


def sample_windows(
    video_path: Path,
    windows: list[tuple[float, float]],
    frames_per_window: int,
) -> list[list[Image.Image]]:
    """Sample ``frames_per_window`` RGB frames for each ``(start, end)`` window.

    The video is opened once and frames are read in increasing timestamp order.
    Returns one list of PIL images per window (a window yields a black frame if
    a target frame cannot be decoded, so downstream batching stays rectangular).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 256
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 256

        # Flatten all requested timestamps, decode in order, then regroup.
        requests: list[tuple[int, float]] = []  # (window_index, timestamp)
        for w_idx, (start, end) in enumerate(windows):
            for ts in _uniform_timestamps(start, end, frames_per_window):
                requests.append((w_idx, ts))
        requests.sort(key=lambda r: r[1])

        decoded: dict[tuple[int, float], Image.Image] = {}
        black = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        for w_idx, ts in requests:
            frame_idx = max(0, int(round(ts * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                decoded[(w_idx, ts)] = black
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[(w_idx, ts)] = Image.fromarray(rgb)

        result: list[list[Image.Image]] = [[] for _ in windows]
        for w_idx, (start, end) in enumerate(windows):
            for ts in _uniform_timestamps(start, end, frames_per_window):
                result[w_idx].append(decoded[(w_idx, ts)])
        return result
    finally:
        cap.release()
