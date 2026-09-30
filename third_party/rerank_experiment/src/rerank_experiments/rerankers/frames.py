"""Frame sampling for media-based rerankers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from ..video import probe_duration, sample_windows
from .base import Candidate

_FRAME_FACTOR = 2
_FPS_MIN_FRAMES = 4


@dataclass(frozen=True)
class SampledVideoArray:
    """Explicit RGB frame sample plus metadata for offline vLLM video inputs."""

    frames: np.ndarray
    frame_indices: list[int]
    source_fps: float
    sample_fps: float
    duration: float


@lru_cache(maxsize=8192)
def _decode(video_path: str, start: float, end: float, num_frames: int) -> tuple[Image.Image, ...]:
    frames = sample_windows(Path(video_path), [(start, end)], num_frames)[0]
    return tuple(frames)


def sample_candidate_frames(candidate: Candidate, num_frames: int) -> list[Image.Image]:
    """Sample ``num_frames`` RGB frames over the candidate's segment window."""
    if candidate.video_path is None:
        raise ValueError("Candidate has no video_path; cannot sample frames")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    start, end = candidate_time_bounds(candidate)
    frames = _decode(str(candidate.video_path), round(start, 3), round(end, 3), num_frames)
    return list(frames)


def candidate_time_bounds(candidate: Candidate) -> tuple[float, float]:
    if candidate.video_path is None:
        raise ValueError("Candidate has no video_path; cannot resolve video duration")
    seg = candidate.segment
    if seg.start is not None and seg.end is not None:
        return float(seg.start), float(seg.end)
    duration = probe_duration(candidate.video_path)
    if duration <= 0:
        raise ValueError(f"Could not determine full-video duration: {candidate.video_path}")
    return 0.0, duration


def _ceil_by_factor(value: float, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def _floor_by_factor(value: float, factor: int) -> int:
    return int(math.floor(value / factor) * factor)


def controlled_frame_indices(
    candidate: Candidate,
    *,
    fps: float,
    max_frames: int,
    total_frames: int,
    video_fps: float,
) -> list[int]:
    """Frame indices for vLLM video rerankers that need explicit sampling.

    This mirrors Qwen-style segment sampling: convert the candidate time window
    to source frame bounds, take roughly ``duration * fps`` frames, clamp to a
    small even minimum and ``max_frames``, then spread them evenly with linspace.
    """
    if video_fps <= 0.0:
        video_fps = max(1.0, float(fps))
    if total_frames <= 0:
        raise ValueError("Cannot sample video with no frames")

    seg = candidate.segment
    if seg.start is None or seg.end is None:
        start_frame = 0
        end_frame = total_frames - 1
    else:
        max_duration = total_frames / video_fps
        start = max(0.0, min(float(seg.start), max_duration))
        end = max(0.0, min(float(seg.end), max_duration))
        start_frame = max(0, int(math.ceil(start * video_fps)))
        end_frame = min(total_frames - 1, int(math.floor(end * video_fps)))
        if end_frame < start_frame:
            end_frame = start_frame

    segment_frames = end_frame - start_frame + 1
    if segment_frames <= 1:
        return [start_frame]

    min_frames = _ceil_by_factor(_FPS_MIN_FRAMES, _FRAME_FACTOR)
    capped_max = max(_FRAME_FACTOR, _floor_by_factor(max_frames, _FRAME_FACTOR))
    nframes = segment_frames / video_fps * float(fps)
    nframes = min(min(max(nframes, min_frames), capped_max), segment_frames)
    nframes = max(1, min(segment_frames, _floor_by_factor(nframes, _FRAME_FACTOR)))
    return np.linspace(start_frame, end_frame, nframes).round().astype(int).tolist()


def sampled_video_array(candidate: Candidate, fps: float, max_frames: int) -> SampledVideoArray:
    """Decode a controlled RGB frame sample for vLLM offline video inputs."""
    if candidate.video_path is None:
        raise ValueError("Candidate has no video_path; cannot sample video frames")

    import cv2

    source = candidate.video_path.resolve()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for frame sampling: {source}")

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = controlled_frame_indices(
            candidate,
            fps=fps,
            max_frames=max_frames,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read frame {frame_idx} from source: {source}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if video_fps <= 0.0:
        video_fps = max(1.0, float(fps))
    if len(indices) <= 1:
        duration = 1.0 / video_fps
    else:
        duration = max(1e-6, (indices[-1] - indices[0] + 1) / video_fps)
    return SampledVideoArray(
        frames=np.stack(frames, axis=0),
        frame_indices=indices,
        source_fps=video_fps,
        sample_fps=len(frames) / duration,
        duration=duration,
    )


def sampled_video(
    candidate: Candidate, fps: float, max_frames: int
) -> tuple[list[Image.Image], float]:
    """fps-based sample of a candidate segment for native-video models.

    Samples ``round(duration * fps)`` frames (>=1), capped at ``max_frames``, and
    returns ``(frames, true_fps)`` where ``true_fps = n_frames / duration`` is the
    honest declared rate to hand the model (== ``fps`` when the cap isn't hit;
    lower when it is). One rule for every model.
    """
    start, end = candidate_time_bounds(candidate)
    duration = max(1e-6, end - start)
    n = min(int(max_frames), max(1, round(duration * float(fps))))
    frames = sample_candidate_frames(candidate, n)
    return frames, len(frames) / duration


def candidate_video_ref(candidate: Candidate, fps: float, max_frames: int) -> dict:
    """Qwen-compatible source-video reference for processor-side segment sampling."""
    if candidate.video_path is None:
        raise ValueError("Candidate has no video_path; cannot build video reference")
    seg = candidate.segment
    ref = {
        "type": "video",
        "video": str(candidate.video_path),
        "fps": float(fps),
        "max_frames": int(max_frames),
    }
    if seg.start is not None and seg.end is not None:
        ref["video_start"] = float(seg.start)
        ref["video_end"] = float(seg.end)
    return ref


def contact_sheet(
    frames: list[Image.Image], cols: int | None = None, cell: int = 336
) -> Image.Image:
    """Tile frames into a single grid image (row-major), each cell ``cell`` px."""
    if not frames:
        raise ValueError("contact_sheet requires at least one frame")
    cols = cols or math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, frame in enumerate(frames):
        tile = frame.convert("RGB").resize((cell, cell), Image.Resampling.BILINEAR)
        sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
    return sheet


def candidate_visual(candidate: Candidate, fps: float, max_frames: int) -> list[Image.Image]:
    """Visual evidence for a candidate as a single tiled contact-sheet image.

    Returned as a one-element list so callers stay uniform. Frames are sampled by
    the same fps rule as native video (the cross-encoders just tile them into one
    image). Family-A cross-encoders take one image per document, while
    Qwen-compatible native-video models can let their processor sample segments
    from the source video.
    """
    frames, _ = sampled_video(candidate, fps, max_frames)
    return [contact_sheet(frames)]


def parallel_map(fn, items: list, workers: int = 8) -> list:
    """Thread-pool map (frame decode / image ops release the GIL), preserving order.

    Used to build per-candidate inputs concurrently so CPU preprocessing overlaps
    with the GPU instead of starving it.
    """
    if len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def to_data_url(image: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image as a base64 JPEG ``data:`` URL (for vLLM chat)."""
    import base64
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clear_frame_cache() -> None:
    _decode.cache_clear()
