"""Incremental video analysis with frame-level prediction export."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .inference import YogaPoseClassifier
from .utils import get_logger, is_supported_video

logger = get_logger()


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass
class FramePrediction:
    frame_index: int
    timestamp_s: float
    label: str
    confidence: float
    person_detected: bool
    inference_ms: float


@dataclass
class VideoAnalysisResult:
    video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_s: float
    predictions: list[FramePrediction] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    dominant_pose: str | None = None
    average_latency_ms: float = 0.0
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "video": self.video_path,
            "processed_frames": self.processed_frames,
            "duration_s": round(self.duration_s, 2),
            "dominant_pose": self.dominant_pose,
            "class_counts": self.class_counts,
            "average_latency_ms": round(self.average_latency_ms, 2),
        }


def open_video(path: str | Path):
    """Open a video file with OpenCV, raising VideoError on failure."""
    import cv2

    p = Path(path)
    if not p.exists():
        raise VideoError(f"Video file not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise VideoError(
            f"Could not open video {p.name}. The codec may be unsupported by the "
            "bundled OpenCV build; try converting to H.264 MP4 with FFmpeg."
        )
    return cap


def analyse_video(
    video_path: str | Path,
    classifier: YogaPoseClassifier,
    *,
    sample_every: int = 5,
    max_frames: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> VideoAnalysisResult:
    """Process a video incrementally, classifying every ``sample_every`` frames.

    Frames are decoded one at a time; the whole video is never held in memory.
    """
    if not is_supported_video(video_path):
        logger.warning("Unusual video extension for %s — attempting to open anyway", video_path)

    cap = open_video(video_path)
    result = VideoAnalysisResult(
        video_path=str(video_path),
        total_frames=int(cap.get(5)) or 0,  # CAP_PROP_FRAME_COUNT
        processed_frames=0,
        fps=float(cap.get(5) or 0.0) or 30.0,
        duration_s=0.0,
    )
    result.fps = float(cap.get(5) or 0.0)
    frame_count = int(cap.get(7))  # CAP_PROP_FRAME_COUNT
    fps = float(cap.get(5)) or 30.0
    result.total_frames = frame_count
    result.fps = fps
    result.duration_s = frame_count / fps if fps else 0.0

    latencies: list[float] = []
    counts: dict[str, int] = {}
    start = time.perf_counter()
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                if max_frames is not None and result.processed_frames >= max_frames:
                    break
                prediction, pose = classifier.predict_frame(frame, smooth=True)
                fp = FramePrediction(
                    frame_index=idx,
                    timestamp_s=idx / fps if fps else 0.0,
                    label=prediction.label,
                    confidence=prediction.confidence,
                    person_detected=pose.detected,
                    inference_ms=prediction.inference_ms,
                )
                result.predictions.append(fp)
                result.processed_frames += 1
                latencies.append(prediction.inference_ms)
                counts[prediction.label] = counts.get(prediction.label, 0) + 1
                if progress_callback and frame_count:
                    progress_callback(min(1.0, idx / max(1, frame_count)))
            idx += 1
    finally:
        cap.release()

    result.elapsed_s = time.perf_counter() - start
    result.class_counts = counts
    result.dominant_pose = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    result.average_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    if progress_callback:
        progress_callback(1.0)
    return result


def predictions_to_dataframe(result: VideoAnalysisResult):
    """Convert frame predictions to a pandas DataFrame."""
    import pandas as pd

    columns = [
        "frame", "timestamp_s", "label", "confidence",
        "person_detected", "inference_ms",
    ]
    if not result.predictions:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "frame": p.frame_index,
                "timestamp_s": round(p.timestamp_s, 3),
                "label": p.label,
                "confidence": round(p.confidence, 4),
                "person_detected": p.person_detected,
                "inference_ms": round(p.inference_ms, 2),
            }
            for p in result.predictions
        ]
    )


def export_predictions_csv(result: VideoAnalysisResult, output_path: str | Path) -> Path:
    df = predictions_to_dataframe(result)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
