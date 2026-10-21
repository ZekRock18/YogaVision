"""Tests for video analysis, skeletonization and invalid input handling."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from yogavision.preprocessing import ImageLoadError
from yogavision.skeletonization import skeletonize_image
from yogavision.video_analysis import (
    FramePrediction,
    VideoError,
    analyse_video,
    export_predictions_csv,
    open_video,
    predictions_to_dataframe,
)


def _fake_video(path: Path, n_frames: int = 20, width: int = 320, height: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter — codec mp4v unsupported")
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _DummyBundle:
    """A stand-in for ModelBundle that the classifier can use for testing."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.image_size = (75, 75)
        self.input_representation = "skeleton"


class _DummyClassifier:
    def __init__(self, labels: list[str]) -> None:
        self.bundle = _DummyBundle(labels)
        self.detector = None
        self.smoother = type("S", (), {"_history": type("H", (), {"append": lambda s, x: None})()})()
        self.confidence_threshold = 0.6

    def predict_frame(self, frame, *, smooth=True, k=3):
        from yogavision.inference import Prediction
        from yogavision.pose_detection import PoseResult

        return (
            Prediction(
                label=self.bundle.labels[0],
                confidence=0.9,
                top_k=[(self.bundle.labels[0], 0.9)],
                above_threshold=True,
                inference_ms=1.0,
            ),
            PoseResult(detected=True, landmarks=None, world_landmarks=None, image_shape=frame.shape[:2]),
        )


class TestSkeletonizeImage:
    def test_returns_canvas(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        skeleton, detected = skeletonize_image(img)
        assert skeleton.shape == img.shape
        assert skeleton.dtype == np.uint8
        # No person -> blank canvas
        assert skeleton.max() == 0
        assert detected is False


class TestImageLoadError:
    def test_load_missing_file_raises(self, tmp_path):
        from yogavision.preprocessing import load_image

        with pytest.raises(ImageLoadError):
            load_image(tmp_path / "nope.jpg")

    def test_load_corrupt_file_raises(self, tmp_path):
        from yogavision.preprocessing import load_image

        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        with pytest.raises(ImageLoadError):
            load_image(p)


class TestVideoOpen:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoError):
            open_video(tmp_path / "nope.mp4")


class TestAnalyseVideo:
    def test_processes_incrementally(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _fake_video(video, n_frames=20)
        classifier = _DummyClassifier(["a", "b"])
        result = analyse_video(video, classifier, sample_every=5, max_frames=4)
        assert result.processed_frames <= 4
        assert result.total_frames == 20
        assert result.dominant_pose is not None

    def test_progress_callback(self, tmp_path):
        video = tmp_path / "clip.mp4"
        _fake_video(video, n_frames=10)
        classifier = _DummyClassifier(["a"])
        calls: list[float] = []
        analyse_video(
            video, classifier, sample_every=2,
            progress_callback=calls.append,
        )
        assert 1.0 in calls  # final call


class TestPredictionsToDataframe:
    def test_empty(self):
        from yogavision.video_analysis import VideoAnalysisResult

        empty = VideoAnalysisResult(
            video_path="x", total_frames=0, processed_frames=0,
            fps=0, duration_s=0, predictions=[],
        )
        df = predictions_to_dataframe(empty)
        assert df.empty
        assert "frame" in df.columns
        assert "label" in df.columns


class TestExportPredictionsCSV:
    def test_writes_csv(self, tmp_path):
        from yogavision.video_analysis import VideoAnalysisResult

        result = VideoAnalysisResult(
            video_path="x", total_frames=2, processed_frames=2, fps=10, duration_s=0.2,
            predictions=[
                FramePrediction(0, 0.0, "a", 0.9, True, 1.0),
                FramePrediction(10, 1.0, "b", 0.8, True, 1.2),
            ],
        )
        out = export_predictions_csv(result, tmp_path / "p.csv")
        assert out.exists()
        content = out.read_text()
        assert "a" in content and "b" in content
        assert "frame" in content
