"""Tests for pose detection module."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogavision.pose_detection import (
    Landmark,
    PoseDetector,
    PoseResult,
    POSE_CONNECTIONS,
    NUM_LANDMARKS,
)


class TestPoseResult:
    def test_pixel_landmarks(self):
        result = PoseResult(
            detected=True,
            landmarks=np.array([[0.5, 0.3, 0.9]] * NUM_LANDMARKS, dtype=np.float32),
            world_landmarks=None,
            image_shape=(480, 640),
        )
        pixels = result.pixel_landmarks()
        assert pixels is not None
        assert pixels.shape == (NUM_LANDMARKS, 2)
        assert pixels[0, 0] == pytest.approx(320.0, rel=1e-3)
        assert pixels[0, 1] == pytest.approx(144.0, rel=1e-3)

    def test_not_detected(self):
        result = PoseResult(detected=False, landmarks=None, world_landmarks=None, image_shape=(480, 640))
        assert result.pixel_landmarks() is None
        assert result.is_visible(0) is False


class TestPoseDetector:
    def test_detect_synthetic_image(self):
        detector = PoseDetector()
        # blank image - no person
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        # May or may not detect; just check it returns a PoseResult
        assert isinstance(result, PoseResult)
        assert result.image_shape == (480, 640)

    def test_skeleton_drawing_does_not_crash(self):
        detector = PoseDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        canvas = detector.draw_skeleton(img, result)
        assert canvas.shape == img.shape
        assert canvas.dtype == np.uint8

    def test_reset(self):
        detector = PoseDetector()
        detector._ensure_pose()
        assert detector._pose is not None
        detector.reset()
        assert detector._pose is None

    def test_close(self):
        detector = PoseDetector()
        detector._ensure_pose()
        detector.close()
        assert detector._pose is None


class TestLandmarkIndices:
    def test_num_landmarks(self):
        assert NUM_LANDMARKS == 33

    def test_connections_reasonable(self):
        assert len(POSE_CONNECTIONS) > 10
        for a, b in POSE_CONNECTIONS:
            assert 0 <= a < NUM_LANDMARKS
            assert 0 <= b < NUM_LANDMARKS
