"""Tests for feedback module."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogavision.feedback import (
    angle_between,
    analyse_feedback,
    joint_angles,
    FeedbackReport,
    FeedbackItem,
)
from yogavision.pose_detection import Landmark, PoseResult


class TestAngleBetween:
    def test_right_angle(self):
        # horizontal left, horizontal right -> 180 deg
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        c = (2.0, 0.0)
        ang = angle_between(a, b, c)
        assert ang == pytest.approx(180.0, abs=1e-3)

    def test_vertical_angle(self):
        a = (0.0, 0.0)
        b = (0.0, 0.0)
        c = (0.0, 1.0)
        ang = angle_between(a, b, c)
        assert ang == pytest.approx(0.0, abs=1e-3)

    def test_90_degrees(self):
        a = (-1.0, 0.0)
        b = (0.0, 0.0)
        c = (0.0, 1.0)
        ang = angle_between(a, b, c)
        assert 89.0 < ang < 91.0


class TestJointAngles:
    def test_no_detection(self):
        result = PoseResult(detected=False, landmarks=None, world_landmarks=None, image_shape=(480, 640))
        angles = joint_angles(result)
        assert angles == {}

    def test_synthetic_landmarks(self):
        landmarks = np.zeros((33, 3), dtype=np.float32)
        # Set visibility high
        landmarks[:, 2] = 1.0
        # left hip (23), left knee (25), left ankle (27) -> 180 deg
        landmarks[Landmark.LEFT_HIP] = [0.4, 0.5, 1.0]
        landmarks[Landmark.LEFT_KNEE] = [0.4, 0.6, 1.0]
        landmarks[Landmark.LEFT_ANKLE] = [0.4, 0.7, 1.0]
        result = PoseResult(detected=True, landmarks=landmarks, world_landmarks=None, image_shape=(480, 640))
        angles = joint_angles(result)
        assert "left_knee" in angles
        assert angles["left_knee"] == pytest.approx(180.0, abs=1.0)


class TestFeedbackReport:
    def test_has_warnings(self):
        report = FeedbackReport()
        assert report.has_warnings is False
        report.items.append(FeedbackItem("warn", "test", "test_rule"))
        assert report.has_warnings is True


class TestAnalyseFeedback:
    def test_no_person(self):
        result = PoseResult(detected=False, landmarks=None, world_landmarks=None, image_shape=(480, 640))
        report = analyse_feedback(result)
        assert any(i.rule == "no_person" for i in report.items)

    def test_sufficient_visibility(self):
        landmarks = np.zeros((33, 3), dtype=np.float32)
        landmarks[:, 2] = 0.8
        result = PoseResult(detected=True, landmarks=landmarks, world_landmarks=None, image_shape=(480, 640))
        report = analyse_feedback(result)
        assert any(i.severity == "good" or i.severity == "warn" for i in report.items)
