"""MediaPipe pose landmark detection and drawing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .utils import get_logger

logger = get_logger()

# MediaPipe Pose landmark indices
class Landmark:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

NUM_LANDMARKS = 33


@dataclass
class PoseResult:
    """Landmark detection output for one frame."""

    detected: bool
    landmarks: np.ndarray | None  # (33, 3) x, y (normalised), visibility
    world_landmarks: np.ndarray | None  # (33, 3) metric coords, may be None
    image_shape: tuple[int, int]  # (height, width)

    def pixel_landmarks(self) -> np.ndarray | None:
        """Return (33, 2) integer pixel coordinates, or None if no detection."""
        if self.landmarks is None:
            return None
        h, w = self.image_shape
        xy = self.landmarks[:, :2].copy()
        xy[:, 0] *= w
        xy[:, 1] *= h
        return xy

    def visibility(self, index: int) -> float:
        if self.landmarks is None:
            return 0.0
        return float(self.landmarks[index, 2])

    def is_visible(self, index: int, threshold: float = 0.5) -> bool:
        return self.visibility(index) >= threshold


class PoseDetector:
    """Thin wrapper around MediaPipe Pose with a lazy singleton graph."""

    _instance: "PoseDetector | None" = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pose: Any = None

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "PoseDetector":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def _ensure_pose(self) -> Any:
        if self._pose is None:
            import mediapipe as mp

            self._mp = mp
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.settings.mp_min_detection_confidence,
                min_tracking_confidence=self.settings.mp_min_tracking_confidence,
            )
        return self._pose

    def reset(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def detect(self, image_bgr: np.ndarray) -> PoseResult:
        """Run pose estimation on a BGR image."""
        pose = self._ensure_pose()
        import cv2

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        try:
            result = pose.process(image_rgb)
        except Exception as exc:  # pragma: no cover - MediaPipe runtime failure
            logger.warning("MediaPipe processing failed: %s", exc)
            return PoseResult(False, None, None, (h, w))
        finally:
            image_rgb.flags.writeable = True

        if not result.pose_landmarks:
            return PoseResult(False, None, None, (h, w))

        landmarks = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        world = None
        if result.pose_world_landmarks:
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks.landmark],
                dtype=np.float32,
            )
        return PoseResult(True, landmarks, world, (h, w))

    def draw_skeleton(
        self,
        image_bgr: np.ndarray,
        result: PoseResult,
        *,
        thickness: int = 2,
        radius: int = 3,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw the skeleton overlay onto a copy of the image."""
        import cv2

        canvas = image_bgr.copy()
        if not result.detected:
            return canvas
        pixels = result.pixel_landmarks()
        for a, b in POSE_CONNECTIONS:
            if result.is_visible(a) and result.is_visible(b):
                pa = (int(pixels[a, 0]), int(pixels[a, 1]))
                pb = (int(pixels[b, 0]), int(pixels[b, 1]))
                cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
        for i, (x, y) in enumerate(pixels):
            if result.is_visible(i):
                cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 255), -1, cv2.LINE_AA)
        return canvas

    def draw_landmarks_canvas(
        self, result: PoseResult, *, thickness: int = 2, radius: int = 3
    ) -> np.ndarray:
        """Render the skeleton onto a blank black canvas (paper-style skeleton image)."""
        h, w = result.image_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if not result.detected:
            return canvas
        return self.draw_skeleton(canvas, result, thickness=thickness, radius=radius)

    def close(self) -> None:
        self.reset()
