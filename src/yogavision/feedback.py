"""Rule-based posture feedback from MediaPipe landmarks.

These rules are heuristic and deliberately kept separate from CNN
classification. A predicted pose label does NOT prove correct execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .pose_detection import Landmark, PoseResult
from .utils import get_logger

logger = get_logger()


@dataclass
class FeedbackItem:
    severity: str  # "info" | "warn" | "good"
    message: str
    rule: str


@dataclass
class FeedbackReport:
    items: list[FeedbackItem] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warn" for i in self.items)


def angle_between(
    a: tuple[float, float] | np.ndarray,
    b: tuple[float, float] | np.ndarray,
    c: tuple[float, float] | np.ndarray,
) -> float:
    """Interior angle at joint ``b`` formed by points a-b-c, in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _pixel(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def joint_angles(result: PoseResult) -> dict[str, float]:
    """Compute left/right elbow, shoulder, hip and knee angles.

    Angles are computed only when all three contributing landmarks are visible.
    """
    angles: dict[str, float] = {}
    if not result.detected or result.landmarks is None:
        return angles
    pixels = result.pixel_landmarks()
    if pixels is None:
        return angles

    def visible(*idx: int, thresh: float = 0.5) -> bool:
        return all(result.is_visible(i, thresh) for i in idx)

    joints = {
        "left_elbow": (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
        "right_elbow": (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
        "left_shoulder": (Landmark.LEFT_ELBOW, Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
        "right_shoulder": (Landmark.RIGHT_ELBOW, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
        "left_hip": (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
        "right_hip": (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
        "left_knee": (Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
        "right_knee": (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    }
    for name, (i, j, k) in joints.items():
        if visible(i, j, k):
            angles[name] = round(
                angle_between(_pixel(pixels[i]), _pixel(pixels[j]), _pixel(pixels[k])), 1
            )
    return angles


def _visible_fraction(result: PoseResult, indices: list[int]) -> float:
    if not result.detected or result.landmarks is None:
        return 0.0
    return sum(1 for i in indices if result.is_visible(i, 0.5)) / len(indices)


def analyse_feedback(
    result: PoseResult,
    settings: Settings | None = None,
    *,
    predicted_label: str | None = None,
) -> FeedbackReport:
    """Apply rule-based posture checks to a pose result."""
    settings = settings or get_settings()
    cfg = settings.feedback_thresholds
    vis_thresh = float(cfg.get("min_landmark_visibility", 0.5))
    symmetry_tol = float(cfg.get("symmetry_tolerance_deg", 20.0))
    shoulder_tilt_tol = float(cfg.get("shoulder_tilt_tolerance_deg", 15.0))
    hip_tilt_tol = float(cfg.get("hip_tilt_tolerance_deg", 15.0))
    min_visible = float(cfg.get("min_visible_fraction", 0.6))

    report = FeedbackReport()

    if not result.detected:
        report.items.append(
            FeedbackItem(
                "warn", "No person detected. Stand so your whole body is in frame.", "no_person"
            )
        )
        return report

    core = [
        Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER,
        Landmark.LEFT_HIP, Landmark.RIGHT_HIP,
        Landmark.LEFT_KNEE, Landmark.RIGHT_KNEE,
        Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE,
    ]
    frac = _visible_fraction(result, core)
    if frac < min_visible:
        report.items.append(
            FeedbackItem(
                "warn",
                "Keep your body fully visible in the frame — several key joints "
                "are not detected.",
                "low_visibility",
            )
        )
        return report

    report.items.append(
        FeedbackItem("good", "Body detected with sufficient landmark visibility.", "detected")
    )

    angles = joint_angles(result)

    for side in ("left", "right"):
        a = angles.get(f"{side}_elbow")
        b = angles.get("left_elbow" if side == "right" else "right_elbow")
        if a is not None and b is not None and abs(a - b) > symmetry_tol:
            report.items.append(
                FeedbackItem(
                    "warn",
                    f"Your elbow angles differ between sides "
                    f"({side[0].upper()}{side[1:]} {a:.0f} deg vs other {b:.0f} deg).",
                    "elbow_asymmetry",
                )
            )

    for joint, tol, label in (
        ("knee", symmetry_tol, "knee"),
        ("hip", symmetry_tol, "hip"),
    ):
        l = angles.get(f"left_{joint}")
        r = angles.get(f"right_{joint}")
        if l is not None and r is not None and abs(l - r) > tol:
            report.items.append(
                FeedbackItem(
                    "warn",
                    f"Your left and right {label} angles differ significantly "
                    f"({l:.0f} deg vs {r:.0f} deg).",
                    f"{joint}_asymmetry",
                )
            )

    pixels = result.pixel_landmarks()
    if pixels is not None:
        ls = pixels[Landmark.LEFT_SHOULDER]
        rs = pixels[Landmark.RIGHT_SHOULDER]
        lh = pixels[Landmark.LEFT_HIP]
        rh = pixels[Landmark.RIGHT_HIP]
        if (
            result.is_visible(Landmark.LEFT_SHOULDER, vis_thresh)
            and result.is_visible(Landmark.RIGHT_SHOULDER, vis_thresh)
        ):
            dx = abs(float(ls[0] - rs[0]))
            dy = abs(float(ls[1] - rs[1]))
            shoulder_diff = math.degrees(math.atan2(dy, max(dx, 1e-6)))
            if shoulder_diff > shoulder_tilt_tol:
                report.items.append(
                    FeedbackItem(
                        "warn",
                        "Your shoulder alignment appears uneven — try to level your shoulders.",
                        "shoulder_tilt",
                    )
                )
        if (
            result.is_visible(Landmark.LEFT_HIP, vis_thresh)
            and result.is_visible(Landmark.RIGHT_HIP, vis_thresh)
        ):
            dx = abs(float(lh[0] - rh[0]))
            dy = abs(float(lh[1] - rh[1]))
            hip_diff = math.degrees(math.atan2(dy, max(dx, 1e-6)))
            if hip_diff > hip_tilt_tol:
                report.items.append(
                    FeedbackItem(
                        "warn",
                        "Your hips appear tilted — try to square your pelvis.",
                        "hip_tilt",
                    )
                )

    if result.landmarks is not None:
        h, w = result.image_shape
        xs = result.landmarks[result.landmarks[:, 2] >= vis_thresh, 0]
        ys = result.landmarks[result.landmarks[:, 2] >= vis_thresh, 1]
        if len(xs):
            span = float(xs.max() - xs.min())
            if span > 0.92:
                report.items.append(
                    FeedbackItem(
                        "warn",
                        "Move slightly farther from the camera — you are close to the frame edge.",
                        "too_close",
                    )
                )
            elif span < 0.15:
                report.items.append(
                    FeedbackItem(
                        "warn",
                        "Move slightly closer to the camera so your full body is visible.",
                        "too_far",
                    )
                )

    if not report.has_warnings:
        report.items.append(
            FeedbackItem("good", "Alignment looks balanced for the detected pose.", "balanced")
        )
    return report


def feedback_rules_documentation(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Return the implemented rules and their thresholds for documentation."""
    settings = settings or get_settings()
    cfg = settings.feedback_thresholds
    return [
        {"rule": "no_person", "condition": "MediaPipe found no pose landmarks"},
        {"rule": "low_visibility", "condition": f"< {cfg.get('min_visible_fraction', 0.6)} of core joints visible"},
        {"rule": "elbow_asymmetry", "condition": f"|L-R elbow angle| > {cfg.get('symmetry_tolerance_deg', 20)} deg"},
        {"rule": "knee_asymmetry", "condition": f"|L-R knee angle| > {cfg.get('symmetry_tolerance_deg', 20)} deg"},
        {"rule": "hip_asymmetry", "condition": f"|L-R hip angle| > {cfg.get('symmetry_tolerance_deg', 20)} deg"},
        {"rule": "shoulder_tilt", "condition": f"shoulder line tilt > {cfg.get('shoulder_tilt_tolerance_deg', 15)} deg"},
        {"rule": "hip_tilt", "condition": f"hip line tilt > {cfg.get('hip_tilt_tolerance_deg', 15)} deg"},
        {"rule": "too_close", "condition": "horizontal landmark span > 92% of frame width"},
        {"rule": "too_far", "condition": "horizontal landmark span < 15% of frame width"},
    ]
