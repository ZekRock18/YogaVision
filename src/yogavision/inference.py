"""Inference pipeline: frame -> prediction with temporal smoothing."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .model import ModelBundle, load_model
from .pose_detection import PoseDetector, PoseResult
from .skeletonization import skeletonize_image
from .utils import get_logger, top_k

logger = get_logger()


@dataclass
class Prediction:
    """One classification result."""

    label: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)
    above_threshold: bool = False
    inference_ms: float = 0.0


class PredictionSmoother:
    """Majority-vote smoothing over the last ``window`` predictions."""

    def __init__(self, window: int = 5) -> None:
        self.window = max(1, int(window))
        self._history: deque[str] = deque(maxlen=self.window)

    def update(self, label: str) -> str:
        self._history.append(label)
        counts = Counter(self._history)
        # Ties broken by most recent occurrence
        return max(counts.items(), key=lambda kv: (kv[1], list(self._history)[::-1].index(kv[0])))[0]

    def reset(self) -> None:
        self._history.clear()


class YogaPoseClassifier:
    """End-to-end classifier: image -> skeleton -> CNN -> label.

    A single instance holds the loaded model and pose detector, so it is cached
    by the UI layer and never reloaded per frame.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        settings: Settings | None = None,
        *,
        confidence_threshold: float | None = None,
        smoothing_window: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bundle = bundle
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.settings.confidence_threshold
        )
        self.smoother = PredictionSmoother(
            smoothing_window if smoothing_window is not None else self.settings.smoothing_window
        )
        self.detector = PoseDetector.get_instance(self.settings)
        self._representation = bundle.input_representation

    def reset_smoothing(self) -> None:
        self.smoother.reset()

    def _prepare_input(self, image_bgr: np.ndarray) -> tuple[np.ndarray, PoseResult]:
        """Build the model input according to the trained representation."""
        result = self.detector.detect(image_bgr)
        h, w = self.bundle.image_size[1], self.bundle.image_size[0]
        if self._representation == "skeleton":
            if result.detected:
                canvas = self.detector.draw_skeleton(
                    np.zeros_like(image_bgr), result, thickness=2, radius=3
                )
            else:
                canvas = np.zeros_like(image_bgr)
            import cv2

            resized = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
        else:
            import cv2

            resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0, result

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        *,
        smooth: bool = True,
        k: int = 3,
    ) -> tuple[Prediction, PoseResult]:
        """Classify one frame and return (prediction, pose_result)."""
        import time

        start = time.perf_counter()
        model_input, pose = self._prepare_input(image_bgr)
        probs = self.bundle.predict_one(model_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        best_idx = int(np.argmax(probs))
        best_label = self.bundle.labels[best_idx]
        confidence = float(probs[best_idx])

        final_label = self.smoother.update(best_label) if smooth else best_label
        # Recompute confidence for the smoothed label using the latest frame
        label_idx = self.bundle.labels.index(final_label)
        final_conf = float(probs[label_idx])

        top = top_k(probs, k)
        prediction = Prediction(
            label=final_label,
            confidence=final_conf,
            top_k=[(self.bundle.labels[i], float(p)) for i, p in top],
            above_threshold=final_conf >= self.confidence_threshold,
            inference_ms=elapsed_ms,
        )
        return prediction, pose

    def predict_image(
        self, image_bgr: np.ndarray, *, k: int = 3, smooth: bool = False
    ) -> tuple[Prediction, PoseResult]:
        return self.predict_frame(image_bgr, smooth=smooth, k=k)


def get_classifier(settings: Settings | None = None) -> YogaPoseClassifier:
    """Load the model bundle and return a classifier instance.

    Raises ModelNotAvailableError when assets are missing; callers are expected
    to surface a friendly setup message.
    """
    settings = settings or get_settings()
    bundle = load_model(settings, input_representation=settings.config.get(
        "model", {}).get("input_representation", "rgb"))
    return YogaPoseClassifier(bundle, settings)


def label_from_probabilities(
    probabilities: np.ndarray, labels: list[str], threshold: float = 0.0
) -> tuple[str, float]:
    """Map a probability vector to (label, confidence)."""
    if probabilities.size == 0:
        return "unknown", 0.0
    idx = int(np.argmax(probabilities))
    return labels[idx], float(probabilities[idx])
