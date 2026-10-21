"""Image loading, validation and preprocessing for the CNN input."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .utils import get_logger

logger = get_logger()


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded."""


@dataclass
class PreprocessResult:
    """Result of preprocessing one image for model inference."""

    model_input: np.ndarray  # (H, W, 3) float32 in [0, 1]
    original: np.ndarray  # (H0, W0, 3) uint8 BGR
    resized: np.ndarray  # (H, W, 3) uint8 BGR


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a uint8 BGR array.

    Raises ImageLoadError when the file is missing, unreadable or corrupt.
    """
    import cv2

    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"Image file not found: {p}")
    data = np.fromfile(p, dtype=np.uint8)
    if data.size == 0:
        raise ImageLoadError(f"Image file is empty: {p}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not decode image (unsupported or corrupt): {p}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageLoadError(f"Unexpected image shape {image.shape}: {p}")
    return image


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode an in-memory image buffer to a uint8 BGR array."""
    import cv2

    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        raise ImageLoadError("Empty image buffer")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError("Could not decode image buffer")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image to (width, height)."""
    import cv2

    return cv2.resize(image, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] to float32 [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess_for_model(
    image: np.ndarray,
    settings: Settings | None = None,
    size: tuple[int, int] | None = None,
) -> PreprocessResult:
    """Resize and normalise an image for the CNN.

    Returns both the model-ready float array and the intermediate uint8 resized
    image so callers can display or skeletonize the same frame.
    """
    settings = settings or get_settings()
    h, w = size or settings.image_size
    resized = resize_image(image, (w, h))
    model_input = normalize(resized)
    return PreprocessResult(model_input=model_input, original=image, resized=resized)


def prepare_batch(images: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Build a (N, H, W, 3) float32 batch from BGR images."""
    if not images:
        return np.zeros((0, size[1], size[0], 3), dtype=np.float32)
    batch = np.stack([normalize(resize_image(img, (size[0], size[1]))) for img in images])
    return batch.astype(np.float32)
