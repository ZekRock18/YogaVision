"""Shared utilities: logging, seed control, image I/O, formatting."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_LOGGER_NAME = "yogavision"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - TF optional at import time
        pass


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTS


def is_supported_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTS


def human_readable_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def chunked(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def top_k(probabilities: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Return indices of the k highest probabilities with their values."""
    if probabilities.size == 0:
        return []
    k = min(k, probabilities.size)
    idx = np.argpartition(probabilities, -k)[-k:]
    idx = idx[np.argsort(probabilities[idx])[::-1]]
    return [(int(i), float(probabilities[i])) for i in idx]
