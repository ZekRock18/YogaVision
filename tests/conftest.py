"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    from yogavision.config import Settings

    return Settings(PROJECT_ROOT / "configs" / "train.yaml")


@pytest.fixture
def blank_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_landmarks():
    """A (33, 3) landmark array with high visibility."""
    lm = np.zeros((33, 3), dtype=np.float32)
    lm[:, 2] = 0.9
    rng = np.random.default_rng(1)
    lm[:, :2] = rng.uniform(0.1, 0.9, size=(33, 2))
    return lm
