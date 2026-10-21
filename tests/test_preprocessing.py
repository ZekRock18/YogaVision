"""Tests for preprocessing module."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogavision.preprocessing import (
    load_image,
    load_image_from_bytes,
    normalize,
    resize_image,
    prepare_batch,
    PreprocessResult,
)


class TestNormalize:
    def test_scales_to_01(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        out = normalize(img)
        assert out.dtype == np.float32
        assert out.max() == 1.0
        assert out.min() == 1.0  # all 255 -> 255/255 = 1.0

    def test_half_value(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = normalize(img)
        assert out.max() == pytest.approx(0.502, abs=0.01)

    def test_zero_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = normalize(img)
        assert out.max() == 0.0


class TestResize:
    def test_upsample(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = resize_image(img, (20, 20))
        assert out.shape == (20, 20, 3)

    def test_downsample(self):
        img = np.full((100, 80, 3), 64, dtype=np.uint8)
        out = resize_image(img, (50, 40))
        assert out.shape == (40, 50, 3)


class TestLoadImageFromBytes:
    def test_valid_png(self):
        import cv2

        img = np.full((32, 32, 3), 120, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        loaded = load_image_from_bytes(buf.tobytes())
        assert loaded.shape == img.shape
        assert loaded.dtype == np.uint8

    def test_empty_buffer_raises(self):
        with pytest.raises(Exception):
            load_image_from_bytes(b"")


class TestPrepareBatch:
    def test_empty_list(self):
        batch = prepare_batch([], (75, 75))
        assert batch.shape == (0, 75, 75, 3)
        assert batch.dtype == np.float32

    def test_single_image(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        batch = prepare_batch([img], (75, 75))
        assert batch.shape == (1, 75, 75, 3)
        assert 0.78 <= batch[0].mean() <= 0.79

    def test_three_images(self):
        imgs = [np.full((50, 50, 3), v, dtype=np.uint8) for v in [10, 128, 255]]
        batch = prepare_batch(imgs, (75, 75))
        assert batch.shape == (3, 75, 75, 3)
