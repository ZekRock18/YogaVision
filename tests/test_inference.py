"""Tests for inference module."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogavision.inference import PredictionSmoother, label_from_probabilities
from yogavision.model import ModelBundle, ModelNotAvailableError


class TestPredictionSmoother:
    def test_majority_vote(self):
        smoother = PredictionSmoother(window=5)
        smoother.update("tree")
        smoother.update("tree")
        smoother.update("cobra")
        smoother.update("tree")
        smoother.update("cobra")
        # tree appears 3 times out of 5
        assert smoother.update("tree") == "tree"

    def test_single_label(self):
        smoother = PredictionSmoother(window=3)
        for _ in range(5):
            smoother.update("warrior")
        assert smoother.update("warrior") == "warrior"

    def test_reset(self):
        smoother = PredictionSmoother(window=3)
        smoother.update("chair")
        smoother.reset()
        assert smoother.update("cobra") == "cobra"

    def test_window_1(self):
        smoother = PredictionSmoother(window=1)
        assert smoother.update("warrior") == "warrior"


class TestLabelFromProbabilities:
    def test_basic(self):
        probs = np.array([0.1, 0.6, 0.2, 0.05, 0.05])
        labels = ["chair", "cobra", "downdog", "tree", "warrior"]
        label, conf = label_from_probabilities(probs, labels)
        assert label == "cobra"
        assert conf == pytest.approx(0.6)

    def test_empty_probs(self):
        label, conf = label_from_probabilities(np.array([]), ["a", "b"])
        assert label == "unknown"
        assert conf == 0.0

    def test_threshold(self):
        probs = np.array([0.05, 0.05, 0.05, 0.05, 0.8])
        labels = ["a", "b", "c", "d", "e"]
        label, conf = label_from_probabilities(probs, labels, threshold=0.9)
        # Should still return label but above_threshold check is separate


class TestModelNotAvailableError:
    def test_message(self):
        err = ModelNotAvailableError("model not found")
        assert "model not found" in str(err)
