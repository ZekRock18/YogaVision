"""Tests for model loading, asset validation and missing-asset handling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yogavision.model import (
    ModelNotAvailableError,
    build_cnn,
    load_label_map,
    load_model,
    model_status,
    save_label_map,
    save_model,
)


class TestBuildCNN:
    def test_output_shape(self):
        model = build_cnn(5, (75, 75))
        assert model.input_shape == (None, 75, 75, 3)
        assert model.output_shape == (None, 5)

    def test_different_class_count(self):
        model = build_cnn(3, (75, 75))
        assert model.output_shape == (None, 3)

    def test_param_count_reasonable(self):
        model = build_cnn(5, (75, 75))
        total = int(sum(np.prod(w.shape) for w in model.get_weights()))
        # The paper's model has ~21.6M parameters
        assert 15_000_000 < total < 30_000_000, f"unexpected param count {total}"


class TestLabelMap:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "label_map.json"
        save_label_map(["a", "b", "c"], path)
        assert path.exists()
        loaded = load_label_map(path)
        assert loaded == ["a", "b", "c"]

    def test_missing_label_map_raises(self, tmp_path):
        with pytest.raises(ModelNotAvailableError) as exc:
            load_label_map(tmp_path / "nope.json")
        assert "not found" in str(exc.value).lower()

    def test_legacy_index_to_label_format(self, tmp_path):
        path = tmp_path / "lm.json"
        path.write_text(json.dumps({"index_to_label": {"0": "x", "1": "y"}}))
        assert load_label_map(path) == ["x", "y"]


class TestModelSaveLoad:
    def test_roundtrip(self, tmp_path):
        model = build_cnn(5, (75, 75))
        path = tmp_path / "m.keras"
        save_model(model, path)
        assert path.exists()

        label_path = tmp_path / "lm.json"
        save_label_map(["a", "b", "c", "d", "e"], label_path)
        bundle = load_model(model_path=path, label_map_path=label_path)
        assert bundle.labels == ["a", "b", "c", "d", "e"]
        assert bundle.image_size == (75, 75)
        assert bundle.model_path == path

    def test_class_count_mismatch_raises(self, tmp_path):
        model = build_cnn(5, (75, 75))
        model_path = tmp_path / "m.keras"
        save_model(model, model_path)
        label_path = tmp_path / "lm.json"
        save_label_map(["only", "two"], label_path)  # 2 labels vs 5 outputs
        with pytest.raises(ModelNotAvailableError) as exc:
            load_model(model_path=model_path, label_map_path=label_path)
        assert "not compatible" in str(exc.value).lower()

    def test_missing_model_raises_actionable_error(self, tmp_path):
        with pytest.raises(ModelNotAvailableError) as exc:
            load_model(
                model_path=tmp_path / "absent.keras",
                label_map_path=tmp_path / "lm.json",
            )
        msg = str(exc.value)
        assert "not found" in msg.lower()
        assert "train" in msg.lower()


class TestModelStatus:
    def test_reports_missing(self, tmp_path):
        status = model_status(
            type("S", (), {"model_path": tmp_path / "a.keras",
                           "label_map_path": tmp_path / "b.json"})()
        )
        assert status["model_exists"] is False
        assert status["label_map_exists"] is False
        assert status["model_size_mb"] is None

    def test_reports_present(self, tmp_path):
        m = tmp_path / "a.keras"
        m.write_bytes(b"x" * 2048)
        status = model_status(
            type("S", (), {"model_path": m, "label_map_path": tmp_path / "b.json"})()
        )
        assert status["model_exists"] is True
        assert status["model_size_mb"] is not None
        assert status["model_size_mb"] >= 0  # 2 KB rounds to 0 MB
