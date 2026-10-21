"""CNN model definition, persistence and loading (YogaConvo2d)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .utils import get_logger

logger = get_logger()


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained model or label map is missing/incompatible."""


@dataclass
class ModelBundle:
    """A loaded model plus its label mapping and metadata."""

    model: Any
    labels: list[str]
    image_size: tuple[int, int]
    input_representation: str
    model_path: Path

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return class probabilities for a (N, H, W, 3) batch."""
        preds = self.model.predict(batch, verbose=0)
        return np.asarray(preds, dtype=np.float32)

    def predict_one(self, image: np.ndarray) -> np.ndarray:
        """Return class probabilities for a single (H, W, 3) image."""
        batch = np.expand_dims(image, axis=0).astype(np.float32)
        return self.predict(batch)[0]


def build_cnn(
    num_classes: int,
    image_size: tuple[int, int] = (75, 75),
    *,
    dropout: float = 0.25,
    dense_dropout: float = 0.5,
) -> Any:
    """Build the YogaConvo2d architecture from the original paper.

    Faithful reproduction: four Conv2D blocks (64/128/128/256) with ReLU and
    'Same' padding, 2x2 max pooling, dropout after each block, a 1024-unit
    dense head, and a softmax classifier over ``num_classes``.
    """
    import tensorflow as tf

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size[1], image_size[0], 3)),
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="Same"
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="Same"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="Same"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(dense_dropout),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def save_label_map(labels: list[str], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": list(labels),
        "num_classes": len(labels),
        "index_to_label": {str(i): lab for i, lab in enumerate(labels)},
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_label_map(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise ModelNotAvailableError(
            f"Label map not found: {p}. Train a model first (see README) or run "
            "python scripts/train.py --config configs/train.yaml"
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not labels:
        mapping = payload.get("index_to_label", {})
        labels = [mapping[str(i)] for i in range(len(mapping))]
    if not labels:
        raise ModelNotAvailableError(f"Label map at {p} contains no labels")
    return list(labels)


def save_model(model: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".h5", ".hdf5"}:
        model.save(p, save_format="h5")
    else:
        model.save(p)
    return p


def load_model(
    settings: Settings | None = None,
    *,
    model_path: str | Path | None = None,
    label_map_path: str | Path | None = None,
    input_representation: str = "rgb",
) -> ModelBundle:
    """Load the trained model and label map, validating compatibility.

    Raises ModelNotAvailableError with actionable guidance when assets are
    missing or the input shape / class count do not match.
    """
    settings = settings or get_settings()
    mp = Path(model_path) if model_path else settings.model_path
    lp = Path(label_map_path) if label_map_path else settings.label_map_path

    if not mp.exists():
        raise ModelNotAvailableError(
            f"Trained model not found at {mp}.\n"
            "Run training first:\n"
            "  python scripts/train.py --config configs/train.yaml\n"
            "or see docs/dataset_setup.md to obtain the dataset."
        )

    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(mp, compile=False)
    except Exception as exc:
        raise ModelNotAvailableError(f"Could not load model from {mp}: {exc}") from exc

    labels = load_label_map(lp)
    expected_size = settings.image_size

    input_shape = getattr(model, "input_shape", None)
    if input_shape:
        shape = list(input_shape)
        if len(shape) == 4:
            _, ih, iw, _ = shape
            if ih is not None and (int(iw), int(ih)) != tuple(expected_size):
                logger.warning(
                    "Model input shape %s differs from configured image size %s",
                    (iw, ih), tuple(expected_size),
                )
                expected_size = (int(iw), int(ih))

    output_shape = getattr(model, "output_shape", None)
    if output_shape:
        out_dim = output_shape[-1]
        if out_dim != len(labels):
            raise ModelNotAvailableError(
                f"Model outputs {out_dim} classes but label map has {len(labels)} "
                f"labels ({lp}). The model and label map are not compatible."
            )

    return ModelBundle(
        model=model,
        labels=labels,
        image_size=expected_size,
        input_representation=input_representation,
        model_path=mp,
    )


def model_status(settings: Settings | None = None) -> dict[str, Any]:
    """Report whether model assets are present (used by the UI)."""
    settings = settings or get_settings()
    mp = settings.model_path
    lp = settings.label_map_path
    return {
        "model_exists": mp.exists(),
        "label_map_exists": lp.exists(),
        "model_path": mp,
        "label_map_path": lp,
        "model_size_mb": round(mp.stat().st_size / 1024 / 1024, 2) if mp.exists() else None,
    }
