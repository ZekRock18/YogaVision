"""Reproducible CNN training pipeline for yoga pose classification.

Faithful reproduction of the methodology in:
  Garg, Saxena & Gupta (2023), "Yoga pose classification: a CNN and MediaPipe
  inspired deep learning approach for real-world application",
  Journal of Ambient Intelligence and Humanized Computing,
  DOI 10.1007/s12652-022-03910-0

Usage:
    python scripts/train.py --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yogavision.config import Settings  # noqa: E402
from yogavision.evaluation import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_classification_report_text,
    save_metrics_json,
)
from yogavision.model import build_cnn, save_label_map, save_model  # noqa: E402
from yogavision.utils import SUPPORTED_IMAGE_EXTS, get_logger, set_seed  # noqa: E402

logger = get_logger()


def discover_classes(dataset_dir: Path) -> list[str]:
    """Return sorted class folder names."""
    classes = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {dataset_dir}")
    return classes


def list_samples(
    dataset_dir: Path, classes: list[str], limit_per_class: int | None = None
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for idx, cls in enumerate(classes):
        class_dir = dataset_dir / cls
        class_paths = [
            p for p in sorted(class_dir.rglob("*"))
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        if limit_per_class is not None:
            class_paths = class_paths[:limit_per_class]
        paths.extend(class_paths)
        labels.extend([idx] * len(class_paths))
    if not paths:
        raise SystemExit(f"No images found under {dataset_dir}")
    return paths, labels


def stratified_split(
    paths: list[Path],
    labels: list[int],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    """Split indices stratified by class, keeping samples from the same source
    image together is unnecessary here because each file is one sample. We split
    per class so no class is missing from any partition."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(paths))
    train_idx, rest_idx = train_test_split(
        indices, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    if test_frac > 0:
        relative = test_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=relative, stratify=[labels[i] for i in rest_idx],
            random_state=seed,
        )
    else:
        val_idx, test_idx = rest_idx, np.array([], dtype=int)

    def pick(idx_arr):
        return [paths[i] for i in idx_arr], [labels[i] for i in idx_arr]

    return (*pick(train_idx), *pick(val_idx), *pick(test_idx))


def load_images(
    paths: list[Path], size: tuple[int, int], *, max_samples: int | None = None
) -> np.ndarray:
    import cv2

    selected = paths if max_samples is None else paths[:max_samples]
    arrays = []
    for p in selected:
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        arrays.append(img.astype(np.float32) / 255.0)
    if not arrays:
        return np.zeros((0, size[1], size[0], 3), dtype=np.float32)
    return np.stack(arrays).astype(np.float32)


def build_model(settings: Settings, num_classes: int):
    size = settings.image_size
    model_cfg = settings.config.get("model", {})
    return build_cnn(
        num_classes,
        image_size=size,
        dropout=float(model_cfg.get("dropout", 0.25)),
        dense_dropout=float(model_cfg.get("dense_dropout", 0.5)),
    )


def train(settings: Settings) -> dict:
    import tensorflow as tf

    seed = int(settings.config.get("seed", 42))
    set_seed(seed)
    settings.ensure_dirs()

    dataset_dir = settings.training_dataset_path
    if not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run: python scripts/prepare_dataset.py\n"
            "See docs/dataset_setup.md"
        )

    classes = discover_classes(dataset_dir)
    logger.info("Classes: %s", classes)
    limit_per_class = settings.config.get("_limit_per_class")
    paths, labels = list_samples(dataset_dir, classes, limit_per_class)
    logger.info("Total samples: %d", len(paths))

    size = settings.image_size
    data_cfg = settings.config.get("data", {})
    val_frac = float(data_cfg.get("validation_split", 0.2))
    test_frac = float(data_cfg.get("test_split", 0.1))

    tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_split(
        paths, labels, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    logger.info(
        "Split: train=%d val=%d test=%d", len(tr_p), len(va_p), len(te_p)
    )

    X_train = load_images(tr_p, size)
    X_val = load_images(va_p, size)
    X_test = load_images(te_p, size)
    y_train = tf.keras.utils.to_categorical(tr_y, num_classes=len(classes))
    y_val = tf.keras.utils.to_categorical(va_y, num_classes=len(classes))

    model = build_model(settings, len(classes))
    train_cfg = settings.config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.001))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)

    out_dir = settings.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = settings.config.get("model", {}).get("model_name", "yoga_cnn.keras")
    model_path = out_dir / model_name
    checkpoint_path = out_dir / "checkpoints" / f"{Path(model_name).stem}.best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=int(train_cfg.get("early_stopping_patience", 8)),
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
            patience=int(train_cfg.get("reduce_lr_patience", 4)), verbose=1,
        ),
    ]

    aug_cfg = settings.config.get("augmentation", {})
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=float(aug_cfg.get("width_shift_range", 0.1)),
        horizontal_flip=bool(aug_cfg.get("horizontal_flip", True)),
    )

    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 16))

    start = time.perf_counter()
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size, seed=seed),
        validation_data=(X_val, y_val) if len(X_val) else None,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start

    save_model(model, model_path)
    save_label_map(classes, settings.label_map_path)
    logger.info("Saved model to %s and label map to %s", model_path, settings.label_map_path)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    results: dict = {
        "seed": seed,
        "classes": classes,
        "num_samples": {"train": len(tr_p), "val": len(va_p), "test": len(te_p)},
        "training_seconds": round(train_seconds, 2),
        "epochs_run": len(history.history.get("loss", [])),
        "history": history_dict,
        "provenance": "reproduction run using YogaVision training pipeline (not the paper's numbers)",
    }

    if len(X_test):
        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        metrics = compute_metrics(te_y, y_pred, classes)
        results["test_metrics"] = metrics
        logger.info(
            "Test accuracy: %.4f | macro F1: %.4f",
            metrics["accuracy"], metrics["f1_macro"],
        )
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), classes,
            settings.plots_dir / "confusion_matrix.png",
            title="Test Confusion Matrix (YogaVision reproduction run)",
        )
        save_classification_report_text(
            metrics, settings.evaluation_dir / "classification_report.txt"
        )

    plot_training_curves(history_dict, settings.plots_dir / "training_curves.png")
    save_metrics_json(results, settings.evaluation_dir / "training_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the yoga pose CNN")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per class (smoke test)")
    args = parser.parse_args()

    from yogavision.config import PROJECT_ROOT

    settings = Settings(args.config if Path(args.config).is_absolute() else PROJECT_ROOT / args.config)
    if args.epochs is not None:
        settings.config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        settings.config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.limit is not None:
        settings.config["_limit_per_class"] = args.limit

    results = train(settings)
    logger.info("Training finished. Metrics written to outputs/evaluation/")
    print(json.dumps({
        "epochs_run": results.get("epochs_run"),
        "num_samples": results.get("num_samples"),
        "test_accuracy": results.get("test_metrics", {}).get("accuracy"),
        "test_f1_macro": results.get("test_metrics", {}).get("f1_macro"),
    }, indent=2))


if __name__ == "__main__":
    main()
