"""Evaluate a trained model on the held-out test split.

Usage:
    python scripts/evaluate.py --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yogavision.config import PROJECT_ROOT, Settings  # noqa: E402
from yogavision.evaluation import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    save_classification_report_text,
    save_metrics_json,
)
from yogavision.model import load_label_map, load_model  # noqa: E402
from yogavision.utils import SUPPORTED_IMAGE_EXTS, get_logger, set_seed  # noqa: E402

logger = get_logger()


def load_test_images(
    dataset_dir: Path,
    classes: list[str],
    size: tuple[int, int],
    val_frac: float,
    test_frac: float,
    seed: int,
):
    """Rebuild the exact train/val/test split used by scripts/train.py.

    The split must be reproduced identically or the test set overlaps the
    training set and the reported metrics are meaningless.
    """
    import cv2
    from sklearn.model_selection import train_test_split

    paths: list[Path] = []
    labels: list[int] = []
    for idx, cls in enumerate(classes):
        class_dir = dataset_dir / cls
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                paths.append(p)
                labels.append(idx)

    indices = np.arange(len(paths))
    train_idx, rest_idx = train_test_split(
        indices, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    relative = test_frac / (val_frac + test_frac)
    _, test_idx = train_test_split(
        rest_idx,
        test_size=relative,
        stratify=[labels[i] for i in rest_idx],
        random_state=seed,
    )

    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    images = []
    for p in test_paths:
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        images.append(img.astype(np.float32) / 255.0)
    return np.stack(images).astype(np.float32), test_labels, test_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--latency-runs", type=int, default=20)
    args = parser.parse_args()

    settings = Settings(PROJECT_ROOT / args.config)
    set_seed(int(settings.config.get("seed", 42)))
    settings.ensure_dirs()

    bundle = load_model(
        settings,
        model_path=args.model,
        input_representation=settings.config.get("model", {}).get(
            "input_representation", "skeleton"
        ),
    )
    logger.info("Loaded model %s with classes %s", bundle.model_path, bundle.labels)

    dataset_dir = settings.training_dataset_path
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset not found: {dataset_dir}")

    val_frac = float(settings.config.get("data", {}).get("validation_split", 0.2))
    test_frac = float(settings.config.get("data", {}).get("test_split", 0.1))
    X_test, y_true, test_paths = load_test_images(
        dataset_dir, bundle.labels, bundle.image_size, val_frac, test_frac,
        int(settings.config.get("seed", 42)),
    )
    logger.info("Test samples: %d", len(X_test))

    preds = bundle.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    metrics = compute_metrics(y_true, y_pred, bundle.labels)

    # Latency measurement on one representative frame, cast to the model's
    # dtype first so the slice does not pay a float64 -> float32 conversion
    # inside the timed region.
    latencies = []
    import time

    sample = np.ascontiguousarray(X_test[:1], dtype=np.float32)
    for _ in range(5):
        bundle.predict(sample)
    for _ in range(max(1, args.latency_runs)):
        start = time.perf_counter()
        bundle.predict(sample)
        latencies.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies)
    metrics["inference_latency_ms"] = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "approx_fps": float(1000.0 / arr.mean()) if arr.mean() else 0.0,
        "runs": int(arr.size),
        "note": "batch size 1, CPU, includes model.predict overhead",
    }
    metrics["provenance"] = "YogaVision reproduction run"

    save_metrics_json(metrics, settings.evaluation_dir / "evaluation_metrics.json")
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), bundle.labels,
        settings.plots_dir / "confusion_matrix.png",
        title="Test Confusion Matrix",
    )
    save_classification_report_text(
        metrics, settings.evaluation_dir / "classification_report.txt"
    )

    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "inference_latency_ms": metrics["inference_latency_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
