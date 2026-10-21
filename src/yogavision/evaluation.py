"""Evaluation metrics, confusion matrix and training-curve plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .utils import get_logger

logger = get_logger()


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    labels: list[str],
) -> dict[str, Any]:
    """Compute accuracy, precision, recall and F1 scores."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        ),
        "num_samples": int(len(y_true)),
    }
    return metrics


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        np.asarray(confusion),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "accuracy" in history:
        axes[0].plot(history["accuracy"], label="Train")
    if "val_accuracy" in history:
        axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left")

    if "loss" in history:
        axes[1].plot(history["loss"], label="Train")
    if "val_loss" in history:
        axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    out.write_text(json.dumps(metrics, indent=2, default=_default), encoding="utf-8")
    return out


def save_classification_report_text(metrics: dict[str, Any], output_path: str | Path) -> Path:
    from sklearn.metrics import classification_report

    report = metrics.get("classification_report", {})
    lines = ["Per-class classification report", "=" * 40, ""]
    header = f"{'class':<15}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines.append(header)
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        lines.append(
            f"{key:<15}{value.get('precision', 0):>10.3f}{value.get('recall', 0):>10.3f}"
            f"{value.get('f1-score', 0):>10.3f}{value.get('support', 0):>10}"
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def measure_latency(classifier, image: np.ndarray, runs: int = 20) -> dict[str, float]:
    """Measure inference latency statistics over repeated calls."""
    import time

    latencies = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        classifier.predict_frame(image, smooth=False)
        latencies.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "runs": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "approx_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def load_metrics_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
