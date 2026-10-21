"""Configuration loading and path resolution for YogaVision."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical yoga pose classes (sorted alphabetically for deterministic label maps)
# Verified against the actual Kaggle dataset folder listing.
YOGA_CLASSES = [
    "downdog",
    "goddess",
    "plank",
    "tree",
    "warrior2",
]


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


class Settings:
    """Runtime configuration resolved from config file and environment."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config: dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            self.config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # -- path helpers -----------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # -- model ------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.resolve(os.getenv("MODEL_PATH", "models/yoga_cnn.keras"))

    @property
    def label_map_path(self) -> Path:
        return self.resolve(os.getenv("LABEL_MAP_PATH", "models/label_map.json"))

    @property
    def model_dir(self) -> Path:
        return self.resolve(self.config.get("model", {}).get("output_dir", "models"))

    @property
    def image_size(self) -> tuple[int, int]:
        size = self.config.get("data", {}).get("image_size", [75, 75])
        return int(size[0]), int(size[1])

    # -- dataset ----------------------------------------------------------
    @property
    def dataset_path(self) -> Path:
        return self.resolve(os.getenv("DATASET_PATH", "data/raw/DATASET"))

    @property
    def training_dataset_path(self) -> Path:
        """The dataset directory that directly contains class folders.

        Handles datasets whose root is a wrapper folder (e.g. DATASET/TRAIN/<class>).
        Prefers a TRAIN folder when one exists.
        """
        base = self.resolve(
            self.config.get("data", {}).get("dataset_dir", "data/processed/yoga_poses_skeleton")
        )

        def has_class_folders(path: Path) -> bool:
            if not path.is_dir():
                return False
            return any(
                p.is_dir()
                and any(
                    q.is_file() and q.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for q in p.iterdir()
                )
                for p in path.iterdir()
            )

        if has_class_folders(base):
            return base

        if not base.is_dir():
            return base
        # Prefer a TRAIN folder, then any other sub-folder containing classes
        children = [p for p in sorted(base.iterdir()) if p.is_dir()]
        for preferred in ("TRAIN", "train"):
            for child in children:
                if child.name.upper() == preferred.upper() and has_class_folders(child):
                    return child
        for child in children:
            if has_class_folders(child):
                return child
        return base

    @property
    def processed_dir(self) -> Path:
        return self.resolve("data/processed")

    # -- outputs ----------------------------------------------------------
    @property
    def outputs_dir(self) -> Path:
        return self.resolve("outputs")

    @property
    def evaluation_dir(self) -> Path:
        return self.outputs_dir / "evaluation"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots"

    @property
    def session_db_path(self) -> Path:
        return self.resolve(os.getenv("SESSION_DB", "outputs/sessions.db"))

    # -- inference --------------------------------------------------------
    @property
    def confidence_threshold(self) -> float:
        return float(self.config.get("inference", {}).get("confidence_threshold", 0.60))

    @property
    def smoothing_window(self) -> int:
        return int(self.config.get("inference", {}).get("smoothing_window", 5))

    # -- feedback ---------------------------------------------------------
    @property
    def feedback_thresholds(self) -> dict[str, Any]:
        return self.config.get("feedback", {})

    @property
    def mp_min_detection_confidence(self) -> float:
        return float(
            self.config.get("mediapipe", {}).get(
                "min_detection_confidence",
                os.getenv("MP_MIN_DETECTION_CONFIDENCE", 0.5),
            )
        )

    @property
    def mp_min_tracking_confidence(self) -> float:
        return float(
            self.config.get("mediapipe", {}).get(
                "min_tracking_confidence",
                os.getenv("MP_MIN_TRACKING_CONFIDENCE", 0.5),
            )
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.dataset_path,
            self.processed_dir,
            self.model_dir,
            self.outputs_dir,
            self.evaluation_dir,
            self.plots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def get_settings(config_path: str | Path | None = None) -> Settings:
    return Settings(config_path)
