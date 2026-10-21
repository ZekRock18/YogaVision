"""Skeletonization: convert RGB photos into MediaPipe skeleton images.

This reproduces the preprocessing described in the original paper
(Garg, Saxena & Gupta, 2023 — DOI 10.1007/s12652-022-03910-0), where the CNN
is trained on skeletonized images rather than raw photographs.

The original repository's ``skeletonization.py`` draws only landmark dots.
Here we render the full pose connection graph, which is the representation the
paper reports and which yields a more informative input for the CNN.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .pose_detection import PoseDetector
from .utils import SUPPORTED_IMAGE_EXTS, get_logger

logger = get_logger()


@dataclass
class SkeletonizeStats:
    processed: int = 0
    skipped: int = 0
    no_person: int = 0
    failed: int = 0


def skeletonize_image(
    image_bgr: np.ndarray,
    detector: PoseDetector | None = None,
    *,
    draw_connections: bool = True,
) -> tuple[np.ndarray, bool]:
    """Return (skeleton_image, person_detected) for one BGR image."""
    detector = detector or PoseDetector.get_instance()
    result = detector.detect(image_bgr)
    if not result.detected:
        blank = np.zeros_like(image_bgr)
        return blank, False
    if draw_connections:
        canvas = detector.draw_skeleton(
            np.zeros_like(image_bgr), result, thickness=2, radius=3
        )
    else:
        canvas = detector.draw_skeleton(
            np.zeros_like(image_bgr), result, thickness=-1, radius=4
        )
    return canvas, True


def skeletonize_dataset(
    source_root: str | Path,
    target_root: str | Path,
    *,
    settings: Settings | None = None,
    progress_callback=None,
) -> SkeletonizeStats:
    """Skeletonize a directory tree of ``class_name/image.jpg`` folders.

    The class folder structure is preserved so the result can be fed straight
    into ``tf.keras.utils.image_dataset_from_directory``.
    """
    import cv2

    settings = settings or get_settings()
    source = Path(source_root)
    target = Path(target_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")

    detector = PoseDetector.get_instance(settings)
    stats = SkeletonizeStats()

    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        out_class = target / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)
        images = [p for p in sorted(class_dir.rglob("*")) if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        for img_path in images:
            rel = img_path.relative_to(class_dir)
            out_path = out_class / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = np.fromfile(img_path, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    stats.failed += 1
                    logger.warning("Unreadable image skipped: %s", img_path)
                    continue
                skeleton, detected = skeletonize_image(image, detector)
                ok = cv2.imwrite(str(out_path), skeleton)
                if not ok:
                    stats.failed += 1
                    continue
                if detected:
                    stats.processed += 1
                else:
                    stats.no_person += 1
            except Exception as exc:
                stats.failed += 1
                logger.warning("Failed on %s: %s", img_path, exc)
            if progress_callback:
                progress_callback(stats.processed + stats.no_person, len(images))
    detector.reset()
    return stats


def skeletonize_single_file(
    input_path: str | Path, output_path: str | Path
) -> tuple[Path, bool]:
    """Skeletonize one image file and write the result."""
    import cv2

    from .preprocessing import load_image

    image = load_image(input_path)
    skeleton, detected = skeletonize_image(image)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), skeleton)
    return out, detected


def copy_non_image_files(source_root: str | Path, target_root: str | Path) -> int:
    """Copy sidecar files (e.g. licence, metadata) that are not images."""
    source = Path(source_root)
    target = Path(target_root)
    copied = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            rel = path.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied += 1
    return copied
