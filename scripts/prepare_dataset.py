"""Prepare the dataset: skeletonize raw photos into the CNN input format.

Steps:
  1. Locate the raw dataset (TRAIN/TEST folders of class sub-directories).
  2. Run MediaPipe skeletonization on every image.
  3. Write skeleton images preserving the class folder structure.
  4. Write a dataset manifest (JSON) with class counts.

Usage:
    python scripts/prepare_dataset.py --raw data/raw/DATASET --out data/processed/yoga_poses_skeleton
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yogavision.config import PROJECT_ROOT  # noqa: E402
from yogavision.skeletonization import skeletonize_dataset  # noqa: E402
from yogavision.utils import SUPPORTED_IMAGE_EXTS, get_logger  # noqa: E402

logger = get_logger()


def find_dataset_root(raw_root: Path) -> Path:
    """Find the directory that directly contains class folders."""
    for candidate in (raw_root, raw_root / "DATASET", raw_root / "DATASET" / "TRAIN"):
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
            return candidate
    return raw_root


def build_manifest(source_root: Path, manifest_path: Path) -> dict:
    """Walk into one level of sub-folders (TRAIN/TEST) and tally per class."""
    classes: dict[str, int] = {}
    for split_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = len(
                [q for q in class_dir.rglob("*") if q.suffix.lower() in SUPPORTED_IMAGE_EXTS]
            )
            if n == 0:
                continue
            classes[class_dir.name] = classes.get(class_dir.name, 0) + n
    manifest = {
        "source": str(source_root),
        "num_classes": len(classes),
        "classes": classes,
        "total_images": sum(classes.values()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeletonize the raw dataset")
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "raw" / "DATASET"))
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "data" / "processed" / "yoga_poses_skeleton")
    )
    parser.add_argument(
        "--representation",
        choices=["skeleton", "rgb"],
        default="skeleton",
        help="'skeleton' reproduces the paper; 'rgb' copies the raw photos.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw)
    if not raw_root.is_dir():
        raise SystemExit(
            f"Raw dataset not found at {raw_root}.\n"
            "Download it first:\n"
            "  python scripts/download_assets.py\n"
            "See docs/dataset_setup.md for details."
        )

    source_root = find_dataset_root(raw_root)
    logger.info("Using dataset root: %s", source_root)

    out_root = Path(args.out)
    if args.representation == "rgb":
        import shutil

        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(source_root, out_root)
        logger.info("Copied raw images to %s", out_root)
    else:
        logger.info("Skeletonizing images (this may take a while)...")

        def progress(done: int, total: int) -> None:
            if total and done % 100 == 0:
                logger.info("  %d / %d", done, total)

        stats = skeletonize_dataset(source_root, out_root, progress_callback=progress)
        logger.info(
            "Skeletonization complete: %d with person, %d without person, %d failed, %d skipped",
            stats.processed, stats.no_person, stats.failed, stats.skipped,
        )

    manifest = build_manifest(out_root, out_root.parent / "dataset_manifest.json")
    logger.info("Manifest: %s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
