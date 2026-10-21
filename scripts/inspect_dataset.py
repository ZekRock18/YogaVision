"""Inspect a dataset folder and report class distribution and data quality."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yogavision.utils import (  # noqa: E402
    SUPPORTED_IMAGE_EXTS,
    get_logger,
    human_readable_size,
)

logger = get_logger()


def inspect_dataset(root: str | Path, *, check_duplicates: bool = False) -> dict:
    import cv2

    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")

    report: dict = {"root": str(root), "classes": {}, "total_images": 0,
                    "corrupt": [], "non_image_files": 0}

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        others = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() not in SUPPORTED_IMAGE_EXTS]
        report["non_image_files"] += len(others)
        shapes: Counter = Counter()
        corrupt: list[str] = []
        for img in images:
            data = img.read_bytes()
            arr = cv2.imdecode(__import__("numpy").frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
            if arr is None:
                corrupt.append(str(img))
                continue
            shapes[arr.shape[:2]] += 1
        report["classes"][class_dir.name] = {
            "images": len(images),
            "corrupt": len(corrupt),
            "shapes": {f"{h}x{w}": c for (h, w), c in shapes.most_common(5)},
            "size_bytes": sum(p.stat().st_size for p in images if p.exists()),
        }
        report["total_images"] += len(images)
        report["corrupt"].extend(corrupt)

    counts = [v["images"] for v in report["classes"].values()]
    if counts:
        report["num_classes"] = len(counts)
        report["min_per_class"] = min(counts)
        report["max_per_class"] = max(counts)
        report["imbalance_ratio"] = round(max(counts) / max(1, min(counts)), 3)
        report["total_size"] = human_readable_size(
            sum(v["size_bytes"] for v in report["classes"].values())
        )

    if check_duplicates:
        import hashlib

        hashes: dict[str, list[str]] = {}
        for img in root.rglob("*"):
            if img.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                h = hashlib.md5(img.read_bytes()).hexdigest()
                hashes.setdefault(h, []).append(str(img))
        report["duplicate_groups"] = {h: p for h, p in hashes.items() if len(p) > 1}

    return report


def print_report(report: dict) -> None:
    print("=" * 70)
    print(f"Dataset inspection: {report['root']}")
    print("=" * 70)
    print(f"Classes          : {report.get('num_classes', 0)}")
    print(f"Total images     : {report.get('total_images', 0)}")
    print(f"Total size       : {report.get('total_size', 'n/a')}")
    print(f"Imbalance ratio  : {report.get('imbalance_ratio', 'n/a')}")
    print(f"Corrupt files    : {len(report.get('corrupt', []))}")
    print("-" * 70)
    print(f"{'class':<20}{'images':>10}{'corrupt':>10}   shapes")
    for name, info in report["classes"].items():
        shapes = ", ".join(f"{k}({v})" for k, v in list(info["shapes"].items())[:2])
        print(f"{name:<20}{info['images']:>10}{info['corrupt']:>10}   {shapes}")
    if report.get("duplicate_groups"):
        print("-" * 70)
        print(f"Duplicate groups : {len(report['duplicate_groups'])}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a yoga pose dataset folder")
    parser.add_argument("root", nargs="?", default="data/raw/DATASET/TRAIN")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument("--duplicates", action="store_true", help="Hash images to find duplicates")
    args = parser.parse_args()

    report = inspect_dataset(args.root, check_duplicates=args.duplicates)
    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
