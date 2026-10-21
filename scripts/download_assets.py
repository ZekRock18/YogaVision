"""Download the research dataset from Kaggle.

Dataset: niharika41298/yoga-poses-dataset (5 classes, TRAIN/TEST folders).
License: see the dataset page; the original paper's repository points here.

Requires Kaggle API credentials: place kaggle.json in %USERPROFILE%\.kaggle\
(or set KAGGLE_USERNAME / KAGGLE_KEY). The Kaggle CLI is optional; this script
uses the public API directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yogavision.config import PROJECT_ROOT, get_settings  # noqa: E402
from yogavision.utils import get_logger  # noqa: E402

logger = get_logger()

DATASET_SLUG = "niharika41298/yoga-poses-dataset"
DATASET_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"


def _credentials() -> tuple[str, str]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        import json

        payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
        return payload["username"], payload["key"]
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if user and key:
        return user, key
    raise SystemExit(
        "Kaggle credentials not found.\n"
        "Create a Kaggle API token (kaggle.com -> Account -> Create New API Token),\n"
        f"save it as {kaggle_json}, or set KAGGLE_USERNAME and KAGGLE_KEY."
    )


def download_dataset(target_dir: str | Path, *, force: bool = False) -> Path:
    """Download and extract the dataset zip into ``target_dir``."""
    import urllib.request

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "yoga-poses-dataset.zip"

    if zip_path.exists() and not force:
        logger.info("Archive already present: %s", zip_path)
    else:
        logger.info("Downloading %s", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        logger.info("Downloaded to %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    extract_dir = target / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    logger.info("Extracted to %s", extract_dir)

    # The zip contains a top-level DATASET/TRAIN|TEST folder
    dataset_root = extract_dir / "DATASET"
    if not dataset_root.exists():
        candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
        dataset_root = candidates[0] if candidates else extract_dir
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the yoga pose dataset")
    parser.add_argument(
        "--target", default=str(PROJECT_ROOT / "data" / "raw"), help="Download destination"
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing archive")
    args = parser.parse_args()

    _credentials()
    root = download_dataset(args.target, force=args.force)
    logger.info("Dataset root: %s", root)
    for sub in sorted(p for p in root.rglob("*") if p.is_dir()):
        count = len([q for q in sub.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if count:
            logger.info("  %s : %d images", sub.relative_to(root), count)


if __name__ == "__main__":
    main()
