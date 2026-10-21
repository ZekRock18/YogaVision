# Dataset Setup

## Source

| Field | Value |
|---|---|
| Dataset | [niharika41298/yoga-poses-dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset) |
| Used by | Garg, Saxena & Gupta (2023), DOI [10.1007/s12652-022-03910-0](https://doi.org/10.1007/s12652-022-03910-0) |
| Classes | 5 — `downdog`, `goddess`, `plank`, `tree`, `warrior2` |
| Structure | `DATASET/TRAIN/<class>/*.jpg`, `DATASET/TEST/<class>/*.jpg` |
| Licence | No explicit licence stated on the dataset page; used here for research/reproduction only |
| Redistribution | Not redistributed by this project |

## Verified counts

From the download used for this project (2026-10-14):

| Class | TRAIN | TEST | Total |
|---|---:|---:|---:|
| downdog | 223 | 96 | 319 |
| goddess | 180 | 80 | 260 |
| plank | 266 | 115 | 381 |
| tree | 160 | 69 | 229 |
| warrior2 | 252 | 109 | 361 |
| **Total** | **1081** | **469** | **1550** |

`TRAIN/tree/00000114.jpg` is a truncated JPEG and was removed, leaving 1080 TRAIN images.

Class imbalance ratio: **1.66** (160 tree vs 266 plank). 179 duplicate image
groups were found by MD5 hashing, so the effective sample count is lower than
1550.

## Download

### Option 1 — anonymous download (worked during this build)

The Kaggle download endpoint accepted an unauthenticated request. The archive is
roughly 287 MB.

```powershell
python scripts/download_assets.py
```

### Option 2 — with Kaggle API credentials

1. Go to <https://www.kaggle.com> → your account picture → **Settings** →
   **Create New API Token**. A `kaggle.json` file downloads.
2. Save it as `%USERPROFILE%\.kaggle\kaggle.json`, or set environment variables:

   ```powershell
   $env:KAGGLE_USERNAME = "your-username"
   $env:KAGGLE_KEY = "your-key"
   ```

3. Run:

   ```powershell
   python scripts/download_assets.py --target data/raw
   ```

If credentials are missing, `download_assets.py` exits with an error explaining
exactly where to put the token.

### Option 3 — manual

1. Download the zip from the Kaggle page in a browser.
2. Place it at `data/raw/yoga-poses-dataset.zip`.
3. Run `python scripts/download_assets.py --target data/raw` (it will detect the
   existing archive and skip re-downloading) or extract manually so that
   `data/raw/DATASET/TRAIN/<class>/` exists.

## Directory layout after setup

```text
data/
├── raw/
│   └── DATASET/
│       ├── TRAIN/
│       │   ├── downdog/*.jpg
│       │   ├── goddess/*.jpg
│       │   ├── plank/*.jpg
│       │   ├── tree/*.jpg
│       │   └── warrior2/*.jpg
│       └── TEST/
│           └── <same 5 classes>
└── processed/
    ├── yoga_poses_skeleton/     # MediaPipe skeleton images
    │   ├── TRAIN/<class>/*.jpg
    │   └── TEST/<class>/*.jpg
    └── dataset_manifest.json
```

## Prepare the model input

The paper trains the CNN on **skeletonized** images, not raw photographs. To
reproduce that:

```powershell
python scripts/prepare_dataset.py --raw data/raw/DATASET --out data/processed/yoga_poses_skeleton
```

This runs MediaPipe Pose over every image and writes a black canvas with the
pose skeleton rendered in white. Expected output on this dataset:

```
Skeletonization complete: 1397 with person, 153 without person, 0 failed, 0 skipped
```

153 images (~10%) produced no detectable pose. They become all-black images,
exactly as in the original repository's script. This is a genuine property of
the data, not a bug.

To train on raw RGB instead (the paper's "Non-MediaPipe" variant):

```powershell
python scripts/prepare_dataset.py --raw data/raw/DATASET --out data/processed/yoga_poses_rgb --representation rgb
```

then set `data.input_representation: rgb` in `configs/train.yaml`.

## Inspect

```powershell
python scripts/inspect_dataset.py data/raw/DATASET/TRAIN --duplicates
python scripts/inspect_dataset.py data/raw/DATASET/TRAIN --json > outputs/dataset_report.json
```

Reports image counts, per-class counts, corrupt files, shape distribution,
imbalance ratio and (with `--duplicates`) MD5 duplicate groups.

## Train

```powershell
python scripts/train.py --config configs/train.yaml
```

Configurable in `configs/train.yaml`: `image_size`, `batch_size`, `epochs`,
`learning_rate`, splits, augmentation, dropout, seed and output directories.

Quick smoke test with a small subset:

```powershell
python scripts/train.py --config configs/train.yaml --epochs 2 --limit 30
```

## Evaluate

```powershell
python scripts/evaluate.py --config configs/train.yaml
```

## Troubleshooting

**`Dataset directory not found`** — the raw dataset was never downloaded. Run
`python scripts/download_assets.py`.

**Skeletonization produces mostly black images** — MediaPipe found no poses.
Check that the images are valid photographs and that `mediapipe` is installed
(`python -c "import mediapipe; print(mediapipe.__version__)"`).

**`No class folders found`** — the dataset layout is unexpected. The script
expects `.../TRAIN/<class>/image.jpg`. Inspect with `python scripts/inspect_dataset.py <path>`.

**Slow skeletonization** — 1550 images take roughly 55 seconds on this machine.
It is single-threaded because MediaPipe holds process-wide state.
