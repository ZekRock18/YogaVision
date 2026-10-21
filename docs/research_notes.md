# Research Notes

**Project:** YogaVision — Real-Time Yoga Pose Classification and Feedback System
**Date compiled:** 2026-10-14

## 1. Paper citation

> Garg, S., Saxena, A., & Gupta, R. (2023). Yoga pose classification: a CNN and
> MediaPipe inspired deep learning approach for real-world application.
> *Journal of Ambient Intelligence and Humanized Computing*, 14(12), 16551–16562.
> Springer. DOI: [10.1007/s12652-022-03910-0](https://doi.org/10.1007/s12652-022-03910-0)

Published online 2022; assigned to the 2023 volume.

## 2. Original implementation

- Repository: <https://github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization>
- Clone URL: `https://github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization.git`
- Licence: MIT (Copyright (c) 2021 SHUBHAM GARG)

Files present in the repository:

| File | Purpose |
|---|---|
| `cnn-yoga.ipynb` | Training notebook — data loading, model definition, training, plots |
| `skeletonization.py` | Standalone script that converts photos to landmark-only images |
| `script.sh` | Setup script: install deps, download dataset via Kaggle API |
| `models/*.py` | Transfer-learning model definitions (VGG16, InceptionV3, InceptionResNetV2, DenseNet, EfficientNet, MobileNetV2, ResNet50) |
| `loss-metrics/*.py` | Metric helper scripts (accuracy, precision, recall, F1, confusion matrix heatmap) |
| `Images/` | Result figures (skel.png, plot.png, confusion.png) |

## 3. Original methodology

The repository's notebook is the authoritative description of the method actually
implemented. Reconstructed from `cnn-yoga.ipynb`:

1. **Dataset** — a directory tree with one sub-folder per class. The README links
   <https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset> (5 classes).
2. **Loading** — `ImageDataGenerator.flow_from_directory` with:
   - `target_size = (75, 75)`
   - `color_mode = 'rgb'`
   - `class_mode = 'categorical'`
   - `batch_size = 16`
   - `rescale = 1./255`
   - `validation_split = 0.2`
3. **Augmentation** — `width_shift_range=0.1`, `horizontal_flip=True`.
4. **Model (YogaConvo2d)** — a `Sequential` CNN:
   - Conv2D(64, 3×3, ReLU, 'Same') → MaxPool2D(2,2) → Dropout(0.25)
   - Conv2D(128, 3×3, ReLU, 'Same') → MaxPool2D(2,2) → Dropout(0.25)
   - Conv2D(256, 3×3, ReLU, 'Same') → MaxPool2D(2,2) → Dropout(0.25)
   - Flatten → Dense(1024, ReLU) → Dropout(0.5) → Dense(5, softmax)
   - Input shape `(75, 75, 3)`.
   - Two additional Conv blocks are present but commented out in the notebook.
5. **Optimisation** — Adam with `learning_rate=0.001`, `loss='categorical_crossentropy'`,
   `metrics=['accuracy']`, `epochs=50`, `batch_size=16`.
6. **Skeletonization** — a separate `skeletonization.py` produces landmark-only
   images with MediaPipe Pose. The notebook is run twice in the paper's
   experiments: once on raw RGB and once on skeletonized images.

### Skeletonization detail

The repository's `skeletonization.py` initialises
`mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)`, converts to
RGB, processes the image, creates a **black canvas of the original size**, and
draws a filled white circle (`radius=5`) at every detected landmark. If no
landmarks are found the output is a completely black image.

Note: the repository script draws **only landmark dots, not the connection
lines**. The paper's figures show full stick figures. YogaVision renders the full
pose connection graph because that is the representation shown in the paper's
figures and it preserves more geometric structure. This is documented as a
deviation in `docs/methodology.md`.

## 4. Dataset

- **Source used by the paper:** <https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset>
- **Structure:** `DATASET/TRAIN/<class>/*.jpg` and `DATASET/TEST/<class>/*.jpg`
- **Classes (5), verified against the actual download:** `downdog`, `goddess`,
  `plank`, `tree`, `warrior2`
  (the notebook's final `Dense(5)` agrees on 5 classes; the folder listing is
  the authority on the names — note these differ from the class names one might
  infer from the paper's prose)
- **Verified counts (this project's download, 2026-10-14):**

| Class | TRAIN | TEST |
|---|---|---|
| downdog | 223 | 96 |
| goddess | 180 | 80 |
| plank | 266 | 115 |
| tree | 160 | 69 |
| warrior2 | 252 | 109 |
| **total** | **1081** | **469** |

- **Licence:** the Kaggle dataset page does not state an explicit licence; it is
  used here for research/reproduction purposes only. YogaVision does not
  redistribute it.
- **Access:** the Kaggle download endpoint accepted an anonymous request during
  this build, so the dataset was obtained directly and verified. The download is
  still gated on Kaggle API credentials for the official `kaggle` CLI path, and
  `scripts/download_assets.py` reports a clear error when those are absent.
- **Data quality findings:** 1 corrupt JPEG in `TRAIN/tree/00000114.jpg`
  (truncated, removed); 179 duplicate image groups detected by MD5 hash;
  class imbalance ratio 1.66 (min 160 tree, max 266 plank).

## 5. Reported results (original paper / repository README)

These are the **authors'** numbers, reproduced verbatim from the repository
README results table. They are **not** YogaVision's results.

| Model | Val Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YogaConvo2d (MediaPipe) | 99.62% | 97.09% | 0.97 | 0.97 | 0.97 |
| YogaConvo2d (Non-MediaPipe) | 89.97% | 89.36% | 0.89 | 0.88 | 0.89 |
| InceptionV3 (MediaPipe) | 95.09% | 94.39% | 0.95 | 0.94 | 0.94 |
| InceptionV3 (Non-MediaPipe) | 94.98% | 94.04% | 0.95 | 0.94 | 0.94 |
| NASNetMobile (MediaPipe) | 96.23% | 94.39% | 0.94 | 0.95 | 0.95 |
| NASNetMobile (Non-MediaPipe) | 94.31% | 92.77% | 0.94 | 0.92 | 0.93 |
| InceptionResNetV2 (MediaPipe) | 96.60% | 95.29% | 0.95 | 0.95 | 0.95 |
| InceptionResNetV2 (Non-MediaPipe) | 89.30% | 86.81% | 0.87 | 0.86 | 0.86 |
| VGG16 (MediaPipe) | 97.36% | 96.86% | 0.97 | 0.97 | 0.97 |
| VGG16 (Non-MediaPipe) | 95.60% | 95.53% | 0.96 | 0.95 | 0.96 |

The paper's full text (Springer, paywalled) was not accessible in this build
environment. All methodology statements above come from the public repository,
which is the code the paper describes.

## 6. Missing assets and reproducibility issues

1. **No trained weights are published.** The repository contains no `.h5`/`.keras`
   model file, so the reported numbers cannot be reproduced by loading a model.
   A model must be trained locally.
2. **Dataset requires Kaggle credentials.** The download is gated on an API token.
3. **No dataset statistics are published.** Image counts per class, resolution
   distribution and exact split sizes are unknown until the dataset is downloaded.
4. **Split ambiguity.** The notebook applies `validation_split=0.2` to *both*
   `train_dir` and `test_dir`, using `test_dir` as the validation source. There is
   no clearly separated test set in the notebook, which is a leakage risk. The
   paper's "test" numbers therefore come from an unclear evaluation procedure.
   YogaVision performs an explicit stratified train/val/test split instead.
5. **Hardcoded paths.** The notebook uses `'../DATASET/TRAIN'` and
   `'../Desktop/DATASET/TEST'`, which only work in the author's layout.
6. **No `ModelCheckpoint` in the active code** (it is commented out), so the
   notebook does not persist the best model.
7. **No fixed random seed** in the notebook, so runs are not reproducible.
8. **MediaPipe version drift.** `mediapipe==0.10.21` (the version pinned here)
   produces slightly different landmark coordinates than the 2022-era version
   used by the authors, so skeleton images are not byte-identical.
9. **Dots vs connections.** The repository script draws dots only; the paper's
   figures show full skeletons. See the deviation note above.

## 7. Proposed extensions implemented by YogaVision

- Full skeleton rendering (connection graph), not dots only.
- Explicit stratified train/validation/test split with no cross-split leakage.
- Configurable seeds and `enable_op_determinism()` for reproducibility.
- `ModelCheckpoint`, `EarlyStopping` and `ReduceLROnPlateau` callbacks.
- Real-time webcam inference with temporal smoothing and a confidence threshold.
- Image upload and video upload analysis with CSV export.
- Rule-based posture feedback (joint angles, symmetry, tilt, framing).
- SQLite session history with CSV export.
- Evaluation dashboard with confusion matrix, per-class report and latency.
- Streamlit UI with a model/dataset status dashboard.
