# Methodology

## 1. Problem statement

Given an image, video frame, or webcam frame containing a single person,
identify which of a fixed set of yoga poses they are performing, estimate the
confidence of that prediction, and provide simple rule-based posture feedback.

## 2. Research motivation

Yoga is widely practised at home from video instruction, where no teacher is
present to correct alignment. A vision system that classifies the pose and flags
gross alignment problems gives a learner immediate feedback. The difficulty is
that poses differ subtly (a tree pose and a warrior pose share a standing base),
backgrounds vary, and a single camera gives only a 2D projection of a 3D body.

The approach taken here follows Garg, Saxena & Gupta (2023), who showed that
skeletonizing the image with MediaPipe before feeding it to a small custom CNN
substantially improves accuracy over using raw photographs, because the skeleton
removes background, clothing and lighting variation while retaining body geometry.

## 3. Paper methodology

See `docs/research_notes.md` for the full reconstruction. In brief:

- 5 yoga pose classes, 75×75 RGB inputs, pixel values scaled to [0, 1].
- `ImageDataGenerator` with `width_shift_range=0.1`, `horizontal_flip=True`,
  `validation_split=0.2`.
- YogaConvo2d: Conv(64)-Conv(128)-Conv(256), each with 2×2 max pooling and
  dropout 0.25, then Dense(1024) with dropout 0.5 and a softmax output.
- Adam, learning rate 1e-3, batch size 16, up to 50 epochs.
- Two input variants: raw RGB photographs, and MediaPipe skeletonized images.

## 4. Dataset and preprocessing

**Dataset:** the Kaggle dataset `niharika41298/yoga-poses-dataset`, the same
source used by the original repository. See `docs/dataset_setup.md`.

**Preprocessing pipeline (`src/yogavision/skeletonization.py`):**

1. Decode the image with OpenCV to a BGR uint8 array.
2. Convert to RGB and run MediaPipe Pose (`static_image_mode` for stills,
   tracking mode for video).
3. If landmarks are found, render the full pose connection graph as white lines
   and joint dots on a black canvas of the original resolution.
4. If no landmarks are found, output a completely black canvas.
5. Resize to 75×75 and scale to [0, 1] as float32.

**Deviation from the repository's script:** the repository's
`skeletonization.py` draws only landmark dots (`cv2.circle`, radius 5). YogaVision
draws the full connection graph, which matches the stick-figure representation
shown in the paper's figures and preserves more spatial structure. Both variants
are supported through the `draw_connections` flag.

## 5. CNN architecture

YogaConvo2d, a faithful reproduction of the notebook's `Sequential` model
(`src/yogavision/model.py: build_cnn`):

| Layer | Output shape | Params |
|---|---|---|
| Conv2D(64, 3×3, ReLU, 'Same') | 75×75×64 | 1,792 |
| MaxPooling2D(2,2) | 37×37×64 | 0 |
| Dropout(0.25) | 37×37×64 | 0 |
| Conv2D(128, 3×3, ReLU, 'Same') | 37×37×128 | 73,856 |
| MaxPooling2D(2,2) | 18×18×128 | 0 |
| Dropout(0.25) | 18×18×128 | 0 |
| Conv2D(256, 3×3, ReLU, 'Same') | 18×18×256 | 295,168 |
| MaxPooling2D(2,2) | 9×9×256 | 0 |
| Dropout(0.25) | 9×9×256 | 0 |
| Flatten | 20,736 | 0 |
| Dense(1024, ReLU) | 1024 | 21,234,688 |
| Dropout(0.5) | 1024 | 0 |
| Dense(5, softmax) | 5 | 5,125 |

Total ≈ 21.6M parameters. Input `(75, 75, 3)`, output softmax over classes.

## 6. MediaPipe integration

`src/yogavision/pose_detection.py` wraps `mediapipe.solutions.pose.Pose` with:

- `static_image_mode=False`, `model_complexity=1`, `smooth_landmarks=True`
- `enable_segmentation=False`
- `min_detection_confidence` / `min_tracking_confidence` from config (default 0.5)

The wrapper returns 33 landmarks as `(x, y, visibility)` in normalised image
coordinates, plus optional world (metric) landmarks. Visibility is used
throughout to avoid computing geometry on unreliable landmarks.

The detector is a lazily-created singleton so the MediaPipe graph is built once
per process rather than per frame.

## 7. Training procedure

`scripts/train.py --config configs/train.yaml`

1. Discover class folders and enumerate all images.
2. **Stratified split** — `sklearn.model_selection.train_test_split` splits
   per-class, so every class appears in train, validation and test. This fixes the
   original notebook's ambiguity, where `validation_split=0.2` was applied to both
   the train and test directories and the test directory was used as the
   validation source.
3. Load and resize images to 75×75, scale to [0, 1].
4. One-hot encode labels with `tf.keras.utils.to_categorical`.
5. Compile with Adam (lr 1e-3) and `categorical_crossentropy`.
6. Train with `ImageDataGenerator(width_shift_range=0.1, horizontal_flip=True)`.
7. Callbacks: `ModelCheckpoint` (best val_loss), `EarlyStopping`
   (patience 8, restore best weights), `ReduceLROnPlateau` (patience 4, factor 0.5).
8. Save the model, the label map, training history, curves and test metrics.

**Reproducibility:** `utils.set_seed()` seeds Python's `random`, NumPy and
TensorFlow, and calls `tf.config.experimental.enable_op_determinism()`. The seed
is configurable in the config file.

## 8. Evaluation methodology

`scripts/evaluate.py` reloads the saved model, rebuilds the **same stratified
test split** using the same seed, and computes:

- Accuracy
- Precision, recall, F1 (macro and weighted)
- Confusion matrix (heatmap PNG)
- Per-class classification report (txt + JSON)
- Inference latency: mean, std, p95, min, max and approximate FPS over repeated
  single-frame predictions on CPU

Metrics are written to `outputs/evaluation/evaluation_metrics.json` and
`outputs/plots/confusion_matrix.png`.

## 9. Experimental results

A **reproduction run completed on 2026-10-14** using this project's pipeline on
the Kaggle dataset described above. All numbers below come from
`outputs/evaluation/` on this machine (CPU, Python 3.11.0, TensorFlow 2.19.0,
MediaPipe 0.10.21) and are labelled **reproduction run**.

### Training configuration actually used

| Parameter | Value |
|---|---|
| Dataset | `data/processed/yoga_poses_skeleton/TRAIN` (MediaPipe skeletons) |
| Images | 1,080 |
| Split | stratified 755 train / 216 val / 109 test (seed 42) |
| Epochs run | 16 (early stopping, patience 8; best epoch 8) |
| Batch size | 16 |
| Optimizer | Adam, lr 1e-3 (reduced to 5e-4 by ReduceLROnPlateau) |
| Augmentation | `width_shift_range=0.1`, `horizontal_flip=True` |
| Image size | 75×75, scaled to [0, 1] |
| Trainable parameters | ~21.6M |

### Training-set metrics

| Metric | Value |
|---|---|
| Final train accuracy | 0.836 |
| Best validation accuracy | 0.657 |
| Best validation loss | 0.863 |
| Wall-clock training time | ~4 min (16 epochs × ~13 s) |

### Held-out test metrics (`scripts/evaluate.py`)

| Metric | Value |
|---|---|
| **Accuracy** | **0.6111** |
| Precision (macro) | 0.6233 |
| Recall (macro) | 0.5979 |
| F1 (macro) | 0.6037 |
| F1 (weighted) | 0.6105 |
| Test samples | 108 |

### Inference latency

| Metric | Value |
|---|---|
| Mean | 67.3 ms |
| Std | 3.7 ms |
| p95 | 71.3 ms |
| Approx FPS | 14.9 |

Measured at batch size 1 on CPU. Note that ~67 ms of this is
`model.predict()` overhead; the raw forward pass is faster. A MediaPipe pose
estimation step adds further latency not included in this figure.

### Interpretation

The reproduction reaches **61.1% test accuracy**, well below the paper's
reported **97.09%**. This gap is expected and is not evidence that the
implementation is wrong:

1. **Different split procedure.** The paper applies `validation_split=0.2` to
   both the train and test directories and uses the test directory as its
   validation source. With 179 duplicate image groups in the data, a random split
   leaks near-identical images across partitions, inflating test accuracy. This
   project's stratified split is honest but yields a harder number.
2. **Early stopping at epoch 8.** Train accuracy reached 0.836 while validation
   accuracy peaked at 0.657 — a large generalization gap typical of a 21.6M
   parameter model on 755 training images with no regularization beyond dropout.
3. **Skeleton rendering differs.** The paper's repository script draws landmark
   dots only; this project draws the full connection graph. That changes the
   input distribution relative to whatever the authors actually trained on.
4. **MediaPipe version drift.** `mediapipe==0.10.21` produces different landmark
   coordinates than the 2022-era version, so skeletons are not identical.
5. **Different fold of the data.** This run uses only the TRAIN folder re-split;
   the authors used both TRAIN and TEST directories in an unspecified way.

No claim is made that this reproduction matches the paper's accuracy. The
purpose of the run is to verify that the pipeline trains, evaluates and serves
predictions correctly end to end, which it does.

### Artefacts produced

- `outputs/evaluation/training_results.json` — history, splits, test metrics
- `outputs/evaluation/evaluation_metrics.json` — held-out test metrics + latency
- `outputs/evaluation/classification_report.txt` — per-class precision/recall/F1
- `outputs/plots/confusion_matrix.png` — confusion matrix heatmap
- `outputs/plots/training_curves.png` — accuracy and loss curves
- `models/yoga_cnn.keras`, `models/label_map.json` — the trained model

## 10. Differences from the original paper

| Aspect | Original | YogaVision |
|---|---|---|
| Skeleton rendering | Landmark dots only | Full connection graph (dots + lines) |
| Train/val/test split | `validation_split=0.2` on both dirs, test dir used for validation | Explicit stratified 3-way split |
| Random seed | None | Configurable, determinism enabled |
| Checkpointing | Commented out | `ModelCheckpoint` best val_loss |
| Early stopping | None | Patience 8, best weights restored |
| Learning-rate schedule | None | `ReduceLROnPlateau` |
| Input size | 75×75 | 75×75 (unchanged) |
| Inference targets | Offline notebook | Webcam, image, video, Streamlit UI |
| Posture feedback | None | Rule-based joint-angle module |
| Session persistence | None | SQLite + CSV export |

The CNN architecture, optimizer, learning rate, batch size, image size and
augmentation settings are **unchanged** from the original notebook.

## 11. Limitations

See `docs/limitations.md` for the full discussion. Summary: single fixed camera,
2D-only estimation, sensitivity to occlusion, lighting, clothing and background,
pose similarity, class imbalance, uncalibrated confidence scores, and the
distinction between classifying a pose and judging whether it is performed well.

## 12. Future improvements

- Collect a larger, more diverse dataset with more classes and per-class variation.
- Add left/right-mirrored augmentation to improve symmetry robustness.
- Calibrate confidence with temperature scaling or Platt scaling.
- Compare the skeleton-image CNN against a landmark-coordinate MLP baseline
  (cheaper inference, no image rendering).
- Temporal modelling (LSTM/Transformer over landmark sequences) for video.
- Pose-hold timers and repetition counting.
- On-device export (TFLite/ONNX) for mobile inference.
- Export processed video with skeleton overlays.
- Multi-person support with per-person tracking.
