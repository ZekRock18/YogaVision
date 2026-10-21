# Limitations

This document states what the system cannot do, so that nobody relies on it
beyond what it supports.

## Camera placement sensitivity

- The system assumes **one person** fully in frame. Two people in the frame
  means MediaPipe locks onto whichever is larger/closer, and the classifier will
  report that person's pose only.
- Distance matters. Too close and limbs leave the frame; too far and landmark
  resolution collapses. The feedback module flags both cases, but the classifier
  still produces an output either way — just an unreliable one.
- Camera height changes the 2D projection. A pose viewed from above looks
  different from the same pose at eye level, and the model has no way to
  compensate.
- The paper's dataset is a fixed set of photographs; a live webcam stream has
  motion blur, exposure changes and rolling-shutter artefacts that the training
  data does not contain.

## Occlusion and landmark detection failures

- MediaPipe Pose needs most major joints visible. Occlusion by furniture,
  another person, or the user's own body (common in twisting poses) causes
  landmarks to drop out or be marked low-visibility.
- On this dataset, **153 of 1550 images (~10%) produced no detectable pose at
  all**. Live performance will be worse in cluttered environments.
- When no pose is detected, the skeleton image is completely black. The CNN
  still returns a class for a black image — the confidence score is the only
  signal that the input was empty, and it is not reliably low.

## Lighting, clothing and background

- Skeletonization removes most background and clothing information, which is the
  main reason the paper's MediaPipe variant outperforms raw RGB. It does not
  remove all of it: low contrast against the background, strong backlighting and
  very dark scenes still degrade landmark accuracy.
- Loose clothing hides limb geometry from the pose estimator, which distorts the
  skeleton before the CNN ever sees it.

## Similarity between poses

The five classes are visually related. `tree` and `warrior2` share a wide
standing base; `plank` and `downdog` both have a straight arm-to-torso line. With
only ~150–270 training images per class, confusions between these pairs are
expected. Reported accuracy on 5 classes does not imply the model would separate
20 classes as cleanly.

## Class imbalance

Class counts range from 160 (`tree`) to 266 (`plank`), an imbalance ratio of
**1.66**. The model is trained on unweighted categorical cross-entropy, so it is
mildly biased toward the majority classes. Additionally, **179 duplicate image
groups** were found by MD5 hash, meaning the effective sample count is
substantially lower than 1550 and the true diversity is lower still.

## Generalization

- Trained on a single small dataset of 1550 images from one source. It has not
  been evaluated on any held-out dataset from a different distribution.
- No cross-dataset or cross-subject validation was performed. Because many
  frames are near-duplicates, random splitting means the test set likely contains
  near-copies of training images. Test accuracy therefore overstates real-world
  generalization. **The paper's 97.09% test accuracy should be read with this in
  mind.**
- No evaluation was performed on people of different body types, ages, or
  mobility levels beyond whatever the dataset happens to contain.

## Confidence calibration

- The softmax outputs are **not calibrated**. A 0.95 probability does not mean a
  95% chance of being correct. No temperature scaling or Platt scaling has been
  applied.
- The default 0.60 confidence threshold is a heuristic chosen for the UI, not a
  statistically derived operating point.
- Temporal smoothing averages labels across frames, which makes the *displayed*
  label more stable but does not make the underlying estimate more accurate.

## Limits of 2D pose estimation

- A single camera gives a 2D projection. Depth is unknown.
- Landmarks in front of and behind each other collapse onto the same image
  position. In `downdog`, the far-side arm and leg are hidden behind the near
  side, so the skeleton shows only half the pose.
- Depth-dependent judgements (whether the knee tracks over the toes, whether the
  pelvis is square, whether the spine is lengthening) cannot be made from this
  input.

## Classification is not pose-quality assessment

This is the most important limitation.

- The classifier answers "which of the 5 poses does this most resemble?" It does
  **not** answer "is this pose performed correctly?"
- A shallow `tree` with the foot on the ankle is still classified `tree`.
- The rule-based feedback module checks only gross symmetry and framing. Its
  rules are geometric heuristics with fixed thresholds, not validated against
  any expert annotation. It has no ground truth to compare against and has never
  been evaluated for accuracy.
- **Not medical advice.** Nothing here diagnoses, treats, or prevents any
  condition, and no injury risk assessment is performed. Do not use this as a
  substitute for instruction from a qualified teacher, particularly if you have
  an injury or a medical condition.

## Implementation limitations

- The webcam path runs a blocking loop inside the Streamlit script. It works,
  but it blocks the tab's other controls while active.
- **Webcam verified on physical hardware on 2026-10-14** (camera index 0,
  640×480): 25 frames processed, a person was detected in 21 of them, mean
  latency 99 ms, approximately 10 FPS including MediaPipe pose estimation.
  This was a live smoke test, not a sustained session, and the subject was not
  performing a specific pose — the classifier reported `plank` throughout, which
  reflects the model's bias and the fact that a seated operator at a desk is a
  poor match for all five classes.
- Video analysis requires a codec that the bundled OpenCV build can decode. Some
  H.265/HEVC and ProRes files will fail to open; converting to H.264 MP4 with
  FFmpeg resolves it.
- Recordings are not saved by default. If recording is enabled, output is
  written to `outputs/uploads/` and must be deleted manually.
- No GPU is required, and none was used. All reported latency figures are
  single-threaded CPU numbers from this machine and will differ elsewhere.
