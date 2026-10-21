"""Build the YogaVision presentation deck.

Uses ppt_helpers for all layout primitives so every slide shares one visual
system (deep navy cards, coloured accent borders, one heading style).

Run:  python scripts/make_presentation.py
"""
from __future__ import annotations

from pathlib import Path

from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from ppt_helpers import (
    C_AMBER, C_BG, C_BLUE, C_CARD, C_CYAN, C_DGREY, C_GREEN, C_GREY, C_NAVY,
    C_PURPLE, C_RED, C_WHITE, add_slide, bg, badge, bottombar, card, feature_grid,
    heading, info_grid, kv_table, metric_cards, picture_or_placeholder, rect,
    save, topbar, txt,
)


def slide_title() -> None:
    s = add_slide()
    bg(s, C_BG)
    rect(s, 0, 0, 13.333, 7.5, C_BG)
    rect(s, 0, 0, 5.5, 7.5, C_NAVY)
    rect(s, 5.5, 0, 0.08, 7.5, C_CYAN)

    txt(s, "YogaVision", 5.8, 1.2, 7.2, 1.0, size=58, bold=True, color=C_WHITE)
    txt(s, "Real-Time Yoga Pose Classification\nand Feedback System",
        5.8, 2.25, 7.2, 1.3, size=24, color=C_CYAN)
    txt(s, "End-to-End CNN and MediaPipe Based Implementation",
        5.8, 3.75, 7.2, 0.5, size=14, color=C_GREY, italic=True)
    txt(s, "A Deep Learning Approach for Real-World Applications",
        5.8, 4.2, 7.2, 0.5, size=13, color=C_GREY)
    txt(s, "Inspired by: Yoga Pose Classification - A CNN and MediaPipe Inspired\n"
           "Deep Learning Approach for Real-World Application\n"
           "Journal of Ambient Intelligence and Humanized Computing, 2022",
        5.8, 4.9, 7.2, 1.2, size=11, color=C_DGREY, italic=True)

    txt(s, "Computer Vision\nProject", 0.4, 3.0, 4.8, 1.5, size=36, bold=True, color=C_CYAN)
    txt(s, "\nPose Detection  |  CNN Classification\nMediaPipe  |  Streamlit Interface",
        0.4, 4.3, 4.8, 1.2, size=14, color=C_GREY)

    rect(s, 0, 7.2, 13.333, 0.3, C_BLUE)
    txt(s, "YogaVision  |  Deep Learning for Human Pose Understanding",
        0.3, 7.22, 12.5, 0.25, size=10, color=C_WHITE)


def slide_agenda() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Project Overview", "What we built and why")

    items = [
        ("01", "Problem Statement", "Real-time yoga pose identification from webcam or image input"),
        ("02", "Research Foundation", "CNN and MediaPipe approach from published literature"),
        ("03", "System Architecture", "Full pipeline from input through pose detection to classification"),
        ("04", "Dataset", "5 yoga pose classes with skeleton image preprocessing"),
        ("05", "Model and Training", "YogaConv2D CNN trained from scratch with early stopping"),
        ("06", "Results and Evaluation", "Quantitative metrics, confusion matrix, training curves"),
        ("07", "Application", "Streamlit interface with webcam, image, and video analysis"),
        ("08", "Limitations and Future Work", "Known constraints, planned improvements, and live demo"),
    ]
    for i, (num, title, desc) in enumerate(items):
        x = 0.3 + (i // 4) * 6.5
        y = 1.15 + (i % 4) * 1.5
        card(s, x, y, 6.2, 1.35, C_CARD, C_BLUE, 0.8)
        badge(s, num, x + 0.08, y + 0.08, 0.55, 0.55, C_BLUE, size=14)
        txt(s, title, x + 0.75, y + 0.12, 5.3, 0.42, size=15, bold=True, color=C_WHITE)
        txt(s, desc, x + 0.75, y + 0.55, 5.3, 0.7, size=11, color=C_GREY)
        rect(s, x + 0.75, y + 0.5, 5.2, 0.02, C_BLUE)

    bottombar(s, 0.22, C_CYAN)


def slide_problem() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Problem Statement", "Why yoga pose classification matters")

    txt(s, "The Challenge", 0.4, 1.15, 5.8, 0.45, size=17, bold=True, color=C_CYAN)
    challenges = [
        "Yoga requires precise body alignment and positioning",
        "Traditional assessment relies on expert human observation",
        "One-on-one instruction is expensive and hard to scale",
        "Lighting, clothing, and camera angle variations reduce accuracy",
        "Real-time feedback is needed for effective home practice",
        "Many poses look similar from different viewing angles",
    ]
    for i, c in enumerate(challenges):
        y = 1.65 + i * 0.78
        rect(s, 0.4, y, 0.05, 0.55, C_CYAN)
        txt(s, c, 0.58, y, 5.6, 0.7, size=13, color=C_WHITE)

    txt(s, "Our Objective", 7.1, 1.15, 5.5, 0.45, size=17, bold=True, color=C_GREEN)
    card(s, 6.9, 1.65, 6.1, 5.3, C_CARD, C_GREEN, 0.8)
    objectives = [
        ("Detect", "Identify human body landmarks in real time using MediaPipe Pose"),
        ("Extract", "Convert raw frames to skeleton images capturing pose geometry"),
        ("Classify", "Predict the yoga pose using a CNN trained on skeleton data"),
        ("Feedback", "Provide joint-angle feedback from detected landmarks"),
        ("Interface", "Deliver results through a web interface with webcam support"),
    ]
    for i, (title, desc) in enumerate(objectives):
        y = 1.85 + i * 1.0
        badge(s, title[0], 7.05, y, 0.6, 0.6, C_GREEN, size=18, text_color=C_BG)
        txt(s, title, 7.75, y + 0.02, 5.0, 0.35, size=14, bold=True, color=C_GREEN)
        txt(s, desc, 7.75, y + 0.38, 5.0, 0.55, size=11, color=C_GREY)

    bottombar(s, 0.22, C_CYAN)


def slide_research() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Research Foundation", "Paper citation and methodology summary")

    card(s, 0.3, 1.1, 12.7, 1.6, C_CARD, C_AMBER, 1.0)
    txt(s, "PAPER CITATION", 0.55, 1.18, 4, 0.32, size=10, bold=True, color=C_AMBER)
    txt(s, "Yoga Pose Classification: A CNN and MediaPipe Inspired Deep Learning\n"
           "Approach for Real-World Application",
        0.55, 1.45, 12.2, 0.7, size=15, bold=True, color=C_WHITE)
    txt(s, "Journal of Ambient Intelligence and Humanized Computing, 2022\n"
           "DOI: 10.1007/s12652-022-03910-0",
        0.55, 2.12, 12.2, 0.5, size=11, color=C_GREY)

    card(s, 0.3, 2.85, 12.7, 0.55, C_CARD, C_CYAN, 0.6)
    txt(s, "GitHub: github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization",
        0.55, 2.95, 12.2, 0.4, size=12, color=C_CYAN)

    txt(s, "Original Methodology", 0.4, 3.6, 6.0, 0.4, size=16, bold=True, color=C_AMBER)
    steps = [
        "Input: raw RGB images of yoga practitioners",
        "MediaPipe Pose extracts 33 body landmarks (x, y, z, visibility)",
        "Skeletonization draws landmarks and connections on a black canvas",
        "Skeleton image (224 x 224 x 3) fed into the CNN",
        "YogaConv2D: custom 4-block convolutional network",
        "Trained with Adam, early stopping, up to 50 epochs",
        "Augmentation: horizontal flip only",
    ]
    for i, step in enumerate(steps):
        txt(s, step, 0.55, 4.05 + i * 0.42, 6.0, 0.38, size=12, color=C_WHITE)

    txt(s, "Paper Reported Results (YogaConv2D + MediaPipe)", 6.9, 3.6, 6.1, 0.4,
        size=15, bold=True, color=C_AMBER)
    kv_table(
        s,
        [("Validation Accuracy", "99.62 %"), ("Test Accuracy", "97.09 %"),
         ("Precision", "0.97"), ("Recall", "0.97"), ("Macro F1", "0.97")],
        left=6.9, top=4.05, width=6.1, key_w=3.4, value_w=2.4, row_h=0.52,
        key_color=C_GREY, value_color=C_GREEN, value_size=14,
    )
    txt(s, "Note: these are the authors' numbers, not YogaVision results. "
           "Full paper text is paywalled, so methodology came from the public repository.",
        0.4, 7.05, 12.5, 0.3, size=10, color=C_DGREY, italic=True)
    bottombar(s, 0.22, C_AMBER)


def slide_architecture() -> None:
    s = add_slide()
    bg(s)
    heading(s, "System Architecture", "End-to-end pipeline from input to classification output")

    stages = [
        ("INPUT", ["Webcam", "Image", "Video"], C_BLUE, 0.4),
        ("MEDIAPIPE\nPOSE", ["33 Landmarks", "Skeleton", "Visibility"], C_CYAN, 2.65),
        ("SKELETON\nIMAGE", ["75 x 75 x 3", "Black Canvas", "Joint Lines"], C_PURPLE, 4.9),
        ("CNN\nCLASSIFIER", ["YogaConv2D", "5 Classes", "Softmax"], C_GREEN, 7.15),
        ("OUTPUT", ["Pose Label", "Confidence", "Top-3"], C_AMBER, 9.4),
    ]
    box_w, box_h = 1.9, 2.8
    for label, details, color, left in stages:
        card(s, left, 1.3, box_w, box_h, C_CARD, color, 1.5)
        rect(s, left, 1.3, box_w, 0.7, color)
        txt(s, label, left + 0.05, 1.45, box_w - 0.1, 0.6, size=11, bold=True,
            color=C_WHITE, align=PP_ALIGN.CENTER)
        for i, d in enumerate(details):
            txt(s, d, left + 0.1, 2.15 + i * 0.55, box_w - 0.2, 0.5, size=11,
                color=C_WHITE, align=PP_ALIGN.CENTER)

    for left in [2.32, 4.57, 6.82, 8.07]:
        txt(s, ">", left, 2.5, 0.35, 0.5, size=24, bold=True, color=C_CYAN,
            align=PP_ALIGN.CENTER)

    txt(s, "Core Modules", 0.4, 4.3, 4.5, 0.4, size=14, bold=True, color=C_CYAN)
    modules = [
        ("pose_detection.py", "MediaPipe Pose wrapper, landmark extraction"),
        ("skeletonization.py", "Converts raw frames to skeleton images"),
        ("model.py", "YogaConv2D architecture and model loading"),
        ("inference.py", "Frame prediction, top-k, confidence thresholding"),
        ("feedback.py", "Joint angle calculation and posture feedback"),
        ("video_analysis.py", "Frame sampling, per-frame predictions, CSV export"),
    ]
    for i, (fname, desc) in enumerate(modules):
        y = 4.75 + i * 0.41
        txt(s, fname, 0.5, y, 2.1, 0.36, size=11, bold=True, color=C_CYAN)
        txt(s, desc, 2.65, y, 3.2, 0.36, size=10, color=C_GREY)

    txt(s, "Technology Stack", 6.0, 4.3, 3.5, 0.4, size=14, bold=True, color=C_GREEN)
    stack = [
        ("OpenCV", "Camera capture and video processing"),
        ("MediaPipe Pose", "Body landmark detection, 33 points"),
        ("TensorFlow / Keras", "CNN definition and training"),
        ("Streamlit", "Web user interface"),
        ("NumPy / Pandas", "Data processing and metrics"),
        ("scikit-learn", "Metrics, splitting, evaluation"),
    ]
    for i, (tech, desc) in enumerate(stack):
        y = 4.75 + i * 0.41
        txt(s, tech, 6.1, y, 1.9, 0.36, size=11, bold=True, color=C_GREEN)
        txt(s, desc, 8.0, y, 3.0, 0.36, size=10, color=C_GREY)

    bottombar(s, 0.22, C_CYAN)


def slide_dataset() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Dataset", "Source, structure, preprocessing, and data quality checks")

    card(s, 0.3, 1.1, 6.1, 1.2, C_CARD, C_CYAN, 0.8)
    txt(s, "SOURCE", 0.5, 1.18, 3, 0.3, size=10, bold=True, color=C_CYAN)
    txt(s, "Kaggle: niharika41298/yoga-poses-dataset", 0.5, 1.45, 5.7, 0.35,
        size=13, bold=True, color=C_WHITE)
    txt(s, "License CC BY-SA 4.0  |  5 classes  |  approximately 900 images",
        0.5, 1.8, 5.7, 0.35, size=11, color=C_GREY)

    txt(s, "Class Distribution (After Preprocessing)", 0.3, 2.45, 6.1, 0.4,
        size=14, bold=True, color=C_CYAN)
    classes = [("downdog", 178), ("goddess", 173), ("plank", 266),
               ("tree", 160), ("warrior2", 203)]
    colors = [C_BLUE, C_CYAN, C_GREEN, C_AMBER, C_PURPLE]
    max_h = 1.5
    for i, (cls, n) in enumerate(classes):
        bar_x = 0.5 + i * 1.22
        bar_h = (n / 266) * max_h
        bar_top = 5.85 - bar_h
        rect(s, bar_x, bar_top, 1.1, bar_h, colors[i])
        txt(s, cls, bar_x, 5.9, 1.1, 0.35, size=10, bold=True, color=C_WHITE,
            align=PP_ALIGN.CENTER)
        txt(s, str(n), bar_x, bar_top - 0.32, 1.1, 0.3, size=11, bold=True,
            color=colors[i], align=PP_ALIGN.CENTER)
    txt(s, "266", 0.0, 4.32, 0.45, 0.3, size=9, color=C_GREY)
    txt(s, "0", 0.0, 5.86, 0.45, 0.3, size=9, color=C_GREY)
    rect(s, 0.45, 5.85, 6.0, 0.02, C_DGREY)
    rect(s, 0.45, 4.36, 6.0, 0.02, C_DGREY)

    txt(s, "Data Split", 6.6, 1.1, 6.5, 0.4, size=14, bold=True, color=C_CYAN)
    kv_table(
        s,
        [("Training", "755 samples   (70%)"), ("Validation", "216 samples   (20%)"),
         ("Test", "109 samples   (10%)")],
        left=6.6, top=1.55, width=6.4, key_w=2.0, value_w=4.1, row_h=0.68,
        border=C_BLUE, value_size=13, value_align=PP_ALIGN.RIGHT,
    )

    txt(s, "Data Quality Findings", 6.6, 3.85, 6.5, 0.4, size=14, bold=True, color=C_AMBER)
    quality = [
        "1 corrupt JPEG detected and removed (truncated file)",
        "179 duplicate image groups detected by MD5 hash",
        "Class imbalance ratio 1.66 (min 160 tree, max 266 plank)",
        "Stratified splitting prevents data leakage between splits",
        "Skeleton images are 75 x 75 pixels, 3 channel, black background",
    ]
    for i, q in enumerate(quality):
        rect(s, 6.6, 4.28 + i * 0.45, 0.04, 0.32, C_AMBER)
        txt(s, q, 6.75, 4.28 + i * 0.45, 6.1, 0.4, size=11, color=C_WHITE)

    bottombar(s, 0.22, C_CYAN)


def slide_model() -> None:
    s = add_slide()
    bg(s)
    heading(s, "CNN Model Architecture",
            "YogaConv2D, a faithful reproduction of the paper's custom CNN")

    txt(s, "Network Architecture", 0.3, 1.1, 6.5, 0.4, size=15, bold=True, color=C_PURPLE)
    layers = [
        ("Input", "Skeleton image   (75 x 75 x 3)", C_DGREY),
        ("Conv2D + ReLU", "32 filters, 3 x 3, padding same", C_BLUE),
        ("MaxPooling2D", "2 x 2, stride 2", C_CYAN),
        ("Conv2D + ReLU", "64 filters, 3 x 3, padding same", C_BLUE),
        ("MaxPooling2D", "2 x 2, stride 2", C_CYAN),
        ("Conv2D + ReLU", "128 filters, 3 x 3, padding same", C_BLUE),
        ("MaxPooling2D", "2 x 2, stride 2", C_CYAN),
        ("Conv2D + ReLU", "256 filters, 3 x 3, padding same", C_BLUE),
        ("GlobalAvgPool2D", "Aggregates spatial features", C_PURPLE),
        ("Dropout", "Rate 0.25, training only", C_DGREY),
        ("Dense + ReLU", "128 units, L2 regularization", C_GREEN),
        ("Dropout", "Rate 0.50", C_DGREY),
        ("Dense + Softmax", "5 units, one per yoga class", C_AMBER),
    ]
    for i, (name, detail, color) in enumerate(layers):
        top = 1.55 + i * 0.41
        card(s, 0.3, top, 6.0, 0.36, C_CARD, color, 0.5)
        rect(s, 0.3, top, 1.75, 0.36, color)
        txt(s, name, 0.33, top + 0.04, 1.7, 0.3, size=9, bold=True, color=C_WHITE,
            align=PP_ALIGN.CENTER)
        txt(s, detail, 2.15, top + 0.04, 4.05, 0.3, size=10, color=C_WHITE)

    txt(s, "Design Rationale", 6.8, 1.1, 6.2, 0.4, size=15, bold=True, color=C_PURPLE)
    rationale = [
        ("Why skeleton images?",
         "MediaPipe extracts geometric pose structure from 33 landmarks and discards "
         "background, clothing, and skin tone, so the CNN learns pose geometry rather "
         "than appearance."),
        ("Why GlobalAveragePooling?",
         "It reduces feature maps to a single vector, removing dependence on exact input "
         "resolution and making the model more tolerant of landmark position variation "
         "than a Flatten layer."),
        ("Why two dropout layers?",
         "0.25 after pooling prevents early overfitting, and 0.50 before the final Dense "
         "regularizes the classification head. Combined with early stopping."),
        ("Why a confidence threshold?",
         "The model has no 'none of the above' class, so a 0.60 threshold suppresses "
         "unreliable predictions instead of always naming one of the five poses."),
    ]
    for i, (q, a) in enumerate(rationale):
        y = 1.55 + i * 1.4
        card(s, 6.6, y, 6.4, 1.32, C_CARD, C_PURPLE, 0.7)
        txt(s, q, 6.75, y + 0.08, 6.1, 0.4, size=12, bold=True, color=C_PURPLE)
        txt(s, a, 6.75, y + 0.46, 6.1, 0.82, size=10, color=C_GREY)

    bottombar(s, 0.22, C_PURPLE)


def slide_training() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Training Procedure",
            "Reproducible pipeline with fixed seeds and documented configuration")

    txt(s, "Training Configuration", 0.3, 1.1, 6.0, 0.4, size=15, bold=True, color=C_GREEN)
    kv_table(
        s,
        [("Optimizer", "Adam"), ("Learning Rate", "0.001, reduced to 0.0005 on plateau"),
         ("Batch Size", "16"), ("Epochs", "50 maximum, early stopped at 16"),
         ("Image Size", "75 x 75 pixels"),
         ("Input", "Skeleton images from MediaPipe"),
         ("Loss Function", "Categorical crossentropy"),
         ("Seed", "42, all random states fixed"),
         ("Augmentation", "Horizontal flip only")],
        left=0.3, top=1.55, width=5.8, key_w=1.7, value_w=3.8, row_h=0.52,
        key_color=C_GREEN, value_size=11, value_align=PP_ALIGN.LEFT,
    )

    txt(s, "Training Pipeline", 6.5, 1.1, 6.5, 0.4, size=15, bold=True, color=C_CYAN)
    steps = [
        ("Step 1", "Dataset Discovery", "Verify class folders, count images, report corrupt files"),
        ("Step 2", "Data Loading", "Load skeleton images, assign integer labels"),
        ("Step 3", "Train Val Test Split", "Stratified 70 / 20 / 10, seed 42"),
        ("Step 4", "Preprocessing", "Rescale pixels to [0, 1], resize to 75 x 75 x 3"),
        ("Step 5", "CNN Training", "YogaConv2D compiled with Adam"),
        ("Step 6", "Callbacks", "EarlyStopping patience 8, ReduceLROnPlateau patience 4"),
        ("Step 7", "Checkpointing", "Save best model by validation loss"),
        ("Step 8", "Evaluation", "Evaluate best checkpoint on held-out test set"),
        ("Step 9", "Export", "Save yoga_cnn.keras and label_map.json"),
    ]
    for i, (step, title, desc) in enumerate(steps):
        y = 1.55 + i * 0.63
        card(s, 6.5, y, 6.5, 0.58, C_CARD, C_CYAN, 0.4)
        badge(s, step, 6.5, y, 0.68, 0.58, C_CYAN, size=9, text_color=C_BG)
        txt(s, title, 7.25, y + 0.04, 2.4, 0.28, size=11, bold=True, color=C_CYAN)
        txt(s, desc, 7.25, y + 0.32, 5.6, 0.25, size=9, color=C_GREY)

    bottombar(s, 0.22, C_GREEN)


def slide_training_results() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Training Results",
            "YogaVision reproduction run, YogaConv2D on skeleton images")

    picture_or_placeholder(s, "outputs/plots/training_curves.png", 0.3, 1.05, 7.2, 5.6,
                           "Training curves image")

    txt(s, "Training Summary", 7.7, 1.1, 5.2, 0.4, size=15, bold=True, color=C_GREEN)
    kv_table(
        s,
        [("Total Samples", "1080"), ("Train / Val / Test", "755 / 216 / 109"),
         ("Epochs Run", "16 of 50 maximum"), ("Training Time", "259.85 seconds"),
         ("Best Train Accuracy", "83.58 %"), ("Best Val Accuracy", "68.52 %"),
         ("Best Val Loss", "0.8634")],
        left=7.5, top=1.55, width=5.4, key_w=2.6, value_w=2.6, row_h=0.68,
        key_color=C_GREEN, value_size=13,
    )
    txt(s, "Accuracy and loss curves over 16 epochs. Validation loss stabilised after "
           "epoch 8 and early stopping triggered at epoch 16.",
        7.7, 6.35, 5.2, 0.8, size=10, color=C_GREY, italic=True)
    bottombar(s, 0.22, C_GREEN)


def slide_evaluation() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Evaluation Metrics",
            "Test set results on 109 held-out samples, YogaVision reproduction run")

    picture_or_placeholder(s, "outputs/plots/confusion_matrix.png", 0.3, 1.05, 6.5, 5.5,
                           "Confusion matrix image")

    txt(s, "Overall Metrics", 7.0, 1.1, 6.0, 0.4, size=15, bold=True, color=C_BLUE)
    kv_table(
        s,
        [("Accuracy", "71.56 %"), ("Macro Precision", "72.76 %"),
         ("Macro Recall", "69.58 %"), ("Macro F1-Score", "70.32 %"),
         ("Weighted F1", "71.22 %")],
        left=6.85, top=1.55, width=6.2, key_w=3.0, value_w=3.0, row_h=0.7,
        key_color=C_GREY, value_color=C_GREEN, value_size=17,
    )

    txt(s, "Per-Class F1 Scores", 7.0, 5.2, 6.0, 0.38, size=14, bold=True, color=C_CYAN)
    class_f1 = [("downdog", 0.829, C_GREEN), ("goddess", 0.600, C_AMBER),
                ("plank", 0.787, C_GREEN), ("tree", 0.645, C_AMBER),
                ("warrior2", 0.655, C_AMBER)]
    for i, (cls, f1, col) in enumerate(class_f1):
        x = 7.0 + (i % 3) * 2.05
        y = 5.62 + (i // 3) * 0.72
        card(s, x, y, 1.95, 0.66, C_CARD, col, 0.5)
        txt(s, cls, x + 0.08, y + 0.04, 1.8, 0.3, size=10, bold=True, color=col)
        txt(s, f"{f1:.1%}", x + 0.08, y + 0.32, 1.8, 0.3, size=13, bold=True,
            color=C_WHITE, align=PP_ALIGN.CENTER)

    bottombar(s, 0.22, C_BLUE)


def slide_comparison() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Comparison: YogaVision vs Original Paper",
            "Understanding the accuracy gap between reproduction and published results")

    headers = ["Metric", "Paper (YogaConv2D + MediaPipe)", "YogaVision (This Project)"]
    col_x = [0.4, 4.4, 9.2]
    col_w = [3.9, 4.7, 3.7]
    rect(s, 0.3, 1.1, 12.7, 0.5, C_AMBER)
    for h, x, w in zip(headers, col_x, col_w):
        txt(s, h, x + 0.1, 1.16, w - 0.2, 0.4, size=12, bold=True, color=C_BG,
            align=PP_ALIGN.CENTER)

    rows = [
        ("Validation Accuracy", "99.62 %", "68.52 %", C_GREEN, C_RED),
        ("Test Accuracy", "97.09 %", "71.56 %", C_GREEN, C_RED),
        ("Macro Precision", "0.97", "72.76 %", C_GREEN, C_RED),
        ("Macro Recall", "0.97", "69.58 %", C_GREEN, C_RED),
        ("Macro F1-Score", "0.97", "70.32 %", C_GREEN, C_RED),
        ("Input Resolution", "224 x 224", "75 x 75", C_GREY, C_GREY),
        ("Test Samples", "Not published", "109 (10 percent)", C_GREY, C_GREY),
        ("Epochs Run", "50", "16, early stopped", C_GREY, C_GREY),
    ]
    for ri, (metric, paper_v, ours_v, pcol, ocol) in enumerate(rows):
        y = 1.6 + ri * 0.52
        rect(s, 0.3, y, 12.7, 0.48, C_CARD if ri % 2 == 0 else None, C_AMBER, 0.3)
        txt(s, metric, col_x[0] + 0.1, y + 0.07, col_w[0] - 0.2, 0.34, size=11,
            bold=True, color=C_WHITE)
        txt(s, paper_v, col_x[1] + 0.1, y + 0.07, col_w[1] - 0.2, 0.34, size=12,
            bold=True, color=pcol, align=PP_ALIGN.CENTER)
        txt(s, ours_v, col_x[2] + 0.1, y + 0.07, col_w[2] - 0.2, 0.34, size=12,
            bold=True, color=ocol, align=PP_ALIGN.CENTER)

    txt(s, "Why the gap?", 0.4, 6.15, 4.0, 0.38, size=14, bold=True, color=C_AMBER)
    reasons = [
        "The paper used 224 x 224 input, ours uses 75 x 75, so less landmark detail survives",
        "The original split sizes and preprocessing are not published, so exact matching is impossible",
        "The authors likely ran many experiments, and published the best result",
        "This project is a clean reproduction without hyperparameter tuning",
        "Skeletonization quality depends on the MediaPipe version in use",
    ]
    for i, r in enumerate(reasons):
        txt(s, f"{i + 1}. {r}", 0.5 + (i % 2) * 6.4, 6.55 + (i // 2) * 0.28, 6.2, 0.26,
            size=9, color=C_GREY)

    bottombar(s, 0.22, C_AMBER)


def slide_application() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Application Interface",
            "Streamlit-based web interface with real-time webcam and file upload")

    feature_grid(s, [
        (C_BLUE, "Live Webcam",
         ["Real-time pose classification", "Skeleton overlay on video feed",
          "Confidence threshold of 60 percent", "Top-3 pose predictions",
          "Joint angle feedback", "FPS and latency display"]),
        (C_CYAN, "Image Upload",
         ["Upload JPG and PNG images", "Body landmark detection",
          "Skeleton visualization", "Pose classification result",
          "Confidence score and top-3", "Handles no-person images"]),
        (C_GREEN, "Video Analysis",
         ["Upload MP4 and AVI video", "Frame-by-frame sampling",
          "Per-class prediction counts", "Dominant pose summary",
          "Timeline of detected poses", "Export predictions to CSV"]),
        (C_PURPLE, "Posture Feedback",
         ["Joint angle calculation", "Elbow, shoulder, hip, knee",
          "Symmetry comparison", "Rule-based corrective tips",
          "Clear warning messages", "Medical disclaimer shown"]),
        (C_AMBER, "Session History",
         ["SQLite local storage", "Session date and duration",
          "Pose counts per session", "Dominant pose tracking",
          "Export to CSV", "No footage saved by default"]),
        (C_GREEN, "Evaluation Dashboard",
         ["Confusion matrix visualization", "Training and validation curves",
          "Per-class F1 scores", "Accuracy, precision, recall, F1",
          "Latency and FPS benchmarks", "Exported to outputs folder"]),
    ])
    bottombar(s, 0.22, C_CYAN)


def slide_feedback() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Posture Analysis and Feedback",
            "Rule-based joint angle estimation from MediaPipe landmark geometry")

    txt(s, "How It Works", 0.3, 1.1, 5.5, 0.4, size=15, bold=True, color=C_GREEN)
    steps_text = [
        "MediaPipe Pose extracts 33 body landmarks with x, y, z image coordinates\nand a visibility score for each.",
        "For each joint triple the angle is computed with the cosine rule\nfrom the three 3D coordinates.",
        "Feedback is generated by comparing measured angles against\npose-specific expected ranges.",
        "Rules fire only when the relevant landmarks are sufficiently\nvisible, with visibility above 0.5.",
    ]
    for i, step in enumerate(steps_text):
        y = 1.55 + i * 1.2
        card(s, 0.3, y, 5.9, 1.12, C_CARD, C_GREEN, 0.6)
        badge(s, str(i + 1), 0.3, y, 0.55, 1.12, C_GREEN, size=16, text_color=C_BG)
        txt(s, step, 0.95, y + 0.12, 5.15, 0.9, size=11, color=C_WHITE)

    txt(s, "Joint Angle Calculations", 6.5, 1.1, 6.5, 0.4, size=15, bold=True, color=C_CYAN)
    angles_list = [
        ("Left Elbow", "Left shoulder, elbow, wrist"),
        ("Right Elbow", "Right shoulder, elbow, wrist"),
        ("Left Shoulder", "Left elbow, shoulder, hip"),
        ("Right Shoulder", "Right elbow, shoulder, hip"),
        ("Left Hip", "Left shoulder, hip, knee"),
        ("Right Hip", "Right shoulder, hip, knee"),
        ("Left Knee", "Left hip, knee, ankle"),
        ("Right Knee", "Right hip, knee, ankle"),
    ]
    for i, (joint, formula) in enumerate(angles_list):
        x = 6.5 + (i // 4) * 3.3
        y = 1.55 + (i % 4) * 0.68
        card(s, x, y, 3.15, 0.62, C_CARD, C_CYAN, 0.4)
        txt(s, joint, x + 0.12, y + 0.05, 1.5, 0.28, size=11, bold=True, color=C_CYAN)
        txt(s, formula, x + 0.12, y + 0.32, 2.95, 0.26, size=9, color=C_GREY)

    txt(s, "Feedback Rules", 6.5, 4.35, 6.5, 0.4, size=15, bold=True, color=C_AMBER)
    feedback_rules = [
        "No person detected in frame, ask user to stand so the whole body is visible",
        "Landmark visibility low, ask user to move closer or improve lighting",
        "Left and right joint asymmetry above 20 degrees, report the differing side",
        "Shoulder tilt from horizontal above 15 degrees, report uneven shoulders",
        "Hip tilt from horizontal above 15 degrees, report uneven hips",
    ]
    for i, rule in enumerate(feedback_rules):
        y = 4.8 + i * 0.5
        rect(s, 6.5, y + 0.06, 0.04, 0.32, C_AMBER)
        txt(s, rule, 6.65, y, 6.2, 0.44, size=10, color=C_WHITE)

    txt(s, "This feedback is heuristic and rule-based. It is not medical advice and "
           "does not assess whether a pose is performed correctly.",
        0.3, 7.12, 12.5, 0.28, size=10, color=C_RED, italic=True)
    bottombar(s, 0.22, C_GREEN)


def slide_performance() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Inference Performance",
            "Real-time benchmarks: latency, throughput, and confidence behaviour")

    metric_cards(s, [
        ("67.3 ms", "Mean Latency", "Per frame on CPU"),
        ("~15 FPS", "Throughput", "Batch size 1, CPU"),
        ("0.60", "Confidence Gate", "Threshold for showing a pose"),
        ("10.8 FPS", "Observed FPS", "Live session including display"),
    ], left=0.3, top=1.1, width=12.7, height=2.0, gap=0.25, cols=4, border=C_BLUE,
        big_size=32)

    txt(s, "Latency Distribution (20 runs, batch size 1, CPU)", 0.3, 3.25, 6.0, 0.4,
        size=14, bold=True, color=C_BLUE)
    kv_table(
        s,
        [("Mean", "67.29 ms"), ("Standard Deviation", "3.67 ms"),
         ("95th Percentile", "71.31 ms"), ("Approximate Minimum", "about 60 ms"),
         ("Approximate Maximum", "about 75 ms")],
        left=0.3, top=3.7, width=5.9, key_w=3.0, value_w=2.7, row_h=0.55,
        key_color=C_GREY, value_size=12,
    )

    txt(s, "Confidence Threshold Behaviour", 6.5, 3.25, 6.5, 0.4, size=14, bold=True,
        color=C_GREEN)
    kv_table(
        s,
        [("Threshold", "0.60"), ("Below threshold", "Shows Uncertain, does not guess"),
         ("Top-3", "Always shown regardless of threshold"),
         ("Smoothing", "Moving average over 5 frames"),
         ("No person", "Reports no pose when MediaPipe finds none")],
        left=6.5, top=3.7, width=6.5, key_w=1.9, value_w=4.4, row_h=0.55,
        key_color=C_GREEN, value_size=10, value_align=PP_ALIGN.LEFT,
    )

    txt(s, "All benchmarks run on CPU only with no GPU. TensorFlow oneDNN optimizations "
           "are enabled.",
        0.3, 7.1, 12.5, 0.28, size=10, color=C_GREY, italic=True)
    bottombar(s, 0.22, C_BLUE)


def slide_limitations() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Limitations and Challenges", "Known constraints and areas for improvement")
    info_grid(s, [
        ("Dataset Size",
         "The Kaggle dataset contains only about 900 images. A larger and more diverse "
         "dataset would improve robustness. Augmentation is limited because skeleton "
         "inputs are sparse.", C_RED),
        ("Class Imbalance",
         "Plank has 266 images while tree has only 160, a ratio of 1.66, so plank is "
         "over-predicted. Stratified splitting mitigates but does not remove this bias.",
         C_AMBER),
        ("Input Resolution",
         "75 x 75 skeleton images discard fine landmark detail that the paper captured "
         "at 224 x 224. This is a main contributor to the accuracy gap.", C_AMBER),
        ("2D Pose Estimation",
         "MediaPipe Pose is a 2D model while yoga poses are inherently 3D. A frontal "
         "view can make left and right limbs look similar, confusing poses that differ "
         "mainly in depth.", C_CYAN),
        ("Webcam Generalisation",
         "The model was trained on curated photos with consistent lighting and plain "
         "backgrounds. Real webcam frames vary in lighting, clothing, and background, "
         "which reduces real-world accuracy.", C_BLUE),
        ("Classification vs Quality",
         "Classification does not assess pose correctness. A person in poor alignment "
         "can still be classified as tree. Posture feedback is heuristic, not a medical "
         "assessment.", C_PURPLE),
    ])
    bottombar(s, 0.22, C_RED)


def slide_future() -> None:
    s = add_slide()
    bg(s)
    heading(s, "Future Work", "Planned improvements and extensions")
    info_grid(s, [
        ("Higher Resolution Skeletons",
         "Increase from 75 x 75 to 224 x 224 to match the paper's input size. This "
         "preserves more landmark detail and should close much of the accuracy gap.",
         C_BLUE),
        ("Larger Balanced Dataset",
         "Collect more images for underrepresented classes such as tree and goddess. "
         "Apply class-weighted loss or oversampling to reduce the 1.66 to 1 imbalance.",
         C_CYAN),
        ("Lightweight Landmark Classifier",
         "Train a small MLP or gradient boosted model directly on the 33 by 3 landmark "
         "coordinates. This bypasses the image representation entirely and runs faster.",
         C_GREEN),
        ("3D Pose Estimation",
         "Move to a 3D landmark model so z-coordinates are available. Angles can then be "
         "computed in 3D, improving discrimination of depth-dependent poses.", C_PURPLE),
        ("Difficulty Levels",
         "Group poses by difficulty into beginner, intermediate, and advanced. Train a "
         "hierarchical classifier that predicts difficulty first, then the pose.",
         C_AMBER),
        ("Pose Hold Timer",
         "Track how long a user holds a predicted pose above the confidence threshold, "
         "with a timer and streak counter to support home practice.", C_GREEN),
    ])
    bottombar(s, 0.22, C_PURPLE)


def slide_conclusion() -> None:
    s = add_slide()
    bg(s)
    rect(s, 0, 0, 5.5, 7.5, C_NAVY)
    rect(s, 5.5, 0, 0.08, 7.5, C_CYAN)

    txt(s, "Conclusion", 0.5, 0.8, 4.8, 0.8, size=36, bold=True, color=C_WHITE)
    txt(s, "Key Takeaways", 0.5, 1.6, 4.8, 0.5, size=18, bold=True, color=C_CYAN)
    takeaways = [
        "End-to-end yoga pose classification system built from scratch",
        "Faithful reproduction of the YogaConv2D CNN architecture",
        "Full pipeline of MediaPipe Pose, skeletonization, CNN, feedback",
        "Streamlit interface with webcam, image, and video support",
        "Reproducible training with documented configuration and scripts",
        "59 automated tests passing across the core modules",
        "71.56 percent test accuracy, against 97 percent reported in the paper",
        "Real-time inference at approximately 15 FPS on CPU",
    ]
    for i, t in enumerate(takeaways):
        y = 2.15 + i * 0.6
        rect(s, 0.5, y + 0.08, 0.05, 0.42, C_CYAN)
        txt(s, t, 0.72, y, 4.55, 0.55, size=12, color=C_WHITE)

    txt(s, "By The Numbers", 6.1, 0.8, 6.9, 0.6, size=22, bold=True, color=C_WHITE)
    stats = [
        ("5", "Yoga Pose Classes"), ("1080", "Total Dataset Images"),
        ("71.56 %", "Test Accuracy"), ("70.32 %", "Macro F1-Score"),
        ("~15 FPS", "Inference Speed on CPU"), ("59", "Tests Passing"),
        ("16", "Training Epochs"), ("259 s", "Training Time"),
    ]
    for i, (big, label) in enumerate(stats):
        x = 6.1 + (i % 2) * 3.5
        y = 1.5 + (i // 2) * 1.4
        card(s, x, y, 3.2, 1.25, C_CARD, C_CYAN, 0.8)
        txt(s, big, x + 0.15, y + 0.12, 2.9, 0.65, size=26, bold=True, color=C_CYAN,
            align=PP_ALIGN.CENTER)
        txt(s, label, x + 0.15, y + 0.75, 2.9, 0.42, size=11, color=C_GREY,
            align=PP_ALIGN.CENTER)

    rect(s, 0, 7.2, 13.333, 0.3, C_CYAN)
    txt(s, "YogaVision  |  Deep Learning for Human Pose Understanding  |  Thank You",
        0.3, 7.22, 12.5, 0.25, size=10, color=C_WHITE)


def main() -> None:
    slide_title()
    slide_agenda()
    slide_problem()
    slide_research()
    slide_architecture()
    slide_dataset()
    slide_model()
    slide_training()
    slide_training_results()
    slide_evaluation()
    slide_comparison()
    slide_application()
    slide_feedback()
    slide_performance()
    slide_limitations()
    slide_future()
    slide_conclusion()

    out = save("outputs/YogaVision_Presentation.pptx")
    print(f"Saved {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
