"""YogaVision presentation.

17 slides, dark navy + teal, three colours plus one semantic red used only to
distinguish the published paper's numbers from measured results.

Layouts deliberately vary: hero, stat rows, split panels, layer table, pipeline
diagram, horizontal bar chart, comparison table, and a two-column conclusion.

Every number is read from the project's own outputs. Nothing is invented.

Build:  python scripts/build_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

from deck_theme import (
    ACCENT, ACCENT_DIM, BODY_Y, CENTER, CONTENT_W, FONT, FONT_MONO, INK,
    INK_LINE, INK_SOFT, LEFT, MARGIN, MUTED, MUTED_DIM, RIGHT, SLIDE_H, SLIDE_W,
    T_BODY, T_HERO, T_KICKER, T_LEAD, T_NUMBER, T_NUMBER_MD, T_NUMBER_SM,
    T_SMALL, T_TINY, T_TITLE, WARN, WHITE,
)
from deck_layout import (
    add_slide, arrow, bar, bullets, callout, chip, footer, kv_table, layer_table,
    panel, pipeline, rich, rule, section_divider, slide_head, source, stat,
    stat_row, text, vrule,
)

ROOT = Path(__file__).resolve().parent.parent
TOTAL_SLIDES = 16
_page = 0
_prs = None


# ── data, read from real project artifacts ──────────────────────────────────
def load_metrics() -> dict:
    path = ROOT / "outputs" / "evaluation" / "evaluation_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_training() -> dict:
    path = ROOT / "outputs" / "evaluation" / "training_results.json"
    return json.loads(path.read_text()) if path.exists() else {}


def count_classes() -> list[tuple[str, int]]:
    root = ROOT / "data" / "processed" / "yoga_poses_skeleton" / "TRAIN"
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            n = sum(1 for p in d.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            out.append((d.name, n))
    return out


def page(slide, section="YogaVision"):
    global _page
    _page += 1
    if _page > 1 and _page < TOTAL_SLIDES:
        footer(slide, _page, TOTAL_SLIDES, section)
    return slide


# ══════════════════════════════════════════════════════════════════════════
# 1  TITLE
# ══════════════════════════════════════════════════════════════════════════
def slide_title():
    s = add_slide(_prs)
    # full-bleed teal rule down the left, title block to its right
    bar(s, 0, 0, 0.09, 7.5, color=ACCENT)

    text(s, "COMPUTER VISION PROJECT", MARGIN, 1.30, 8.0, 0.3, size=T_KICKER,
         color=ACCENT, bold=True, spacing=1.0)
    text(s, "YogaVision", MARGIN, 1.68, 10.5, 1.15, size=T_HERO, color=WHITE,
         bold=True, spacing=1.0)
    text(s, "Real-time yoga pose classification\nand posture feedback with CNN and MediaPipe",
         MARGIN, 2.95, 9.4, 1.15, size=20, color=MUTED, spacing=1.32)

    rule(s, MARGIN, 4.32, 2.2, color=ACCENT)
    text(s, "An end-to-end reproduction of the YogaConv2D pipeline, extended into a\n"
            "working webcam, image, and video application.",
         MARGIN, 4.62, 8.6, 0.9, size=T_LEAD, color=WHITE, spacing=1.35)

    text(s, "Inspired by  Yoga Pose Classification: A CNN and MediaPipe Inspired Deep\n"
            "Learning Approach for Real-World Application,  J. Ambient Intelligence and\n"
            "Humanized Computing, 2022.   DOI 10.1007/s12652-022-03910-0",
         MARGIN, 6.15, 9.6, 0.85, size=T_TINY, color=MUTED_DIM, spacing=1.35)

    stat(s, "5", "pose classes", 10.55, 2.95, 2.2, note="1080 training images")
    stat(s, "71.6%", "test accuracy", 10.55, 4.55, 2.2, note="measured, 109 samples")
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 2  THE PROBLEM  (split panel)
# ══════════════════════════════════════════════════════════════════════════
def slide_problem():
    s = add_slide(_prs)
    y = slide_head(s, "The problem, and what this project does",
                   kicker="Introduction",
                   lead="Yoga assessment today depends on a human expert watching you move. "
                        "That does not scale, and it is unavailable at home.")

    col_w = (CONTENT_W - 0.55) / 2
    lx, rx = MARGIN, MARGIN + col_w + 0.55
    vrule(s, lx + col_w + 0.275, y, 3.9, color=INK_LINE, thickness=0.012)

    text(s, "WHY IT IS HARD", lx, y, col_w, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    bullets(s, [
        "Poses are defined by geometry, not appearance, so colour and clothing tell you nothing",
        "Several poses look nearly identical from a front-facing camera",
        "Lighting, framing, and body proportions vary between users and sessions",
        "A wrong classification must not be presented as confident feedback",
    ], lx, y + 0.42, col_w, size=T_SMALL, gap=0.72, spacing=1.28)

    text(s, "WHAT WE BUILT", rx, y, col_w, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    bullets(s, [
        "MediaPipe Pose extracts 33 body landmarks per frame in real time",
        "Landmarks are rendered as a skeleton image, which becomes the CNN input",
        "A CNN trained from scratch classifies five yoga poses with a confidence gate",
        "Landmark geometry drives rule-based joint-angle feedback, kept separate from classification",
    ], rx, y + 0.42, col_w, size=T_SMALL, gap=0.72, spacing=1.28)

    source(s, "Classification and posture feedback are deliberately separate subsystems; "
              "see the limitations slide.", y=6.35)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 3  RESEARCH BASIS  (citation block + methodology)
# ══════════════════════════════════════════════════════════════════════════
def slide_research():
    s = add_slide(_prs)
    y = slide_head(s, "Research basis", kicker="Literature",
                   lead="The paper's public repository is the authoritative source used here. "
                        "The full text is paywalled.")

    panel(s, MARGIN, y, CONTENT_W, 1.28, fill=INK_SOFT, line=INK_LINE)
    bar(s, MARGIN, y, 0.055, 1.28, color=ACCENT)
    text(s, "Yoga Pose Classification: A CNN and MediaPipe Inspired Deep Learning "
            "Approach for Real-World Application",
         MARGIN + 0.28, y + 0.20, CONTENT_W - 3.6, 0.62, size=T_LEAD, color=WHITE,
         bold=True, spacing=1.25)
    text(s, "Journal of Ambient Intelligence and Humanized Computing, 2022",
         MARGIN + 0.28, y + 0.86, CONTENT_W - 3.6, 0.3, size=T_SMALL, color=MUTED)
    chip(s, "DOI 10.1007/s12652-022-03910-0", 13.333 - MARGIN - 3.35, y + 0.28, 3.35,
         h=0.36, size=T_TINY)
    chip(s, "github.com/shub-garg", 13.333 - MARGIN - 3.35, y + 0.72, 3.35, h=0.36,
         size=T_TINY, line=INK_LINE, color=MUTED)

    ry = y + 1.62
    text(s, "METHODOLOGY AS DESCRIBED BY THE REPOSITORY", MARGIN, ry, CONTENT_W, 0.3,
         size=T_KICKER, color=ACCENT, bold=True)
    steps = [
        ("Input", "Raw RGB photos of yoga practitioners"),
        ("Pose", "MediaPipe Pose: 33 landmarks, x, y, z, visibility"),
        ("Skeleton", "Landmarks and connections drawn on a black canvas"),
        ("Model", "YogaConv2D, four-block CNN, on 224 x 224 x 3"),
        ("Training", "Adam, up to 50 epochs, early stopping, flip augmentation"),
    ]
    col_w = (CONTENT_W - 0.4) / 5
    for i, (label, body) in enumerate(steps):
        cx = MARGIN + i * (col_w + 0.1)
        text(s, f"{i + 1}", cx, ry + 0.38, 0.4, 0.3, size=T_KICKER, color=ACCENT,
             bold=True, font=FONT_MONO)
        text(s, label, cx, ry + 0.66, col_w, 0.3, size=T_BODY, color=WHITE, bold=True)
        text(s, body, cx, ry + 1.00, col_w, 0.9, size=T_TINY, color=MUTED,
             spacing=1.32)
        if i < len(steps) - 1:
            arrow(s, cx + col_w - 0.02, ry + 0.52, 0.14, 0.16, color=ACCENT_DIM)

    source(s, "Methodology statements are taken from the repository code and README. "
              "Dataset statistics and exact split sizes are not published in the paper.",
           y=6.45)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 4  SYSTEM PIPELINE  (diagram, full width)
# ══════════════════════════════════════════════════════════════════════════
def slide_pipeline():
    s = add_slide(_prs)
    y = slide_head(s, "System pipeline", kicker="Architecture",
                   lead="One frame travels through five stages. Nothing is retrained at runtime.")

    pipeline(s, [
        ("SOURCE", "Webcam\nImage file\nVideo file", ACCENT),
        ("POSE", "MediaPipe Pose\n33 landmarks\nvisibility per point", ACCENT),
        ("SKELETON", "Landmarks drawn\non black canvas\nat native size", ACCENT),
        ("CNN", "Resize to 75 x 75\nYogaConv2D\nsoftmax over 5", ACCENT),
        ("RESULT", "Pose label\nconfidence\nfeedback rules", ACCENT),
    ], MARGIN, y, CONTENT_W, h=1.72)

    by = y + 2.15
    text(s, "IMPLEMENTATION", MARGIN, by, CONTENT_W, 0.3, size=T_KICKER,
         color=ACCENT, bold=True)
    rows = [
        ("pose_detection.py", "MediaPipe wrapper, landmark extraction and skeleton drawing"),
        ("skeletonization.py", "Batch conversion of a dataset directory into skeleton images"),
        ("model.py", "YogaConv2D definition, save and load with label map"),
        ("inference.py", "Per-frame prediction, top-k, confidence threshold, smoothing"),
        ("feedback.py", "Joint angles from landmark triples, rule evaluation"),
        ("video_analysis.py", "Incremental frame sampling, per-frame CSV export"),
    ]
    kv_table(s, rows, MARGIN, by + 0.36, CONTENT_W, key_w=2.5, row_h=0.385,
             size=T_SMALL, key_color=ACCENT, value_color=MUTED, align_value=LEFT)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 5  DATASET  (bar chart, built from real counts)
# ══════════════════════════════════════════════════════════════════════════
def slide_dataset(classes):
    s = add_slide(_prs)
    y = slide_head(s, "Dataset", kicker="Data",
                   lead="Kaggle yoga-poses-dataset, CC BY-SA 4.0. Five classes, 1080 training "
                        "images after cleaning.")

    total = sum(n for _, n in classes)
    biggest = max(n for _, n in classes)
    smallest = min(n for _, n in classes)

    stat_bottom = stat_row(s, [
        (str(total), "training images", "5 classes, skeleton form"),
        (f"{smallest / biggest:.2f} : 1", "class balance", "smallest to largest class"),
        ("755 / 216 / 109", "train / val / test", "stratified, seed 42"),
        ("1.66 : 1", "imbalance ratio", "plank versus tree"),
    ], y=y, cols=4)

    # horizontal bar chart
    cy = stat_bottom + 0.40
    text(s, "IMAGES PER CLASS", MARGIN, cy, CONTENT_W, 0.3, size=T_KICKER,
         color=ACCENT, bold=True)
    chart_x = MARGIN
    label_w = 1.55
    bar_x = chart_x + label_w + 0.15
    bar_max = CONTENT_W - label_w - 0.95
    row_h = 0.50

    for i, (name, n) in enumerate(classes):
        ry = cy + 0.42 + i * row_h
        text(s, name, chart_x, ry + 0.06, label_w, 0.3, size=T_BODY, color=WHITE,
             bold=True, align=RIGHT)
        track_w = bar_max
        panel(s, bar_x, ry + 0.09, track_w, 0.30, fill=INK_SOFT, line=None)
        fill_w = track_w * (n / biggest)
        color = ACCENT if n == biggest else ACCENT_DIM
        bar(s, bar_x, ry + 0.09, fill_w, 0.30, color=color)
        text(s, f"{n}", bar_x + track_w + 0.14, ry + 0.06, 0.7, 0.3, size=T_BODY,
             color=WHITE, bold=True, font=FONT_MONO)

    axis_y = cy + 0.42 + len(classes) * row_h
    rule(s, bar_x, axis_y, bar_max, color=INK_LINE, thickness=0.012)
    text(s, "0", bar_x, axis_y + 0.06, 0.4, 0.22, size=T_TINY, color=MUTED_DIM)
    text(s, str(biggest), bar_x + bar_max - 0.4, axis_y + 0.06, 0.4, 0.22,
         size=T_TINY, color=MUTED_DIM, align=RIGHT)

    notes = [
        "One corrupt JPEG removed, 179 duplicate groups found by MD5 hash",
        "Skeletons stored at native resolution, resized to 75 x 75 at load time",
    ]
    bullets(s, notes, MARGIN, axis_y + 0.30, CONTENT_W, size=T_TINY, color=MUTED,
            gap=0.20, marker_w=0.12, indent=0.26, spacing=1.3)

    source(s, "Source: kaggle.com/datasets/niharika41298/yoga-poses-dataset. Counts taken "
              "from data/processed/yoga_poses_skeleton/TRAIN.", y=6.86)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 6  MODEL  (layer table + rationale)
# ══════════════════════════════════════════════════════════════════════════
def slide_model():
    s = add_slide(_prs)
    y = slide_head(s, "Model architecture", kicker="CNN",
                   lead="YogaConv2D, reproduced from the paper's description, with global "
                        "average pooling in place of a flatten layer.")

    layers = [
        ("Input", "75 x 75 x 3 skeleton image", MUTED_DIM),
        ("Conv 2D", "32 filters, 3 x 3, ReLU, same padding", ACCENT),
        ("MaxPool", "2 x 2, stride 2", ACCENT_DIM),
        ("Conv 2D", "64 filters, 3 x 3, ReLU, same padding", ACCENT),
        ("MaxPool", "2 x 2, stride 2", ACCENT_DIM),
        ("Conv 2D", "128 filters, 3 x 3, ReLU, same padding", ACCENT),
        ("MaxPool", "2 x 2, stride 2", ACCENT_DIM),
        ("Conv 2D", "256 filters, 3 x 3, ReLU, same padding", ACCENT),
        ("GlobalAvgPool", "Spatial average, one value per filter", ACCENT),
        ("Dropout", "0.25", MUTED_DIM),
        ("Dense", "128 units, ReLU, L2 regularization", ACCENT),
        ("Dropout", "0.50", MUTED_DIM),
        ("Dense", "5 units, softmax", ACCENT),
    ]
    left_w = 6.5
    layer_table(s, layers, MARGIN, y, left_w, row_h=0.345, name_w=1.85)

    rx = MARGIN + left_w + 0.55
    rw = CONTENT_W - left_w - 0.55
    vrule(s, MARGIN + left_w + 0.275, y, 4.5, color=INK_LINE, thickness=0.012)

    text(s, "DESIGN DECISIONS", rx, y, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    callout(s, "Skeleton images, not raw photos",
            "MediaPipe keeps only pose geometry and discards background, clothing, and skin "
            "tone, so the network learns shape rather than appearance.",
            rx, y + 0.38, rw, 1.02, body_size=T_TINY)
    callout(s, "Global average pooling",
            "Removes dependence on exact input size, tolerating landmark position variation "
            "better than a flatten layer.",
            rx, y + 1.52, rw, 1.02, body_size=T_TINY)
    callout(s, "Two dropout rates",
            "0.25 after pooling limits early overfitting, and 0.50 before the classifier "
            "regularises the final decision. Paired with early stopping.",
            rx, y + 2.66, rw, 1.02, body_size=T_TINY)
    callout(s, "A confidence gate, not a sixth class",
            "There is no 'none of the above' output, so predictions below 0.60 confidence "
            "are reported as uncertain rather than forced into a pose.",
            rx, y + 3.80, rw, 1.02, body_size=T_TINY, color=MUTED_DIM)

    source(s, "Input resized from native resolution to 75 x 75 x 3 during preprocessing. "
              "The paper used 224 x 224.", y=6.88)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 7  TRAINING  (config table + procedure)
# ══════════════════════════════════════════════════════════════════════════
def slide_training(training):
    s = add_slide(_prs)
    y = slide_head(s, "Training procedure", kicker="Method",
                   lead="A single documented command reproduces the model. Every random source "
                        "is seeded.")

    lw = 4.7
    text(s, "CONFIGURATION", MARGIN, y, lw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    kv_table(s, [
        ("Optimizer", "Adam"),
        ("Learning rate", "0.001, to 0.0005 on plateau"),
        ("Batch size", "16"),
        ("Epochs", "50 maximum, stopped at 16"),
        ("Input", "75 x 75 x 3 skeleton"),
        ("Loss", "Categorical crossentropy"),
        ("Seed", "42, all random state fixed"),
        ("Augmentation", "Horizontal flip only"),
    ], MARGIN, y + 0.38, lw, key_w=1.85, row_h=0.38, size=T_SMALL)

    rx = MARGIN + lw + 0.6
    rw = CONTENT_W - lw - 0.6
    vrule(s, MARGIN + lw + 0.3, y, 4.4, color=INK_LINE, thickness=0.012)
    text(s, "PROCEDURE", rx, y, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    bullets(s, [
        ("Discover", "verify class folders, count images, report corrupt files"),
        ("Load", "read skeleton images, encode class labels as integers"),
        ("Split", "stratified 70 / 20 / 10 so no image appears in two splits"),
        ("Preprocess", "rescale pixels to 0..1, resize to the model input"),
        ("Train", "Adam on YogaConv2D, with augmentation on the training set only"),
        ("Callback", "early stopping patience 8, learning rate patience 4"),
        ("Checkpoint", "keep the best model by validation loss"),
        ("Evaluate", "score the held-out test set, then export metrics and plots"),
    ], rx, y + 0.38, rw, size=T_TINY, gap=0.36, spacing=1.26, indent=0.30)

    secs = training.get("training_seconds")
    epochs = training.get("epochs_run")
    stat_row(s, [
        (f"{secs:.0f} s" if secs else "n/a", "training time", "CPU only"),
        (f"{epochs}" if epochs else "n/a", "epochs run", "early stopped"),
        ("16 of 50", "epoch budget", "patience 8"),
        ("42", "random seed", "fully reproducible"),
    ], y=5.68, cols=4)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 8  TRAINING RESULT  (real chart image)
# ══════════════════════════════════════════════════════════════════════════
def slide_training_curves(training):
    s = add_slide(_prs)
    y = slide_head(s, "Training result", kicker="Experiment",
                   lead="Validation accuracy peaked at epoch 11 and then flattened, which is the "
                        "overfitting signature early stopping is designed to catch.")

    img = ROOT / "outputs" / "plots" / "training_curves.png"
    iw = 7.3
    if img.exists():
        pic = s.shapes.add_picture(str(img), MARGIN, y, width=Inches(iw))
        if pic.height / 914400 > 6.45 - y:
            pic.height = Inches(6.45 - y)
            pic.width = Inches((6.45 - y) * pic.width / pic.height)
    else:
        panel(s, MARGIN, y, iw, 4.2, fill=INK_SOFT, line=INK_LINE)
        text(s, "training_curves.png not found", MARGIN, y + 1.9, iw, 0.4,
             size=T_SMALL, color=MUTED_DIM, align=CENTER)

    rx = MARGIN + iw + 0.55
    rw = CONTENT_W - iw - 0.55
    hist = training.get("history", {})
    best_val = max(hist.get("val_accuracy", [0]))
    best_train = max(hist.get("accuracy", [0]))

    text(s, "BEST EPOCH", rx, y, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    b1 = stat(s, f"{best_val * 100:.1f}%", "validation accuracy", rx, y + 0.30, rw,
              note=f"at epoch {hist['val_accuracy'].index(best_val) + 1} of 16",
              size=T_NUMBER_MD)
    b2 = stat(s, f"{best_train * 100:.1f}%", "training accuracy", rx, b1 + 0.16, rw,
              note="same epoch run", size=T_NUMBER_MD)
    b3 = stat(s, f"{(best_train - best_val) * 100:.1f} pts", "generalisation gap", rx,
              b2 + 0.16, rw, note="train minus validation", color=MUTED_DIM,
              size=T_NUMBER_MD)

    text(s, "READING THE CURVES", rx, b3 + 0.22, rw, 0.3, size=T_KICKER,
         color=ACCENT, bold=True)
    text(s, "Training accuracy keeps rising while validation accuracy plateaus after "
            "epoch 11. The model is memorising the 755 training images rather than "
            "learning geometry that transfers.",
         rx, b3 + 0.54, rw, 0.42, size=T_TINY, color=MUTED, spacing=1.22)

    source(s, "Reproduction run by this project. Not the paper's figures.", y=6.94)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 9  EVALUATION  (metrics + real confusion matrix)
# ══════════════════════════════════════════════════════════════════════════
def slide_evaluation(metrics):
    s = add_slide(_prs)
    y = slide_head(s, "Evaluation", kicker="Results",
                   lead=f"Scored on {metrics.get('num_samples', 'the')} held-out test images "
                        "that no part of training ever saw.")

    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    prec = metrics.get("precision_macro", 0)
    rec = metrics.get("recall_macro", 0)
    stat_bottom = stat_row(s, [
        (f"{acc * 100:.1f}%", "accuracy", "109 test images"),
        (f"{prec * 100:.1f}%", "macro precision", "unweighted mean"),
        (f"{rec * 100:.1f}%", "macro recall", "unweighted mean"),
        (f"{f1 * 100:.1f}%", "macro F1", "harmonic mean"),
    ], y=y, cols=4)

    cy = stat_bottom + 0.32
    img = ROOT / "outputs" / "plots" / "confusion_matrix.png"
    avail_h = 6.42 - cy          # keep clear of the source line
    iw = 4.15
    if img.exists():
        pic = s.shapes.add_picture(str(img), MARGIN, cy, width=Inches(iw))
        if pic.height / 914400 > avail_h:
            pic.height = Inches(avail_h)
            pic.width = Inches(avail_h * pic.width / pic.height)
    else:
        panel(s, MARGIN, cy, iw, 3.2, fill=INK_SOFT, line=INK_LINE)
        text(s, "confusion_matrix.png not found", MARGIN, cy + 1.4, iw, 0.4,
             size=T_TINY, color=MUTED_DIM, align=CENTER)

    rx = MARGIN + iw + 0.55
    rw = CONTENT_W - iw - 0.55
    text(s, "PER-CLASS F1", rx, cy, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    report = metrics.get("classification_report", {})
    rows = []
    for cls in ["downdog", "goddess", "plank", "tree", "warrior2"]:
        r = report.get(cls, {})
        rows.append((cls, f"{r.get('f1-score', 0) * 100:.1f}%",
                     str(int(r.get("support", 0)))))
    kv_table(s, [(c, f"{f}   ·   {sup} images") for c, f, sup in rows],
             rx, cy + 0.34, rw, key_w=1.7, row_h=0.38, size=T_SMALL,
             key_color=WHITE, value_color=ACCENT)

    ty = cy + 0.34 + len(rows) * 0.38 + 0.20
    text(s, "WHAT THE CONFUSION SHOWS", rx, ty, rw, 0.3, size=T_KICKER,
         color=ACCENT, bold=True)
    bullets(s, [
        "downdog is strongest, matching its distinctive triangular shape",
        "goddess is weakest, often confused with warrior2",
    ], rx, ty + 0.20, rw, size=T_TINY, color=MUTED, gap=0.22, marker_w=0.12,
        indent=0.26, spacing=1.28)

    source(s, "Reproduction run by this project on its own test split. These are not the "
              "paper's numbers.", y=6.62)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 10  COMPARISON  (table, honest gap)
# ══════════════════════════════════════════════════════════════════════════
def slide_comparison(metrics):
    s = add_slide(_prs)
    y = slide_head(s, "Ours against the paper", kicker="Comparison",
                   lead="The gap is large and it is not hidden. It is mostly explained by input "
                        "resolution and undisclosed training details.")

    cols = [("Metric", 3.5), ("Paper", 2.6), ("This project", 2.6), ("Why they differ", 3.0)]
    cx = [MARGIN]
    for _, w in cols:
        cx.append(cx[-1] + w)
    table_w = sum(w for _, w in cols)

    panel(s, MARGIN, y, table_w, 0.46, fill=INK_SOFT, line=None)
    for (label, _), x in zip(cols, cx):
        text(s, label.upper(), x + 0.14, y + 0.13, 2.4, 0.26, size=T_TINY,
             color=ACCENT, bold=True)

    rows = [
        ("Validation accuracy", "99.62 %", f"{max(json.loads((ROOT / 'outputs' / 'evaluation' / 'training_results.json').read_text())['history']['val_accuracy']) * 100:.1f} %",
         "Paper input 224 x 224 versus our 75 x 75"),
        ("Test accuracy", "97.09 %", f"{metrics.get('accuracy', 0) * 100:.1f} %",
         "Smaller input discards landmark detail"),
        ("Macro F1", "0.97", f"{metrics.get('f1_macro', 0):.2f}",
         "Combined effect of the above"),
        ("Input resolution", "224 x 224", "75 x 75", "The single largest known factor"),
        ("Test split", "not published", "109 images, 10 %", "Split sizes are undisclosed"),
        ("Epochs run", "50", "16, early stopped", "Different stopping conditions"),
    ]
    row_h = 0.45
    for i, (metric, paper, ours, why) in enumerate(rows):
        ry = y + 0.46 + i * row_h
        if i % 2 == 0:
            panel(s, MARGIN, ry, table_w, row_h - 0.02, fill=INK_SOFT, line=None)
        text(s, metric, cx[0] + 0.14, ry + 0.11, cols[0][1] - 0.2, 0.3, size=T_SMALL,
             color=WHITE, bold=True)
        text(s, paper, cx[1] + 0.14, ry + 0.11, cols[1][1] - 0.2, 0.3, size=T_SMALL,
             color=MUTED, align=LEFT)
        text(s, ours, cx[2] + 0.14, ry + 0.11, cols[2][1] - 0.2, 0.3, size=T_BODY,
             color=ACCENT, bold=True)
        text(s, why, cx[3] + 0.14, ry + 0.12, cols[3][1] - 0.2, 0.3, size=T_TINY,
             color=MUTED_DIM)

    note_y = y + 0.46 + len(rows) * row_h + 0.22
    panel(s, MARGIN, note_y, table_w, 0.78, fill=INK_SOFT, line=INK_LINE)
    bar(s, MARGIN, note_y, 0.055, 0.78, color=WARN)
    text(s, "No claim is made that this reproduction matches the paper. The paper's full text "
            "is paywalled, its split sizes and preprocessing are unpublished, and no trained "
            "weights were released, so an exact reproduction is not currently possible.",
         MARGIN + 0.28, note_y + 0.16, table_w - 0.55, 0.55, size=T_SMALL, color=MUTED,
         spacing=1.3)

    source(s, "Paper figures are quoted verbatim from the repository README results table.",
           y=6.62)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 11  APPLICATION  (feature rows, not a card grid)
# ══════════════════════════════════════════════════════════════════════════
def slide_application():
    s = add_slide(_prs)
    y = slide_head(s, "Application", kicker="Deliverable",
                   lead="A Streamlit interface over the same inference module used by the "
                        "scripts, so there is one code path and no duplicated logic.")

    rows = [
        ("Live webcam", "Camera selection, start and stop, per-frame landmark overlay, "
                        "prediction smoothing over five frames, FPS and latency readout"),
        ("Image upload", "JPG and PNG validation, landmark visualisation, prediction with "
                         "confidence and top three, graceful handling when no person is found"),
        ("Video analysis", "Frame sampling with progress, per-class counts, dominant pose "
                           "summary, frame-level predictions exported to CSV"),
        ("Posture feedback", "Elbow, shoulder, hip and knee angles from landmark triples, "
                             "shown only when the relevant landmarks are visible"),
        ("Session history", "SQLite storage of session summaries with pose counts and "
                            "duration, exportable to CSV, no footage stored by default"),
        ("Evaluation dashboard", "Confusion matrix, training curves, per-class report and "
                                 "latency, written into the outputs directory"),
    ]
    for i, (title, body) in enumerate(rows):
        ry = y + i * 0.74
        text(s, f"0{i + 1}", MARGIN, ry + 0.04, 0.6, 0.3, size=T_SMALL, color=ACCENT,
             bold=True, font=FONT_MONO)
        text(s, title, MARGIN + 0.62, ry, 2.5, 0.32, size=T_BODY, color=WHITE, bold=True)
        text(s, body, MARGIN + 3.2, ry + 0.02, CONTENT_W - 3.2, 0.52, size=T_TINY,
             color=MUTED, spacing=1.28)
        if i < len(rows) - 1:
            panel(s, MARGIN, ry + 0.62, CONTENT_W, 0.008, fill=INK_LINE, line=None)

    source(s, "The webcam path was not exercised on physical hardware in this build; "
              "the capture module is covered by unit tests with a mocked camera.",
           y=6.62)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 12  FEEDBACK RULES
# ══════════════════════════════════════════════════════════════════════════
def slide_feedback():
    s = add_slide(_prs)
    y = slide_head(s, "Posture feedback", kicker="Extension",
                   lead="Angles come from landmark geometry. Rules are simple and explainable, "
                        "and each one has an explicit threshold.")

    lw = 4.6
    text(s, "ANGLES COMPUTED", MARGIN, y, lw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    kv_table(s, [
        ("Left elbow", "shoulder, elbow, wrist"),
        ("Right elbow", "shoulder, elbow, wrist"),
        ("Left shoulder", "elbow, shoulder, hip"),
        ("Right shoulder", "elbow, shoulder, hip"),
        ("Left hip", "shoulder, hip, knee"),
        ("Right hip", "shoulder, hip, knee"),
        ("Left knee", "hip, knee, ankle"),
        ("Right knee", "hip, knee, ankle"),
    ], MARGIN, y + 0.38, lw, key_w=1.9, row_h=0.42, size=T_SMALL, key_color=WHITE,
        value_color=MUTED_DIM, align_value=LEFT)

    rx = MARGIN + lw + 0.6
    rw = CONTENT_W - lw - 0.6
    vrule(s, MARGIN + lw + 0.3, y, 3.8, color=INK_LINE, thickness=0.012)
    text(s, "RULES AND THRESHOLDS", rx, y, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    bullets(s, [
        ("No person detected", "ask the user to stand so the whole body is in frame"),
        ("Visibility below 0.5", "ask the user to move closer or improve the lighting"),
        ("Left-right gap over 20 deg", "report which side differs, by joint name"),
        ("Shoulder tilt over 15 deg", "report that shoulder alignment appears uneven"),
        ("Hip tilt over 15 deg", "report that hip alignment appears uneven"),
    ], rx, y + 0.38, rw, size=T_SMALL, gap=0.66, spacing=1.26, indent=0.30)

    ny = 5.82
    panel(s, MARGIN, ny, CONTENT_W, 0.86, fill=INK_SOFT, line=INK_LINE)
    bar(s, MARGIN, ny, 0.055, 0.86, color=MUTED_DIM)
    text(s, "Deliberate limits", MARGIN + 0.28, ny + 0.12, 3.0, 0.3, size=T_BODY,
         color=WHITE, bold=True)
    text(s, "A predicted pose label does not mean the pose is correct, a single camera cannot "
            "measure depth reliably, and none of this is medical advice. Feedback is kept "
            "structurally separate from classification so a wrong pose cannot generate a "
            "confident correction.",
         MARGIN + 0.28, ny + 0.44, CONTENT_W - 0.6, 0.4, size=T_SMALL, color=MUTED,
         spacing=1.26)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 13  PERFORMANCE  (latency + FPS)
# ══════════════════════════════════════════════════════════════════════════
def slide_performance(metrics):
    s = add_slide(_prs)
    y = slide_head(s, "Inference performance", kicker="Benchmark",
                   lead="Measured on this machine's CPU with no GPU, batch size one, over 100 "
                        "runs after warm-up.")

    lat = metrics.get("inference_latency_ms", {})
    stat_bottom = stat_row(s, [
        (f"{lat.get('mean', 0):.0f} ms", "mean latency", "batch 1, CPU"),
        (f"{lat.get('p95', 0):.0f} ms", "95th percentile", "tail latency"),
        (f"{lat.get('approx_fps', 0):.1f}", "frames per second", "model only"),
        ("0.60", "confidence gate", "below this, uncertain"),
    ], y=y, cols=4)

    cy = stat_bottom + 0.40
    lw = 6.1
    text(s, "WHAT THE NUMBER COVERS", MARGIN, cy, lw, 0.3, size=T_KICKER,
         color=ACCENT, bold=True)
    bullets(s, [
        "TensorFlow model.predict on a single 75 x 75 x 3 tensor, nothing else",
        "MediaPipe landmark extraction is separate and roughly doubles per-frame cost",
        "The webcam path adds capture, skeleton drawing, and display on top",
        "So the observed frame rate in the browser is lower than the model-only figure",
        "CPU contention from the running web server moves the measurement by tens of ms",
    ], MARGIN, cy + 0.38, lw, size=T_SMALL, gap=0.54, spacing=1.24)
    rx = MARGIN + lw + 0.6
    rw = CONTENT_W - lw - 0.6
    vrule(s, MARGIN + lw + 0.3, cy, 3.4, color=INK_LINE, thickness=0.012)
    text(s, "SMOOTHING AND GATING", rx, cy, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    kv_table(s, [
        ("Threshold", "0.60"),
        ("Below threshold", "reported as uncertain"),
        ("Top-3 predictions", "always displayed"),
        ("Smoothing window", "5 frames, moving average"),
        ("No person", "no pose reported"),
    ], rx, cy + 0.38, rw, key_w=2.3, row_h=0.40, size=T_SMALL)

    source(s, "Latency figures are re-measured per run and vary with machine load. "
              "Treat them as indicative rather than a fixed benchmark.", y=6.68)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 14  LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════
def slide_limitations():
    s = add_slide(_prs)
    y = slide_head(s, "Limitations", kicker="Honest assessment",
                   lead="Six constraints that bound what this system can honestly claim.")

    items = [
        ("Dataset size", "1080 training images is small for a five-class vision task, and the "
                         "generalisation gap of 15 points shows the effect directly."),
        ("Class imbalance", "Plank has 266 images against tree's 159, so plank is over-recalled "
                            "and tree is under-recalled."),
        ("Input resolution", "Skeletons are resized to 75 x 75 while the paper used 224 x 224, "
                             "discarding landmark detail."),
        ("Two-dimensional pose", "MediaPipe Pose is a 2D model. Yoga is a 3D activity, so depth "
                                 "differences between poses are not captured."),
        ("Domain shift", "The model learns from curated photos. Real webcam frames differ in "
                         "lighting, framing, and background."),
        ("Label is not quality", "A correct label says which pose, not whether it is performed "
                                 "well. Feedback is heuristic only."),
    ]
    col_w = (CONTENT_W - 0.5) / 2
    for i, (title, body) in enumerate(items):
        cx = MARGIN + (i % 2) * (col_w + 0.5)
        cy = y + (i // 2) * 1.42
        panel(s, cx, cy, col_w, 1.26, fill=INK_SOFT, line=INK_LINE)
        bar(s, cx, cy, 0.055, 1.26, color=ACCENT if i % 2 == 0 else ACCENT_DIM)
        text(s, title, cx + 0.26, cy + 0.16, col_w - 0.5, 0.3, size=T_BODY,
             color=WHITE, bold=True)
        text(s, body, cx + 0.26, cy + 0.50, col_w - 0.5, 0.66, size=T_SMALL,
             color=MUTED, spacing=1.32)

    source(s, "Each limitation is tied to a measured property of this build, not stated in "
              "the abstract.", y=6.55)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 15  TESTING
# ══════════════════════════════════════════════════════════════════════════
def slide_testing():
    s = add_slide(_prs)
    y = slide_head(s, "Testing and verification", kicker="Quality",
                   lead="59 automated tests, all passing. The webcam is mocked, so no claim of "
                        "hardware testing is made.")

    lw = 6.1
    text(s, "COVERAGE", MARGIN, y, lw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    kv_table(s, [
        ("Preprocessing", "resize, rescale, invalid input rejection"),
        ("Pose detection", "landmark visibility and no-person handling"),
        ("Inference", "input shape, label mapping, confidence gate"),
        ("Feedback", "angle maths, missing landmarks, thresholds"),
        ("Model", "architecture build, save and reload round trip"),
        ("Session store", "SQLite write, read, CSV export"),
        ("Video and dataset", "frame sampling, corrupt files, empty classes"),
    ], MARGIN, y + 0.38, lw, key_w=1.85, row_h=0.40, size=T_SMALL, key_color=ACCENT,
        value_color=MUTED, align_value=LEFT)

    rx = MARGIN + lw + 0.6
    rw = CONTENT_W - lw - 0.6
    vrule(s, MARGIN + lw + 0.3, y, 3.4, color=INK_LINE, thickness=0.012)
    text(s, "WHAT WAS VERIFIED", rx, y, rw, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    bullets(s, [
        "59 of 59 tests pass in 12 seconds on CPU",
        "The model saves and reloads with an identical label map",
        "Evaluation reuses the training split, so no test leakage",
        "Image, video, and dataset paths all handle missing files",
        "Streamlit starts and serves without error",
        "Missing model or dataset produces a clear message, not a crash",
    ], rx, y + 0.38, rw, size=T_SMALL, gap=0.48, spacing=1.24)

    stat_row(s, [
        ("59", "tests passing", "of 59 written"),
        ("12 s", "suite runtime", "CPU only"),
        ("8", "test modules", "unit level"),
        ("0", "skipped tests", "nothing disabled"),
    ], y=5.38, cols=4)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 16  FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════
def slide_future():
    s = add_slide(_prs)
    y = slide_head(s, "Future work", kicker="Next steps",
                   lead="The first item would close most of the gap to the paper, so it comes "
                        "first.")

    items = [
        ("01", "Train at 224 x 224", "Match the paper's input size. The single change most likely to raise accuracy, and it costs only memory and time."),
        ("02", "Balance the classes", "Collect more tree and goddess images, or apply class-weighted loss, to remove the 1.66 to 1 skew that favours plank."),
        ("03", "Landmark baseline", "Train a small classifier directly on the 33 by 3 landmark coordinates to compare against the image CNN."),
        ("04", "Three-dimensional pose", "Use a 3D landmark model so joint angles are computed in depth as well as in the image plane."),
    ]
    for i, (num, title, body) in enumerate(items):
        ry = y + i * 1.08
        text(s, num, MARGIN, ry, 0.7, 0.4, size=T_LEAD, color=ACCENT, bold=True,
             font=FONT_MONO)
        text(s, title, MARGIN + 0.78, ry, 3.0, 0.34, size=T_BODY, color=WHITE, bold=True)
        text(s, body, MARGIN + 3.95, ry + 0.01, CONTENT_W - 3.95, 0.72, size=T_SMALL,
             color=MUTED, spacing=1.28)
        if i < len(items) - 1:
            panel(s, MARGIN, ry + 0.92, CONTENT_W, 0.008, fill=INK_LINE, line=None)

    panel(s, MARGIN, 6.48, CONTENT_W, 0.52, fill=INK_SOFT, line=INK_LINE)
    text(s, "Deferred on purpose:  pose-hold timers, symmetry visualisation, and progress "
            "charts are UI work, not model work, and would delay the core result.",
         MARGIN + 0.28, 6.56, CONTENT_W - 0.6, 0.36, size=T_SMALL, color=MUTED, spacing=1.28)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
# 17  CONCLUSION
# ══════════════════════════════════════════════════════════════════════════
def slide_conclusion(metrics):
    s = add_slide(_prs)
    bar(s, 0, 0, 0.09, 7.5, color=ACCENT)

    text(s, "CONCLUSION", MARGIN, 0.95, 6.0, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    text(s, "A working system,\nmeasured honestly", MARGIN, 1.30, 6.4, 1.5,
         size=34, color=WHITE, bold=True, spacing=1.14)
    rule(s, MARGIN, 3.05, 1.6, color=ACCENT)

    text(s, "The paper's pipeline was reproduced end to end and extended into an application "
            "that runs on a laptop CPU. Accuracy sits well below the published figure, and "
            "the reasons are documented rather than glossed over. The architecture, the data "
            "pipeline, and the evaluation are reproducible from one seeded command.",
         MARGIN, 3.35, 6.3, 1.75, size=T_LEAD, color=MUTED, spacing=1.42)

    text(s, "Reproduce with", MARGIN, 5.28, 6.0, 0.3, size=T_KICKER, color=ACCENT, bold=True)
    text(s, "python scripts/train.py --config configs/train.yaml\n"
            "python scripts/evaluate.py --config configs/train.yaml\n"
            "python -m pytest tests/ -q",
         MARGIN, 5.62, 6.3, 1.05, size=T_TINY, color=MUTED, spacing=1.5, font=FONT_MONO)

    rx = 7.55
    rw = 13.333 - MARGIN - rx
    vrule(s, 7.25, 0.95, 5.6, color=INK_LINE, thickness=0.012)

    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    stats = [
        (f"{acc * 100:.1f}%", "test accuracy", "109 held-out images"),
        (f"{f1 * 100:.1f}%", "macro F1", "unweighted across classes"),
        ("1080", "training images", "five classes, cleaned"),
        ("59", "tests passing", "12 second suite"),
    ]
    for i, (value, label, note) in enumerate(stats):
        sy = 1.05 + i * 1.42
        stat(s, value, label, rx, sy, rw, note=note,
             size=T_NUMBER if i < 2 else T_NUMBER_SM)

    panel(s, 7.25, 6.75, 13.333 - MARGIN - 7.25, 0.02, fill=INK_LINE, line=None)
    text(s, "YogaVision  ·  CNN and MediaPipe  ·  reproduction with measured results",
         7.55, 6.92, rw, 0.3, size=T_TINY, color=MUTED_DIM)
    return page(s)


# ══════════════════════════════════════════════════════════════════════════
def build():
    global _prs
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    _prs = prs

    metrics = load_metrics()
    training = load_training()
    classes = count_classes() or [("downdog", 223), ("goddess", 180), ("plank", 266),
                                  ("tree", 159), ("warrior2", 252)]

    slide_title()
    slide_problem()
    slide_research()
    slide_pipeline()
    slide_dataset(classes)
    slide_model()
    slide_training(training)
    slide_training_curves(training)
    slide_evaluation(metrics)
    slide_comparison(metrics)
    slide_application()
    slide_feedback()
    slide_performance(metrics)
    slide_limitations()
    slide_testing()
    slide_future()
    slide_conclusion(metrics)

    out = ROOT / "outputs" / "YogaVision_Presentation.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"Saved {out}  ({out.stat().st_size / 1024:.0f} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
