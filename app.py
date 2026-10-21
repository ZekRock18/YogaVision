"""YogaVision — Real-Time Yoga Pose Classification and Feedback System.

Streamlit application. Run with:

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from yogavision.camera import CameraCapture, CameraError, list_cameras
from yogavision.config import PROJECT_ROOT, Settings, get_settings
from yogavision.evaluation import (
    load_metrics_json,
)
from yogavision.feedback import analyse_feedback, feedback_rules_documentation, joint_angles
from yogavision.inference import YogaPoseClassifier
from yogavision.model import ModelNotAvailableError, load_model, model_status
from yogavision.pose_detection import PoseDetector
from yogavision.preprocessing import load_image, load_image_from_bytes
from yogavision.session_store import SessionStore
from yogavision.utils import (
    format_duration,
    get_logger,
    is_supported_image,
    is_supported_video,
    top_k,
)
from yogavision.video_analysis import analyse_video, export_predictions_csv, predictions_to_dataframe

logger = get_logger()

st.set_page_config(
    page_title="YogaVision — Yoga Pose Classification",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- CSS
CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 2rem; max-width: 1400px; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        color: #e0e0e0;
    }
    .metric-card h4 { margin: 0 0 0.4rem 0; font-size: 0.85rem; color: #94a3b8;
                      text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #38bdf8; }
    .metric-card .sub { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
    .pose-banner {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-radius: 14px; padding: 1.4rem; text-align: center;
        color: #fff; margin: 0.8rem 0;
    }
    .pose-banner .pose { font-size: 2.2rem; font-weight: 800; text-transform: capitalize; }
    .pose-banner .conf { font-size: 1rem; color: #cbd5e1; }
    .status-ok { color: #4ade80; font-weight: 600; }
    .status-missing { color: #f87171; font-weight: 600; }
    .feedback-warn { background: #451a03; border-left: 4px solid #f59e0b;
                     padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.3rem 0; color: #fde68a; }
    .feedback-good { background: #052e16; border-left: 4px solid #4ade80;
                     padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.3rem 0; color: #bbf7d0; }
    .feedback-info { background: #0c1a2e; border-left: 4px solid #38bdf8;
                     padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.3rem 0; color: #bae6fd; }
    .stButton>button { width: 100%; border-radius: 8px; }
    h1 { color: #38bdf8; }
    .warning-box { background: #431407; border: 1px solid #ea580c; border-radius: 8px;
                   padding: 1rem; color: #fed7aa; margin: 0.6rem 0; }
    .info-box { background: #0c2d48; border: 1px solid #38bdf8; border-radius: 8px;
                padding: 1rem; color: #bae6fd; margin: 0.6rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- session state
DEFAULTS = {
    "camera_index": 0,
    "camera_running": False,
    "frame_count": 0,
    "session_id": None,
    "session_start": None,
    "pose_counts": {},
    "latencies": [],
    "last_frames": [],
    "selected_pose": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner=False)
def _get_settings() -> Settings:
    return Settings(PROJECT_ROOT / "configs" / "train.yaml")


@st.cache_resource(show_spinner=False)
def _get_store() -> SessionStore:
    return SessionStore()


@st.cache_resource(show_spinner=False)
def _get_detector() -> PoseDetector:
    return PoseDetector.get_instance(_get_settings())


@st.cache_resource(show_spinner=False)
def _get_classifier() -> YogaPoseClassifier:
    """Load the model once and reuse across reruns."""
    settings = _get_settings()
    bundle = load_model(
        settings,
        input_representation=settings.config.get("model", {}).get(
            "input_representation", "skeleton"
        ),
    )
    return YogaPoseClassifier(bundle, settings)


def classifier_or_message():
    """Return the classifier or render a setup message and return None."""
    try:
        return _get_classifier()
    except ModelNotAvailableError as exc:
        st.error("**Model not available**")
        st.markdown(str(exc))
        st.markdown(
            '<div class="info-box"><b>Quick start:</b><br>'
            "1. <code>python scripts/download_assets.py</code><br>"
            "2. <code>python scripts/prepare_dataset.py</code><br>"
            "3. <code>python scripts/train.py --config configs/train.yaml</code><br>"
            "See <code>docs/dataset_setup.md</code> for details.</div>",
            unsafe_allow_html=True,
        )
        return None


# ---------------------------------------------------------------- UI helpers
def metric_card(title: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><h4>{title}</h4>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def pose_banner(label: str, confidence: float, above: bool) -> None:
    display = label if above else "uncertain"
    color = "#4ade80" if above else "#fbbf24"
    st.markdown(
        f'<div class="pose-banner"><div class="pose" style="color:{color}">{display}</div>'
        f'<div class="conf">confidence {confidence * 100:.1f}%'
        + ("" if above else " — below threshold") + "</div></div>",
        unsafe_allow_html=True,
    )


def render_feedback(report) -> None:
    for item in report.items:
        cls = {"warn": "feedback-warn", "good": "feedback-good", "info": "feedback-info"}.get(
            item.severity, "feedback-info"
        )
        st.markdown(
            f'<div class="{cls}">{item.message}</div>', unsafe_allow_html=True
        )


def render_top_k(prediction) -> None:
    st.write("**Top predictions**")
    for label, prob in prediction.top_k:
        st.progress(float(prob), text=f"{label} — {prob * 100:.1f}%")


# ---------------------------------------------------------------- pages
def page_dashboard() -> None:
    st.title("YogaVision")
    st.markdown(
        "Real-time yoga pose classification and rule-based posture feedback, "
        "built on the methodology of Garg, Saxena & Gupta (2023), "
        "*J. Ambient Intelligence and Humanized Computing*, "
        "[DOI 10.1007/s12652-022-03910-0](https://doi.org/10.1007/s12652-022-03910-0)."
    )

    settings = _get_settings()
    status = model_status(settings)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(
            "Model status",
            "Ready" if status["model_exists"] else "Missing",
            f"{status['model_path']}" + (f" ({status['model_size_mb']} MB)" if status["model_size_mb"] else ""),
        )
    with col2:
        dataset_dir = settings.dataset_path
        metric_card(
            "Dataset status",
            "Present" if dataset_dir.is_dir() else "Missing",
            str(dataset_dir),
        )
    with col3:
        store = _get_store()
        stats = store.summary_stats()
        metric_card("Sessions logged", str(stats["session_count"]), f"{stats['total_frames']} frames total")

    st.subheader("Supported poses")
    labels = None
    try:
        from yogavision.model import load_label_map

        labels = load_label_map(settings.label_map_path)
    except Exception:
        labels = settings.config.get("data", {}).get("class_names", [])
    if labels:
        cols = st.columns(min(5, len(labels)))
        for i, lab in enumerate(labels):
            with cols[i % len(cols)]:
                metric_card(lab, "", "trained class")
    else:
        st.warning("Label map not available — train the model to populate this list.")

    st.subheader("Quick start")
    st.markdown(
        '<div class="info-box">'
        "1. <b>Live Webcam</b> — start a real-time classification session.<br>"
        "2. <b>Image Analysis</b> — upload a JPG or PNG and classify one pose.<br>"
        "3. <b>Video Analysis</b> — process a video and export frame-level predictions.<br>"
        "4. <b>Training</b> — inspect the dataset and retrain the model.<br>"
        "5. <b>Evaluation</b> — view metrics from completed runs.<br>"
        "6. <b>Session History</b> — review and export past sessions."
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("How to position yourself")
    st.markdown(
        "- Stand **2–3 metres** from the camera so your whole body is in frame.\n"
        "- Face the camera square-on for the best landmark visibility.\n"
        "- Wear contrasting clothing and use even lighting.\n"
        "- Avoid busy backgrounds behind you.\n"
        "- Keep your hands and feet inside the frame."
    )

    st.subheader("Recent sessions")
    sessions = store.list_sessions(limit=5)
    if sessions:
        df = pd.DataFrame(
            [
                {
                    "started": s.started_at[:19].replace("T", " "),
                    "source": s.source,
                    "frames": s.frames_processed,
                    "duration": format_duration(s.duration_s),
                    "dominant": s.dominant_pose or "-",
                    "avg latency (ms)": round(s.avg_latency_ms, 1),
                }
                for s in sessions
            ]
        )
        st.dataframe(df)
    else:
        st.info("No sessions recorded yet.")


def page_webcam() -> None:
    st.title("Live Webcam Classification")
    st.markdown("Start a real-time session. The model is loaded once and cached.")

    classifier = classifier_or_message()
    if classifier is None:
        return

    settings = _get_settings()

    with st.sidebar:
        st.subheader("Camera settings")
        try:
            cameras = list_cameras()
        except Exception:
            cameras = [0]
        if not cameras:
            cameras = [0]
        camera_index = st.selectbox("Camera", cameras, index=0, key="cam_select")
        conf_threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, float(settings.confidence_threshold), 0.05
        )
        show_skeleton = st.checkbox("Show skeleton overlay", value=True)
        show_feedback = st.checkbox("Show posture feedback", value=True)
        target_fps = st.slider("Target FPS", 5, 30, 15)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        start = st.button("Start camera", type="primary")
    with col_b:
        stop = st.button("Stop camera")
    with col_c:
        st.write(f"Frames processed: **{st.session_state.frame_count}**")

    if stop:
        st.session_state.camera_running = False
        if st.session_state.session_id:
            _end_session_record(classifier)
        st.rerun()

    if start:
        st.session_state.camera_running = True
        st.session_state.camera_index = camera_index
        st.session_state.frame_count = 0
        st.session_state.pose_counts = {}
        st.session_state.latencies = []
        store = _get_store()
        st.session_state.session_id = store.start_session(source="webcam").session_id
        st.session_state.session_start = time.time()

    if not st.session_state.camera_running:
        st.info("Press **Start camera** to begin. Frames are processed locally and never stored.")
        return

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    fb_placeholder = st.empty()

    classifier.confidence_threshold = conf_threshold
    cam = CameraCapture(st.session_state.camera_index, width=640, height=480)
    try:
        cam.open()
    except CameraError as exc:
        st.session_state.camera_running = False
        st.error(str(exc))
        return

    st.session_state.camera_running = True
    try:
        while st.session_state.camera_running:
            ok, frame = cam.read()
            if not ok:
                st.error("Could not read a frame from the camera. Stopping.")
                st.session_state.camera_running = False
                break

            prediction, pose = classifier.predict_frame(frame, smooth=True)
            st.session_state.frame_count += 1
            st.session_state.latencies.append(prediction.inference_ms)
            if prediction.above_threshold:
                st.session_state.pose_counts[prediction.label] = (
                    st.session_state.pose_counts.get(prediction.label, 0) + 1
                )

            display = frame
            if show_skeleton and pose.detected:
                display = classifier.detector.draw_skeleton(frame, pose)

            frame_placeholder.image(
                display[:, :, ::-1], caption="Live feed", channels="RGB"
            )

            lat = np.mean(st.session_state.latencies[-30:]) if st.session_state.latencies else 0.0
            fps = 1000.0 / lat if lat > 0 else 0.0
            with info_placeholder.container():
                pose_banner(prediction.label, prediction.confidence, prediction.above_threshold)
                c1, c2, c3 = st.columns(3)
                c1.metric("Inference", f"{prediction.inference_ms:.0f} ms")
                c2.metric("Approx FPS", f"{fps:.1f}")
                c3.metric("Frames", str(st.session_state.frame_count))
                if prediction.top_k:
                    render_top_k(prediction)

            if show_feedback:
                report = analyse_feedback(pose, settings, predicted_label=prediction.label)
                with fb_placeholder.container():
                    render_feedback(report)

            time.sleep(1.0 / target_fps if target_fps else 0)
    finally:
        cam.release()
        _end_session_record(classifier)
        st.session_state.camera_running = False


def _end_session_record(classifier) -> None:
    if not st.session_state.session_id:
        return
    store = _get_store()
    duration = (
        time.time() - st.session_state.session_start
        if st.session_state.session_start
        else 0.0
    )
    avg_lat = float(np.mean(st.session_state.latencies)) if st.session_state.latencies else 0.0
    store.end_session(
        st.session_state.session_id,
        frames_processed=st.session_state.frame_count,
        pose_counts=dict(st.session_state.pose_counts),
        duration_s=duration,
        avg_latency_ms=avg_lat,
    )
    st.session_state.session_id = None
    st.session_state.session_start = None


def page_image() -> None:
    st.title("Image Analysis")
    st.markdown("Upload a JPG, JPEG or PNG image of a single yoga pose.")

    classifier = classifier_or_message()
    if classifier is None:
        return
    settings = _get_settings()

    uploaded = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    if uploaded is None:
        return

    try:
        image = load_image_from_bytes(uploaded.read())
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    st.subheader("Original image")
    st.image(image[:, :, ::-1], channels="RGB")

    with st.spinner("Detecting pose and classifying..."):
        prediction, pose = classifier.predict_image(image)
        skeleton_display = None
        if pose.detected:
            skeleton_display = classifier.detector.draw_skeleton(image, pose)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Landmark overlay")
        if pose.detected and skeleton_display is not None:
            st.image(skeleton_display[:, :, ::-1], channels="RGB")
        else:
            st.warning(
                "No person detected in this image. Make sure the whole body is "
                "visible against a plain background."
            )
    with col2:
        st.subheader("Classification")
        if pose.detected:
            pose_banner(prediction.label, prediction.confidence, prediction.above_threshold)
            render_top_k(prediction)
            st.subheader("Joint angles")
            angles = joint_angles(pose)
            if angles:
                st.dataframe(
                    pd.DataFrame([{"joint": k, "angle_deg": v} for k, v in angles.items()])
                )
            st.subheader("Posture feedback")
            render_feedback(analyse_feedback(pose, settings, predicted_label=prediction.label))
        else:
            st.warning("Classification skipped because no pose was detected.")


def page_video() -> None:
    st.title("Video Analysis")
    st.markdown("Upload an MP4 (or other OpenCV-decodable format) to analyse frame by frame.")

    classifier = classifier_or_message()
    if classifier is None:
        return

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    suffix = Path(uploaded.name).suffix.lower()
    if not is_supported_video(uploaded.name):
        st.warning(f"Unusual extension '{suffix}' — attempting to open anyway.")

    tmp_dir = PROJECT_ROOT / "outputs" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / uploaded.name
    video_path.write_bytes(uploaded.read())
    st.video(str(video_path))

    sample_every = st.slider("Analyse every Nth frame", 1, 30, 5)
    max_frames = st.number_input(
        "Max frames to analyse (0 = all)", min_value=0, value=0, step=50
    )

    if st.button("Analyse video", type="primary"):
        progress_bar = st.progress(0.0, text="Starting analysis...")
        try:
            result = analyse_video(
                video_path,
                classifier,
                sample_every=int(sample_every),
                max_frames=int(max_frames) if max_frames else None,
                progress_callback=lambda f: progress_bar.progress(f, text=f"Analysing... {f*100:.0f}%"),
            )
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            return
        progress_bar.progress(1.0, text="Done")

        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames analysed", result.processed_frames)
        c2.metric("Duration", f"{result.duration_s:.1f} s")
        c3.metric("Dominant pose", result.dominant_pose or "-")
        c4.metric("Avg latency", f"{result.average_latency_ms:.1f} ms")

        st.subheader("Detected pose distribution")
        if result.class_counts:
            import pandas as pd

            df = pd.DataFrame(
                sorted(result.class_counts.items(), key=lambda kv: kv[1], reverse=True),
                columns=["pose", "frames"],
            )
            st.bar_chart(df.set_index("pose"))
        else:
            st.info("No poses were classified.")

        st.subheader("Frame-level predictions")
        df = predictions_to_dataframe(result)
        st.dataframe(df)

        csv_path = export_predictions_csv(
            result, PROJECT_ROOT / "outputs" / f"{Path(uploaded.name).stem}_predictions.csv"
        )
        with open(csv_path, "rb") as fh:
            st.download_button(
                "Download predictions CSV",
                data=fh.read(),
                file_name=csv_path.name,
                mime="text/csv",
            )
        st.caption(f"Saved to {csv_path}")


def page_training() -> None:
    st.title("Training and Dataset")
    st.markdown(
        "Inspect the dataset, then retrain the model. Training runs in a separate "
        "process so the UI stays responsive."
    )
    settings = _get_settings()

    st.subheader("Dataset status")
    dataset_dir = settings.resolve(settings.config["data"]["dataset_dir"])
    raw_dir = settings.resolve(settings.config["data"]["raw_dir"])
    for label, path in (("Raw dataset", raw_dir), ("Processed dataset", dataset_dir)):
        if path.is_dir():
            n = sum(1 for p in path.rglob("*") if is_supported_image(p))
            st.success(f"{label}: {path} — {n} images")
        else:
            st.warning(f"{label}: not found at {path}")

    st.markdown("**Setup commands**")
    st.code(
        "python scripts/download_assets.py\n"
        "python scripts/prepare_dataset.py --raw data/raw/DATASET "
        "--out data/processed/yoga_poses_skeleton\n"
        "python scripts/inspect_dataset.py data/processed/yoga_poses_skeleton",
        language="powershell",
    )

    st.subheader("Training configuration")
    st.json(settings.config)

    st.subheader("Run training")
    st.markdown(
        "Training is executed as a subprocess so this page stays interactive. "
        "It can take several minutes on CPU."
    )
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
    limit = st.number_input(
        "Limit samples per class (0 = no limit)", min_value=0, value=0, step=10
    )
    if st.button("Start training", type="primary"):
        import subprocess

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", "configs/train.yaml",
            "--epochs", str(int(epochs)),
        ]
        if limit:
            cmd += ["--limit", str(int(limit))]
        with st.spinner("Training in progress — this may take a while..."):
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
            )
        st.subheader("Training output")
        st.code(proc.stdout[-8000:] or "(no stdout)")
        if proc.returncode != 0:
            st.error(f"Training exited with code {proc.returncode}")
            st.code(proc.stderr[-8000:])
        else:
            st.success("Training finished. Metrics saved to outputs/evaluation/.")
            _get_classifier.clear()


def page_evaluation() -> None:
    st.title("Evaluation")
    st.markdown(
        "Metrics below come from **reproduction runs** of the YogaVision pipeline "
        "on this machine. They are not the original paper's reported numbers."
    )
    settings = _get_settings()

    metrics_path = settings.evaluation_dir / "evaluation_metrics.json"
    training_path = settings.evaluation_dir / "training_results.json"
    metrics = load_metrics_json(metrics_path)
    training = load_metrics_json(training_path)

    if metrics is None and training is None:
        st.warning(
            "No evaluation results found. Run training and evaluation first:\n"
            "`python scripts/train.py --config configs/train.yaml`\n"
            "`python scripts/evaluate.py --config configs/train.yaml`"
        )
        return

    if metrics:
        st.subheader("Held-out test metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision (macro)", f"{metrics.get('precision_macro', 0):.4f}")
        c3.metric("Recall (macro)", f"{metrics.get('recall_macro', 0):.4f}")
        c4.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
        c5.metric("F1 (weighted)", f"{metrics.get('f1_weighted', 0):.4f}")

        lat = metrics.get("inference_latency_ms")
        if lat:
            st.subheader("Inference latency")
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean", f"{lat.get('mean', 0):.2f} ms")
            c2.metric("p95", f"{lat.get('p95', 0):.2f} ms")
            c3.metric("Approx FPS", f"{lat.get('approx_fps', 0):.1f}")
            st.caption(lat.get("note", ""))

        cm_path = settings.plots_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.subheader("Confusion matrix")
            st.image(str(cm_path))

        report_path = settings.evaluation_dir / "classification_report.txt"
        if report_path.exists():
            st.subheader("Per-class report")
            st.code(report_path.read_text(encoding="utf-8"))

    curves = settings.plots_dir / "training_curves.png"
    if curves.exists():
        st.subheader("Training curves")
        st.image(str(curves))

    if training:
        st.subheader("Training run details")
        st.json(
            {
                "epochs_run": training.get("epochs_run"),
                "num_samples": training.get("num_samples"),
                "training_seconds": training.get("training_seconds"),
                "provenance": training.get("provenance"),
            }
        )

    st.subheader("Paper reference values (not our results)")
    st.markdown(
        "The original paper reports **97.09% test accuracy** with 0.97 precision, "
        "recall and F1 for its best MediaPipe model (YogaConvo2d). "
        "These are the authors' numbers on their dataset — see `docs/research_notes.md`. "
        "Our reproduction metrics above are computed independently and are not "
        "directly comparable unless the same dataset, splits and preprocessing are used."
    )


def page_history() -> None:
    st.title("Session History")
    store = _get_store()
    sessions = store.list_sessions(limit=200)

    if not sessions:
        st.info("No sessions recorded yet.")
        return

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "started": s.started_at[:19].replace("T", " "),
                "source": s.source,
                "frames": s.frames_processed,
                "duration": format_duration(s.duration_s),
                "dominant": s.dominant_pose or "-",
                "avg latency (ms)": round(s.avg_latency_ms, 1),
                "pose counts": ", ".join(f"{k}:{v}" for k, v in s.pose_counts.items()) or "-",
                "id": s.session_id,
            }
            for s in sessions
        ]
    )
    st.dataframe(df)

    csv_path = store.export_csv(PROJECT_ROOT / "outputs" / "session_history.csv")
    with open(csv_path, "rb") as fh:
        st.download_button(
            "Export sessions CSV", data=fh.read(), file_name="session_history.csv", mime="text/csv"
        )

    st.subheader("Delete")
    selected = st.selectbox("Session to delete", options=[s.session_id for s in sessions])
    if st.button("Delete selected session", type="secondary"):
        store.delete_session(selected)
        st.rerun()
    if st.button("Clear all history", type="secondary"):
        store.clear_history()
        st.rerun()


def page_about() -> None:
    st.title("About and Research Methodology")

    st.subheader("Paper")
    st.markdown(
        "**Yoga pose classification: a CNN and MediaPipe inspired deep learning "
        "approach for real-world application**\n\n"
        "Shubham Garg, Aman Saxena, Richa Gupta. "
        "*Journal of Ambient Intelligence and Humanized Computing*, vol. 14, "
        "pp. 16551–16562 (2023).\n\n"
        "DOI: [10.1007/s12652-022-03910-0](https://doi.org/10.1007/s12652-022-03910-0)\n\n"
        "Original implementation: "
        "[github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization]"
        "(https://github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization)"
    )

    st.subheader("Pipeline")
    st.markdown(
        "Camera / image / video → OpenCV frame decode → MediaPipe Pose landmark "
        "extraction → skeleton rendering (paper's input representation) → "
        "75×75 normalised tensor → YogaConvo2d CNN → softmax probabilities → "
        "confidence threshold and temporal smoothing → visualisation → "
        "rule-based posture feedback → session persistence."
    )

    st.subheader("Model architecture (YogaConvo2d)")
    st.markdown(
        "Four Conv2D blocks (64 → 128 → 256 filters, 3×3, ReLU, 'Same' padding), "
        "each followed by 2×2 max pooling and dropout, then a 1024-unit dense "
        "head and a softmax output. Trained with Adam (lr 1e-3), batch size 16, "
        "categorical cross-entropy."
    )

    st.subheader("Posture feedback rules")
    settings = _get_settings()
    rules = feedback_rules_documentation(settings)
    import pandas as pd

    st.dataframe(pd.DataFrame(rules))
    st.warning(
        "Posture feedback is heuristic and rule-based. A predicted pose label does "
        "NOT prove the pose is performed correctly. This is not medical advice and "
        "single-camera 2D estimation cannot reliably assess three-dimensional alignment."
    )

    st.subheader("Documentation")
    for name, path in (
        ("Research notes", "docs/research_notes.md"),
        ("Methodology", "docs/methodology.md"),
        ("Dataset setup", "docs/dataset_setup.md"),
        ("Limitations", "docs/limitations.md"),
    ):
        p = PROJECT_ROOT / path
        if p.exists():
            with st.expander(name):
                st.markdown(p.read_text(encoding="utf-8"))
        else:
            st.info(f"{name} not yet written ({path})")


# ---------------------------------------------------------------- navigation
PAGES = {
    "Dashboard": page_dashboard,
    "Live Webcam": page_webcam,
    "Image Analysis": page_image,
    "Video Analysis": page_video,
    "Training": page_training,
    "Evaluation": page_evaluation,
    "Session History": page_history,
    "About": page_about,
}

with st.sidebar:
    st.title("YogaVision")
    st.caption("Yoga pose classification")
    selection = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=["house", "camera-video", "image", "film", "cpu", "bar-chart", "clock-history", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#0ea5e9"},
        },
    )
    st.divider()
    status = model_status(_get_settings())
    if status["model_exists"]:
        st.success("Model ready")
    else:
        st.error("Model missing")
    st.caption(f"v1.0.0 — CPU, no cloud inference")

PAGES[selection]()
