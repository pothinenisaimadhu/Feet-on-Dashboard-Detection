"""
Streamlit UI — OOP (Out-of-Position) Passenger Detection
Supports: webcam live feed  |  video file upload
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import tempfile
import time
import numpy as np
import streamlit as st
from pathlib import Path

from pipeline import (
    PassengerDetector, PoseEstimator, DashboardROI,
    score_frame, TemporalSmoother, render, OOPEngine,
)
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OOP Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .verdict-pos {
        background: #ff4444; color: white;
        padding: 12px 24px; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; letter-spacing: 1px;
    }
    .verdict-neg {
        background: #00c853; color: white;
        padding: 12px 24px; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; letter-spacing: 1px;
    }
    .metric-box {
        background: #1e2130; border-radius: 8px;
        padding: 10px 16px; margin: 4px 0;
    }
    .stProgress > div > div { background-color: #ff4444; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=64)
    st.title("OOP Detector")
    st.caption("Out-of-Position Passenger Detection")
    st.divider()

    mode = st.radio("Input source", ["📁 Upload Video", "📷 Webcam"], index=0)
    st.divider()

    st.subheader("⚙️ Settings")
    show_skeleton = st.toggle("Show joints & leg lines", value=True)
    show_roi      = st.toggle("Show dashboard ROI", value=True)
    conf_thresh   = st.slider("Confidence display threshold", 0.0, 1.0, 0.0, 0.05)
    st.divider()

    st.subheader("📊 Thresholds")
    st.caption(f"HYST_ON  : {config.HYST_ON}")
    st.caption(f"HYST_OFF : {config.HYST_OFF}")
    st.caption(f"EMA α    : {config.EMA_ALPHA}")
    st.caption(f"KP conf  : {config.KP_MIN_CONF}")


# ── Shared model cache (load once) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return PassengerDetector(), PoseEstimator()


detector, pose = load_models()


# ── Helper: process one frame and return annotated BGR + stats ────────────────
def process_frame(
    frame: np.ndarray,
    roi: DashboardROI,
    smoother: TemporalSmoother,
    last_box,
    miss_streak: int,
    frame_idx: int,
    high_ankle: bool,
    _last_conf: float,
) -> tuple[np.ndarray, bool, float, str, any, int, float]:

    h, w = frame.shape[:2]
    need_detect = (
        last_box is None
        or frame_idx % config.DETECT_EVERY == 0
        or miss_streak >= config.DETECT_MISS_MAX
    )
    if need_detect:
        box, conf = detector.detect(frame)
        if box is not None:
            last_box, _last_conf, miss_streak = box, conf, 0
        else:
            miss_streak += 1
    else:
        conf = _last_conf

    crop_rect = None
    if last_box is not None:
        _, crop_rect = detector.crop_for_pose(frame, last_box)

    pd = pose.analyse(frame, last_box, crop_rect)
    has_pose = pd["landmarks"] is not None
    raw_pos, score, debug = score_frame(pd, roi, high_ankle_mode=high_ankle)
    is_pos = smoother.update(raw_pos, has_pose, score)

    # Optionally suppress joints/ROI based on sidebar toggles
    _pd = pd if show_skeleton else {k: None if "px" in k else v for k, v in pd.items()}
    annotated = render(
        frame.copy(),
        last_box if show_skeleton else None,
        _pd,
        roi if show_roi else _NullROI(w, h),
        is_pos, score, debug, _last_conf, frame_idx,
    )
    return annotated, is_pos, score, debug, last_box, miss_streak, _last_conf


class _NullROI:
    """ROI stub that draws nothing and never matches."""
    def __init__(self, w, h): pass
    def contains(self, x, y): return False
    def draw(self, frame, **_): return frame


# ── Stats sidebar updater ─────────────────────────────────────────────────────
def sidebar_stats(score: float, is_pos: bool, frame_idx: int,
                  pos_count: int, total: int):
    with st.sidebar:
        st.divider()
        st.subheader("📈 Live Stats")
        verdict_html = (
            '<div class="verdict-pos">⚠️ FEET ON DASHBOARD</div>'
            if is_pos else
            '<div class="verdict-neg">✅ NORMAL POSTURE</div>'
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
        st.metric("Confidence", f"{score:.2f}")
        st.progress(min(score, 1.0))
        st.metric("Frame", frame_idx)
        if total > 0:
            st.metric("Positive rate", f"{100*pos_count//total}%")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — VIDEO UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if "Upload" in mode:
    st.header("📁 Video Upload")
    uploaded = st.file_uploader(
        "Drop an MP4 / AVI / MOV file", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is None:
        st.info("Upload a video to begin analysis.")
        st.stop()

    # Save to temp file
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    src_path = tmp.name

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    run_btn    = col_btn1.button("▶ Analyse", type="primary", use_container_width=True)
    export_btn = col_btn2.button("💾 Export", use_container_width=True)

    if not run_btn and not export_btn:
        st.video(uploaded)
        st.stop()

    cap = cv2.VideoCapture(src_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi      = DashboardROI(w, h)
    smoother = TemporalSmoother()

    # Bootstrap: sample every 4th frame, max 30 frames
    with st.spinner("Bootstrapping ankle profile…"):
        engine_tmp = OOPEngine.__new__(OOPEngine)
        engine_tmp.detector     = detector
        engine_tmp.pose         = pose
        engine_tmp._last_box    = None
        engine_tmp._last_conf   = 0.0
        engine_tmp._miss_streak = 0
        high_ankle = engine_tmp._bootstrap_high_ankle(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mode_tag = "HIGH-ANKLE" if high_ankle else "STANDARD"
    st.caption(f"Detection mode: **{mode_tag}**  |  {total_frames} frames  |  {fps:.1f} fps")

    # ── Process all frames first, update UI every DISPLAY_EVERY frames ───────
    DISPLAY_EVERY = 8   # redraw UI only every N frames — big speedup on Cloud
    SCALE         = 0.5 # downscale preview to reduce encode/transfer cost

    out_tmp = None
    writer  = None
    if export_btn:
        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        writer = cv2.VideoWriter(
            out_tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h),
        )

    frame_ph  = st.empty()
    prog_bar  = st.progress(0)
    stat_cols = st.columns(4)
    score_ph  = stat_cols[0].empty()
    pos_ph    = stat_cols[1].empty()
    rate_ph   = stat_cols[2].empty()
    mode_ph   = stat_cols[3].empty()

    last_box    = None
    miss_streak = 0
    _last_conf  = 0.0
    frame_idx   = 0
    pos_count   = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, is_pos, score, debug, last_box, miss_streak, _last_conf = process_frame(
            frame, roi, smoother, last_box, miss_streak,
            frame_idx, high_ankle, _last_conf,
        )
        pos_count += int(is_pos)

        if frame_idx % DISPLAY_EVERY == 0:
            # Downscale preview before sending to browser
            preview = cv2.resize(annotated, (int(w * SCALE), int(h * SCALE)))
            frame_ph.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True,
            )
            prog_bar.progress(min((frame_idx + 1) / max(total_frames, 1), 1.0))
            score_ph.metric("Confidence", f"{score:.2f}")
            pos_ph.metric("Verdict", "⚠️ OOP" if is_pos else "✅ Normal")
            rate_ph.metric("Pos rate", f"{100*pos_count//(frame_idx+1)}%")
            mode_ph.metric("Mode", mode_tag)

        if writer:
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    prog_bar.progress(1.0)
    st.success(f"Done — {frame_idx} frames processed. Positive rate: {100*pos_count//max(frame_idx,1)}%")

    if out_tmp:
        with open(out_tmp.name, "rb") as f:
            st.download_button(
                "⬇️ Download annotated video",
                data=f,
                file_name=f"oop_{Path(uploaded.name).stem}.mp4",
                mime="video/mp4",
                type="primary",
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.header("📷 Webcam Live Feed")

    cam_idx = st.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)

    col1, col2 = st.columns([1, 1])
    start_btn = col1.button("▶ Start", type="primary", use_container_width=True)
    stop_btn  = col2.button("⏹ Stop",  use_container_width=True)

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False
    if start_btn:
        st.session_state.cam_running = True
    if stop_btn:
        st.session_state.cam_running = False

    if not st.session_state.cam_running:
        st.info("Press **Start** to begin live detection.")
        st.stop()

    cap = cv2.VideoCapture(int(cam_idx))
    if not cap.isOpened():
        st.error(f"Cannot open camera {cam_idx}. Check the index or permissions.")
        st.session_state.cam_running = False
        st.stop()

    # Read one frame to get dimensions
    ok, probe = cap.read()
    if not ok:
        st.error("Camera opened but could not read a frame.")
        cap.release()
        st.stop()

    h, w = probe.shape[:2]
    roi      = DashboardROI(w, h)
    smoother = TemporalSmoother()
    high_ankle = False   # webcam: no bootstrap, use standard mode

    frame_ph = st.empty()
    stat_cols = st.columns(4)
    score_ph = stat_cols[0].empty()
    pos_ph   = stat_cols[1].empty()
    fps_ph   = stat_cols[2].empty()
    mode_ph  = stat_cols[3].empty()

    last_box    = None
    miss_streak = 0
    _last_conf  = 0.0
    frame_idx   = 0
    pos_count   = 0
    t0          = time.time()

    # Re-push the probe frame as frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while st.session_state.cam_running:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, is_pos, score, debug, last_box, miss_streak, _last_conf = process_frame(
            frame, roi, smoother, last_box, miss_streak,
            frame_idx, high_ankle, _last_conf,
        )

        pos_count += int(is_pos)
        elapsed = time.time() - t0
        live_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        frame_ph.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True,
        )
        score_ph.metric("Confidence", f"{score:.2f}")
        pos_ph.metric("Verdict",  "⚠️ OOP" if is_pos else "✅ Normal")
        fps_ph.metric("FPS", f"{live_fps:.1f}")
        mode_ph.metric("Frame", frame_idx)

        sidebar_stats(score, is_pos, frame_idx, pos_count, frame_idx + 1)

        frame_idx += 1

        # Honour stop button between frames
        if stop_btn:
            break

    cap.release()
    st.info("Webcam stopped.")
