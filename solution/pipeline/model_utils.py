"""
Model bootstrap — auto-download all weights on first run.

Model sources:
  yolov8n.pt            — Ultralytics hub (auto, ~6 MB)
  yolov8n-pose.pt       — Ultralytics hub (auto, ~6 MB)
  pose_landmarker_lite  — Google MediaPipe CDN (auto, ~5 MB)

No manual downloads needed; each loader checks for the file and
fetches it only when missing.
"""

import os
import urllib.request
from pathlib import Path

# ── MediaPipe ──────────────────────────────────────────────────────────────
_MP_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_HERE, "..", "pose_landmarker_lite.task"))


def ensure_model() -> str:
    """Return path to pose_landmarker_lite.task, downloading if absent."""
    if not os.path.exists(MODEL_PATH):
        print("[model_utils] Downloading MediaPipe PoseLandmarker (~5 MB)...")
        urllib.request.urlretrieve(_MP_URL, MODEL_PATH)
        print(f"[model_utils] Saved to {MODEL_PATH}")
    return MODEL_PATH


# ── YOLO ───────────────────────────────────────────────────────────────────
def ensure_yolo(filename: str) -> str:
    """
    Return a path or model name that YOLO() can load.
    Checks two locations relative to this file (submission root).
    If not found locally, returns the bare filename so Ultralytics
    auto-downloads it from its hub on first use.
    """
    for base in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, "..")):
        p = os.path.normpath(os.path.join(base, filename))
        if os.path.isfile(p):
            return p
    # Not found locally — Ultralytics will download automatically
    print(f"[model_utils] {filename} not found locally — Ultralytics will download it.")
    return filename
