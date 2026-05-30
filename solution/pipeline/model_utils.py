"""
Downloads the MediaPipe PoseLandmarker model file if not already present.
Called automatically by pose.py on first use.
"""

import os
import urllib.request

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")
MODEL_PATH = os.path.normpath(MODEL_PATH)


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe PoseLandmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Saved to", MODEL_PATH)
    return MODEL_PATH
