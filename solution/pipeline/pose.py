"""
Pose estimation — YOLOv8n-pose (primary) + MediaPipe (fallback).

YOLO pose is more reliable in this cabin dataset (thick clothing, top-down view).
"""

import os
import numpy as np
from ultralytics import YOLO
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks import python as mp_tasks
from .model_utils import ensure_model
import config

# COCO keypoint indices
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16


def _knee_angle_px(hip, knee, ankle) -> float:
    a = np.array(hip, dtype=float)
    b = np.array(knee, dtype=float)
    c = np.array(ankle, dtype=float)
    ab = a - b
    cb = c - b
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def _pose_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.join(here, "..", ".."), os.path.join(here, "..")):
        p = os.path.normpath(os.path.join(base, "yolov8n-pose.pt"))
        if os.path.isfile(p):
            return p
    return "yolov8n-pose.pt"


class PoseEstimator:
    def __init__(self):
        self._yolo = YOLO(_pose_model_path())
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model()),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=config.MP_NUM_POSES,
            min_pose_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
        )
        self._mp = mp_vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _empty() -> dict:
        return dict(
            landmarks=None,
            left_knee_angle=0.0,
            right_knee_angle=0.0,
            left_ankle_px=None,
            right_ankle_px=None,
            left_knee_px=None,
            right_knee_px=None,
            left_hip_y_norm=0.0,
            right_hip_y_norm=0.0,
            left_ankle_y_norm=0.0,
            right_ankle_y_norm=0.0,
            left_ankle_x_norm=0.0,
            right_ankle_x_norm=0.0,
            pose_source="none",
        )

    def _from_yolo(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        out = self._empty()
        results = self._yolo(
            frame, imgsz=config.IMGSZ, conf=config.POSE_CONF, verbose=False
        )[0]

        if results.keypoints is None or len(results.keypoints) == 0:
            return out

        kps_all = results.keypoints.data.cpu().numpy()
        best_idx = None
        best_hip_x = 0.0

        for i, kp in enumerate(kps_all):
            if kp[_L_HIP, 2] < config.KP_MIN_CONF and kp[_R_HIP, 2] < config.KP_MIN_CONF:
                continue
            avg_hip_x = (kp[_L_HIP, 0] + kp[_R_HIP, 0]) / 2 / w
            if avg_hip_x >= config.PASSENGER_HIP_X_MIN and avg_hip_x > best_hip_x:
                best_hip_x = avg_hip_x
                best_idx = i

        if best_idx is None:
            return out

        kp = kps_all[best_idx]
        out["landmarks"] = kp
        out["pose_source"] = "yolo"

        for side, hip_i, knee_i, ankle_i in [
            ("left", _L_HIP, _L_KNEE, _L_ANKLE),
            ("right", _R_HIP, _R_KNEE, _R_ANKLE),
        ]:
            if min(kp[hip_i, 2], kp[knee_i, 2], kp[ankle_i, 2]) < config.KP_MIN_CONF:
                continue
            hip_pt = (int(kp[hip_i, 0]), int(kp[hip_i, 1]))
            knee_pt = (int(kp[knee_i, 0]), int(kp[knee_i, 1]))
            ankle_pt = (int(kp[ankle_i, 0]), int(kp[ankle_i, 1]))
            out[f"{side}_knee_angle"] = _knee_angle_px(hip_pt, knee_pt, ankle_pt)
            out[f"{side}_ankle_px"] = ankle_pt
            out[f"{side}_knee_px"] = knee_pt
            out[f"{side}_hip_y_norm"] = hip_pt[1] / h
            out[f"{side}_ankle_y_norm"] = ankle_pt[1] / h
            out[f"{side}_ankle_x_norm"] = ankle_pt[0] / w

        return out

    def _from_mediapipe(
        self, frame: np.ndarray, crop_rect: tuple[int, int, int, int] | None
    ) -> dict:
        h, w = frame.shape[:2]
        off_x, off_y = 0, 0
        work = frame
        if crop_rect is not None:
            x1, y1, x2, y2 = crop_rect
            if x2 > x1 and y2 > y1:
                work = frame[y1:y2, x1:x2]
                off_x, off_y = x1, y1
        ch, cw = work.shape[:2]
        if ch < 32 or cw < 32:
            return self._empty()

        out = self._empty()
        mp_img = Image(
            image_format=ImageFormat.SRGB,
            data=np.ascontiguousarray(work[:, :, ::-1]),
        )
        result = self._mp.detect(mp_img)
        if not result or not result.pose_landmarks:
            return out

        L = mp_vision.PoseLandmark
        passenger_lm = None
        best_x = -1.0
        for lm in result.pose_landmarks:
            # If we already cropped to the passenger box, skip the hip_x filter —
            # the crop is already passenger-isolated so all landmarks belong to them.
            if crop_rect is not None:
                passenger_lm = lm
                break
            avg_hip_x = (lm[L.LEFT_HIP].x + lm[L.RIGHT_HIP].x) / 2
            if avg_hip_x >= config.PASSENGER_HIP_X_MIN and avg_hip_x > best_x:
                passenger_lm = lm
                best_x = avg_hip_x
        if passenger_lm is None:
            return out

        lm = passenger_lm
        out["landmarks"] = lm
        out["pose_source"] = "mediapipe"

        def px(lmk):
            return (int(lmk.x * cw) + off_x, int(lmk.y * ch) + off_y)

        for side, hl, kl, al in [
            ("left", L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
            ("right", L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
        ]:
            hip_pt = px(lm[hl])
            ankle_pt = px(lm[al])
            out[f"{side}_knee_angle"] = _knee_angle_px(
                hip_pt, px(lm[kl]), ankle_pt
            )
            out[f"{side}_ankle_px"] = ankle_pt
            out[f"{side}_knee_px"] = px(lm[kl])
            out[f"{side}_hip_y_norm"] = hip_pt[1] / h
            out[f"{side}_ankle_y_norm"] = ankle_pt[1] / h
            out[f"{side}_ankle_x_norm"] = ankle_pt[0] / w

        return out

    def analyse(
        self,
        frame: np.ndarray,
        passenger_box=None,
        crop_rect: tuple[int, int, int, int] | None = None,
    ) -> dict:
        out = self._from_yolo(frame)
        if out["landmarks"] is None:
            out = self._from_mediapipe(frame, crop_rect)
        return out
