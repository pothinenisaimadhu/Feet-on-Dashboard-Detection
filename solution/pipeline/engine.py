"""
Shared inference engine — load YOLO + MediaPipe once, process many videos.
"""

import os
import cv2
from .detector import PassengerDetector
from .pose import PoseEstimator
from .roi import DashboardROI
from .scorer import score_frame
from .smoother import TemporalSmoother
from .renderer import render
import config


class OOPEngine:
    def __init__(self):
        self.detector = PassengerDetector()
        self.pose = PoseEstimator()
        self._last_box = None
        self._last_conf = 0.0
        self._miss_streak = 0

    def process_video(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError("Cannot open video: " + input_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        roi = DashboardROI(w, h)
        smoother = TemporalSmoother()
        high_ankle_mode = self._bootstrap_high_ankle(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._last_box = None
        self._last_conf = 0.0
        self._miss_streak = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            need_detect = (
                self._last_box is None
                or frame_idx % config.DETECT_EVERY == 0
                or self._miss_streak >= config.DETECT_MISS_MAX
            )
            if need_detect:
                box, conf = self.detector.detect(frame)
                if box is not None:
                    self._last_box, self._last_conf = box, conf
                    self._miss_streak = 0
                else:
                    self._miss_streak += 1
            passenger_box, sel_conf = self._last_box, self._last_conf

            crop_rect = None
            if passenger_box is not None:
                _, crop_rect = self.detector.crop_for_pose(frame, passenger_box)

            pose_data = self.pose.analyse(frame, passenger_box, crop_rect)
            has_pose = pose_data["landmarks"] is not None
            raw_positive, score, debug = score_frame(
                pose_data, roi, high_ankle_mode=high_ankle_mode
            )
            is_positive = smoother.update(raw_positive, has_pose, score)

            frame = render(
                frame, passenger_box, pose_data, roi,
                is_positive, score, debug, sel_conf, frame_idx,
            )
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

    def _bootstrap_high_ankle(self, cap) -> bool:
        """Profile3-style: median ankle_y always high -> shelf/ROI scoring path."""
        ankle_ys: list[float] = []
        knee_angles: list[float] = []
        n = 0
        while n < 120:
            ok, frame = cap.read()
            if not ok:
                break
            # Bug 3 fix: use passenger-filtered detection during bootstrap
            box, _ = self.detector.detect(frame)
            _, crop_rect = self.detector.crop_for_pose(frame, box) if box is not None else (None, None)
            pd = self.pose.analyse(frame, box, crop_rect)
            if pd["landmarks"] is None:
                n += 1
                continue
            for side in ("left", "right"):
                ay = pd.get(f"{side}_ankle_y_norm")
                if ay:
                    ankle_ys.append(ay)
                ka = pd.get(f"{side}_knee_angle")
                if ka:
                    knee_angles.append(ka)
            n += 1
        if len(ankle_ys) < 6:
            return False
        import statistics
        med_ay   = statistics.median(ankle_ys)
        med_knee = statistics.median(knee_angles) if knee_angles else 0.0
        return (
            med_ay >= config.HIGH_ANKLE_BASELINE
            and med_knee >= config.HIGH_ANKLE_KNEE_MED_MIN
        )
