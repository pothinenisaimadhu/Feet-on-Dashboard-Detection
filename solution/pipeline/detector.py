"""
Passenger selector — YOLOv8n person detection + side filter.

Each frame: detect all persons, keep those on the passenger side (right),
pick the largest box as the front passenger selection.
"""

import numpy as np
from ultralytics import YOLO
from .model_utils import ensure_yolo
import config


def pad_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Expand box by BBOX_PAD_RATIO, clamped to frame."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = config.BBOX_PAD_RATIO
    x1 = max(0, x1 - bw * pad)
    y1 = max(0, y1 - bh * pad)
    x2 = min(w, x2 + bw * pad)
    y2 = min(h, y2 + bh * pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class PassengerDetector:
    def __init__(self):
        self._model = YOLO(ensure_yolo(config.PERSON_MODEL))

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Returns (box [x1,y1,x2,y2], confidence) for the front passenger,
        or (None, 0.0) if not found.
        """
        h, w = frame.shape[:2]
        results = self._model(frame, imgsz=config.IMGSZ,
                              conf=config.PERSON_CONF,
                              classes=[0], verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None, 0.0

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        mid_x = w / 2
        if config.PASSENGER_SIDE == "right":
            idx = [i for i, b in enumerate(boxes) if (b[0] + b[2]) / 2 > mid_x]
        else:
            idx = [i for i, b in enumerate(boxes) if (b[0] + b[2]) / 2 < mid_x]

        if not idx:
            return None, 0.0

        best = max(idx, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        return boxes[best].astype(np.float32), float(confs[best])

    def crop_for_pose(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Padded crop and (x1, y1, x2, y2) in full-frame coords."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, pad_box(box, w, h))
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
