"""
Dashboard ROI — fixed polygon derived from camera + cabin geometry.

The polygon is defined in normalised [0,1] coordinates so it scales
with any resolution.  Call `build(w, h)` once per video to get the
pixel-space polygon and a fast point-in-polygon test.
"""

import numpy as np
import cv2
import config


class DashboardROI:
    def __init__(self, frame_w: int, frame_h: int):
        pts_norm = np.array(config.DASHBOARD_ROI_NORM, dtype=np.float32)
        self.polygon = (pts_norm * np.array([frame_w, frame_h])).astype(np.int32)

    def contains(self, x: float, y: float) -> bool:
        """True if pixel point (x, y) is inside the dashboard polygon."""
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def draw(self, frame: np.ndarray, color=(0, 165, 255), alpha=0.15) -> np.ndarray:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.polygon], True, color, 2)
        return frame
