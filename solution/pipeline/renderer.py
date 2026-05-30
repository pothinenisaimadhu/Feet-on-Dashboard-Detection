"""
Output renderer — draws ROI, ankle/knee joints only, and verdict label.
No full-body skeleton to keep the overlay clean.
"""

import cv2
import numpy as np
import config


def render(
    frame: np.ndarray,
    passenger_box: np.ndarray | None,
    pose_data: dict,
    roi,
    is_positive: bool,
    score: float,
    debug_text: str,
    sel_conf: float = 0.0,
    frame_idx: int = 0,
) -> np.ndarray:
    # 1. Dashboard ROI overlay
    roi.draw(frame)

    # 2. Passenger selection frame (YOLO person box)
    if passenger_box is not None:
        x1, y1, x2, y2 = map(int, passenger_box)
        verdict = config.COLOR_POS if is_positive else config.COLOR_NEG
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_SEL, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), verdict, 1)
        label = f"PASSENGER {sel_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), config.COLOR_SEL, -1)
        cv2.putText(frame, label, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. Ankle joints — large circles, colour by ROI membership
    for key in ("left_ankle_px", "right_ankle_px"):
        pt = pose_data.get(key)
        if pt is None:
            continue
        in_roi = roi.contains(pt[0], pt[1])
        color  = config.COLOR_POS if in_roi else (255, 255, 0)
        cv2.circle(frame, pt, 10, color, -1)
        cv2.circle(frame, pt, 10, (255, 255, 255), 2)   # white outline

    # 4. Knee joints — smaller, always yellow
    for key in ("left_knee_px", "right_knee_px"):
        pt = pose_data.get(key)
        if pt is None:
            continue
        cv2.circle(frame, pt, 7, (0, 200, 255), -1)

    # 5. Leg line: knee → ankle
    for side in ("left", "right"):
        kpt = pose_data.get(f"{side}_knee_px")
        apt = pose_data.get(f"{side}_ankle_px")
        if kpt and apt:
            color = config.COLOR_POS if roi.contains(apt[0], apt[1]) else (200, 200, 200)
            cv2.line(frame, kpt, apt, color, 2)

    # 6. Verdict label
    label         = "FEET ON DASHBOARD" if is_positive else "NORMAL POSTURE"
    verdict_color = config.COLOR_POS if is_positive else config.COLOR_NEG
    cv2.putText(frame, label,
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, verdict_color, config.FONT_THICKNESS, cv2.LINE_AA)

    # 7. Debug readout (knee angle + elevation + pose backend)
    src = pose_data.get("pose_source", "")
    dbg = debug_text + (f" [{src}]" if src and src != "none" else "")
    cv2.putText(frame, dbg,
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE * 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 8. Confidence + frame index
    cv2.putText(frame, "Confidence: " + str(round(score, 2)),
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE * 0.6, verdict_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {frame_idx}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE * 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
