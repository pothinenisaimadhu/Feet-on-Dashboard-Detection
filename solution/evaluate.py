"""
Quick accuracy report on filename labels (positive_* / negative_*).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import cv2
from pathlib import Path
from pipeline.engine import OOPEngine
from pipeline.roi import DashboardROI
from pipeline.scorer import score_frame
from pipeline.smoother import TemporalSmoother
import config

INPUT = Path(__file__).parent.parent / "input"


def eval_video(engine: OOPEngine, vp: Path) -> dict:
    cap = cv2.VideoCapture(str(vp))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = DashboardROI(w, h)
    sm = TemporalSmoother()
    engine._last_box = None
    engine._miss_streak = 0
    high_ankle = engine._bootstrap_high_ankle(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    raw_p = sm_p = pose_ok = n = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        need = (
            engine._last_box is None
            or idx % config.DETECT_EVERY == 0
            or engine._miss_streak >= config.DETECT_MISS_MAX
        )
        if need:
            box, conf = engine.detector.detect(frame)
            if box is not None:
                engine._last_box, engine._last_conf = box, conf
                engine._miss_streak = 0
            else:
                engine._miss_streak += 1
        box = engine._last_box
        crop = engine.detector.crop_for_pose(frame, box)[1] if box is not None else None
        pd = engine.pose.analyse(frame, box, crop)
        has_pose = pd["landmarks"] is not None
        pose_ok += int(has_pose)
        r, sc, _ = score_frame(pd, roi, high_ankle_mode=high_ankle)
        s = sm.update(r, has_pose, sc)
        raw_p += int(r)
        sm_p += int(s)
        n += 1
        idx += 1
    cap.release()
    return dict(frames=n, raw=raw_p, smooth=sm_p, pose_rate=pose_ok / n if n else 0)


def main() -> None:
    engine = OOPEngine()
    neg_smooth, pos_smooth = [], []
    neg_pose, pos_pose = [], []

    print(f"{'Video':<32} {'Label':<8} {'Pose%':>6} {'Raw%':>6} {'Smooth%':>8}")
    print("-" * 68)

    for vp in sorted(INPUT.rglob("*.mp4")):
        label = "POS" if "positive" in vp.stem else "NEG"
        m = eval_video(engine, vp)
        rel = vp.relative_to(INPUT)
        rp = 100 * m["raw"] / m["frames"]
        sp = 100 * m["smooth"] / m["frames"]
        pp = 100 * m["pose_rate"]
        print(f"{rel!s:<32} {label:<8} {pp:5.1f} {rp:5.1f} {sp:7.1f}")
        if label == "NEG":
            neg_smooth.append(sp)
            neg_pose.append(pp)
        else:
            pos_smooth.append(sp)
            pos_pose.append(pp)

    print("-" * 68)
    print(f"NEG avg smooth+ : {sum(neg_smooth)/len(neg_smooth):.1f}%  (lower is better)")
    print(f"POS avg smooth+ : {sum(pos_smooth)/len(pos_smooth):.1f}%  (higher is better)")
    print(f"NEG avg pose det: {sum(neg_pose)/len(neg_pose):.1f}%")
    print(f"POS avg pose det: {sum(pos_pose)/len(pos_pose):.1f}%")


if __name__ == "__main__":
    main()
