"""
Evaluate pipeline on clean vs noisy val images.

Metrics per noise type:
  - pose_detected   : % frames where YOLOv8n-pose found the passenger skeleton
  - ankle_visible   : % frames where ankle confidence >= KP_MIN_CONF
  - score_mean      : mean raw score across frames
  - positive_rate   : % frames classified POSITIVE (raw, before temporal smoothing)

Run:
    python solution/eval_noise.py
"""

import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from pipeline import PassengerDetector, PoseEstimator, DashboardROI
from pipeline.scorer import score_frame
import config

NOISY_DIR = Path("dataset/images/noisy")
CLEAN_DIR = Path("dataset/images/val")
FRAME_W, FRAME_H = 1920, 1080   # original resolution (images are full-res)

detector = PassengerDetector()
pose     = PoseEstimator()

# Use a dummy ROI at full resolution
roi = DashboardROI(FRAME_W, FRAME_H)

NOISE_TYPES = ["gaussian", "salt_pepper", "blur", "dark", "combined"]

def eval_images(img_paths: list, label: str):
    total = 0
    pose_detected = 0
    ankle_visible = 0
    score_sum = 0.0
    positive_count = 0

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        total += 1

        pbox     = detector.detect(img)
        pdata    = pose.analyse(img, pbox)

        if pdata["landmarks"] is not None:
            pose_detected += 1

        # check ankle confidence — only possible from YOLO numpy array
        if pdata["pose_source"] == "yolo" and pdata["landmarks"] is not None:
            import numpy as np
            kp = pdata["landmarks"]
            if isinstance(kp, np.ndarray) and kp.shape[0] > 16:
                l_conf = float(kp[15, 2])
                r_conf = float(kp[16, 2])
                if max(l_conf, r_conf) >= config.KP_MIN_CONF:
                    ankle_visible += 1
        elif pdata["left_ankle_px"] or pdata["right_ankle_px"]:
            ankle_visible += 1

        raw_pos, score, _ = score_frame(pdata, roi)
        score_sum += score
        if raw_pos:
            positive_count += 1

    if total == 0:
        print(f"  {label:<20}  no images found")
        return

    print(
        f"  {label:<20}  "
        f"n={total:>4}  "
        f"pose={100*pose_detected//total:>3}%  "
        f"ankle={100*ankle_visible//total:>3}%  "
        f"score={score_sum/total:.3f}  "
        f"pos_rate={100*positive_count//total:>3}%"
    )

print(f"\n{'type':<20}  {'n':>6}  {'pose':>5}  {'ankle':>6}  {'score':>6}  {'pos_rate':>9}")
print("-" * 65)

# Clean baseline
clean_imgs = sorted(CLEAN_DIR.glob("*.jpg"))
eval_images(clean_imgs, "clean (baseline)")

# Per noise type
for noise in NOISE_TYPES:
    noisy_imgs = sorted(NOISY_DIR.glob(f"*_{noise}.jpg"))
    eval_images(noisy_imgs, noise)
