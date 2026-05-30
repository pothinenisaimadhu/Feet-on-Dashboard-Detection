# Feet on Dashboard Detection — OOP Case 1

## Approach

Hybrid pipeline: **YOLO passenger selection → YOLOv8n-pose (MediaPipe fallback) → adaptive geometry scoring → temporal smoothing**.

```
Frame
  └─► Passenger Selector (YOLOv8n)       — largest person on right → selection box
        └─► Pose (YOLOv8n-pose)          — ankle/knee keypoints (MediaPipe if YOLO fails)
              └─► Bootstrap cabin profile  — standard vs leg-span scoring mode
                    └─► Dashboard ROI + geometry
                          └─► Temporal smoother (11-frame, 6/11 vote, pose required)
                                └─► Renderer → output MP4
```

### Why this works without custom training

The camera and cabin (Hyundai Santa Fe) are fixed, so the dashboard occupies a
predictable region in every frame.  A static normalised polygon (`DASHBOARD_ROI_NORM`
in `config.py`) captures this region.  When ankle/knee keypoints land inside that
polygon the score rises above the threshold and the frame is classified POSITIVE.

### Scoring (per frame, per leg)

1. **`clearly_normal` gate first** — straight leg / low ankle_y returns NEGATIVE before any positive rule (ROI cannot override).
2. **Fused geometry** — bent knee + forward_drop (`ankle_y − hip_y`) + negative elevation; shin horizontal ratio as weak boost.
3. **ROI is supporting only** — requires bent knee + forward extension, not ankle-in-box alone.
4. **High-ankle cabin mode** (auto bootstrap): uses forward_drop + knee bend when `ankle_y` is always high.
5. **Temporal**: EMA confidence + hysteresis (`HYST_ON` / `HYST_OFF`), not hard majority vote.

`DETECT_EVERY` reuses YOLO passenger box between frames; `DETECT_MISS_MAX` forces refresh on tracking loss.

---

## Setup

From the `submission/` folder:

```bash
python -m pip install -r solution/requirements.txt
```

Place `yolov8n.pt` and `yolov8n-pose.pt` in `submission/` (included).  
MediaPipe downloads on first run as pose fallback.

```bash
python solution/evaluate.py   # accuracy table on input/
```

---

## Run

```bash
cd submission
python solution/run.py

# Single video (output keeps profile path)
python solution/run.py input/profile1/positive_1.mp4
```

Output videos are written to `output/<profile>/<video>.mp4` with:
- Orange dashboard ROI overlay
- **Amber passenger selection frame** (YOLO person box) + confidence
- Ankle/knee markers and leg segments
- Label: **FEET ON DASHBOARD** (red) or **NORMAL POSTURE** (green)
- Per-frame confidence and frame index

---

## Tuning

All parameters live in `solution/config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `DASHBOARD_ROI_NORM` | lower-right quad | Dashboard region in frame |
| `ANKLE_Y_THRESHOLD` | 0.75 | Forward foot position |
| `KNEE_BENT_THRESHOLD` | 115 | Bent-leg detection (degrees) |
| `TEMPORAL_WINDOW` | 15 | Smoothing window |
| `POSITIVE_MAJORITY` | 8 | Votes needed for POSITIVE |
| `PASSENGER_SIDE` | `"right"` | Passenger is right half of frame |
| `PASSENGER_HIP_X_MIN` | 0.55 | Passenger side filter |
| `KP_MIN_CONF` | 0.20 | Keypoint conf (geometry validates) |
| `FORWARD_DROP_MIN` | 0.18 | ankle_y − hip_y for forward foot |
| `HYST_ON` / `HYST_OFF` | 0.55 / 0.35 | Temporal hysteresis |
| `DETECT_EVERY` | 2 | YOLO selection every N frames |
| `DETECT_MISS_MAX` | 8 | Re-detect after missed boxes |

---

## Limitations & Future Work

- Dashboard ROI is hand-tuned; a learned detector would generalise to other vehicles.
- Occlusion (e.g. blanket over feet) can cause false negatives — a dedicated foot
  detector (YOLOv11n fine-tuned on `foot_on_dashboard` class) would close this gap.
- ByteTrack / DeepSORT would improve robustness when multiple passengers are present.
- Distortion correction using the provided camera intrinsics could improve keypoint
  accuracy near frame edges.

---

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Euro NCAP OOP Protocol: https://www.euroncap.com
