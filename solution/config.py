"""
Central configuration — all tunable parameters in one place.
"""

# ── Passenger selector (YOLOv8n person detector, class 0) ─────────────────
PERSON_MODEL   = "yolov8n.pt"
PERSON_CONF    = 0.40
IMGSZ          = 320
DETECT_EVERY   = 4       # YOLO selection every N frames; reuse box between
DETECT_MISS_MAX = 12     # force re-detect after N frames without a valid box
PASSENGER_SIDE = "right"
BBOX_PAD_RATIO = 0.08

# ── YOLOv8n-pose ──────────────────────────────────────────────────────────
POSE_CONF    = 0.25
KP_MIN_CONF  = 0.20   # low conf OK — geometry validates; avoids FN on occluded feet

# ── MediaPipe Pose (fallback) ──────────────────────────────────────────────
MP_MIN_DETECTION_CONF = 0.40
MP_MIN_TRACKING_CONF  = 0.40
MP_NUM_POSES          = 2

# Passenger-side filter (raised from 0.45 — driver bleed caused FP)
PASSENGER_HIP_X_MIN = 0.55

# ── Scoring — standard mode ───────────────────────────────────────────────
ANKLE_Y_THRESHOLD     = 0.72
ANKLE_Y_STRONG        = 0.80
ANKLE_Y_NORMAL_MAX    = 0.52
ANKLE_Y_ROI_MIN       = 0.68
KNEE_BENT_THRESHOLD   = 120
KNEE_STRAIGHT_THRESHOLD = 132
ELEV_NEGATIVE_THRESHOLD = -0.12
ELEV_NORMAL_MIN       = 0.08
FORWARD_DROP_MIN      = 0.18   # ankle_y - hip_y (foot forward in frame)

# ── High-ankle cabin mode (auto bootstrap) ────────────────────────────────
HIGH_ANKLE_BASELINE   = 0.88
HIGH_ANKLE_KNEE_MED_MIN = 135  # bootstrap: profile3 normal posture reads straighter
KNEE_EXTENDED_MIN     = 125
KNEE_TUCKED_THRESHOLD = 105    # very bent tucked leg → normal

# ── Dashboard ROI (tighter — less lower-seat bleed) ─────────────────────────
DASHBOARD_ROI_NORM = [
    (0.52, 0.72),
    (1.00, 0.72),
    (1.00, 0.98),
    (0.52, 0.98),
]

# ── Temporal smoothing (EMA + hysteresis) ───────────────────────────────────
EMA_ALPHA  = 0.35
HYST_ON    = 0.55   # activate FEET ON DASHBOARD
HYST_OFF   = 0.35   # deactivate (hysteresis gap prevents flicker)

# Legacy names kept for evaluate scripts that import them
TEMPORAL_WINDOW   = 11
POSITIVE_MAJORITY = 6

# ── Rendering ─────────────────────────────────────────────────────────────
FONT_SCALE     = 1.2
FONT_THICKNESS = 2
COLOR_POS  = (0,   0, 255)
COLOR_NEG  = (0, 200,   0)
COLOR_SEL  = (255, 200,   0)
