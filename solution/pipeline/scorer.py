"""
Scoring — negative gate first, then fused leg geometry (not ankle-only OR chains).

Signals use hip/knee/ankle relations; ROI is supporting evidence only.
"""

import config


def _shin_forward_ratio(knee_px, ankle_px) -> float | None:
    """Horizontal / vertical shin vector; larger → foot extended toward dashboard."""
    if knee_px is None or ankle_px is None:
        return None
    dx = abs(ankle_px[0] - knee_px[0])
    dy = abs(ankle_px[1] - knee_px[1]) + 1e-6
    return dx / dy


def _clearly_normal(ankle_y: float, knee_angle: float, elev: float) -> bool:
    """Early gate — normal seated posture; blocks all positive paths."""
    if ankle_y < config.ANKLE_Y_NORMAL_MAX:
        return True
    if knee_angle > config.KNEE_STRAIGHT_THRESHOLD and elev > config.ELEV_NORMAL_MIN:
        return True
    return False


def _score_leg_standard(
    ankle_y: float,
    knee_angle: float,
    elev: float,
    in_roi: bool,
    forward_drop: float,
    shin_ratio: float | None,
) -> tuple[bool, float, str]:
    """
    Standard cabin (variable ankle_y).
    Positive requires bent leg + forward foot; ROI alone never triggers.
    """
    bent = knee_angle < config.KNEE_BENT_THRESHOLD
    forward = forward_drop >= config.FORWARD_DROP_MIN
    low_hip = elev < config.ELEV_NEGATIVE_THRESHOLD

    # Fused OOP: need bent knee AND evidence of forward extension
    geometry_oop = bent and forward and low_hip
    strong_oop = (
        ankle_y >= config.ANKLE_Y_STRONG
        and bent
        and low_hip
    )
    # ROI only counts with full leg geometry (not ankle-in-box alone)
    roi_oop = (
        in_roi
        and ankle_y >= config.ANKLE_Y_ROI_MIN
        and bent
        and forward
    )

    is_oop = geometry_oop or strong_oop or roi_oop

    score_geom = min(forward_drop / 0.25, 1.0) if forward else 0.0
    score_bent = 1.0 - min(knee_angle / config.KNEE_BENT_THRESHOLD, 1.0) if bent else 0.0
    score_roi = 0.35 if roi_oop else (0.15 if in_roi else 0.0)
    shin_boost = min((shin_ratio or 0) / 2.0, 0.25)
    combined = min(0.4 * score_geom + 0.35 * score_bent + score_roi + shin_boost, 1.0)
    if not is_oop:
        combined *= 0.15

    debug = (
        f"knee={int(knee_angle)} fwd={round(forward_drop, 2)} "
        f"elev={round(elev, 2)} roi={'Y' if in_roi else 'N'}"
    )
    return is_oop, combined, debug


def _clearly_normal_high_ankle(
    ankle_y: float, knee_angle: float, in_roi: bool
) -> bool:
    """High-ankle cabin: foot not on dashboard shelf."""
    if not in_roi:
        return True
    if knee_angle < config.KNEE_TUCKED_THRESHOLD:
        return True
    return False


def _score_leg_high_ankle(
    ankle_y: float,
    knee_angle: float,
    in_roi: bool,
) -> tuple[bool, float, str]:
    """
    High ankle_y cabin (profile3): ankle_y always elevated.
    Discriminator: ankle inside tight ROI + extended leg on shelf.
    """
    shelf_oop = (
        in_roi
        and ankle_y >= config.ANKLE_Y_ROI_MIN
        and knee_angle >= config.KNEE_EXTENDED_MIN
    )

    is_oop = shelf_oop
    combined = 0.0
    if in_roi:
        combined += 0.45
    if ankle_y >= config.ANKLE_Y_ROI_MIN:
        combined += 0.30
    if knee_angle >= config.KNEE_EXTENDED_MIN:
        combined += 0.25
    if not is_oop:
        combined *= 0.15

    debug = (
        f"[hi] knee={int(knee_angle)} ankle_y={round(ankle_y, 2)} "
        f"roi={'Y' if in_roi else 'N'}"
    )
    return is_oop, min(combined, 1.0), debug


def score_frame(
    pose_data: dict,
    roi,
    high_ankle_mode: bool = False,
) -> tuple[bool, float, str]:
    """
    Returns (is_positive, confidence_0_to_1, debug_text).
    """
    if pose_data["landmarks"] is None:
        return False, 0.0, "no pose"

    best_score = 0.0
    best_debug = ""
    triggered = False

    for side in ("left", "right"):
        ankle_px = pose_data.get(f"{side}_ankle_px")
        knee_px = pose_data.get(f"{side}_knee_px")
        if ankle_px is None:
            continue

        knee_angle = pose_data.get(f"{side}_knee_angle", 0.0)
        ankle_y = pose_data.get(f"{side}_ankle_y_norm", 0.0)
        hip_y = pose_data.get(f"{side}_hip_y_norm", 0.0)
        elev = hip_y - ankle_y
        forward_drop = ankle_y - hip_y
        in_roi = roi.contains(ankle_px[0], ankle_px[1])
        shin_ratio = _shin_forward_ratio(knee_px, ankle_px)

        # ── 1. Negative gate FIRST (ROI cannot override) ──────────────────
        if high_ankle_mode:
            if _clearly_normal_high_ankle(ankle_y, knee_angle, in_roi):
                continue
        elif _clearly_normal(ankle_y, knee_angle, elev):
            continue

        # ── 2. Positive escalation only if not clearly normal ─────────────
        if high_ankle_mode:
            fire, combined, debug = _score_leg_high_ankle(
                ankle_y, knee_angle, in_roi
            )
        else:
            fire, combined, debug = _score_leg_standard(
                ankle_y, knee_angle, elev, in_roi, forward_drop, shin_ratio
            )

        debug = f"{side} {debug}"

        if fire:
            triggered = True
        if combined > best_score:
            best_score = combined
            best_debug = debug

    return triggered, min(best_score, 1.0), best_debug
