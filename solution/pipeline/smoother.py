"""
Temporal smoother — EMA confidence + hysteresis (more stable than hard majority).
"""

import config


class TemporalSmoother:
    def __init__(self):
        self._ema = 0.0
        self._positive = False

    def update(
        self,
        frame_positive: bool,
        has_pose: bool,
        confidence: float = 0.0,
    ) -> bool:
        if not has_pose:
            target = 0.0
        elif frame_positive:
            target = confidence
        else:
            target = 0.0

        a = config.EMA_ALPHA
        self._ema = a * target + (1.0 - a) * self._ema

        if self._positive:
            if self._ema < config.HYST_OFF:
                self._positive = False
        else:
            if self._ema >= config.HYST_ON:
                self._positive = True

        return self._positive

    def reset(self) -> None:
        self._ema = 0.0
        self._positive = False
