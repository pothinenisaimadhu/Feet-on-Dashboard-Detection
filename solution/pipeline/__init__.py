"""
__init__.py — expose pipeline modules as a package.
"""

from .detector    import PassengerDetector
from .pose        import PoseEstimator
from .roi         import DashboardROI
from .scorer      import score_frame
from .smoother    import TemporalSmoother
from .renderer    import render
from .model_utils import ensure_model
from .engine      import OOPEngine

__all__ = [
    "PassengerDetector",
    "PoseEstimator",
    "DashboardROI",
    "score_frame",
    "TemporalSmoother",
    "render",
    "ensure_model",
    "OOPEngine",
]
