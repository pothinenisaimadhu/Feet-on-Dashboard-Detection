"""
process_video.py — process a single video end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.engine import OOPEngine

_engine: OOPEngine | None = None


def _get_engine() -> OOPEngine:
    global _engine
    if _engine is None:
        _engine = OOPEngine()
    return _engine


def process_video(input_path: str, output_path: str) -> None:
    _get_engine().process_video(input_path, output_path)
    print("  done  " + os.path.basename(input_path) + "  ->  " + output_path, flush=True)
