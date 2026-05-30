"""
run.py — entry point.

Usage:
    python run.py                          # process all videos under ../input/
    python run.py path/to/video.mp4        # process a single video
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from process_video import process_video

INPUT_ROOT  = Path(__file__).parent.parent / "input"
OUTPUT_ROOT = Path(__file__).parent.parent / "output"


def run_all() -> None:
    videos = sorted(INPUT_ROOT.rglob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found under {INPUT_ROOT}")
        return

    for video_path in videos:
        rel        = video_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / rel
        print(f"Processing {rel} ...", flush=True)
        process_video(str(video_path), str(output_path))

    print("\nAll videos processed.")


def run_single(video_path: str) -> None:
    p = Path(video_path)
    try:
        rel = p.resolve().relative_to(INPUT_ROOT.resolve())
        output_path = OUTPUT_ROOT / rel
    except ValueError:
        output_path = OUTPUT_ROOT / p.name
    process_video(str(p), str(output_path))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_all()
