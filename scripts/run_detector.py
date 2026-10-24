#!/usr/bin/env python3
"""Run tactical video ball detector on frames with missing ATD ball.

Auto-skips if video, weights, or GPU are absent.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from games import GAME, video_path
from mtah.healing.detector import is_detector_environment_ready, run_ball_detector


def main():
    parser = argparse.ArgumentParser(description="Run tactical ball detector on missing frames")
    parser.add_argument("--weights", default=str(ROOT / "weights/ball_yolo.pt"), help="Path to YOLO weights")
    parser.add_argument("--video", default=None, help="Path to tactical MP4 video")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA GPU is not available")
    args = parser.parse_args()

    v_path = args.video or video_path(game=GAME)
    w_path = args.weights

    ready, reason = is_detector_environment_ready(v_path, w_path, require_gpu=args.require_gpu)
    if not ready:
        print(f"[SKIP] Detector auto-skipped: {reason}.")
        print("       The pipeline will rely on short-gap Kalman/spline healing without detector.")
        return

    # If ready, run detector
    run_ball_detector(v_path, w_path, missing_frame_indices=[])


if __name__ == "__main__":
    main()
