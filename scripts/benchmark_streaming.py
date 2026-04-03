import argparse
import os
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import app  # noqa: E402


def _mock_detection():
    return (
        "happy",
        0.78,
        {"x": 120, "y": 80, "w": 220, "h": 220},
        True,
        {
            "happy": 78.0,
            "neutral": 12.0,
            "sad": 4.0,
            "angry": 2.0,
            "surprise": 2.0,
            "fear": 1.0,
            "disgust": 1.0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Synthetic streaming benchmark for process_frame_async.")
    parser.add_argument("--frames", type=int, default=300, help="Number of benchmark frames.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames.")
    args = parser.parse_args()

    sid = "bench-sid"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("app.detect_emotion", side_effect=lambda _frame: _mock_detection()):
        with patch.object(app.socketio, "emit", return_value=None):
            for _ in range(max(0, args.warmup)):
                app.process_frame_async(frame, sid)

            started = time.perf_counter()
            for _ in range(max(1, args.frames)):
                app.process_frame_async(frame, sid)
            elapsed = time.perf_counter() - started

    fps = (args.frames / elapsed) if elapsed > 0 else 0.0
    ms_per_frame = (elapsed * 1000.0 / args.frames) if args.frames > 0 else 0.0
    print(f"[BENCH] frames={args.frames} elapsed={elapsed:.3f}s fps={fps:.2f} ms/frame={ms_per_frame:.2f}")
    print(f"[BENCH] runtime_metrics={app._runtime_metrics_snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

