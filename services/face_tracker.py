import math
import threading

import numpy as np


def _as_box(face_location):
    if not isinstance(face_location, dict):
        return None
    try:
        x = int(face_location.get("x", 0))
        y = int(face_location.get("y", 0))
        w = int(face_location.get("w", 0))
        h = int(face_location.get("h", 0))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _center_distance_ratio(a, b, frame_w, frame_h):
    if frame_w <= 0 or frame_h <= 0:
        return 0.0
    acx = a["x"] + (a["w"] / 2.0)
    acy = a["y"] + (a["h"] / 2.0)
    bcx = b["x"] + (b["w"] / 2.0)
    bcy = b["y"] + (b["h"] / 2.0)
    distance = math.sqrt(((acx - bcx) ** 2) + ((acy - bcy) ** 2))
    diagonal = math.sqrt((frame_w**2) + (frame_h**2))
    if diagonal <= 0:
        return 0.0
    return distance / diagonal


class SessionFaceTracker:
    """Lightweight single-face session tracker for box smoothing and short-term hold."""

    def __init__(
        self,
        enabled=True,
        smoothing_alpha=0.65,
        max_missing_frames=3,
        reacquire_iou_threshold=0.05,
        max_center_jump_ratio=0.45,
    ):
        self.enabled = bool(enabled)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.1, 1.0))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.reacquire_iou_threshold = float(np.clip(reacquire_iou_threshold, 0.0, 1.0))
        self.max_center_jump_ratio = float(np.clip(max_center_jump_ratio, 0.05, 1.0))
        self.lock = threading.Lock()
        self.state = {}

    def clear(self, sid):
        with self.lock:
            self.state.pop(sid, None)

    def _smooth_box(self, prev_box, next_box):
        alpha = self.smoothing_alpha
        return {
            "x": int(round((alpha * next_box["x"]) + ((1.0 - alpha) * prev_box["x"]))),
            "y": int(round((alpha * next_box["y"]) + ((1.0 - alpha) * prev_box["y"]))),
            "w": int(round((alpha * next_box["w"]) + ((1.0 - alpha) * prev_box["w"]))),
            "h": int(round((alpha * next_box["h"]) + ((1.0 - alpha) * prev_box["h"]))),
        }

    def update(self, sid, face_location, frame_width=0, frame_height=0):
        """
        Returns:
        - stabilized_face_location (dict or None)
        - is_tracked (bool): true when output came from hold/smoothing of prior state
        """
        box = _as_box(face_location)
        if not self.enabled:
            return box, False

        with self.lock:
            session_state = self.state.get(sid)

            if box is None:
                if not session_state:
                    return None, False
                missed = int(session_state.get("missed", 0)) + 1
                session_state["missed"] = missed
                if missed > self.max_missing_frames:
                    self.state.pop(sid, None)
                    return None, False
                return dict(session_state["box"]), True

            if not session_state:
                self.state[sid] = {"box": box, "missed": 0}
                return box, False

            prev_box = session_state["box"]
            overlap = _iou(prev_box, box)
            jump_ratio = _center_distance_ratio(prev_box, box, int(frame_width), int(frame_height))

            if overlap < self.reacquire_iou_threshold and jump_ratio > self.max_center_jump_ratio:
                # Large jump with low overlap: treat as new face acquisition.
                session_state["box"] = box
                session_state["missed"] = 0
                return box, False

            stabilized = self._smooth_box(prev_box, box)
            session_state["box"] = stabilized
            session_state["missed"] = 0
            return dict(stabilized), True

