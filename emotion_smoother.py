import threading
import numpy as np


class EmotionSmoother:
    """
    EMA smoother with confidence gating + hysteresis.

    Notes:
    - `window_size` is accepted for backward compatibility with old tests/callers.
    - Scores can be 0..1 or 0..100; class keeps native scale unchanged.
    """

    def __init__(
        self,
        alpha=0.4,
        confidence_threshold=0.35,
        switch_in_frames=3,
        switch_out_frames=4,
        stable_hold_ratio=0.9,
        stale_reset_frames=15,
        min_dominance_margin=2.5,
        window_size=None
    ):
        self.alpha = float(np.clip(alpha, 0.05, 0.95))
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        self.switch_in_frames = max(1, int(switch_in_frames))
        self.switch_out_frames = max(1, int(switch_out_frames))
        self.stable_hold_ratio = float(np.clip(stable_hold_ratio, 0.0, 1.0))
        self.stale_reset_frames = max(1, int(stale_reset_frames))
        self.min_dominance_margin = max(0.0, float(min_dominance_margin))

        # Backward compatibility: accepted but no longer used in EMA mode.
        self.window_size = window_size

        self.ema_scores = {}
        self.stable_emotion = "unknown"
        self.candidate_emotion = None
        self.candidate_count = 0
        self.stale_frame_count = 0
        self.lock = threading.Lock()

    def _reset_stable_state(self, clear_scores=False):
        self.stable_emotion = "unknown"
        self.candidate_emotion = None
        self.candidate_count = 0
        if clear_scores:
            self.ema_scores = {}

    def _dominance_margin(self, scores):
        if not scores:
            return 0.0
        ordered = sorted((float(v) for v in scores.values()), reverse=True)
        if not ordered:
            return 0.0
        top1 = ordered[0]
        top2 = ordered[1] if len(ordered) > 1 else 0.0
        margin = max(0.0, top1 - top2)
        # Support both 0..1 and 0..100 score scales.
        if top1 <= 1.5:
            margin *= 100.0
        return margin

    def _switch_requirement(self, target_emotion):
        # Entering an expressive state is faster than switching away from a stable state.
        if self.stable_emotion in ("unknown", "neutral") and target_emotion not in ("unknown", "neutral"):
            return self.switch_in_frames
        return self.switch_out_frames

    def _update_hysteresis(self, dominant_emotion):
        if not dominant_emotion or dominant_emotion == "unknown":
            return

        if self.stable_emotion == dominant_emotion:
            self.candidate_emotion = None
            self.candidate_count = 0
            return

        if self.candidate_emotion == dominant_emotion:
            self.candidate_count += 1
        else:
            self.candidate_emotion = dominant_emotion
            self.candidate_count = 1

        required_frames = self._switch_requirement(dominant_emotion)
        if self.candidate_count >= required_frames:
            self.stable_emotion = dominant_emotion
            self.candidate_emotion = None
            self.candidate_count = 0

    def update(self, emotion_scores, face_detected=True, detection_confidence=1.0):
        with self.lock:
            if not face_detected or not emotion_scores:
                self.stale_frame_count += 1
                if self.stale_frame_count >= self.stale_reset_frames:
                    self._reset_stable_state(clear_scores=True)
                return False
            if detection_confidence is None or float(detection_confidence) < self.confidence_threshold:
                self.stale_frame_count += 1
                if self.stale_frame_count >= self.stale_reset_frames:
                    self._reset_stable_state(clear_scores=True)
                return False

            self.stale_frame_count = 0
            if not self.ema_scores:
                self.ema_scores = {emotion: float(score) for emotion, score in emotion_scores.items()}
            else:
                all_emotions = set(self.ema_scores.keys()) | set(emotion_scores.keys())
                next_scores = {}
                for emotion in all_emotions:
                    prev_score = float(self.ema_scores.get(emotion, 0.0))
                    current_score = float(emotion_scores.get(emotion, 0.0))
                    next_scores[emotion] = (self.alpha * current_score) + ((1.0 - self.alpha) * prev_score)
                self.ema_scores = next_scores

            frame_dominant = max(emotion_scores, key=emotion_scores.get)
            if self.min_dominance_margin > 0:
                margin = self._dominance_margin(emotion_scores)
                if margin < self.min_dominance_margin:
                    return True

            self._update_hysteresis(frame_dominant)
            return True

    def get_averaged_scores(self):
        with self.lock:
            return dict(self.ema_scores)

    def get_dominant_emotion(self, averaged_scores=None, prefer_stable=True):
        with self.lock:
            if averaged_scores is None:
                averaged_scores = self.ema_scores
            if not averaged_scores:
                return "unknown", 0.0

            dominant = max(averaged_scores, key=averaged_scores.get)
            dominant_score = float(averaged_scores.get(dominant, 0.0))

            if prefer_stable and self.stable_emotion in averaged_scores and self.stable_emotion != "unknown":
                stable_score = float(averaged_scores.get(self.stable_emotion, 0.0))
                if self.stable_emotion == dominant:
                    return self.stable_emotion, stable_score
                if stable_score >= (dominant_score * self.stable_hold_ratio):
                    return self.stable_emotion, stable_score

            return dominant, dominant_score
