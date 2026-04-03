import copy
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app


class TestPipelineFeatures(unittest.TestCase):
    def setUp(self):
        self.original_config = copy.deepcopy(app.CONFIG)
        self.sid = "test-sid-pipeline"

        with app.processing_lock:
            app.session_processing_flags.pop(self.sid, None)
        with app.frame_skip_lock:
            app.session_frame_counters.pop(self.sid, None)
            app.session_last_results.pop(self.sid, None)
            app.session_last_smoothed.pop(self.sid, None)
        with app.smoother_lock:
            app.session_smoothers.pop(self.sid, None)
        with app.emotion_history_lock:
            app.session_emotion_state.pop(self.sid, None)
            app.emotion_history = [e for e in app.emotion_history if e.get("sid") != self.sid]

    def tearDown(self):
        app.CONFIG.clear()
        app.CONFIG.update(self.original_config)

        with app.processing_lock:
            app.session_processing_flags.pop(self.sid, None)
        with app.frame_skip_lock:
            app.session_frame_counters.pop(self.sid, None)
            app.session_last_results.pop(self.sid, None)
            app.session_last_smoothed.pop(self.sid, None)
        with app.smoother_lock:
            app.session_smoothers.pop(self.sid, None)
        with app.emotion_history_lock:
            app.session_emotion_state.pop(self.sid, None)
            app.emotion_history = [e for e in app.emotion_history if e.get("sid") != self.sid]

    def _set_minimal_preprocess_config(self, gamma_enabled=True):
        app.CONFIG["image_preprocessing"] = {
            "enabled": True,
            "gamma_correction": {
                "enabled": gamma_enabled,
                "target_brightness": 127
            },
            "clahe": {"enabled": False},
            "denoise": {"enabled": False},
            "sharpen": {"enabled": False}
        }

    def test_preprocess_gamma_brightens_dark_frame(self):
        self._set_minimal_preprocess_config(gamma_enabled=True)
        dark_frame = np.full((120, 160, 3), 30, dtype=np.uint8)

        processed = app.preprocess_frame(dark_frame)

        self.assertGreater(float(processed.mean()), float(dark_frame.mean()))

    def test_preprocess_gamma_darkens_overexposed_frame(self):
        self._set_minimal_preprocess_config(gamma_enabled=True)
        bright_frame = np.full((120, 160, 3), 220, dtype=np.uint8)

        processed = app.preprocess_frame(bright_frame)

        self.assertLess(float(processed.mean()), float(bright_frame.mean()))

    def test_ema_confidence_gating_blocks_low_confidence_updates(self):
        smoother = app.EmotionSmoother(alpha=0.35, confidence_threshold=0.5)
        first_scores = {"happy": 90.0, "sad": 10.0}
        second_scores = {"happy": 10.0, "sad": 90.0}

        updated = smoother.update(first_scores, face_detected=True, detection_confidence=0.95)
        self.assertTrue(updated)
        baseline = smoother.get_averaged_scores()

        updated = smoother.update(second_scores, face_detected=True, detection_confidence=0.3)
        self.assertFalse(updated)
        gated_scores = smoother.get_averaged_scores()

        self.assertEqual(baseline, gated_scores)

    def test_hysteresis_requires_consecutive_frames_before_switch(self):
        smoother = app.EmotionSmoother(
            alpha=0.35,
            confidence_threshold=0.0,
            switch_in_frames=3,
            switch_out_frames=4
        )
        happy_scores = {"happy": 80.0, "neutral": 20.0}
        sad_scores = {"sad": 90.0, "happy": 10.0}

        for _ in range(2):
            smoother.update(happy_scores, face_detected=True, detection_confidence=0.9)
        self.assertEqual(smoother.stable_emotion, "unknown")

        smoother.update(happy_scores, face_detected=True, detection_confidence=0.9)
        self.assertEqual(smoother.stable_emotion, "happy")

        for _ in range(3):
            smoother.update(sad_scores, face_detected=True, detection_confidence=0.9)
        self.assertEqual(smoother.stable_emotion, "happy")

        smoother.update(sad_scores, face_detected=True, detection_confidence=0.9)
        self.assertEqual(smoother.stable_emotion, "sad")

    def test_frame_skipping_reuses_last_result_when_n_is_2(self):
        app.CONFIG.setdefault("emotion", {})
        app.CONFIG["emotion"]["analyze_every_n_frames"] = 2

        mock_result = (
            "happy",
            0.91,
            {"x": 10, "y": 10, "w": 40, "h": 40},
            True,
            {"happy": 91.0, "neutral": 9.0}
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("app.detect_emotion", return_value=mock_result) as mock_detect:
            with patch.object(app.socketio, "emit") as mock_emit:
                app.process_frame_async(frame, self.sid)
                app.process_frame_async(frame, self.sid)

        self.assertEqual(mock_detect.call_count, 1)
        self.assertEqual(mock_emit.call_count, 2)

        first_payload = mock_emit.call_args_list[0].args[1]
        second_payload = mock_emit.call_args_list[1].args[1]
        self.assertEqual(first_payload["emotion"], "happy")
        self.assertEqual(second_payload["emotion"], "happy")
        self.assertTrue(second_payload["face_detected"])

    def test_hsemotion_contempt_maps_to_neutral_by_default(self):
        gpu_config = {"contempt_maps_to": "neutral"}
        idx_to_class = {0: "Contempt", 1: "Happiness"}
        probabilities = np.array([0.7, 0.3], dtype=np.float32)

        scores = app._hsemotion_scores_to_output(probabilities, idx_to_class, gpu_config)

        self.assertGreater(scores["neutral"], scores["disgust"])
        self.assertGreater(scores["neutral"], scores["happy"])

    def test_disgust_guardrail_prefers_happy_on_close_scores(self):
        scores = {
            "angry": 1.0,
            "disgust": 43.0,
            "fear": 2.0,
            "happy": 40.0,
            "sad": 3.0,
            "surprise": 1.0,
            "neutral": 10.0,
        }
        gpu_config = {
            "disgust_guardrails": {
                "enabled": True,
                "min_disgust_score": 45,
                "happy_rescue_ratio": 0.88,
                "second_choice_ratio": 0.92,
            }
        }

        adjusted = app._apply_gpu_disgust_guardrails(scores, gpu_config)
        dominant = max(adjusted, key=adjusted.get)

        self.assertEqual(dominant, "happy")

    @patch("app.get_deepface_module")
    def test_cpu_path_disgust_guardrail_applies(self, mock_get_df):
        app.CONFIG.setdefault("gpu_emotion", {})
        app.CONFIG["gpu_emotion"]["enabled"] = False
        app.CONFIG.setdefault("emotion", {})
        app.CONFIG["emotion"]["disgust_guardrails"] = {
            "enabled": True,
            "min_disgust_score": 43,
            "happy_rescue_ratio": 0.82,
            "second_choice_ratio": 0.90,
        }

        mock_df = MagicMock()
        mock_df.analyze.return_value = [
            {
                "emotion": {"disgust": 43.0, "happy": 40.0, "neutral": 17.0},
                "dominant_emotion": "disgust",
                "region": {"x": 10, "y": 10, "w": 32, "h": 32},
            }
        ]
        mock_get_df.return_value = mock_df

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        emotion, _confidence, _loc, detected, raw_scores = app.detect_emotion(frame)

        self.assertTrue(detected)
        self.assertEqual(emotion, "happy")
        self.assertGreater(raw_scores.get("happy", 0.0), raw_scores.get("disgust", 0.0))

    def test_display_confidence_calibration_boosts_clear_winner(self):
        scores = {
            "happy": 35.0,
            "neutral": 6.0,
            "sad": 4.0,
            "angry": 2.0,
            "surprise": 2.0,
            "fear": 1.0,
            "disgust": 1.0,
        }
        calibrated = app.compute_display_confidence(
            scores,
            dominant_emotion="happy",
            fallback_confidence=0.35
        )
        self.assertGreater(calibrated, 0.5)

    def test_display_confidence_calibration_fallback_on_empty_scores(self):
        calibrated = app.compute_display_confidence({}, dominant_emotion="happy", fallback_confidence=0.37)
        self.assertAlmostEqual(calibrated, 0.37, places=4)


if __name__ == "__main__":
    unittest.main()
