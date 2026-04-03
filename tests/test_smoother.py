import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emotion_smoother import EmotionSmoother

class TestEmotionSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = EmotionSmoother(
            alpha=0.35,
            confidence_threshold=0.5,
            switch_in_frames=3,
            switch_out_frames=4
        )

    def test_update_and_ema_scores(self):
        updated = self.smoother.update({'happy': 100.0, 'sad': 0.0}, face_detected=True, detection_confidence=0.9)
        self.assertTrue(updated)
        avg = self.smoother.get_averaged_scores()
        self.assertAlmostEqual(avg['happy'], 100.0)
        self.assertAlmostEqual(avg['sad'], 0.0)

        # EMA: 0.35*50 + 0.65*100 = 82.5
        updated = self.smoother.update({'happy': 50.0, 'sad': 50.0}, face_detected=True, detection_confidence=0.9)
        self.assertTrue(updated)
        avg = self.smoother.get_averaged_scores()
        self.assertAlmostEqual(avg['happy'], 82.5, places=2)
        self.assertAlmostEqual(avg['sad'], 17.5, places=2)

    def test_confidence_gating_ignores_low_confidence_frame(self):
        self.smoother.update({'happy': 90.0, 'sad': 10.0}, face_detected=True, detection_confidence=0.9)
        baseline = self.smoother.get_averaged_scores()
        updated = self.smoother.update({'happy': 0.0, 'sad': 100.0}, face_detected=True, detection_confidence=0.2)
        self.assertFalse(updated)
        self.assertEqual(baseline, self.smoother.get_averaged_scores())

    def test_hysteresis_prevents_rapid_flips(self):
        # Need 3 consecutive frames to switch from unknown -> happy
        for _ in range(2):
            self.smoother.update({'happy': 80.0, 'neutral': 20.0}, face_detected=True, detection_confidence=0.9)
        dom, _ = self.smoother.get_dominant_emotion(self.smoother.get_averaged_scores())
        self.assertEqual(dom, 'happy')  # score-dominant can be happy
        self.assertEqual(self.smoother.stable_emotion, 'unknown')  # but stable state not switched yet

        self.smoother.update({'happy': 80.0, 'neutral': 20.0}, face_detected=True, detection_confidence=0.9)
        self.assertEqual(self.smoother.stable_emotion, 'happy')

        # Need 4 consecutive frames to switch away from stable happy -> sad
        for _ in range(3):
            self.smoother.update({'sad': 90.0, 'happy': 10.0}, face_detected=True, detection_confidence=0.9)
        self.assertEqual(self.smoother.stable_emotion, 'happy')
        self.smoother.update({'sad': 90.0, 'happy': 10.0}, face_detected=True, detection_confidence=0.9)
        self.assertEqual(self.smoother.stable_emotion, 'sad')

    def test_stale_reset_clears_stable_emotion(self):
        smoother = EmotionSmoother(
            alpha=0.35,
            confidence_threshold=0.8,
            switch_in_frames=3,
            switch_out_frames=4,
            stale_reset_frames=3
        )
        for _ in range(3):
            smoother.update({'happy': 80.0, 'neutral': 20.0}, face_detected=True, detection_confidence=0.95)
        self.assertEqual(smoother.stable_emotion, 'happy')

        for _ in range(3):
            smoother.update({'happy': 20.0, 'sad': 80.0}, face_detected=True, detection_confidence=0.2)
        self.assertEqual(smoother.stable_emotion, 'unknown')
        self.assertEqual(smoother.get_averaged_scores(), {})

    def test_min_dominance_margin_blocks_stable_switch_on_near_ties(self):
        smoother = EmotionSmoother(
            alpha=0.35,
            confidence_threshold=0.0,
            switch_in_frames=2,
            switch_out_frames=2,
            min_dominance_margin=5.0,
        )

        smoother.update({'disgust': 43.0, 'happy': 41.0, 'neutral': 16.0}, face_detected=True, detection_confidence=0.9)
        smoother.update({'happy': 42.0, 'disgust': 41.0, 'neutral': 17.0}, face_detected=True, detection_confidence=0.9)

        self.assertEqual(smoother.stable_emotion, 'unknown')

if __name__ == '__main__':
    unittest.main()
