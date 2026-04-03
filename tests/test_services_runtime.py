import json
import os
import sys
import tempfile
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.evaluation import build_calibration_artifact  # noqa: E402
from services.face_tracker import SessionFaceTracker  # noqa: E402
from services.runtime_calibration import RuntimeCalibration  # noqa: E402


class TestServicesRuntime(unittest.TestCase):
    def test_build_calibration_artifact_has_expected_structure(self):
        records = []
        for _ in range(40):
            records.append(
                {
                    "true_label": "happy",
                    "predicted_label": "happy",
                    "predicted_confidence": 0.75,
                    "scores": {"happy": 75.0, "neutral": 25.0},
                }
            )
        for _ in range(20):
            records.append(
                {
                    "true_label": "happy",
                    "predicted_label": "disgust",
                    "predicted_confidence": 0.62,
                    "scores": {"happy": 38.0, "disgust": 42.0, "neutral": 20.0},
                }
            )

        artifact = build_calibration_artifact(records)
        self.assertIn("metrics", artifact)
        self.assertIn("confidence", artifact)
        self.assertIn("class_bias", artifact)
        self.assertGreaterEqual(artifact.get("dataset_samples", 0), 60)

    def test_class_bias_multiplier_reduces_overpredicted_disgust(self):
        records = []
        for _ in range(80):
            records.append(
                {
                    "true_label": "happy",
                    "predicted_label": "disgust",
                    "predicted_confidence": 0.6,
                    "scores": {"happy": 35.0, "disgust": 45.0, "neutral": 20.0},
                }
            )
        for _ in range(20):
            records.append(
                {
                    "true_label": "disgust",
                    "predicted_label": "disgust",
                    "predicted_confidence": 0.7,
                    "scores": {"disgust": 70.0, "neutral": 30.0},
                }
            )

        artifact = build_calibration_artifact(records)
        disgust_multiplier = artifact["class_bias"]["multipliers"]["disgust"]
        self.assertLess(disgust_multiplier, 1.0)

    def test_runtime_calibration_applies_confidence_and_bias(self):
        artifact = {
            "confidence": {"enabled": True, "slope": 1.4, "intercept": 0.1},
            "class_bias": {
                "enabled": True,
                "multipliers": {"happy": 1.2, "disgust": 0.8, "neutral": 1.0},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = os.path.join(tmp, "calibration.json")
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f)

            cfg = {"emotion": {"data_calibration": {"enabled": True, "artifact_path": artifact_path}}}
            calibration = RuntimeCalibration(cfg)

            adjusted = calibration.apply_class_bias({"happy": 40.0, "disgust": 50.0, "neutral": 10.0})
            self.assertLess(adjusted["disgust"], 50.0)
            self.assertGreater(adjusted["happy"], 40.0)
            self.assertAlmostEqual(sum(adjusted.values()), 100.0, places=4)

            calibrated_conf = calibration.calibrate_confidence(0.6)
            self.assertNotAlmostEqual(calibrated_conf, 0.6, places=4)

    def test_face_tracker_smooths_and_holds_short_gaps(self):
        tracker = SessionFaceTracker(
            enabled=True,
            smoothing_alpha=0.5,
            max_missing_frames=2,
            reacquire_iou_threshold=0.0,
            max_center_jump_ratio=1.0,
        )

        first, tracked = tracker.update("sid", {"x": 10, "y": 10, "w": 100, "h": 100}, 640, 480)
        self.assertFalse(tracked)
        self.assertEqual(first["x"], 10)

        second, tracked = tracker.update("sid", {"x": 30, "y": 10, "w": 100, "h": 100}, 640, 480)
        self.assertTrue(tracked)
        self.assertGreater(second["x"], 10)
        self.assertLess(second["x"], 30)

        gap1, tracked = tracker.update("sid", None, 640, 480)
        self.assertTrue(tracked)
        self.assertIsNotNone(gap1)

        gap2, tracked = tracker.update("sid", None, 640, 480)
        self.assertTrue(tracked)
        self.assertIsNotNone(gap2)

        gap3, tracked = tracker.update("sid", None, 640, 480)
        self.assertFalse(tracked)
        self.assertIsNone(gap3)


if __name__ == "__main__":
    unittest.main()

