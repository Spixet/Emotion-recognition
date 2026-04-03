import unittest
import sys
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app
from app import detect_emotion

class TestEmotionDetection(unittest.TestCase):
    def setUp(self):
        self._original_gpu_config = dict(app.CONFIG.get('gpu_emotion', {}))
        app.CONFIG.setdefault('gpu_emotion', {})['enabled'] = False

    def tearDown(self):
        app.CONFIG['gpu_emotion'] = self._original_gpu_config
    
    @patch('app.get_deepface_module')
    def test_detect_emotion_no_face(self, mock_get_module):
        # Mock DeepFace module
        mock_df = MagicMock()
        mock_df.analyze.side_effect = ValueError("Face could not be detected")
        mock_get_module.return_value = mock_df

        # Create a dummy black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        emotion, confidence, loc, detected, raw_emotions = detect_emotion(frame)

        self.assertEqual(emotion, "unknown")
        self.assertFalse(detected)
        self.assertEqual(raw_emotions, {})

    @patch('app.get_deepface_module')
    def test_detect_emotion_success(self, mock_get_module):
        # Mock DeepFace module
        mock_df = MagicMock()
        mock_df.analyze.return_value = [{
            'emotion': {'happy': 90.0, 'sad': 10.0},
            'dominant_emotion': 'happy',
            'region': {'x': 10, 'y': 10, 'w': 50, 'h': 50}
        }]
        mock_get_module.return_value = mock_df

        # Create a dummy image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        emotion, confidence, loc, detected, raw_emotions = detect_emotion(frame)

        self.assertEqual(emotion, "happy")
        self.assertEqual(confidence, 0.9)
        self.assertTrue(detected)
        self.assertIsNotNone(loc)
        self.assertIn('happy', raw_emotions)

if __name__ == '__main__':
    unittest.main()
