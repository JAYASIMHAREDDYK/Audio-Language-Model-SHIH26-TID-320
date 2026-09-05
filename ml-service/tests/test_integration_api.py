import os
import sys
import unittest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_SERVICE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(ML_SERVICE_DIR, ".."))

if ML_SERVICE_DIR not in sys.path:
    sys.path.insert(0, ML_SERVICE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import main

class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertTrue(data["model_loaded"], "Core ALM model should be loaded during API lifespan startup.")

    def test_analyze_endpoint_real_audio(self):
        real_wav_path = os.path.join(ML_SERVICE_DIR, "datasets", "alm_nhce", "wav", "alm_nhce_000001.wav")
        if not os.path.exists(real_wav_path):
            # Fallback to generating test WAV file
            import numpy as np, scipy.io.wavfile as wavfile, tempfile
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            wav_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wavfile.write(tmp.name, sr, (wav_data * 32767).astype(np.int16))
            real_wav_path = tmp.name

        with open(real_wav_path, "rb") as f:
            response = self.client.post(
                "/analyze",
                files={"audio_file": ("alm_nhce_000001.wav", f, "audio/wav")},
                data={
                    "question": "What sound event is present?",
                    "language_hint": "hi",
                    "session_id": "test_session_01"
                }
            )

        self.assertEqual(response.status_code, 200, f"API returned error: {response.text}")
        data = response.json()

        # Validate response schema keys
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("speech", data)
        self.assertIn("speakers", data)
        self.assertIn("audio_events", data)
        self.assertIn("paralinguistic", data)
        self.assertIn("scene", data)
        self.assertIn("evidence", data)

        # Validate types and non-empty values
        self.assertIsInstance(data["answer"], str)
        self.assertTrue(len(data["answer"]) > 0)
        self.assertIsInstance(data["confidence"], (int, float))
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertIsInstance(data["evidence"], list)

if __name__ == "__main__":
    unittest.main()
