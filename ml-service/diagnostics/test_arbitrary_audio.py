import os
import sys
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import httpx

import tempfile

def generate_synthetic_audio_files():
    tmp_dir = os.path.join(tempfile.gettempdir(), "arbitrary_tests")
    os.makedirs(tmp_dir, exist_ok=True)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Sample A: 440 Hz Sine wave (Speech-like frequency tone)
    audio_a = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path_a = os.path.abspath(os.path.join(tmp_dir, "arbitrary_sample_01_speech.wav"))
    wavfile.write(path_a, sr, (audio_a * 32767).astype(np.int16))

    # 2. Sample B: Dual tone 880 Hz + 1200 Hz (Alert tone + speech)
    audio_b = (0.3 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    path_b = os.path.abspath(os.path.join(tmp_dir, "arbitrary_sample_02_alert.wav"))
    wavfile.write(path_b, sr, (audio_b * 32767).astype(np.int16))

    # 3. Sample C: Low frequency 150 Hz (Engine/ambient noise)
    audio_c = (0.5 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))).astype(np.float32)
    path_c = os.path.abspath(os.path.join(tmp_dir, "arbitrary_sample_03_ambient.wav"))
    wavfile.write(path_c, sr, (audio_c * 32767).astype(np.int16))

    return [path_a, path_b, path_c]

def run_arbitrary_audio_tests():
    print("=" * 80)
    print("      PHASE 20 — ARBITRARY AUDIO & FASTAPI INFERENCE TEST")
    print("=" * 80)

    audio_paths = generate_synthetic_audio_files()
    print(f"Generated {len(audio_paths)} completely new arbitrary audio test files.")

    api_url = "http://localhost:8000/analyze"

    # Test 1: Upload 3 arbitrary audio files via FastAPI POST /analyze
    print("\n--- TEST 1: ARBITRARY UPLOADS VIA FASTAPI POST /analyze ---")
    for path in audio_paths:
        fname = os.path.basename(path)
        with open(path, "rb") as f:
            files = {"audio_file": (fname, f, "audio/wav")}
            data = {"question": "What sound is present?", "session_id": "test_sess_arb_01"}
            try:
                response = httpx.post(api_url, files=files, data=data, timeout=10.0)
                if response.status_code == 200:
                    json_res = response.json()
                    print(f"  [SUCCESS] Upload File: {fname}")
                    print(f"            Answer: '{json_res['answer']}' (Conf: {json_res['confidence']})")
                    print(f"            Events: {json_res.get('audio_events')}")
                else:
                    print(f"  [FAIL] HTTP {response.status_code}: {response.text}")
            except Exception as e:
                print(f"  [ERROR] Endpoint call failed: {e}")

    # Test 2: Same Arbitrary Audio, 5 Sequential Questions
    print("\n--- TEST 2: SAME ARBITRARY AUDIO, 5 SEQUENTIAL QUESTIONS ---")
    target_path = audio_paths[0]
    multi_questions = [
        "What sound is present?",
        "How many speakers are present?",
        "What is the speaker's emotional tone?",
        "What can be inferred from the speech and background sounds together?",
        "Where is the speaker likely to be?"
    ]

    for idx, q in enumerate(multi_questions, 1):
        with open(target_path, "rb") as f:
            files = {"audio_file": (os.path.basename(target_path), f, "audio/wav")}
            data = {"question": q, "session_id": "test_sess_arb_multi"}
            try:
                response = httpx.post(api_url, files=files, data=data, timeout=10.0)
                if response.status_code == 200:
                    res_json = response.json()
                    print(f"  Q{idx} ('{q}') -> '{res_json['answer']}' (Conf: {res_json['confidence']})")
                else:
                    print(f"  Q{idx} [FAIL] HTTP {response.status_code}")
            except Exception as e:
                print(f"  Q{idx} [ERROR] {e}")

    print("\n[PHASE 20 ARBITRARY AUDIO INFERENCE TEST COMPLETE]")

if __name__ == "__main__":
    run_arbitrary_audio_tests()
