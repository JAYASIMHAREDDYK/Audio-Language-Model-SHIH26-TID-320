# Demo Guide — Smart Horizon 2026 Audio Language Model

**Project:** SH-DST-02  
**Team ID:** SHIH26-TID-320

---

## Quick Start

### 1. Start the ML Service

```bash
cd ml-service
pip install -r requirements.txt
python main.py
```

The service starts on `http://localhost:8000`.

### 2. Start the Backend

```bash
cd backend
npm install
node src/app.js
```

Backend starts on `http://localhost:4000`.

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend starts on `http://localhost:5173`.

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict (Conformer Features + Experimental Transcription)

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio_file=@demo/hindi.wav" \
  -F "language=hi"
```

### Full ALM Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@demo/hindi.wav" \
  -F "question=What is happening in this audio?" \
  -F "language_hint=hi"
```

---

## Demo Walkthrough (3-5 minutes)

### Step 1 — Architecture Overview

Show the pipeline diagram:

```
Audio Input → 16kHz Preprocessing → 80-bin Log-Mel → 4× Subsampling
    → 12 Conformer Encoder Blocks → (B, T', 512)
    → ASR Adapter (512 → 256) → (B, T', 256)
    → ALM Temporal Fusion → Final Analysis
```

Parallel CTC path from Conformer for transcription (experimental).

### Step 2 — Upload Audio

Upload a Hindi WAV file through the frontend or via curl:

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio_file=@demo/hindi.wav" \
  -F "language=hi"
```

### Step 3 — Show Conformer Processing

The response shows exact tensor shapes:

```json
{
  "conformer_features_shape": [1, 26, 512],
  "alm_features_shape": [1, 26, 256],
  "model": "12-layer Deep Multilingual Conformer (80.24M params)"
}
```

### Step 4 — Show ALM Output

Run full analysis:

```json
{
  "answer": "The audio contains Hindi speech...",
  "confidence": 0.85,
  "speech": { "transcript": "...", "language": "hi" },
  "audio_events": [{"label": "speech"}, {"label": "ambient_room_noise"}],
  "scene": { "environment": "Indoor Acoustic Environment" }
}
```

### Step 5 — Show Multilingual Capability

Run with audio from different languages:

```bash
# Telugu
curl -X POST http://localhost:8000/predict -F "audio_file=@demo/telugu.wav" -F "language=te"

# Mandarin
curl -X POST http://localhost:8000/predict -F "audio_file=@demo/mandarin.wav" -F "language=zh"
```

### Step 6 — Show Performance

Run the benchmark:

```bash
python benchmark_final.py
```

Expected output:
- 2s audio: ~19ms latency, 105× real-time
- 5s audio: ~35ms latency, 143× real-time
- 10s audio: ~60ms latency, 167× real-time

### Step 7 — Show Test Results

```bash
python -m pytest tests/ -v
```

All 26+ tests passing.

---

## Key Technical Claims (Verified)

- ✅ Custom 12-layer Deep Multilingual Conformer (80.24M parameters)
- ✅ 7-language support (hi, te, ta, bn, mr, kn, zh)
- ✅ 2,122-token native Unicode vocabulary
- ✅ 4.198 GB curated corpus (19.565 hours, speaker-disjoint)
- ✅ CTC training pipeline with proven overfit capability
- ✅ ALM temporal feature integration: (B,T',512) → (B,T',256)
- ✅ 26+ automated tests passing
- ✅ Real-time inference (105-167× real-time on RTX 4060)

## Honest Limitations

- ⚠ Full-data CTC training did not converge within hackathon budget
- ⚠ Transcription output is experimental (not production-grade)
- ⚠ Model architecture and pipeline are verified; accuracy optimization is ongoing
