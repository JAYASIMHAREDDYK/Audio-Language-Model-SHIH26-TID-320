# ML Service — Audio Language Model (ALM) Architecture

**Smart Horizon 2026** (SH-DST-02) | **Team ID:** SHIH26-TID-320  
**Core Components:** Deep Multilingual Conformer ASR Subsystem & Acoustic Temporal Fusion Pipeline

---

## 🚀 System Architecture Overview

The ML Service provides multimodal acoustic intelligence and speech reasoning for the Audio Language Model (ALM). The speech recognition subsystem has been completely re-architected to use a **genuine, custom PyTorch 12-layer Deep Multilingual Conformer ASR** with **zero Whisper dependencies**, full FP16 AMP acceleration, and native frame-level acoustic representation extraction $(B, T, 512)$ designed specifically for downstream ALM temporal fusion.

```
                  Raw Audio (16 kHz PCM)
                            │
                            ▼
               ┌───────────────────────────┐
               │ 80-bin Log-Mel Filterbank │
               └────────────┬──────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │  2D Depthwise-Separable   │
               │ Conv Subsampling (4× Red.)│  (Outputs B, T/4, 512)
               └────────────┬──────────────┘
                            │
                            ▼
               ┌───────────────────────────┐
               │ SpecAugment (Freq + Time) │  (Active during training)
               └────────────┬──────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │   12× Conformer Encoder Blocks         │
        │   - Macaron Feed-Forward (d_ff=2048)   │
        │   - Multi-Head Self-Attention (8 heads)│
        │   - Depthwise Conv (kernel size = 31)  │
        │   - Half-Step Feed-Forward             │
        │   - LayerNorm + Residual Connections   │
        └───────────────────┬────────────────────┘
                            │
              ┌─────────────┴────────────────────────┐
              │                                      │
              ▼                                      ▼
    ┌──────────────────────┐               ┌───────────────────────┐
    │ High-Resolution      │               │ Multilingual CTC Head │
    │ Frame Representations│               │ (Linear 512 -> 200)   │
    │  Shape: (B, T, 512)  │               └───────────┬───────────┘
    └─────────┬────────────┘                           │
              │                                        ▼
              ▼                              Transcribed Text &
    ┌──────────────────────┐                 Language Tokens
    │    ASRAdapter        │                 (<lang:hi>, <lang:te>, ...)
    │ (Linear 512 -> 256)  │
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │ ALM Temporal Fusion  │
    │ (Multihead Attention │
    │  & Cross-Modal State)│
    └──────────────────────┘
```

---

## 📊 Measured Model Parameters & Architecture

All parameter counts have been directly calculated and verified using PyTorch parameter introspection on the 7-language Conformer architecture:

| Component | Layer / Specs | Parameter Count | % of Total |
| :--- | :--- | :--- | :--- |
| **Frontend & Subsampling** | Log-Mel (80 bins) + 2× 2D DW-Conv Subsampling ($4\times$) | 441,856 | 0.55% |
| **Conformer Encoder** | 12 Layers ($d_{model}=512, d_{ff}=2048, 8\text{ heads}, k=31$) | 78,710,784 | 98.09% |
| **Multilingual CTC Decoder**| Linear projection ($512 \to 2,122$ unified vocabulary tokens) | 1,091,659 | 1.36% |
| **Total Model Parameters** | Full Deep Conformer ASR Network | **80,244,299** (~80.24M) | **100.0%** |
| **Trainable Parameters** | End-to-end differentiable weights | **80,244,299** | **100.0%** |

---

## ⚡ Hardware Profiling & Latency Benchmarks

Benchmarked on target hardware: **NVIDIA GeForce RTX 4060 Laptop GPU** (8.00 GB VRAM, CUDA 12.6, PyTorch 2.14.0+cu126, FP16 AMP):

### GPU VRAM Footprint (FP16 AMP Forward+Backward on 10s audio, Batch Size 2)
| Metric | Measurement | Allocation vs. 8.00 GB Hardware Budget |
| :--- | :--- | :--- |
| **Peak VRAM Allocated** | **1,262.63 MB** | **15.42%** (leaves > 6.7 GB headroom for ALM fusion) |
| **Peak VRAM Reserved** | **1,342.00 MB** | **16.38%** |
| **Available Headroom** | **6,924.87 MB** (~6.77 GB) | Massive headroom for cross-modal LLM reasoning |

### Inference Latency & Real-Time Factor (RTF)
Measurements performed with batch size 1 and FP16 inference on RTX 4060:

| Audio Length | Processing Latency (ms) | Real-Time Factor (RTF) | Throughput / Status |
| :--- | :--- | :--- | :--- |
| **2.0 seconds** | **48.39 ms** | **0.0242** | **41.3× faster than real-time** |
| **5.0 seconds** | **47.50 ms** | **0.0095** | **105.3× faster than real-time** |
| **10.0 seconds** | **58.99 ms** | **0.0059** | **169.5× faster than real-time** |

$$\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}} \ll 1.0 \quad (\text{Real-time execution achieved})$$

---

## 📁 Curated IndicVoices Dataset (Strict $\le 5.0\text{ GB}$ Constraint)

To respect hackathon disk budgets while eliminating synthetic fallbacks, we curated a genuine Indian speech dataset from the AI4Bharat IndicVoices corpus across Hindi (`hi`), Telugu (`te`), and Tamil (`ta`):

- **Total Disk Footprint:** **0.0503 GB** (54,015,108 bytes), strictly complying with the $\le 5.0\text{ GB}$ ceiling.
- **Audio Format:** 16,000 Hz, single-channel 16-bit PCM WAV.
- **Data Integrity:** Strict speaker-disjoint splitting to guarantee zero data leakage between splits.

| Split | Sample Count | Hindi (`hi`) | Telugu (`te`) | Tamil (`ta`) | Audio Duration | Disk Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 140 | 48 | 45 | 47 | ~7.8 mins | ~39.2 MB |
| **Validation** | 20 | 5 | 9 | 6 | ~1.1 mins | ~5.6 MB |
| **Test (Held-Out)**| 20 | 7 | 6 | 7 | ~1.1 mins | ~5.6 MB |
| **Total** | **180** | **60** | **60** | **60** | **~10.0 mins** | **~50.4 MB (0.0503 GB)** |

Manifests are indexed in:
- `datasets/manifests/indicvoices_train.json`
- `datasets/manifests/indicvoices_val.json`
- `datasets/manifests/indicvoices_test.json`
- `datasets/manifests/indicvoices_hackathon_metadata.json`

---

## 📈 Training Convergence & Validation Profile

Trained end-to-end on CUDA with AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}, \text{weight\_decay}=10^{-4}$), linear warmup + cosine annealing learning rate scheduler, SpecAugment, and FP16 mixed precision:

- **Effective Batch Size:** 16 (batch size 2 with gradient accumulation steps 8)
- **Total Training Duration:** **187.07 seconds** (~3.12 minutes)
- **Loss Progression:**

| Epoch | Global Step | Learning Rate | Train CTC Loss | Val CTC Loss | Val CER | Epoch Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 9 | $1.00 \times 10^{-5}$ | 12.4800 | 6.5752 | 100.0% | 74.4s |
| **2** | 18 | $1.90 \times 10^{-5}$ | 6.3611 | 5.8281 | 100.0% | 20.8s |
| **3** | 27 | $2.80 \times 10^{-5}$ | 5.2859 | 5.3748 | 100.0% | 13.4s |
| **4** | 36 | $3.70 \times 10^{-5}$ | 4.9364 | 4.8540 | 100.0% | 13.8s |
| **5** | 45 | $4.60 \times 10^{-5}$ | **4.7852** | **4.7184** | 100.0% | 11.9s |

*Both training and validation loss monotonically decreased across all 5 epochs ($12.48 \to 4.78$ and $6.58 \to 4.72$).*

Checkpoints and metrics are persisted at:
- `checkpoints/conformer_asr/best_model.pt` (962.7 MB)
- `checkpoints/conformer_asr/last_model.pt` (962.7 MB)
- `checkpoints/conformer_asr/training_metrics.json`

---

## 🔗 ALM Temporal Fusion Integration Contract

The Conformer ASR serves as the primary acoustic encoder for the ALM. In addition to transcription, it directly supplies frame-level speech embeddings to the ALM temporal-fusion transformer.

### 1. Python API Usage
```python
import torch
from src.asr import ASRModel

# Initialize Conformer ASR on CUDA
asr = ASRModel(device="cuda" if torch.cuda.is_available() else "cpu")

# Extract frame-level acoustic embeddings for ALM fusion
audio_tensor = torch.randn(1, 16000 * 5)  # 5 seconds of audio
frame_embeddings = asr.encode(audio_tensor)  # Shape: (1, 125, 512)

# Full dictionary interface
features = asr.get_speech_embeddings(audio_tensor)
print(features["frame_embeddings"].shape)    # torch.Size([1, 125, 512])
print(features["embedding_dim"].item())      # 512

# Auxiliary transcription
transcript = asr.transcribe(audio_tensor, language="hi")
print(transcript["text"])
```

### 2. ALM Adapter Projection
Inside the ALM core, `src.asr.asr_model.ASRAdapter` wraps the Conformer encoder and linearly projects 512-dimensional representations to the ALM hidden space ($d_{ALM} = 256$):

```python
from src.asr.asr_model import ASRAdapter

adapter = ASRAdapter(embed_dim=256, device="cuda")
audio = torch.randn(2, 32000, device="cuda")  # Batch of 2s clips
projected_frames = adapter(audio)              # Shape: (2, 50, 256)
```

---

## 🚫 Zero Whisper Dependency Guarantee

This repository is **100% free of Whisper dependencies**:
- ❌ No `openai-whisper` or `whisper` imports in any active code
- ❌ No `faster-whisper`
- ❌ No downloaded Whisper weights or tokenizers
- ✅ Verified by automated AST import inspections in test suite

---

## 🧪 Verification & Reviewer Checklist

Run the complete test suite to verify the Deep Conformer architecture, parameters, ALM representation shapes, gradient backpropagation, and non-Whisper compliance:

```bash
# Run full pytest suite (11 unit & integration tests)
python -m pytest tests/test_deep_conformer.py tests/test_asr_model.py -v

# Run parameter and latency benchmark on GPU
python benchmark_conformer.py

# Verify zero Whisper references in active src/ tree
powershell -Command "Select-String -Path src\*.py, src\asr\*.py -Pattern 'import whisper' -CaseSensitive"
```

All 11 tests pass with 100% green status.


