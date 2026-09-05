# Experimental Results — Deep Multilingual Conformer ASR

**Project:** SH-DST-02 — Audio Language Model  
**Team ID:** SHIH26-TID-320  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB VRAM)

---

## Verified Components

| Component | Status | Details |
|-----------|--------|---------|
| Tokenizer Round-Trip | ✅ PASS | 100.00% across all 7 languages, 0.0% UNK |
| CTC Pipeline Tests | ✅ PASS | 26/26 tests passing |
| Gradient Health | ✅ PASS | All 12 layers finite, non-zero, norms ∈ [8.7, 52.1] |
| CTC Length Validity | ✅ PASS | 100.0% valid length ratio |
| Data Alignment | ✅ PASS | 140/140 samples verified |
| ALM Adapter | ✅ PASS | (B,T',512) → (B,T',256) verified |

---

## Tiny-Data Overfit Verification

The model successfully overfit a small 8-sample multilingual subset, proving that the architecture and CTC pipeline are functionally correct.

| Metric | Initial | Final |
|--------|---------|-------|
| CTC Loss | 17.5028 | 0.1985 |
| CER | 100.00% | 7.46% |
| WER | 100.00% | 21.18% |
| Steps | — | 160 |
| Time | — | 72.5s |

### Per-Language Overfit Results

| Language | CER | WER |
|----------|-----|-----|
| Kannada (kn) | 0.00% | 0.00% |
| Bengali (bn) | 0.00% | 0.00% |
| Marathi (mr) | 2.15% | 11.76% |
| Hindi (hi) | 5.19% | 14.29% |
| Mandarin Chinese (zh) | 5.56% | 5.56% |

---

## Full-Data Training Results — EXPERIMENTAL LIMITATION

> **The full-data CTC training runs did not reach acceptable generalization within the available training budget.** The model successfully overfit a tiny multilingual subset, demonstrating that the architecture and CTC implementation are functional, but full-corpus ASR accuracy remains an ongoing optimization task.

### Training Run Summary

| Run | LR | Epochs | Train Loss | Val Loss | CER | Blank % | Status |
|-----|-----|--------|-----------|----------|-----|---------|--------|
| Exp B (5 ep) | 1e-4 | 5 | 5.81 → 4.63 | 4.97 → 4.58 | 100% | 92-100% | Blank collapse |
| Exp B (resumed) | 3e-4 | 6→12 | 4.56 → 4.15 | 4.54 → 4.22 | 100% | 89-92% | Stale scheduler |
| Exp B (fresh) | 5e-4 | 13 | 5.61 → 4.83 | 5.08 → 5.09 | 99% | 95-100% | Repetitive predictions |

### Root Cause Analysis

1. **CTC Blank Collapse Valley**: CTC loss rapidly learns to emit ~95% blank tokens, dropping loss to ~4.5. The model must push loss below ~3.0 for meaningful character emission — this requires sustained high learning rate and many more gradient steps than available.

2. **Cosine LR Decay**: The cosine learning rate scheduler decayed the LR to its floor before the model could escape the blank valley, starving gradients.

3. **Dataset Scale vs. Model Capacity**: An 80M-parameter model on ~1,167 training samples (3.6 hours) with 2,122-token multilingual vocabulary requires hundreds of epochs to converge — well beyond the hackathon time budget.

### Key Diagnostic Finding: SpecAugment Silence Floor Bug

Discovered and fixed a critical bug where SpecAugment mask values were set to `0.0` in log-Mel space. Since `log(mel + 1e-5)` maps silence to `-11.5129`, a mask value of `0.0` represents maximum energy (`10^0 = 1.0`), which corrupted BatchNorm statistics. Fixed to use `-11.5129` (true silence).

---

## Dataset

| Property | Value |
|----------|-------|
| Total Size | 4.198 GB |
| Duration | 19.565 hours |
| Utterances | 6,132 |
| Train | 4,922 |
| Validation | 601 |
| Test | 609 |
| Sample Rate | 16 kHz mono |
| Split Protocol | Speaker-disjoint |

### Language Distribution

| Language | Code | Script |
|----------|------|--------|
| Hindi | hi | Devanagari |
| Telugu | te | Telugu |
| Tamil | ta | Tamil |
| Bengali | bn | Bengali |
| Marathi | mr | Devanagari |
| Kannada | kn | Kannada |
| Mandarin Chinese | zh | Simplified Chinese |

---

## Model Architecture

```
12-layer Deep Multilingual Conformer
├── Frontend: 80-bin Log-Mel + 4× Conv2D Subsampling (6.4M params)
├── Encoder: 12 Conformer Blocks (72.7M params)
│   ├── d_model = 512
│   ├── d_ff = 2048
│   ├── 8 attention heads
│   ├── Convolution kernel = 31
│   └── Macaron FFN structure (½ FFN + MHSA + Conv + ½ FFN)
├── CTC Head: 2,122 vocabulary (1.1M params)
└── Total: 80,244,299 parameters
```

---

## Performance

Measured on NVIDIA GeForce RTX 4060 Laptop GPU:

| Duration | Latency | RTF | Speed |
|----------|---------|-----|-------|
| 2s | ~19ms | 0.0095 | 105× real-time |
| 5s | ~35ms | 0.0070 | 143× real-time |
| 10s | ~60ms | 0.0060 | 167× real-time |
