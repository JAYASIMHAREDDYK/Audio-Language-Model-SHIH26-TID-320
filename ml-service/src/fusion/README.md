# Learned Multimodal & Temporal Fusion Module

Smart Horizon 2026 — SH-DST-02 — Team ID: SHIH26-TID-320

## Overview

The `fusion` module performs continuous temporal alignment, multimodal feature projection, and spatio-temporal cross-attention fusion over representations from specialized submodels.

## Architecture & Components

1. **`temporal_alignment.py` (`TemporalAligner`)**:
   - Resamples multi-modal frame sequences (speech, events, speakers, paralinguistics, spectrogram) to a uniform grid (`target_len=64`).
   - Injects continuous sinusoidal timestamp positional encodings.

2. **`multimodal_projection.py` (`MultimodalProjector`)**:
   - Projection heads mapping 5 modality streams into a shared embedding dimension (`fusion_dim=256`).

3. **`temporal_fusion.py` (`TemporalFusionAdapter`)**:
   - Multi-head self-attention & cross-attention Transformer Encoder over aligned modal feature streams.

## Usage Interface

```python
from src.fusion import TemporalAligner, MultimodalProjector, TemporalFusionAdapter

aligner = TemporalAligner(target_len=64, embed_dim=256)
projector = MultimodalProjector(fusion_dim=256)
fusion = TemporalFusionAdapter(embed_dim=256)

aligned = aligner(speech_emb, event_emb, speaker_emb, para_emb, audio_emb)
projected = projector(aligned["speech"], aligned["events"], aligned["speakers"], aligned["paralinguistic"], aligned["audio"])
fused_representation = fusion(projected) # (B, T_aligned, 256)
```
