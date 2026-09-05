# Paralinguistic & Emotion Model (MELD Corpus)

Smart Horizon 2026 — SH-DST-02 — Team ID: SHIH26-TID-320

## Overview

The Paralinguistic perception module extracts vocal emotion state, vocal arousal, speaking style, and urgency cues directly from speech acoustics.

### Architecture

- **`encoder.py`**: Acoustic prosody & vocal biometric feature encoder (`ParalinguisticEncoder`).
- **`emotion.py`**: Classifier (`EmotionClassifier`) supporting MELD 7 emotion categories (`neutral`, `happy`, `sad`, `angry`, `fearful`, `surprised`, `disgusted`).
- **`paralinguistic_model.py`**: Main integration wrapper implementing `ParalinguisticModel` required interface.
- **`inference.py`**: Standalone inference pipeline.
- **`train.py`**: CLI training script (`--dataset`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--output`, `--max-samples`).

## Interface Contract for Core ALM

```python
class ParalinguisticModel(nn.Module):
    def analyze(self, audio: torch.Tensor) -> dict:
        """
        Returns:
        {
          "emotion": "neutral",
          "confidence": 0.82,
          "arousal": "low",
          "speaking_style": "conversational",
          "urgency": "low"
        }
        """
        ...

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, T_frames, embed_dim)
        """
        ...
```

## Dataset Budget & Sources

- **Source**: MELD Audio Dataset (`https://affective-meld.github.io/`).
- **Dataset Subset**: Audio portion under 500 MB limit.
- **Manifest**: `datasets/meld/manifest.json`.
- **Emotions**: strictly official MELD classes (`neutral`, `happy`, `sad`, `angry`, `fearful`, `surprised`, `disgusted`).
