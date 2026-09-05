# Non-Speech Sound Event Understanding Model (FSD50K)

Smart Horizon 2026 — SH-DST-02 — Team ID: SHIH26-TID-320

## Overview

The Sound Event Perception module detects non-speech environmental sound events, producing temporal start/end timestamps, event confidences, and continuous acoustic embeddings for Core ALM multimodal reasoning.

### Architecture

- **`ontology.py`**: Defines target ontology (18 classes) and ID mappings.
- **`encoder.py`**: Continuous acoustic event feature encoder (`SoundEventEncoder`).
- **`classifier.py`**: Multi-label temporal classifier (`SoundEventClassifier`).
- **`event_model.py`**: Main integration wrapper implementing `SoundEventModel` required interface.
- **`inference.py`**: Standalone inference pipeline.
- **`train.py`**: CLI training script (`--dataset`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--output`, `--max-samples`).

## Target Ontology (18 Classes)

- `aircraft`, `helicopter`, `car`, `bus`, `train`, `vehicle`
- `car_horn`, `siren`, `alarm`, `dog`, `crowd`, `footsteps`
- `engine`, `machinery`, `speech`, `music`, `rain`, `thunder`

## Interface Contract for Core ALM

```python
class SoundEventModel(nn.Module):
    def detect(self, audio: torch.Tensor) -> dict:
        """
        Returns:
        {
          "events": [
            {
              "label": "aircraft",
              "start": 4.2,
              "end": 8.5,
              "confidence": 0.91
            }
          ]
        }
        """
        ...

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, T_frames, embed_dim)
        """
        ...
```

## Primary Dataset Budget & Sources

- **Primary Dataset**: FSD50K (`https://fsannotator.upf.edu/fsd/downloads/`).
- **Dataset Subset**: Target ontology subset under ~2 GB limit.
- **Manifest**: `datasets/fsd50k/manifest.json`.
