# Speaker Model & Diarization Module (AMI Meeting Corpus)

Smart Horizon 2026 — SH-DST-02 — Team ID: SHIH26-TID-320

## Overview

The Speaker perception module provides speaker biometrics encoding and turn diarization for multi-speaker meeting audio streams.

### Architecture

- **`encoder.py`**: Continuous speaker biometric feature encoder (`SpeakerEncoder`).
- **`diarization.py`**: Speaker turn clustering & segmentation (`SpeakerDiarizer`).
- **`speaker_model.py`**: Main integration wrapper implementing `SpeakerModel` required interface.
- **`inference.py`**: Standalone inference pipeline.
- **`train.py`**: CLI training script (`--dataset`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--output`, `--max-samples`).

## Interface Contract for Core ALM

```python
class SpeakerModel(nn.Module):
    def diarize(self, audio: torch.Tensor) -> dict:
        """
        Returns:
        {
          "speakers": [
            { "id": "S1", "start": 0.0, "end": 4.2 },
            { "id": "S2", "start": 4.3, "end": 7.8 }
          ],
          "speaker_count": 2
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

- **Source**: AMI Meeting Corpus (`https://groups.inf.ed.ac.uk/ami/corpus/`).
- **Dataset Subset**: Audio headset mix, small subset under 1 GB limit.
- **Manifest**: `datasets/ami/manifest.json`.
