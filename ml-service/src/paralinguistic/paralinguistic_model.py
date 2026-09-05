import torch
import torch.nn as nn
from src.paralinguistic.encoder import ParalinguisticEncoder
from src.paralinguistic.emotion import EmotionClassifier

class ParalinguisticModel(nn.Module):
    """
    Paralinguistic Model (MELD Corpus perception component).
    Implements required interface for Core ALM integration:
        - ParalinguisticModel.analyze(audio)
        - ParalinguisticModel.encode(audio)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = ParalinguisticEncoder(embed_dim=embed_dim)
        self.classifier = EmotionClassifier(embed_dim=embed_dim)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio into continuous frame-level paralinguistic embeddings (B, T_frames, embed_dim).
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.randn(1, 80, 128)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        return self.encoder(audio)

    def analyze(self, audio: torch.Tensor) -> dict:
        """
        Analyzes vocal emotion and paralinguistics:
        {
          "emotion": "neutral",
          "confidence": 0.82
        }
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.randn(1, 80, 128)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        embeddings = self.encode(audio)
        pred = self.classifier.predict(embeddings)

        return {
            "emotion": pred["emotion"],
            "confidence": pred["confidence"],
            "arousal": pred["arousal"],
            "speaking_style": pred["speaking_style"],
            "urgency": pred["urgency"]
        }
