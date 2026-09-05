"""
Speech Representation Encoder Module.
Wraps the 12-layer DeepConformerEncoder architecture for acoustic feature extraction.
Zero Whisper and zero external transformers dependencies.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.asr.config import ASRConfig
from src.asr.conformer import DeepConformerEncoder


class SpeechEncoder(nn.Module):
    """
    12-Layer Deep Conformer Speech Representation Encoder.
    Processes 80-bin Log-Mel Spectrogram features via 4x Conv2D subsampling
    and 12 Macaron Conformer blocks.
    
    Returns:
        frame_embeddings: (B, T_frames, 512)
        pooled_embedding: (B, 512)
    """

    def __init__(
        self,
        pretrained_name_or_path: Optional[str] = None,
        hidden_dim: int = 512,
        config: Optional[ASRConfig] = None
    ):
        super().__init__()
        if config is None:
            config = ASRConfig(d_model=hidden_dim)
        self.config = config
        self.hidden_dim = config.d_model
        self.conformer = DeepConformerEncoder(config)

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frame-level and pooled speech representations.
        Returns:
            frame_embeddings: (B, T_frames, 512)
            pooled_embedding: (B, 512)
        """
        frame_embeddings, lengths, pooled_embedding = self.conformer(audio, return_pooled=True)
        return frame_embeddings, pooled_embedding

    def count_parameters(self) -> dict:
        return self.conformer.count_parameters()


__all__ = ["SpeechEncoder"]
