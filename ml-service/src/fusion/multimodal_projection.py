import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    """
    Multimodal Projection Layer.
    Projects speech, event, speaker, paralinguistic, and audio representations into a unified hidden space.
    """
    def __init__(self, in_dims: dict = None, fusion_dim: int = 256):
        super().__init__()
        if in_dims is None:
            in_dims = {
                "speech": 256,
                "events": 256,
                "speakers": 256,
                "paralinguistic": 256,
                "audio": 256
            }
        
        self.fusion_dim = fusion_dim
        self.speech_proj = nn.Sequential(
            nn.Linear(in_dims.get("speech", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.event_proj = nn.Sequential(
            nn.Linear(in_dims.get("events", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.speaker_proj = nn.Sequential(
            nn.Linear(in_dims.get("speakers", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.para_proj = nn.Sequential(
            nn.Linear(in_dims.get("paralinguistic", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(in_dims.get("audio", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

    def forward(self, speech_emb: torch.Tensor, event_emb: torch.Tensor,
                speaker_emb: torch.Tensor, para_emb: torch.Tensor,
                audio_emb: torch.Tensor) -> dict:
        """
        Projects all 5 modalities into unified fusion dimension (B, T, fusion_dim).
        """
        return {
            "speech": self.speech_proj(speech_emb),
            "events": self.event_proj(event_emb),
            "speakers": self.speaker_proj(speaker_emb),
            "paralinguistic": self.para_proj(para_emb),
            "audio": self.audio_proj(audio_emb)
        }
