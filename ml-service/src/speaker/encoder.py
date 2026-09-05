import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    """
    Speaker Biometric Embedding Encoder (ResNet / ECAPA-TDNN based).
    Extracts continuous frame-level and segment-level speaker feature embeddings.
    """
    def __init__(self, in_channels=80, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 80, T) acoustic spectrogram
        Returns:
            speaker_embeddings: (B, T_out, embed_dim)
        """
        if x.ndim == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.ndim == 3 and x.size(2) == 80:
            x = x.transpose(1, 2)
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 80, 1)

        feat = self.conv_layers(x) # (B, embed_dim, T_out)
        feat_t = feat.transpose(1, 2) # (B, T_out, embed_dim)
        proj = self.projection(feat_t)
        return F.normalize(self.layer_norm(proj), p=2, dim=-1)
