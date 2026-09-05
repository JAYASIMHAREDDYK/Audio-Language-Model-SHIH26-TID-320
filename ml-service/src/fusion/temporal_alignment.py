import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """
    Continuous Sinusoidal Temporal Positional Encoding for multi-second audio timelines.
    """
    def __init__(self, embed_dim=256, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class TemporalAligner(nn.Module):
    """
    Temporal Aligner for continuous audio streams.
    Synchronizes multi-rate embedding streams (speech, event, speaker, paralinguistic, raw audio)
    onto a unified frame grid with exact timestamp encodings.
    """
    def __init__(self, target_len=128, embed_dim=256):
        super().__init__()
        self.target_len = target_len
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim=embed_dim, max_len=5000)

    def align_feature_stream(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resample input tensor (B, T_in, D) to (B, target_len, D) along the time axis.
        """
        if feat.size(1) == target_len:
            return feat
        # (B, T, D) -> (B, D, T)
        feat_t = feat.transpose(1, 2)
        resampled = F.interpolate(feat_t, size=target_len, mode='linear', align_corners=False)
        return resampled.transpose(1, 2)

    def forward(self, speech_emb: torch.Tensor, event_emb: torch.Tensor,
                speaker_emb: torch.Tensor, para_emb: torch.Tensor,
                audio_emb: torch.Tensor, target_len: int = None) -> dict:
        """
        Args:
            speech_emb: (B, T_s, D)
            event_emb: (B, T_e, D)
            speaker_emb: (B, T_sp, D)
            para_emb: (B, T_p, D)
            audio_emb: (B, T_a, D)
            target_len: optional target time length
        Returns:
            aligned_dict with all modalities aligned to target_len + timestamp encodings.
        """
        T = target_len or self.target_len

        s_aligned = self.align_feature_stream(speech_emb, T)
        e_aligned = self.align_feature_stream(event_emb, T)
        sp_aligned = self.align_feature_stream(speaker_emb, T)
        p_aligned = self.align_feature_stream(para_emb, T)
        a_aligned = self.align_feature_stream(audio_emb, T)

        # Add continuous temporal positional encodings (timestamps)
        s_aligned = self.pos_encoder(s_aligned)
        e_aligned = self.pos_encoder(e_aligned)
        sp_aligned = self.pos_encoder(sp_aligned)
        p_aligned = self.pos_encoder(p_aligned)
        a_aligned = self.pos_encoder(a_aligned)

        return {
            "speech": s_aligned,
            "events": e_aligned,
            "speakers": sp_aligned,
            "paralinguistic": p_aligned,
            "audio": a_aligned,
            "timestamps_seq_len": T
        }
