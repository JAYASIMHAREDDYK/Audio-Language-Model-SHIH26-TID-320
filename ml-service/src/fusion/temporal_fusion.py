import torch
import torch.nn as nn

class ModalityEmbeddingToken(nn.Module):
    """
    Modality Type Embedding indicator (Speech=0, Events=1, Speakers=2, Paralinguistic=3, Audio=4).
    """
    def __init__(self, num_modalities=5, embed_dim=256):
        super().__init__()
        self.modality_emb = nn.Embedding(num_modalities, embed_dim)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        # x: (B, T, D)
        mod_token = self.modality_emb(torch.tensor(modality_id, device=x.device))
        return x + mod_token.view(1, 1, -1)

class TemporalFusionAdapter(nn.Module):
    """
    Learned Multimodal & Temporal Fusion Adapter.
    Performs dynamic spatio-temporal self-attention & cross-modal fusion over speech, event, speaker,
    paralinguistic, and audio representation streams.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_indicator = ModalityEmbeddingToken(num_modalities=5, embed_dim=embed_dim)
        
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.normal_(self.temporal_pos_emb, std=0.02)
        
        # Concat projection layer to preserve individual modality streams
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Cross-Modal Multi-Head Attention layers
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        # Transformer Encoder for Joint Spatio-Temporal Sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, aligned_dict: dict) -> torch.Tensor:
        """
        Args:
            aligned_dict: Dictionary containing aligned tensors for 'speech', 'events', 'speakers', 'paralinguistic', 'audio'
        Returns:
            fused_multimodal_tensor: (B, T_aligned, embed_dim)
        """
        s = self.modality_indicator(aligned_dict["speech"], 0)
        e = self.modality_indicator(aligned_dict["events"], 1)
        sp = self.modality_indicator(aligned_dict["speakers"], 2)
        p = self.modality_indicator(aligned_dict["paralinguistic"], 3)
        a = self.modality_indicator(aligned_dict["audio"], 4)

        # Preserve distinct modality information via concatenation + projection
        concat_fused = torch.cat([s, e, sp, p, a], dim=-1)
        base_fused = self.fusion_proj(concat_fused)

        # Cross-Attention refinement
        attn_out, _ = self.cross_attn(query=base_fused, key=base_fused, value=base_fused)
        fused = base_fused + attn_out

        # Deep Temporal Transformer Pass
        T_aligned = fused.size(1)
        fused = fused + self.temporal_pos_emb[:, :T_aligned, :]
        fused = self.temporal_transformer(fused)
        return self.norm(fused)
