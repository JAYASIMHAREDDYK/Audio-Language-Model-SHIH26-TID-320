import torch
import torch.nn as nn

class ALMFusionAdapter(nn.Module):
    """
    Core ALM Question-Multimodal Fusion Adapter.
    Performs bidirectional cross-attention between user question tokens and fused spatio-temporal audio representations.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.q2audio_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.audio2q_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, proj_fused: torch.Tensor, proj_question: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj_fused: (B, T_audio, embed_dim)
            proj_question: (B, L_q, embed_dim)
        Returns:
            question_conditioned_context: (B, L_q + T_audio, embed_dim)
        """
        # Question query over audio context
        q_attn, _ = self.q2audio_attn(query=proj_question, key=proj_fused, value=proj_fused)
        q_context = self.norm1(proj_question + q_attn)

        # Audio query over question context
        a_attn, _ = self.audio2q_attn(query=proj_fused, key=proj_question, value=proj_question)
        a_context = self.norm2(proj_fused + a_attn)

        # Concatenate into full question-conditioned sequence
        joint_seq = torch.cat([q_context, a_context], dim=1) # (B, L_q + T_audio, embed_dim)
        
        out = self.norm3(joint_seq + self.mlp(joint_seq))
        return out
