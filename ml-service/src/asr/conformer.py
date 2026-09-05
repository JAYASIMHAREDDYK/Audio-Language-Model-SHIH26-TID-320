"""
DeepConformerEncoder & Conformer Architecture Modules.
Implements a 12-layer Macaron-style Conformer Encoder combining multi-head self-attention
(global temporal modeling) and depthwise-separable 1D convolution (local acoustic modeling).

Architecture Overview:
----------------------
DeepConformerEncoder
    ├── Input: (B, T_mel, 80)
    ├── 4× Conv2D Subsampling -> (B, T/4, 512)
    ├── Sinusoidal Positional Encoding
    ├── 12 × ConformerBlock:
    │     ├── Macaron FFN 1 (0.5x scaling, d_ff=2048)
    │     ├── Multi-Head Self-Attention (8 Heads, d_k=64)
    │     ├── Depthwise Separable ConvModule (Kernel 31)
    │     ├── Macaron FFN 2 (0.5x scaling, d_ff=2048)
    │     └── LayerNorm (Pre-LN Sandwich Structure)
    └── Output: Frame Embeddings (B, T/4, 512)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.asr.config import ASRConfig
from src.asr.frontend import LogMelFrontend, Conv2dSubsampling, PositionalEncoding, SpecAugment


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit activation."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w(x)) * self.v(x)


class ConformerFeedForward(nn.Module):
    """
    Macaron-Style Conformer Feed-Forward Module.
    Uses SwiGLU / GELU activations and applies an explicit 0.5 half-step scaling multiplier.
    
    Structure:
        LayerNorm -> Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model) -> Dropout -> 0.5 * Output
    """
    
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model) scaled by 0.5 for Macaron structure
        """
        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)
        return 0.5 * x  # Explicit 0.5 Macaron half-step scaling multiplier


class ConformerSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) Module for Global Temporal Modeling.
    Supports padding key masks for variable-length audio sequences.
    
    Parameters:
        d_model: 512 hidden dimension
        num_heads: 8 heads (head dimension 64)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T) boolean mask (True for padded positions)
        Returns:
            out: (B, T, d_model)
        """
        b, t, _ = x.size()
        x_norm = self.layer_norm(x)

        # Multi-head projections
        q = self.q_proj(x_norm).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_k)
        k = self.k_proj(x_norm).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_k)
        v = self.v_proj(x_norm).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T) -> reshape to (B, 1, 1, T)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)  # (B, T, d_model)

        out = self.out_proj(context)
        return self.dropout(out)


class ConformerConvModule(nn.Module):
    """
    Depthwise-Separable Convolution Module for Local Acoustic Modeling.
    
    Structure:
        LayerNorm -> Pointwise Conv1D (d_model -> 2*d_model) -> GLU ->
        Depthwise Conv1D (Kernel 31, Stride 1, Padding 15) -> BatchNorm1d -> SiLU/Swish ->
        Pointwise Conv1D (d_model -> d_model) -> Dropout
    """
    
    def __init__(self, d_model: int = 512, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise Conv 1: Projects d_model to 2*d_model for GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise Conv1D: Local temporal convolution with kernel_size = 31
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1, padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()  # Swish activation
        
        # Pointwise Conv 2: Projects back to d_model
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        x_norm = self.layer_norm(x)
        # Permute to (B, d_model, T) for 1D convolution
        x_conv = x_norm.transpose(1, 2)
        
        x_conv = self.pointwise_conv1(x_conv)  # (B, 2*d_model, T)
        x_conv = self.glu(x_conv)              # (B, d_model, T)
        x_conv = self.depthwise_conv(x_conv)   # (B, d_model, T)
        x_conv = self.batch_norm(x_conv)
        x_conv = self.act(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv)
        
        # Permute back to (B, T, d_model)
        return x_conv.transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Macaron-Style Conformer Encoder Block.
    Combines FFN 1 (0.5x) + Self-Attention + ConvModule + FFN 2 (0.5x) with Pre-LN Residual Connections.
    """
    
    def __init__(self, config: ASRConfig):
        super().__init__()
        self.ffn1 = ConformerFeedForward(config.d_model, config.d_ff, config.dropout)
        self.self_attn = ConformerSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.conv_module = ConformerConvModule(config.d_model, config.conv_kernel_size, config.dropout)
        self.ffn2 = ConformerFeedForward(config.d_model, config.d_ff, config.dropout)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T) padding mask
        Returns:
            out: (B, T, d_model)
        """
        # 1. Macaron Feed-Forward 1 (0.5x) + Residual
        x = x + self.ffn1(x)
        
        # 2. Multi-Head Self-Attention + Residual
        x = x + self.self_attn(x, key_padding_mask=key_padding_mask)
        
        # 3. Depthwise ConvModule + Residual
        x = x + self.conv_module(x)
        
        # 4. Macaron Feed-Forward 2 (0.5x) + Residual
        x = x + self.ffn2(x)
        
        # 5. Final Block Layer Normalization
        return self.final_layer_norm(x)


class DeepConformerEncoder(nn.Module):
    """
    12-Layer Deep Conformer Speech Encoder.
    Processes 80-bin Log-Mel features via 4x Conv2D subsampling, positional encodings,
    and 12 stacked Conformer blocks.
    
    Primary Output:
        Continuous Acoustic Representation H in R^(B x T_frames x 512) for ALM Temporal Fusion.
    """
    
    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        
        # Acoustic Frontend
        self.frontend = LogMelFrontend(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.subsampling = Conv2dSubsampling(
            in_channels=1,
            out_channels=config.d_model,
            n_mels=config.n_mels
        )
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout
        )
        self.spec_augment = SpecAugment()
        self.use_spec_augment = True
        
        # 12 Stacked Conformer Encoder Blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Attentive Global Pooling Head (Optional for Utterance-Level Representation)
        self.pool_attn = nn.Linear(config.d_model, 1)

    def count_parameters(self) -> dict:
        """Calculate exact parameter counts across frontend, encoder blocks, and total system."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frontend_params = sum(p.numel() for p in self.subsampling.parameters())
        encoder_params = sum(p.numel() for p in self.blocks.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frontend_parameters": frontend_params,
            "encoder_parameters": encoder_params,
            "num_layers": self.config.num_layers,
            "d_model": self.config.d_model,
            "num_heads": self.config.num_heads,
            "d_ff": self.config.d_ff,
            "conv_kernel_size": self.config.conv_kernel_size
        }

    def forward(
        self,
        audio_waveform: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        return_pooled: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_waveform: (B, N) or (B, 1, N) 16kHz audio waveform
            audio_lengths: (B,) optional waveform lengths
            return_pooled: If True, also computes attentive global pooled vector (B, 512)
        Returns:
            frame_embeddings: (B, T_frames, 512) frame-level acoustic representation
            lengths: (B,) temporal frame counts
            pooled_embedding: (B, 512) if return_pooled is True, else zeros tensor
        """
        # 1. Log-Mel Extraction: (B, N) -> (B, 80, T_mel)
        mel_features, mel_lengths = self.frontend(audio_waveform, lengths=audio_lengths)
        
        # 2. SpecAugment during training
        if self.training and getattr(self, "use_spec_augment", True):
            mel_features = self.spec_augment(mel_features)
        
        # 3. 4x Temporal Subsampling: -> (B, T_sub, 512)
        x, lengths = self.subsampling(mel_features, mel_lengths)
        
        # 3. Sinusoidal Positional Encoding
        x = self.pos_encoder(x)
        
        # 4. Construct Key Padding Mask for Variable Lengths
        max_len = x.size(1)
        # Create boolean mask where True = Padded Index
        seq_range = torch.arange(max_len, device=x.device).unsqueeze(0)  # (1, T)
        key_padding_mask = seq_range >= lengths.unsqueeze(1)              # (B, T)
        
        # 5. Pass through 12 Stacked Conformer Blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            
        frame_embeddings = self.final_norm(x)  # (B, T_frames, 512)
        
        # 6. Optional Attentive Global Pooling
        if return_pooled:
            attn_weights = self.pool_attn(frame_embeddings)  # (B, T, 1)
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))
            attn_probs = F.softmax(attn_weights, dim=1)
            pooled_embedding = torch.sum(frame_embeddings * attn_probs, dim=1)  # (B, 512)
        else:
            pooled_embedding = torch.zeros((x.size(0), self.config.d_model), device=x.device)
            
        return frame_embeddings, lengths, pooled_embedding
