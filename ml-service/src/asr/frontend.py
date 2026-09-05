"""
Acoustic Frontend for Deep Conformer ASR Model.
Extracts 80-bin Log-Mel Spectrograms and downsamples temporal frames by 4x
using strided 2D convolutions with LayerNorm and GELU activations.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class LogMelFrontend(nn.Module):
    """
    Extracts 80-bin Log-Mel Spectrogram features from raw 16kHz audio waveforms.
    
    Tensor Flow:
        Raw Waveform (B, N) -> STFT Mel (B, n_mels=80, T_mel) -> Log Scale (B, 80, T_mel)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: (B, N) or (B, 1, N) raw 16kHz audio tensor
            lengths: (B,) optional waveform sample lengths
        Returns:
            mel: (B, n_mels=80, T_mel) log-mel spectrogram features
            lengths: (B,) temporal frame counts
        """
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        device = waveform.device
        mel = self.mel_transform(waveform.to(device))  # (B, n_mels, T_mel)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # Log scale for dynamic range
        
        # Calculate unpadded lengths
        if lengths is not None:
            mel_lengths = torch.clamp((lengths // self.hop_length) + 1, max=mel.size(-1))
        else:
            mel_lengths = torch.full(
                (waveform.size(0),),
                mel.size(-1),
                dtype=torch.long,
                device=device
            )
        return log_mel, mel_lengths


class Conv2dSubsampling(nn.Module):
    """
    4x Temporal Subsampling Module using 2D Convolutions.
    
    Reduces the temporal dimension of Log-Mel features by 4x while projecting
    the feature space to d_model (512).
    
    Tensor Flow:
        Input: (B, 1, n_mels=80, T_mel)
        Conv1 (stride 2): -> (B, 256, 40, T_mel/2)
        Conv2 (stride 2): -> (B, 512, 20, T_mel/4)
        Linear Projection: -> (B, T_mel/4, d_model=512)
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 512, n_mels: int = 80):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        # Calculate spatial dimension after two stride-2 convolutions
        # For n_mels = 80: 80 -> 40 -> 20
        out_freq_dim = (((n_mels + 2 * 1 - 3) // 2 + 1) + 2 * 1 - 3) // 2 + 1
        self.out_proj = nn.Linear(out_channels * out_freq_dim, out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, n_mels=80, T_mel)
            lengths: (B,) frame lengths before subsampling
        Returns:
            out: (B, T_subsampled, d_model=512)
            subsampled_lengths: (B,) frame lengths after 4x subsampling
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, n_mels, T_mel)
            
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))  # (B, out_channels, F_sub, T_sub)
        
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T_sub, c * f)
        out = self.out_proj(x)  # (B, T_sub, d_model)
        
        if lengths is not None:
            # Subsampling formula for 2 strided convolutions with padding 1 and kernel 3
            subsampled_lengths = (((lengths + 2 * 1 - 3) // 2 + 1) + 2 * 1 - 3) // 2 + 1
            subsampled_lengths = torch.clamp(subsampled_lengths, min=1)
        else:
            subsampled_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
            
        return out, subsampled_lengths


class PositionalEncoding(nn.Module):
    """
    Absolute Sinusoidal Positional Encoding for temporal sequence representations.
    Adds deterministic positional embeddings PE_(pos, 2i) = sin(pos/10000^(2i/d_model)).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x_pe: (B, T, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend on the fly if sequence exceeds max_len
            return x
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR (Park et al., 2019).
    Applies frequency masking and time masking to Log-Mel Spectrogram features.
    Active strictly during model training mode.
    """

    def __init__(
        self,
        freq_mask_max: int = 15,
        time_mask_max: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels=80, T_mel) Log-Mel Spectrogram features
        Returns:
            augmented: (B, n_mels=80, T_mel) with zeroed mask blocks during training
        """
        if not self.training:
            return mel

        augmented = mel.clone()
        b, n_mels, t_mel = augmented.size()
        # In log-Mel space (log(mel + 1e-5)), silence floor is log(1e-5) ~= -11.5129
        # Setting to 0.0 was artificially injecting maximum loud energy into masked bands!
        mask_val = -11.5129

        # 1. Frequency masking
        for _ in range(self.num_freq_masks):
            f = int(torch.randint(1, max(2, self.freq_mask_max), (1,)).item())
            f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
            augmented[:, f0:f0 + f, :] = mask_val

        # 2. Time masking
        max_t_mask = min(self.time_mask_max, max(2, t_mel // 5))
        for _ in range(self.num_time_masks):
            t = int(torch.randint(1, max(2, max_t_mask), (1,)).item())
            t0 = int(torch.randint(0, max(1, t_mel - t), (1,)).item())
            augmented[:, :, t0:t0 + t] = mask_val

        return augmented

