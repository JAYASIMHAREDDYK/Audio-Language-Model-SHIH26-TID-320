import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    """
    Core Audio Spectrogram Encoder.
    Extracts dense frame-level acoustic representation directly from raw audio waveforms or Log-Mel Spectrograms.
    Completely independent of Whisper.
    """
    def __init__(self, in_channels=80, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-stage Convolutional Feature Extractor
        self.spec_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Projection to embed_dim
        self.fc_proj = nn.LazyLinear(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def extract_mel_spectrogram(self, waveform: torch.Tensor, sample_rate: int = 16000, n_mels: int = 80) -> torch.Tensor:
        """
        Converts 1D audio waveform (B, T_samples) to 2D Mel-Spectrogram (B, n_mels, T_frames).
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Simple STFT-based mel energy fallback if torchaudio mel is unavailable
        B, S = waveform.shape
        win_length = 400
        hop_length = 160
        n_fft = 512

        window = torch.hann_window(win_length, device=waveform.device)
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
        spectrogram = torch.abs(stft) ** 2 # (B, freq_bins, T_frames)
        
        # Interpolate frequency dimension to n_mels
        mel_spec = F.interpolate(spectrogram, size=(n_mels, spectrogram.shape[2]), mode='bilinear', align_corners=False) if spectrogram.ndim == 4 else F.interpolate(spectrogram, size=n_mels, mode='linear', align_corners=False).transpose(1, 2)
        if mel_spec.shape[1] != n_mels and mel_spec.shape[2] == n_mels:
            mel_spec = mel_spec.transpose(1, 2)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T_frames) or (B, n_mels, T_frames) or raw waveform (B, T_samples)
        Returns:
            audio_embeddings: (B, T_out, embed_dim)
        """
        if x.ndim == 2:
            # Raw audio waveform (B, T_samples)
            x = self.extract_mel_spectrogram(x)
        
        if x.ndim == 3:
            x = x.unsqueeze(1) # (B, 1, n_mels, T_frames)
            
        conv_out = self.spec_conv(x) # (B, C, F', T')
        B, C, F_dim, T_dim = conv_out.shape
        conv_out = conv_out.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        
        proj = self.fc_proj(conv_out)
        return self.norm(proj)
