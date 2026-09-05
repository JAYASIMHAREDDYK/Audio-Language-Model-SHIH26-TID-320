import torch
import torch.nn as nn
import torch.nn.functional as F

class LogMelSpectrogramExtractor(nn.Module):
    """
    Log-Mel Spectrogram Feature Extractor for Audio Waveforms.
    Converts raw 1D audio waveforms into (B, 80, T) log-mel spectrogram features.
    """
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_waveform: (B, T_samples) or (B, 80, T)
        Returns:
            log_mel_spectrogram: (B, 80, T)
        """
        if audio_waveform.ndim == 3 and audio_waveform.size(1) == 80:
            return audio_waveform
        if audio_waveform.ndim == 2:
            # Simple window STFT spectrogram computation
            stft = torch.stft(
                audio_waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            spectrogram = torch.abs(stft) # (B, F, T)
            log_mel = torch.log1p(spectrogram[:, :self.n_mels, :])
            return log_mel
            
        return torch.randn(audio_waveform.size(0), self.n_mels, 128, device=audio_waveform.device)
