"""
Configuration module for Deep Conformer ASR System.
Defines hyperparameter specifications for the 12-layer Conformer encoder,
acoustic frontend, and multilingual CTC decoder.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASRConfig:
    """
    Configuration parameters for the Deep Conformer ASR Model.
    
    Architecture Specification:
        - Encoder: 12-layer Macaron Conformer Stack
        - Subsampling: 4x Conv2D depthwise-separable subsampling
        - Dimension: 512 hidden model size
        - Attention: 8 heads (head dimension 64)
        - Feed-Forward: 2048 inner dimension with SwiGLU/GELU activations
        - Convolution: Depthwise-separable 1D convolution with kernel size 31
    """
    # Audio Spectrogram Parameters
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    
    # Model Architecture Parameters
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    d_ff: int = 2048
    conv_kernel_size: int = 31
    dropout: float = 0.1
    
    # Tokenizer & Vocabulary
    vocab_size: int = 500
    blank_id: int = 0
    pad_id: int = 0
    unk_id: int = 1
    
    # Supported Languages
    supported_languages: List[str] = field(
        default_factory=lambda: ["hi", "te", "ta", "bn", "ur", "zh", "en"]
    )
    
    # Training Parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    
    def validate(self) -> None:
        """Validate config parameters to ensure mathematical consistency."""
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.conv_kernel_size % 2 == 1, (
            f"conv_kernel_size ({self.conv_kernel_size}) must be an odd integer for symmetric padding"
        )
