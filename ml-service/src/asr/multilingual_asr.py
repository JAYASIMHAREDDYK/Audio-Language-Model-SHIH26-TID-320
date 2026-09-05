import os
import io
import time
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.asr.preprocessing import (
    load_and_preprocess_audio,
    normalize_language_code,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
)
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.config import ASRConfig
from src.asr.conformer import DeepConformerEncoder
from src.asr.ctc import CTCDecoder


class ASRModel(nn.Module):
    """
    Deep Multilingual Non-Whisper Conformer ASR Model.
    Supports 7 Target Languages: Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, and English.
    
    Architecture:
        - 80-bin Log-Mel Spectrogram Frontend
        - 4x Conv2D Subsampling
        - 12-layer Macaron Conformer Stack (d_model=512, 8 heads, d_ff=2048, conv kernel 31)
        - Connectionist Temporal Classification (CTC) Decoding Head

    Primary Interface for ALM Multimodal Fusion:
        - encode(audio): Extracts (B, T, 512) speech representations for temporal fusion.
        - transcribe(audio, language): Produces multilingual transcription & token alignments.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        hidden_dim: int = 512,
        vocab_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[ASRConfig] = None
    ):
        super().__init__()
        
        # 1. Device Setup
        if device is None:
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_type = str(device)
        self.target_device = torch.device(self.device_type)

        # 2. Multilingual Tokenizer
        self.tokenizer = MultilingualTokenizer(vocab_path=vocab_path)
        self.vocab_size = self.tokenizer.vocab_size()

        # 3. 12-Layer Deep Conformer Speech Representation Encoder
        if config is None:
            config = ASRConfig(d_model=hidden_dim, vocab_size=self.vocab_size)
        self.config = config
        self.hidden_dim = config.d_model
        self.encoder = DeepConformerEncoder(config)

        # 4. CTC Classification & Decoding Head
        self.decoder = CTCDecoder(
            d_model=self.hidden_dim,
            vocab_size=self.vocab_size,
            blank_id=self.tokenizer.blank_id
        )
        self.ctc_decoder = self.decoder

        self.to(self.target_device)

    @property
    def embedding_dim(self) -> int:
        """Returns the speech embedding dimension (d_model=512)."""
        return self.hidden_dim

    # =========================================================================
    # CORE ALM CONTRACT: SPEECH REPRESENTATION ENCODER
    # =========================================================================

    def encode(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO],
        pool: bool = False
    ) -> torch.Tensor:
        """
        Extract high-resolution speech representations for Core ALM Temporal Fusion.
        This is the primary endpoint consumed by the multimodal fusion pipeline.

        Args:
            audio: Audio file path, raw bytes, numpy waveform, or PyTorch tensor.
            pool: If True, returns global pooled embedding (B, 512). If False, returns (B, T, 512) frame embeddings.

        Returns:
            torch.Tensor: Frame-level embeddings of shape (B, T, 512) or pooled (B, 512).
        """
        self.eval()
        with torch.no_grad():
            waveform, _ = self._prepare_audio_tensor(audio)
            frame_embeddings, lengths, pooled_embedding = self.encoder(waveform, return_pooled=pool)
            if pool:
                return pooled_embedding
            return frame_embeddings

    def get_speech_embeddings(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO]
    ) -> Dict[str, torch.Tensor]:
        """
        Rich dictionary output for ALM multimodal fusion pipeline containing both
        frame-level representations (B, T, 512) and pooled sentence embeddings (B, 512).
        """
        self.eval()
        with torch.no_grad():
            waveform, sr = self._prepare_audio_tensor(audio)
            frame_embeddings, lengths, pooled_embedding = self.encoder(waveform, return_pooled=True)
            return {
                "frame_embeddings": frame_embeddings,
                "pooled_embedding": pooled_embedding,
                "embedding_dim": torch.tensor(self.hidden_dim, device=frame_embeddings.device),
                "num_frames": torch.tensor(frame_embeddings.size(1), device=frame_embeddings.device),
                "sample_rate": torch.tensor(sr, device=frame_embeddings.device)
            }

    # =========================================================================
    # MULTILINGUAL TRANSCRIPTION INTERFACE
    # =========================================================================

    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe multilingual audio in any of the 7 supported languages.

        Args:
            audio: Audio input (file path, raw bytes, numpy array, or tensor).
            language: Optional language hint ('hi', 'te', 'ta', 'bn', 'ur', 'zh', 'en').

        Returns:
            Dict containing:
                - text: Decoded transcript string
                - language: Detected or requested language code
                - confidence: Average token confidence score (0.0 to 1.0)
                - duration_seconds: Audio duration in seconds
                - embedding_shape: Shape of the generated frame embeddings
        """
        start_time = time.perf_counter()
        lang_code = normalize_language_code(language) if language else "en"

        self.eval()
        with torch.no_grad():
            waveform, sr = self._prepare_audio_tensor(audio)
            duration = float(waveform.size(-1)) / float(sr)

            # 1. Acoustic encoding via Deep Conformer (B, T_frames, 512)
            frame_embeddings, lengths, _ = self.encoder(waveform, return_pooled=False)

            # 2. CTC projection & greedy decode
            logits = self.decoder(frame_embeddings)
            decode_results = self.decoder.decode_greedy(logits, self.tokenizer, input_lengths=lengths)

            result = decode_results[0] if decode_results else {"text": "", "token_ids": [], "confidence": 0.0}
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            return {
                "text": result.get("text", ""),
                "language": lang_code,
                "confidence": result.get("confidence", 0.0),
                "tokens": [self.tokenizer.id_to_token.get(tid, "") for tid in result.get("token_ids", [])],
                "duration_seconds": round(duration, 3),
                "latency_ms": elapsed_ms,
                "embedding_shape": list(frame_embeddings.shape)
            }

    # =========================================================================
    # TRAINING FORWARD PASS
    # =========================================================================

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass returning logits, frame embeddings, and CTC loss if targets are given.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        raw_audio_lens = audio_lengths if audio_lengths is not None else input_lengths
        frame_embeddings, enc_lengths, pooled_embedding = self.encoder(
            audio, audio_lengths=raw_audio_lens, return_pooled=False
        )
        logits = self.decoder(frame_embeddings)

        outputs = {
            "logits": logits,
            "frame_embeddings": frame_embeddings,
            "lengths": enc_lengths,
            "audio_lengths": enc_lengths,
            "input_lengths": enc_lengths,
            "pooled_embedding": pooled_embedding,
        }

        if targets is not None and target_lengths is not None:
            eff_input_lengths = enc_lengths
            loss = self.decoder.compute_loss(logits, targets, eff_input_lengths, target_lengths)
            outputs["loss"] = loss

        return outputs

    # =========================================================================
    # PARAMETER COUNTING & DIAGNOSTICS
    # =========================================================================

    def count_parameters(self) -> Dict[str, Any]:
        """Calculate exact parameter counts across frontend, encoder blocks, and CTC head."""
        enc_counts = self.encoder.count_parameters()
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        tot_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": tot_params,
            "trainable_parameters": trainable,
            "frontend_parameters": enc_counts["frontend_parameters"],
            "encoder_parameters": enc_counts["encoder_parameters"],
            "decoder_parameters": dec_params,
            "num_layers": self.config.num_layers,
            "d_model": self.config.d_model,
            "num_heads": self.config.num_heads,
            "d_ff": self.config.d_ff,
            "conv_kernel_size": self.config.conv_kernel_size,
        }

    # =========================================================================
    # HELPERS & SERIALIZATION
    # =========================================================================

    def _prepare_audio_tensor(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO]
    ) -> Tuple[torch.Tensor, int]:
        """Standardize any audio input to 16kHz tensor on self.target_device."""
        if isinstance(audio, torch.Tensor) and audio.ndim in (1, 2) and audio.size(-1) > 0:
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            return audio.to(self.target_device, dtype=torch.float32), TARGET_SAMPLE_RATE

        waveform, sr = load_and_preprocess_audio(audio, target_sr=TARGET_SAMPLE_RATE)
        return waveform.to(self.target_device, dtype=torch.float32), sr

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save comprehensive training checkpoint for seamless resumability."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "hidden_dim": self.hidden_dim,
            "vocab": self.tokenizer.token_to_id,
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "metrics": metrics or {},
            "metadata": metadata or {}
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load model state dict, tokenizer vocabulary, optimizer, scheduler, and scaler."""
        checkpoint = torch.load(path, map_location=self.target_device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        if "vocab" in checkpoint:
            self.tokenizer.token_to_id = checkpoint["vocab"]
            self.tokenizer._refresh_mappings()
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        return checkpoint
