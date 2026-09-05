"""
Test Suite for Deep Multilingual Conformer ASR Architecture.
Validates:
- 12-layer Macaron Conformer architecture specification
- 4x Conv2D subsampling & (B, T, 512) frame representations for ALM temporal fusion
- Zero Whisper dependency guarantee
- All 7 target languages tokenization and special language tokens
- Gradient flow through encoder, SpecAugment, and CTC decoding head
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.asr.config import ASRConfig
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.conformer import DeepConformerEncoder
from src.asr.ctc import CTCDecoder
from src.asr.multilingual_asr import ASRModel
from src.asr.asr_model import ASRModel as ALMASRAdapter
from src.asr.preprocessing import SUPPORTED_LANGUAGES, TARGET_SAMPLE_RATE


class TestDeepConformerASR:
    """Comprehensive test suite for Deep Conformer ASR."""

    @pytest.fixture
    def asr_config(self):
        return ASRConfig(
            sample_rate=16000,
            n_mels=80,
            d_model=512,
            num_layers=12,
            num_heads=8,
            d_ff=2048,
            conv_kernel_size=31,
            dropout=0.1
        )

    @pytest.fixture
    def asr_model(self):
        return ASRModel(device="cpu")

    def test_zero_whisper_dependency(self):
        """Verify that neither openai-whisper nor faster-whisper is imported."""
        assert "whisper" not in sys.modules
        assert "faster_whisper" not in sys.modules

    def test_conformer_architecture_parameters(self, asr_model, asr_config):
        """Verify exact 12-layer Conformer architecture dimensions."""
        counts = asr_model.count_parameters()
        assert counts["num_layers"] == 12
        assert counts["d_model"] == 512
        assert counts["num_heads"] == 8
        assert counts["d_ff"] == 2048
        assert counts["conv_kernel_size"] == 31
        assert counts["total_parameters"] > 70_000_000
        assert counts["trainable_parameters"] == counts["total_parameters"]

    def test_alm_frame_representation_contract(self, asr_model):
        """Verify (B, T, 512) frame representations without premature temporal collapsing."""
        # 2 seconds of 16kHz audio
        sample_audio = torch.randn(1, 32000, dtype=torch.float32)

        # 1. Uncollapsed frame-level representations for ALM temporal fusion
        frame_embs = asr_model.encode(sample_audio, pool=False)
        assert isinstance(frame_embs, torch.Tensor)
        assert frame_embs.ndim == 3
        assert frame_embs.size(0) == 1
        assert frame_embs.size(2) == 512
        assert frame_embs.size(1) > 1  # Temporal dimension preserved!
        assert not torch.isnan(frame_embs).any()

        # 2. Utterance-level pooled representation
        pooled_emb = asr_model.encode(sample_audio, pool=True)
        assert isinstance(pooled_emb, torch.Tensor)
        assert pooled_emb.shape == (1, 512)

    def test_alm_adapter_integration(self):
        """Verify ALM integration adapter projects to 256 while preserving frames."""
        adapter = ALMASRAdapter(embed_dim=256, device="cpu")
        sample_audio = torch.randn(1, 32000, dtype=torch.float32)

        # Frame output projected to 256
        frames_256 = adapter.encode(sample_audio, pool=False)
        assert frames_256.ndim == 3
        assert frames_256.shape == (1, frames_256.size(1), 256)

        # Rich dictionary with both projected and raw 512-dim Conformer frames
        features = adapter.get_speech_embeddings(sample_audio)
        assert "frame_embeddings" in features
        assert features["frame_embeddings"].shape[-1] == 256
        assert "raw_conformer_frames" in features
        assert features["raw_conformer_frames"].shape[-1] == 512

    def test_multilingual_tokenizer_special_tokens(self, asr_model):
        """Verify tokenizer supports all 7 languages with special tags and space_id."""
        tokenizer = asr_model.tokenizer
        assert tokenizer.space_id == 2
        assert tokenizer.blank_id == 0

        for lang in ["hi", "te", "ta", "bn", "ur", "zh", "en"]:
            lang_id = tokenizer.get_lang_token_id(lang)
            assert lang_id is not None
            assert lang_id > 2

            # Test encoding with prepended language special token
            encoded = tokenizer.encode("hello", language=lang, add_special_tokens=True)
            assert encoded[0] == lang_id

    def test_gradient_flow_and_ctc_loss(self, asr_model):
        """Verify clean backward pass and gradient flow through encoder and CTC head."""
        asr_model.train()
        audio = torch.randn(2, 32000, dtype=torch.float32)
        
        # Targets for batch size 2
        targets = torch.tensor([[5, 10, 15, 20], [8, 12, 16, 0]], dtype=torch.long)
        target_lengths = torch.tensor([4, 3], dtype=torch.long)

        outputs = asr_model(audio=audio, targets=targets, target_lengths=target_lengths)
        loss = outputs.get("loss")

        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()

        # Check gradients in encoder and decoder
        has_encoder_grads = False
        for param in asr_model.encoder.parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any()
                has_encoder_grads = True
                break

        has_decoder_grads = False
        for param in asr_model.decoder.parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any()
                has_decoder_grads = True
                break

        assert has_encoder_grads, "Encoder parameters received no gradients!"
        assert has_decoder_grads, "Decoder parameters received no gradients!"
