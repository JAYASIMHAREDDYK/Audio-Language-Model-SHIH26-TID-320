import pytest
import sys
import numpy as np
import torch

from src.asr.multilingual_asr import ASRModel
from src.asr.preprocessing import SUPPORTED_LANGUAGES, TARGET_SAMPLE_RATE


class TestMultilingualASRModel:
    """Test suite for Non-Whisper ASRModel and Speech Embeddings."""

    @pytest.fixture
    def asr_model(self):
        return ASRModel(device="cpu")

    def test_no_whisper_dependency(self):
        """Verify that neither openai-whisper nor faster-whisper is imported or required."""
        assert "whisper" not in sys.modules
        assert "faster_whisper" not in sys.modules

    def test_all_seven_languages_supported(self, asr_model):
        """Verify all mandatory target languages are supported in vocabulary and configuration."""
        mandatory_langs = {"hi", "te", "ta", "bn", "mr", "kn", "zh"}
        assert mandatory_langs.issubset(set(SUPPORTED_LANGUAGES))
        for lang in mandatory_langs:
            token_id = asr_model.tokenizer.get_lang_token_id(lang)
            assert token_id != asr_model.tokenizer.unk_id

    def test_encode_speech_embeddings_for_alm_fusion(self, asr_model):
        """Verify ASRModel.encode() produces high-dimensional embeddings for Jay's temporal fusion."""
        # 2 seconds of 16kHz audio
        sample_audio = torch.randn(1, 32000, dtype=torch.float32)

        # 1. Frame-level embeddings: (Batch, Time_Frames, Embedding_Dim)
        frame_embs = asr_model.encode(sample_audio, pool=False)
        assert isinstance(frame_embs, torch.Tensor)
        assert frame_embs.ndim == 3
        assert frame_embs.size(0) == 1
        assert frame_embs.size(2) == asr_model.embedding_dim
        assert not torch.isnan(frame_embs).any()
        assert not torch.isinf(frame_embs).any()

        # 2. Pooled acoustic vector: (Batch, Embedding_Dim)
        pooled_emb = asr_model.encode(sample_audio, pool=True)
        assert isinstance(pooled_emb, torch.Tensor)
        assert pooled_emb.ndim == 2
        assert pooled_emb.shape == (1, asr_model.embedding_dim)
        assert not torch.isnan(pooled_emb).any()

    def test_get_speech_embeddings_dict(self, asr_model):
        """Verify rich dictionary output structure for Core ALM pipeline."""
        sample_audio = np.zeros(16000, dtype=np.float32)
        features = asr_model.get_speech_embeddings(sample_audio)

        assert "frame_embeddings" in features
        assert "pooled_embedding" in features
        assert "embedding_dim" in features
        assert features["embedding_dim"].item() == asr_model.embedding_dim

    def test_transcribe_all_languages(self, asr_model):
        """Verify transcribe() accepts and executes for each of the 7 languages."""
        dummy_audio = torch.randn(1, 16000, dtype=torch.float32)

        for lang in SUPPORTED_LANGUAGES:
            result = asr_model.transcribe(dummy_audio, language=lang)
            assert isinstance(result, dict)
            assert "text" in result
            assert "language" in result
            assert result["language"] == lang
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0
            assert "duration_seconds" in result
            assert result["duration_seconds"] > 0
