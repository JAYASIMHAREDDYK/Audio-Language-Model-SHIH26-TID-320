import pytest
import numpy as np
import torch

from src.asr.preprocessing import (
    load_and_preprocess_audio,
    normalize_text,
    normalize_language_code,
    trim_audio_silence,
    resample_audio_numpy,
    TARGET_SAMPLE_RATE,
)


class TestASRPreprocessing:
    """Test suite for audio standardizer and multilingual text normalizer."""

    def test_audio_resampling_and_channel_mixing(self):
        """Test stereo to mono conversion and resampling to 16 kHz."""
        # 44.1 kHz stereo audio (2, 44100)
        stereo_44k = np.random.randn(2, 44100).astype(np.float32)
        waveform, sr = load_and_preprocess_audio(stereo_44k, orig_sr=44100, target_sr=16000)

        assert sr == 16000
        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 2
        assert waveform.size(0) == 1
        # Approx 1 second -> 16000 samples (+/- 1)
        assert abs(waveform.size(1) - 16000) <= 2

    def test_silence_trimming(self):
        """Test energy-based silence trimming."""
        silence = np.zeros(8000, dtype=np.float32)
        tone = (0.5 * np.sin(np.linspace(0, 10, 8000))).astype(np.float32)
        audio = np.concatenate([silence, tone, silence])

        trimmed = trim_audio_silence(audio, threshold=0.01)
        assert len(trimmed) < len(audio)

    def test_text_normalization_hindi(self):
        """Test Hindi (Devanagari) punctuation and whitespace cleaning."""
        raw = "  पुलिस को जल्दी बुलाओ! यहाँ आपातकाल है... ।  "
        norm = normalize_text(raw, language="hi")
        assert "पुलिस को जल्दी बुलाओ यहाँ आपातकाल है" in norm
        assert "!" not in norm
        assert "।" not in norm

    def test_text_normalization_telugu(self):
        """Test Telugu script normalization."""
        raw = "దయచేసి సహాయం చేయండి!! "
        norm = normalize_text(raw, language="te")
        assert norm == "దయచేసి సహాయం చేయండి"

    def test_text_normalization_tamil(self):
        """Test Tamil script normalization."""
        raw = "தயவுசெய்து உதவுங்கள்!! "
        norm = normalize_text(raw, language="ta")
        assert norm == "தயவுசெய்து உதவுங்கள்"

    def test_text_normalization_bengali(self):
        """Test Bengali script normalization."""
        raw = "দয়া করে সাহায্য করুন! "
        norm = normalize_text(raw, language="bn")
        assert norm == "দয়া করে সাহায্য করুন"

    def test_text_normalization_urdu(self):
        """Test Urdu script diacritic stripping and Arabic char harmonization."""
        # Urdu with Arabic Kaf & diacritics
        raw = "مَدَد کَرو!"
        norm = normalize_text(raw, language="ur")
        assert "!" not in norm
        # Diacritics should be stripped
        assert "\u064E" not in norm  # Fathah removed

    def test_text_normalization_mandarin(self):
        """Test Mandarin Chinese punctuation cleanup."""
        raw = "快救人！这里有危险，请立即报警。"
        norm = normalize_text(raw, language="zh")
        assert "！" not in norm
        assert "，" not in norm
        assert "。" not in norm
        assert "快救人" in norm

    def test_text_normalization_english(self):
        """Test English lowercasing and punctuation stripping."""
        raw = "HELP ME, PLEASE! IT'S AN EMERGENCY..."
        norm = normalize_text(raw, language="en")
        assert norm == "help me please it's an emergency"
