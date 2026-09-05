"""
Lahari Multilingual Non-Whisper ASR Module
Supports 7 Target Languages: Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, and English.
Exports public interfaces including ASRModel and preprocessing functions.
"""

from src.asr.multilingual_asr import ASRModel
from src.asr.encoder import SpeechEncoder
from src.asr.decoder import CTCDecoder
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.preprocessing import (
    load_and_preprocess_audio,
    normalize_text,
    normalize_language_code,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
)
from src.asr.evaluate import calculate_wer, calculate_cer, evaluate_dataset

__all__ = [
    "ASRModel",
    "SpeechEncoder",
    "CTCDecoder",
    "MultilingualTokenizer",
    "load_and_preprocess_audio",
    "normalize_text",
    "normalize_language_code",
    "SUPPORTED_LANGUAGES",
    "TARGET_SAMPLE_RATE",
    "calculate_wer",
    "calculate_cer",
    "evaluate_dataset",
]
