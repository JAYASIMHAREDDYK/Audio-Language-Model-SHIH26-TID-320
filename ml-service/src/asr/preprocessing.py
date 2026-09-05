import io
import re
import os
import unicodedata
import wave
from typing import Union, Tuple, Optional
import numpy as np
import torch

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

TARGET_SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ["hi", "te", "ta", "bn", "mr", "kn", "zh", "en", "ur"]

LANGUAGE_ALIASES = {
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "bengali": "bn",
    "bangla": "bn",
    "marathi": "mr",
    "kannada": "kn",
    "urdu": "ur",
    "mandarin": "zh",
    "chinese": "zh",
    "cmn": "zh",
    "cmn_hans_cn": "zh",
    "english": "en",
    "hi_in": "hi",
    "te_in": "te",
    "ta_in": "ta",
    "bn_in": "bn",
    "mr_in": "mr",
    "kn_in": "kn",
}


def normalize_language_code(lang: Optional[str]) -> str:
    """Normalize language name or ISO code to standard 2-letter code."""
    if not lang:
        return "en"
    clean = lang.strip().lower()
    return LANGUAGE_ALIASES.get(clean, clean if clean in SUPPORTED_LANGUAGES else "en")


# ==========================================
# 1. AUDIO PREPROCESSING (16 kHz Mono Float32)
# ==========================================

def _read_audio_generic(source: Union[str, os.PathLike, bytes, bytearray, io.BytesIO]) -> Tuple[np.ndarray, int]:
    """Read audio from file path or bytes using soundfile, scipy, or built-in wave."""
    if SOUNDFILE_AVAILABLE:
        try:
            if isinstance(source, (bytes, bytearray)):
                source = io.BytesIO(source)
            data, sr = sf.read(source, dtype="float32")
            return data, sr
        except Exception:
            pass

    if SCIPY_AVAILABLE:
        try:
            if isinstance(source, (bytes, bytearray)):
                source = io.BytesIO(source)
            sr, data = wavfile.read(source)
            # Convert integer types to float32 [-1.0, 1.0]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            return data, sr
        except Exception:
            pass

    # Built-in wave module fallback
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, (str, os.PathLike)):
        source = str(source)

    with wave.open(source, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            data = np.frombuffer(raw_bytes, dtype=np.float32)

        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return data, framerate


def load_and_preprocess_audio(
    audio: Union[str, os.PathLike, bytes, bytearray, io.BytesIO, np.ndarray, torch.Tensor],
    orig_sr: Optional[int] = None,
    target_sr: int = TARGET_SAMPLE_RATE,
    trim_silence: bool = False,
    normalize_volume: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Load and preprocess audio from any format into a standardized (1, T) float32 tensor at target_sr.

    Args:
        audio: Audio file path, bytes buffer, numpy array, or PyTorch tensor.
        orig_sr: Source sample rate if audio is raw numpy/tensor without metadata (default: None).
        target_sr: Standardized sample rate (default: 16000 Hz).
        trim_silence: Whether to trim leading and trailing silence.
        normalize_volume: Whether to apply peak/RMS volume normalization.

    Returns:
        Tuple of (waveform: torch.Tensor of shape (1, num_samples), sample_rate: int)
    """
    sr = orig_sr or target_sr
    waveform_np = None

    if isinstance(audio, (str, os.PathLike, bytes, bytearray, io.BytesIO)):
        waveform_np, detected_sr = _read_audio_generic(audio)
        sr = orig_sr or detected_sr
    elif isinstance(audio, np.ndarray):
        waveform_np = audio.astype(np.float32)
    elif isinstance(audio, torch.Tensor):
        waveform_np = audio.detach().cpu().numpy().astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # Convert shape to 1D float32
    if waveform_np.ndim == 2:
        # Multi-channel audio: convert to mono by averaging channels
        if waveform_np.shape[0] < waveform_np.shape[1] and waveform_np.shape[0] in (1, 2, 4, 6, 8):
            waveform_np = np.mean(waveform_np, axis=0)
        else:
            waveform_np = np.mean(waveform_np, axis=1)
    elif waveform_np.ndim > 2:
        waveform_np = waveform_np.flatten()

    # Resample if needed
    if sr != target_sr:
        waveform_np = resample_audio_numpy(waveform_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Optional silence trimming (energy-based)
    if trim_silence and len(waveform_np) > 0:
        waveform_np = trim_audio_silence(waveform_np)

    # Volume normalization
    if normalize_volume and len(waveform_np) > 0:
        max_val = np.max(np.abs(waveform_np))
        if max_val > 1e-6:
            waveform_np = waveform_np / max_val * 0.95

    # Ensure float32 tensor of shape (1, T)
    waveform_tensor = torch.from_numpy(waveform_np).to(dtype=torch.float32)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    return waveform_tensor, sr


def resample_audio_numpy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resampling without external heavy dependencies."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / float(orig_sr)
    num_target_samples = int(np.round(duration * target_sr))
    if num_target_samples == 0:
        return np.zeros(0, dtype=np.float32)
    orig_times = np.linspace(0.0, duration, len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num_target_samples, endpoint=False)
    resampled = np.interp(target_times, orig_times, audio).astype(np.float32)
    return resampled


def trim_audio_silence(audio: np.ndarray, threshold: float = 0.01, frame_size: int = 512) -> np.ndarray:
    """Trim leading and trailing silence based on energy threshold."""
    if len(audio) < frame_size * 2:
        return audio
    
    # Calculate energy in frames
    num_frames = len(audio) // frame_size
    trimmed = audio[:num_frames * frame_size]
    frames = trimmed.reshape(num_frames, frame_size)
    energy = np.mean(frames ** 2, axis=1)
    
    non_silent_indices = np.where(energy > (threshold ** 2))[0]
    if len(non_silent_indices) == 0:
        return audio
    
    start_idx = max(0, non_silent_indices[0] * frame_size - frame_size)
    end_idx = min(len(audio), (non_silent_indices[-1] + 2) * frame_size)
    return audio[start_idx:end_idx]


# ==========================================
# 2. MULTILINGUAL TEXT NORMALIZATION
# ==========================================

# Urdu diacritics & character mapping
URDU_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
URDU_CHAR_MAP = {
    "\u0643": "\u06A9",  # Arabic Kaf -> Urdu Keheh
    "\u0649": "\u06CC",  # Arabic Alef Maksura -> Urdu Yeh
    "\u064A": "\u06CC",  # Arabic Yeh -> Urdu Yeh
    "\u0629": "\u06C1",  # Arabic Teh Marbuta -> Urdu Gol Heh
    "\u0647": "\u06C1",  # Arabic Heh -> Urdu Gol Heh
}

# Indic punctuation and symbols to clean
INDIC_PUNCT_RE = re.compile(r"[।॥।\.,!?;:\"\'\(\)\[\]\{\}\-—_\\/`~@#$%^&*+=<>|\n\r\t]")
PUNCT_RE = re.compile(r"[\.,!?;:\"\'\(\)\[\]\{\}\-—_\\/`~@#$%^&*+=<>|\n\r\t]")


def normalize_text(text: str, language: str = "en") -> str:
    """
    Standardize text across 7 target languages (hi, te, ta, bn, ur, zh, en).

    Normalizes Unicode representations (NFC/NFKC), removes unwanted punctuation,
    handles script-specific nuances (Devanagari nuktas, Urdu diacritics, Chinese spacing).
    """
    if not text:
        return ""

    lang = normalize_language_code(language)
    
    # Base Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    if lang == "hi":
        return _normalize_hindi(text)
    elif lang == "mr":
        return _normalize_marathi(text)
    elif lang == "kn":
        return _normalize_kannada(text)
    elif lang == "te":
        return _normalize_telugu(text)
    elif lang == "ta":
        return _normalize_tamil(text)
    elif lang == "bn":
        return _normalize_bengali(text)
    elif lang == "ur":
        return _normalize_urdu(text)
    elif lang == "zh":
        return _normalize_mandarin(text)
    else:  # en or fallback
        return _normalize_english(text)


def _normalize_hindi(text: str) -> str:
    """Normalize Hindi (Devanagari script)."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_marathi(text: str) -> str:
    """Normalize Marathi (Devanagari script with Marathi extensions)."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_kannada(text: str) -> str:
    """Normalize Kannada script."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_telugu(text: str) -> str:
    """Normalize Telugu script."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_tamil(text: str) -> str:
    """Normalize Tamil script."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_bengali(text: str) -> str:
    """Normalize Bengali script."""
    text = INDIC_PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_urdu(text: str) -> str:
    """Normalize Urdu (Perso-Arabic script)."""
    text = URDU_DIACRITICS.sub("", text)
    for k, v in URDU_CHAR_MAP.items():
        text = text.replace(k, v)
    text = PUNCT_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_mandarin(text: str) -> str:
    """Normalize Mandarin Chinese (Hanzi character spacing and punctuation)."""
    text = re.sub(r"[，。！？、；：“”‘’（）《》【】—…\.,!?;:\'\"/\\~`@#$%^&*()_+=<>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_english(text: str) -> str:
    """Normalize English text: lowercase and clean punctuation."""
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
