import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

from src.asr.preprocessing import load_and_preprocess_audio, TARGET_SAMPLE_RATE, normalize_language_code, normalize_text
from src.asr.tokenizer import MultilingualTokenizer


class MultilingualASRDataset(Dataset):
    """
    Multilingual ASR Dataset loader reading from JSON or JSONL manifests across 7+ languages.
    Supports keys 'audio' or 'audio_filepath', 'text', 'language', 'speaker_id', 'duration'.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: MultilingualTokenizer,
        max_samples: Optional[int] = None,
        target_sr: int = TARGET_SAMPLE_RATE,
        synthetic_fallback: bool = False
    ):
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.synthetic_fallback = synthetic_fallback
        self.samples: List[Dict[str, Any]] = []

        if os.path.exists(manifest_path):
            self.samples = self._load_manifest(manifest_path)
        elif synthetic_fallback:
            # Generate synthetic dummy samples ONLY when explicitly requested for tests
            self.samples = self._generate_synthetic_samples()
        else:
            raise FileNotFoundError(
                f"Dataset manifest not found at '{manifest_path}' and synthetic_fallback is False."
            )

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def _load_manifest(self, path: str) -> List[Dict[str, Any]]:
        """Load manifest from either JSON or JSONL file format."""
        samples: List[Dict[str, Any]] = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("["):
                    samples = json.loads(content)
                elif content.startswith("{"):
                    data = json.loads(content)
                    samples = data.get("samples", [data])
                else:
                    # Fallback line-delimited
                    for line in content.splitlines():
                        if line.strip():
                            samples.append(json.loads(line))
        return samples

    def _generate_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Fallback synthetic dataset for tests and pipeline validation."""
        sample_texts = [
            ("नमस्ते भारत", "hi"),
            ("నమస్కారం ప్రపంచం", "te"),
            ("வணக்கம் உலகம்", "ta"),
            ("নমস্কার বিশ্ব", "bn"),
            ("नमस्कार महाराष्ट्र", "mr"),
            ("ನಮಸ್ಕಾರ ಕರ್ನಾಟಕ", "kn"),
            ("你好世界紧急报警", "zh"),
            ("hello world emergency alert", "en"),
        ]
        items = []
        for text, lang in sample_texts:
            items.append({
                "audio_filepath": f"synthetic_{lang}.wav",
                "audio": f"synthetic_{lang}.wav",
                "text": text,
                "language": lang,
                "speaker_id": f"spk_{lang}_01",
                "duration": 2.0
            })
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def get_duration(self, idx: int) -> float:
        """Return audio duration in seconds for dynamic batching."""
        item = self.samples[idx]
        if "duration" in item and float(item["duration"]) > 0:
            return float(item["duration"])
        # Estimate duration from audio file if duration not recorded
        audio_path = item.get("audio") or item.get("audio_filepath", "")
        if os.path.exists(audio_path):
            try:
                sz = os.path.getsize(audio_path)
                # For 16kHz 16-bit mono PCM: 32,000 bytes/sec
                return max(0.5, float(sz) / 32000.0)
            except Exception:
                pass
        return 3.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_path = item.get("audio") or item.get("audio_filepath", "")
        text = item.get("text", "")
        lang = normalize_language_code(item.get("language", "en"))

        # Load audio or generate dummy tone
        if os.path.exists(audio_path):
            waveform, _ = load_and_preprocess_audio(audio_path, target_sr=self.target_sr)
        elif self.synthetic_fallback:
            # Generate 1.5s dummy sine audio for missing files/tests
            duration = float(item.get("duration", 1.5))
            num_samples = int(duration * self.target_sr)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            sine_wave = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            waveform = torch.from_numpy(sine_wave).unsqueeze(0)
        else:
            raise FileNotFoundError(f"Speech audio file does not exist: {audio_path}")

        clean_text = normalize_text(text, language=lang)
        token_ids = self.tokenizer.encode(clean_text, language=lang, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.space_id]

        return {
            "waveform": waveform.squeeze(0),  # (T,)
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": clean_text,
            "raw_text": text,
            "language": lang,
            "speaker_id": item.get("speaker_id", "unknown"),
            "duration": float(item.get("duration", len(waveform.squeeze(0)) / float(self.target_sr)))
        }


class DynamicBatchSampler(Sampler[List[int]]):
    """
    Duration-aware Dynamic Batch Sampler.
    Groups utterances into batches bounded by `max_batch_audio_seconds` and `max_batch_size`.
    Sorts/buckets utterances by length to minimize padding waste and eliminate VRAM spikes.
    """

    def __init__(
        self,
        dataset: MultilingualASRDataset,
        max_batch_audio_seconds: float = 60.0,
        max_batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.max_audio_seconds = max_batch_audio_seconds
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Precompute lengths / durations
        self.durations: List[float] = [dataset.get_duration(i) for i in range(len(dataset))]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            # Add slight duration jitter to prevent deterministic identical batches every epoch
            indices.sort(key=lambda i: self.durations[i] + rng.uniform(-0.5, 0.5))
        else:
            indices.sort(key=lambda i: self.durations[i])

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_duration: float = 0.0

        for idx in indices:
            dur = self.durations[idx]
            # If adding this item exceeds budget and current batch is non-empty, flush batch
            if current_batch and (
                current_duration + dur > self.max_audio_seconds or len(current_batch) >= self.max_batch_size
            ):
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0

            current_batch.append(idx)
            current_duration += dur

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        # Approximate length estimation
        total_dur = sum(self.durations)
        avg_batch_dur = max(5.0, self.max_audio_seconds * 0.75)
        return max(1, int(total_dur / avg_batch_dur))


def collate_asr_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate variable length audio and token sequences into padded tensors for CTC training.
    """
    waveforms = [item["waveform"] for item in batch]
    token_seqs = [item["token_ids"] for item in batch]
    texts = [item["text"] for item in batch]
    languages = [item["language"] for item in batch]

    # Pad audio waveforms
    audio_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    max_audio_len = max(len(w) for w in waveforms)
    padded_audio = torch.zeros(len(waveforms), max_audio_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded_audio[i, :len(w)] = w

    # Pad target token IDs (use pad_id = 3 for padding, NOT blank_id = 0)
    target_lengths = torch.tensor([len(t) for t in token_seqs], dtype=torch.long)
    max_target_len = max(len(t) for t in token_seqs)
    padded_targets = torch.full((len(token_seqs), max_target_len), fill_value=3, dtype=torch.long)
    for i, t in enumerate(token_seqs):
        padded_targets[i, :len(t)] = t

    return {
        "audio": padded_audio,               # (Batch, Max_Samples)
        "audio_lengths": audio_lengths,       # (Batch,)
        "targets": padded_targets,           # (Batch, Max_Tokens)
        "target_lengths": target_lengths,     # (Batch,)
        "texts": texts,
        "languages": languages
    }

